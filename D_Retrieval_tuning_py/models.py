import re

import torch
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from config import CHROMA_DIR, CONFIG


def _default_collection_name(embedding_model_name):
    return CONFIG["run_name"] + "_" + re.sub(r"[^0-9a-zA-Z]+", "_", embedding_model_name)


def build_vectorstore(splits, embedding_model_name=None, collection_name=None):
    embedding_model_name = embedding_model_name or CONFIG["embedding_model"]
    collection_name = collection_name or _default_collection_name(embedding_model_name)

    # 임베딩은 CPU에서 실행 — GPU는 LLM(4bit)이 통째로 쓸 수 있게 비워둠
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"임베딩 모델 로드 완료: {embedding_model_name}")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    existing_count = vectorstore._collection.count()

    if existing_count == len(splits):
        print(f"벡터스토어 캐시 재사용 (컬렉션: {collection_name}, {existing_count}개 청크) — 재임베딩 생략")
        return vectorstore

    if existing_count > 0:
        print(f"캐시된 청크 수({existing_count})가 현재 청크 수({len(splits)})와 달라 다시 구축합니다")
        vectorstore.delete_collection()
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

    vectorstore.add_documents(splits)
    print(f"벡터스토어 구축 및 캐시 저장 완료 (Chroma, {len(splits)}개 청크)")
    return vectorstore


def build_dense_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type=CONFIG["search_type"],
        search_kwargs=CONFIG["search_kwargs"],
    )
    print(f"Dense Retriever 설정 완료: search_type={CONFIG['search_type']}, kwargs={CONFIG['search_kwargs']}")
    return retriever


def build_hybrid_retriever(vectorstore, splits, weights=None):
    """BM25(키워드 기반) + Dense(임베딩 기반) 앙상블 검색."""
    weights = weights or CONFIG["hybrid_weights"]
    dense_retriever = build_dense_retriever(vectorstore)

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = CONFIG["search_kwargs"]["k"]

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=weights,
    )
    print(f"Hybrid Retriever(BM25+Dense) 설정 완료: weights={weights}, k={CONFIG['search_kwargs']['k']}")
    return ensemble


_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANK_CANDIDATE_K = 10  # 1차로 넉넉히 뽑은 뒤 reranker로 top_k만 추림


def build_reranked_retriever(vectorstore):
    """Dense로 넉넉히(k=10) 뽑은 뒤 Cross-Encoder reranker로 top_k만 추리는 전략."""
    base_retriever = vectorstore.as_retriever(
        search_type=CONFIG["search_type"],
        search_kwargs={"k": _RERANK_CANDIDATE_K},
    )
    cross_encoder = HuggingFaceCrossEncoder(model_name=_RERANKER_MODEL, model_kwargs={"device": "cpu"})
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=CONFIG["search_kwargs"]["k"])
    retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
    print(f"Reranked Retriever 설정 완료: candidate_k={_RERANK_CANDIDATE_K} -> reranker({_RERANKER_MODEL}) -> top_k={CONFIG['search_kwargs']['k']}")
    return retriever


def build_retriever(vectorstore, splits=None, strategy=None):
    strategy = strategy or CONFIG["retrieval_strategy"]
    if strategy == "dense":
        return build_dense_retriever(vectorstore)
    if strategy == "dense_rerank":
        return build_reranked_retriever(vectorstore)
    if strategy.startswith("hybrid"):
        assert splits is not None, "hybrid 전략은 BM25 구축을 위해 splits가 필요합니다"
        if strategy == "hybrid":
            return build_hybrid_retriever(vectorstore, splits)
        # "hybrid_0.3_0.7" 같은 이름에서 가중치 파싱
        parts = strategy.split("_")[1:]
        weights = [float(p) for p in parts]
        return build_hybrid_retriever(vectorstore, splits, weights=weights)
    raise ValueError(f"알 수 없는 retrieval_strategy: {strategy}")


_HAN_PATTERN = re.compile(r"[一-鿿㐀-䶿]")  # CJK 한자(중국어) — 한글(Hangul)과는 다른 유니코드 블록


def _han_token_ids(tokenizer):
    vocab_size = len(tokenizer)
    decoded = tokenizer.batch_decode([[i] for i in range(vocab_size)])
    return [i for i, s in enumerate(decoded) if _HAN_PATTERN.search(s)]


def build_llm():
    # 로컬 GPU에 직접 다운로드해서 실행 (HF Inference API 크레딧 소진으로 로컬 실행으로 전환)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model"])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["llm_model"],
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Qwen 계열 모델이 간혹 답변 도중 중국어로 전환되는 현상 방지 —
    # 한자 토큰을 생성 후보에서 아예 제외 (프롬프트 지시보다 훨씬 확실함)
    suppress_tokens = _han_token_ids(tokenizer)
    print(f"한자 토큰 {len(suppress_tokens)}개 생성 억제 설정 완료")

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=CONFIG["temperature"] > 0,
        **({"temperature": CONFIG["temperature"]} if CONFIG["temperature"] > 0 else {}),
        suppress_tokens=suppress_tokens,
        return_full_text=False,
    )

    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen_pipeline))
    print(f"LLM 로드 완료 (로컬 GPU, 4bit): {CONFIG['llm_model']}")
    return llm
