import re

import torch
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from peft import PeftModel
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
    """embedding_model_name에 파인튜닝된 로컬 모델 경로를 넘기면 그 모델로 임베딩."""
    embedding_model_name = embedding_model_name or CONFIG["base_embedding_model"]
    collection_name = collection_name or _default_collection_name(embedding_model_name)

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


def build_retriever(vectorstore, splits=None, strategy=None):
    strategy = strategy or CONFIG["retrieval_strategy"]
    if strategy == "dense":
        return build_dense_retriever(vectorstore)
    if strategy == "hybrid":
        assert splits is not None, "hybrid 전략은 BM25 구축을 위해 splits가 필요합니다"
        return build_hybrid_retriever(vectorstore, splits, weights=CONFIG["hybrid_weights"])
    raise ValueError(f"알 수 없는 retrieval_strategy: {strategy}")


_HAN_PATTERN = re.compile(r"[一-鿿㐀-䶿]")  # CJK 한자(중국어) — 한글(Hangul)과는 다른 유니코드 블록


def _han_token_ids(tokenizer):
    vocab_size = len(tokenizer)
    decoded = tokenizer.batch_decode([[i] for i in range(vocab_size)])
    return [i for i, s in enumerate(decoded) if _HAN_PATTERN.search(s)]


def load_base_model_and_tokenizer(llm_model_name=None):
    """4bit 양자화된 베이스 모델 + 토크나이저 로드. QLoRA 학습/평가 양쪽에서 재사용."""
    llm_model_name = llm_model_name or CONFIG["base_llm_model"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    return model, tokenizer


def build_llm(llm_model_name=None, lora_adapter_path=None):
    """lora_adapter_path를 넘기면 베이스 모델 위에 파인튜닝된 LoRA 어댑터를 얹어서 로드."""
    llm_model_name = llm_model_name or CONFIG["base_llm_model"]
    model, tokenizer = load_base_model_and_tokenizer(llm_model_name)

    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print(f"LoRA 어댑터 적용 완료: {lora_adapter_path}")

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
    tag = f"{llm_model_name} + LoRA({lora_adapter_path})" if lora_adapter_path else llm_model_name
    print(f"LLM 로드 완료 (로컬 GPU, 4bit): {tag}")
    return llm
