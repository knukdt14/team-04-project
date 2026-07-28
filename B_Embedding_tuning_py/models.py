import re

import torch
from langchain_chroma import Chroma
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
    # collection_name은 임베딩 모델마다 달라야 함 — 같은 이름을 재사용하면
    # 벡터 차원이 다른 임베딩이 섞여 들어가면서 Chroma가 크래시남
    return CONFIG["run_name"] + "_" + re.sub(r"[^0-9a-zA-Z]+", "_", embedding_model_name)


def build_vectorstore(splits, embedding_model_name=None, collection_name=None):
    embedding_model_name = embedding_model_name or CONFIG["embedding_model"]
    collection_name = collection_name or _default_collection_name(embedding_model_name)

    # 임베딩은 CPU에서 실행 — GPU는 LLM(4bit)이 통째로 쓸 수 있게 비워둠
    # normalize_embeddings=True: 코사인 유사도 기준 검색에 권장(측정상 F1엔 중립적이었지만 해될 것 없음)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"임베딩 모델 로드 완료: {embedding_model_name}")

    # persist_directory에 컬렉션이 이미 있고 청크 수가 같으면 재임베딩하지 않고 그대로 재사용
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


def build_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type=CONFIG["search_type"],
        search_kwargs=CONFIG["search_kwargs"],
    )
    print(f"Retriever 설정 완료: search_type={CONFIG['search_type']}, kwargs={CONFIG['search_kwargs']}")
    return retriever


_HAN_PATTERN = re.compile(r"[一-鿿㐀-䶿]")  # CJK 한자(중국어) — 한글(Hangul)과는 다른 유니코드 블록


def _han_token_ids(tokenizer):
    # 어휘 전체를 훑어서 한자가 포함된 토큰 id만 골라냄 (한글 답변에는 영향 없음)
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
