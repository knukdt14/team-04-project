import torch
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
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

from config import CONFIG

# 법 조문 질문은 본문 요건(예: 제34조)과 벌칙/과태료(예: 제81조) 조항이
# 문서 내에서 멀리 떨어져 있는 경우가 많아, 원 질문 하나만으로는 top-k 검색이
# 둘 중 하나를 놓치기 쉬움. 서로 다른 관점의 하위 질의를 만들어 함께 검색해
# 커버리지를 넓히려는 목적.
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "당신은 법률 질문을 검색에 유리하게 다듬어주는 어시스턴트입니다.\n"
        "아래 질문에 대해 벡터 검색으로 관련 법 조문을 더 잘 찾을 수 있도록,\n"
        "서로 다른 관점의 하위 질문 2개를 한국어로 만들어 주세요.\n"
        "- 하나는 요건/절차 관점\n"
        "- 하나는 벌칙·과태료·처벌 관점\n"
        "각 질문은 한 줄에 하나씩만 작성하고, 번호나 설명 없이 질문 문장만 출력하세요.\n"
        "원본 질문: {question}"
    ),
)


def build_vectorstore(splits, embedding_model_name=None, distance_metric=None, collection_name=None):
    embedding_model_name = embedding_model_name or CONFIG["embedding_model"]
    distance_metric = distance_metric or CONFIG["distance_metric"]
    collection_name = collection_name or CONFIG["run_name"]

    # 임베딩은 CPU에서 실행 — GPU는 LLM(4bit)이 통째로 쓸 수 있게 비워둠
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
    )
    print(f"임베딩 모델 로드 완료: {embedding_model_name}")

    # collection_name은 distance_metric마다 달라야 함 — 같은 이름을 재사용하면
    # 이전 실행의 인덱스(hnsw:space)가 그대로 남아있어 metric이 안 바뀐 것처럼 보일 수 있음
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": distance_metric},
    )
    print(f"벡터스토어 구축 완료 (Chroma, distance_metric={distance_metric})")
    return vectorstore


def build_retriever(vectorstore, search_type=None, search_kwargs=None):
    search_type = search_type or CONFIG["search_type"]
    search_kwargs = dict(search_kwargs or CONFIG["search_kwargs"])

    if search_type == "similarity_score_threshold":
        search_kwargs.setdefault("score_threshold", 0.5)
    elif search_type == "mmr":
        search_kwargs.setdefault("fetch_k", max(search_kwargs.get("k", 3) * 2, 10))

    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    print(f"Retriever 설정 완료: search_type={search_type}, kwargs={search_kwargs}")
    return retriever


def build_multi_query_retriever(retriever, query_llm):
    # 원 질문 + 하위 질문 2개로 각각 검색 후 합집합(중복 제거)을 반환.
    # query_llm은 max_new_tokens을 짧게 잡은 전용 LLM — 최종 답변용 llm과는 다름
    # (하위 질문 몇 줄 생성하는 데 256토큰까지 쓸 필요가 없어 응답 시간을 아낌).
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=query_llm,
        prompt=MULTI_QUERY_PROMPT,
        include_original=True,
    )
    print("멀티쿼리 Retriever 구성 완료 (하위 질문 2개 + 원본)")
    return multi_retriever


def _build_chat_llm(model, tokenizer, max_new_tokens):
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=CONFIG["temperature"] > 0,
        **({"temperature": CONFIG["temperature"]} if CONFIG["temperature"] > 0 else {}),
        return_full_text=False,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen_pipeline))


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

    # 모델 가중치는 한 번만 로드하고, 파이프라인만 두 개(생성 길이가 다름)로 나눠 재사용
    # → VRAM은 추가로 안 쓰면서 하위 질문 생성 호출을 훨씬 짧게 끝냄
    llm = _build_chat_llm(model, tokenizer, max_new_tokens=256)
    query_llm = _build_chat_llm(model, tokenizer, max_new_tokens=80)
    print(f"LLM 로드 완료 (로컬 GPU, 4bit): {CONFIG['llm_model']}")
    return llm, query_llm
