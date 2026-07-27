import torch
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from config import CONFIG


def build_embeddings(cfg=None):
    cfg = cfg or CONFIG
    # 팀 공통 baseline 임베딩(B 담당 축) — 여기서는 고정해서 사용
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg["embedding_model"],
        model_kwargs={"device": "cpu"},
    )
    print(f"임베딩 모델 로드 완료: {cfg['embedding_model']}")
    return embeddings


def build_vectorstore(splits, embeddings=None, collection_name=None, cfg=None):
    cfg = cfg or CONFIG
    embeddings = embeddings or build_embeddings(cfg)
    collection_name = collection_name or cfg["run_name"]

    # collection_name은 청킹 설정마다 달라야 함 — 같은 이름을 재사용하면
    # 청크 구성이 다른 데이터가 섞여 들어가면서 이전 실험 결과가 오염됨
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
    )
    print("벡터스토어 구축 완료 (Chroma)")
    return vectorstore


def build_retriever(vectorstore, cfg=None):
    cfg = cfg or CONFIG
    retriever = vectorstore.as_retriever(
        search_type=cfg["search_type"],
        search_kwargs=cfg["search_kwargs"],
    )
    print(f"Retriever 설정 완료: search_type={cfg['search_type']}, kwargs={cfg['search_kwargs']}")
    return retriever


def build_llm(cfg=None):
    # 팀 baseline과 동일 — 로컬 GPU에 직접 다운로드해서 4bit 양자화로 실행
    # (B의 rag_chain.py와 동일 방식. HF Inference API/OpenAI 크레딧 문제 회피)
    cfg = cfg or CONFIG

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["llm_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["llm_model"],
        quantization_config=quantization_config,
        device_map="auto",
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=cfg["temperature"] > 0,
        **({"temperature": cfg["temperature"]} if cfg["temperature"] > 0 else {}),
        return_full_text=False,
    )

    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen_pipeline))
    print(f"LLM 로드 완료 (로컬 GPU, 4bit): {cfg['llm_model']}")
    return llm
