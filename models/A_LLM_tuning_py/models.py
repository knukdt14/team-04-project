import gc

import torch
from langchain_chroma import Chroma
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from config import CONFIG

# 이 PC에 NVIDIA GPU(8GB VRAM)가 있어 GPU + 4bit 양자화로 실행. embedding은 담당 축이 아니므로 BASELINE 고정.


def build_vectorstore(splits, collection_name=None):
    collection_name = collection_name or CONFIG["run_name"]

    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    print(f"임베딩 모델 로드 완료 (고정): {CONFIG['embedding_model']}")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
    )
    print("벡터스토어 구축 완료 (Chroma)")
    return vectorstore


def build_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type=CONFIG["search_type"],
        search_kwargs=CONFIG["search_kwargs"],
    )
    print(f"Retriever 설정 완료 (고정): search_type={CONFIG['search_type']}, kwargs={CONFIG['search_kwargs']}")
    return retriever


# 모델(가중치)은 llm_model 단위로만 캐시 — temperature는 생성 파라미터일 뿐 가중치와 무관하므로
# 같은 모델을 temperature만 바꿔 여러 번 써도 GPU에 한 번만 올라감.
_model_cache = {}


def _load_model(llm_model):
    if llm_model in _model_cache:
        return _model_cache[llm_model]

    tokenizer = AutoTokenizer.from_pretrained(llm_model)

    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(llm_model)

    _model_cache[llm_model] = (model, tokenizer)
    print(f"모델 로드 완료 ({'GPU 4bit' if torch.cuda.is_available() else 'CPU'}): {llm_model}")
    return model, tokenizer


def clear_llm_cache():
    """다음 모델을 위해 GPU 메모리를 비움 (여러 모델을 순차 비교할 때 VRAM 누적 방지).

    주의: 이 함수를 호출하는 쪽에서 llm/rag_chain 등 모델을 참조하는 변수를 먼저
    del 하거나 스코프 밖으로 내보내야 실제로 GPU 메모리가 회수됨 (파이썬 참조가 남아있으면
    empty_cache()를 불러도 소용없음).
    """
    _model_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("모델 캐시 비움 (GPU 메모리 해제)")


def build_llm(llm_model=None, temperature=None):
    """A 담당 축: llm_model과 temperature를 바꿔가며 비교."""
    llm_model = llm_model or CONFIG["llm_model"]
    temperature = CONFIG["temperature"] if temperature is None else temperature

    model, tokenizer = _load_model(llm_model)

    pipeline_kwargs = {"max_new_tokens": 512, "return_full_text": False}
    if temperature and temperature > 0:
        pipeline_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        pipeline_kwargs.update({"do_sample": False})  # greedy decoding

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs,
    )
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen_pipeline))
    print(f"LLM 준비 완료: {llm_model} (temperature={temperature})")
    return llm
