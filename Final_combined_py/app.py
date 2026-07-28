"""자동차관리법 RAG 챗봇 — Streamlit UI.
실행: cd Final_combined_py && streamlit run app.py
"""
import time

import streamlit as st
from dotenv import load_dotenv

from config import CONFIG
from data_prep import load_and_split_markdown
from models import build_llm, build_retriever, build_vectorstore
from rag_chain import build_rag_chain

load_dotenv()

st.set_page_config(page_title="자동차관리법 챗봇", page_icon="🚗", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f6f8fb 0%, #eef1f7 100%); color: #1f2937; }
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label { color: #1f2937; }
    .hero {
        text-align: center;
        padding: 1.6rem 1rem 0.4rem 1rem;
    }
    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #2b5fd9, #6c3ce9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0;
    }
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 0.6rem 0.4rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        text-align: center;
    }
    div[data-testid="stMetric"] * { color: #1f2937 !important; }
    div[data-testid="stChatMessage"] {
        background: white;
        border-radius: 14px;
        padding: 0.6rem 0.8rem;
        color: #1f2937;
    }
    div[data-testid="stChatMessage"] p { color: #1f2937; }
    section[data-testid="stSidebar"] { color: #1f2937; }
    section[data-testid="stSidebar"] * { color: #1f2937; }
    .example-caption {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="RAG 파이프라인 로딩 중 (최초 1회, LLM 4bit 로드 포함 약 30초~1분)...")
def load_pipeline():
    splits = load_and_split_markdown()
    vectorstore = build_vectorstore(splits)
    retriever = build_retriever(vectorstore, splits)
    llm = build_llm()
    return build_rag_chain(retriever, llm)


rag_chain = load_pipeline()

st.markdown(
    """
    <div class="hero">
        <h1>🚗 자동차관리법 챗봇</h1>
        <p>RAG 기반 법률 QA · KoE5 임베딩 + Qwen2.5-7B(QLoRA 파인튜닝) + Hybrid 검색</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("BERTScore F1", "0.7020")
col2.metric("임베딩", "KoE5")
col3.metric("LLM", "Qwen2.5-7B+QLoRA")
col4.metric("검색 전략", "Hybrid 0.3/0.7")

st.divider()

EXAMPLE_QUESTIONS = [
    "자동차 튜닝(구조·장치 변경)을 할 때 사전 승인이 필요한 항목은 뭔가요?",
    "정기검사 기간을 놓치면 과태료가 얼마나 나오나요?",
    "중고차 살 때 성능·상태 점검기록부 보증 기간은 얼마나 되나요?",
    "차를 폐차할 때 말소등록은 언제까지 신청해야 하나요?",
]

with st.sidebar:
    st.header("ℹ️ 파이프라인 정보")
    st.markdown(
        f"""
        - **임베딩**: `{CONFIG['embedding_model']}`
        - **LLM**: `{CONFIG['llm_model']}` (QLoRA 파인튜닝 적용)
        - **청킹**: `{CONFIG['chunk_size']}자 / overlap {CONFIG['chunk_overlap']}`
        - **검색 전략**: `hybrid` (BM25 {CONFIG['hybrid_weights'][0]} / Dense {CONFIG['hybrid_weights'][1]})
        - **평가 결과**: BERTScore F1 **0.7020** (10문항 기준)
        """
    )
    st.divider()
    st.header("💡 예시 질문")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"ex_{q[:10]}"):
            st.session_state.pending_question = q
    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

AVATARS = {"user": "🙋", "assistant": "🚗"}

if not st.session_state.messages:
    st.markdown('<p class="example-caption">왼쪽 사이드바의 예시 질문을 눌러보거나, 아래 입력창에 직접 질문해보세요.</p>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])

typed_question = st.chat_input("자동차관리법에 대해 궁금한 점을 물어보세요")
question = st.session_state.pop("pending_question", None) or typed_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("답변 생성 중..."):
            start = time.time()
            answer = rag_chain.invoke(question)
            elapsed = time.time() - start
        st.markdown(answer)
        st.caption(f"⏱️ 응답시간: {elapsed:.1f}초")
    st.session_state.messages.append({"role": "assistant", "content": answer})
