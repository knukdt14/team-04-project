import os

import pymupdf4llm
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import CONFIG, MD_CACHE_PATH, PDF_PATH


def load_and_split():
    """기존 방식: PyPDFLoader(페이지 단위 원문 추출) + 일반 텍스트 분할."""
    assert os.path.exists(PDF_PATH), f"PDF 파일을 찾을 수 없습니다: {PDF_PATH} (경로를 확인하세요)"
    print(f"PDF 경로 확인 완료: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"로드된 페이지 수: {len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
    )
    splits = text_splitter.split_documents(docs)
    print(f"청킹 완료: {len(splits)}개 조각 (chunk_size={CONFIG['chunk_size']}, overlap={CONFIG['chunk_overlap']})")
    return splits


def load_and_split_markdown():
    """PDF를 마크다운으로 변환한 뒤 분할 — 페이지 경계에서 조문이 끊기는 문제 완화.
    chunk_size/overlap은 기존과 동일하게 유지, 분할 기준만 마크다운 구조를 인식하도록 변경."""
    assert os.path.exists(PDF_PATH), f"PDF 파일을 찾을 수 없습니다: {PDF_PATH} (경로를 확인하세요)"
    print(f"PDF 경로 확인 완료: {PDF_PATH}")

    if os.path.exists(MD_CACHE_PATH):
        with open(MD_CACHE_PATH, encoding="utf-8") as f:
            md_text = f.read()
        print(f"마크다운 캐시 재사용: {MD_CACHE_PATH} ({len(md_text)}자) — PDF 재변환 생략")
    else:
        md_text = pymupdf4llm.to_markdown(PDF_PATH)
        os.makedirs(os.path.dirname(MD_CACHE_PATH), exist_ok=True)
        with open(MD_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"마크다운 변환 완료 및 캐시 저장: {len(md_text)}자 -> {MD_CACHE_PATH}")

    docs = [Document(page_content=md_text, metadata={"source": PDF_PATH})]

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
    )
    splits = text_splitter.split_documents(docs)
    print(f"청킹 완료: {len(splits)}개 조각 (chunk_size={CONFIG['chunk_size']}, overlap={CONFIG['chunk_overlap']})")
    return splits
