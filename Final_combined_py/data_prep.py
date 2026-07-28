import os

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import CONFIG, MD_CACHE_PATH, PDF_PATH


def load_and_split_markdown(chunk_size=None, chunk_overlap=None):
    """PDF -> 마크다운 변환 후 문자 수 기준 분할.
    chunk_size/overlap을 생략하면 CONFIG 값(800/100)을 사용."""
    chunk_size = chunk_size or CONFIG["chunk_size"]
    chunk_overlap = chunk_overlap if chunk_overlap is not None else CONFIG["chunk_overlap"]

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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(docs)
    print(f"청킹 완료: {len(splits)}개 조각 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return splits
