import os
import re

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import CONFIG, MD_CACHE_PATH, PDF_PATH

# "제34조(...)", "- 제27조의2(...)" 처럼 줄 시작에 오는 조문 번호를 감지 (C_Chunking_tuning_py와 동일)
_ARTICLE_SPLIT_PATTERN = re.compile(r"(?m)(?=^(?:- )?제\d+조(?:의\d+)?\()")


def _load_markdown():
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
    return md_text


def load_and_split_article():
    """조문("제O조") 단위로 먼저 분할 -> chunk_size를 넘는 조문만 하위 분할.
    합성 QA 데이터 생성 시 그라운딩 단위로 사용 (조문 하나 = 의미적으로 완결된 단위)."""
    md_text = _load_markdown()

    raw_parts = _ARTICLE_SPLIT_PATTERN.split(md_text)
    raw_parts = [p for p in raw_parts if p.strip()]
    print(f"[article] 조문 단위 1차 분할: {len(raw_parts)}개 조각")

    sub_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
    )

    docs = []
    for part in raw_parts:
        if len(part) <= CONFIG["chunk_size"]:
            docs.append(Document(page_content=part, metadata={"source": PDF_PATH}))
        else:
            docs.extend(sub_splitter.split_documents(
                [Document(page_content=part, metadata={"source": PDF_PATH})]
            ))

    print(f"[article] 최종 청킹 완료: {len(docs)}개 조각 (초과 조문만 chunk_size={CONFIG['chunk_size']}로 하위 분할)")
    return docs


def load_and_split_markdown(chunk_size=None, chunk_overlap=None):
    """RAG 평가용: 문자 수 기준 분할 (Final_combined_py와 동일한 chunk_800/100 baseline)."""
    chunk_size = chunk_size or CONFIG["chunk_size"]
    chunk_overlap = chunk_overlap if chunk_overlap is not None else CONFIG["chunk_overlap"]

    md_text = _load_markdown()
    docs = [Document(page_content=md_text, metadata={"source": PDF_PATH})]

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(docs)
    print(f"[character] 청킹 완료: {len(splits)}개 조각 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return splits
