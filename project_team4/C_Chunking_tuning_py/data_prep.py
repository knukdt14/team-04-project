import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from config import CONFIG, PDF_PATH


def build_splitter(cfg=None, embeddings=None):
    """cfg["splitter"]에 따라 알맞은 TextSplitter를 만들어 반환.

    semantic 전략은 임베딩 유사도로 자르기 때문에 embeddings 객체가 필요합니다
    (models.build_embeddings()로 만든 것을 그대로 넘겨서 재사용하세요).
    """
    cfg = cfg or CONFIG
    kind = cfg["splitter"]

    if kind == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"]
        )
    elif kind == "character":
        return CharacterTextSplitter(
            chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"], separator="\n"
        )
    elif kind == "token":
        return TokenTextSplitter(
            chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"]
        )
    elif kind == "semantic":
        # 임베딩 기반 의미 단위 분할 — langchain_experimental 필요
        # (해당 패키지는 유지보수 종료(sunset) 예정이니 추후 대체 라이브러리 검토 필요)
        from langchain_experimental.text_splitter import SemanticChunker

        if embeddings is None:
            raise ValueError(
                "semantic 분할은 embeddings 객체가 필요합니다. "
                "models.build_embeddings()로 만든 뒤 이 함수에 넘겨주세요."
            )
        return SemanticChunker(
            embeddings,
            breakpoint_threshold_type=cfg.get("semantic_breakpoint_type", "percentile"),
            breakpoint_threshold_amount=cfg.get("semantic_breakpoint_amount", 95),
        )
    else:
        raise ValueError(f"알 수 없는 splitter: {kind}")


def load_and_split(cfg=None, embeddings=None):
    cfg = cfg or CONFIG
    assert os.path.exists(PDF_PATH), f"PDF 파일을 찾을 수 없습니다: {PDF_PATH} (경로를 확인하세요)"
    print(f"PDF 경로 확인 완료: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"로드된 페이지 수: {len(docs)}")

    splitter = build_splitter(cfg, embeddings=embeddings)
    splits = splitter.split_documents(docs)
    print(
        f"청킹 완료: {len(splits)}개 조각 "
        f"(splitter={cfg['splitter']}, chunk_size={cfg.get('chunk_size')}, overlap={cfg.get('chunk_overlap')})"
    )
    return splits
