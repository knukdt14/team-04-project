import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CONFIG, PDF_PATH


def load_and_split():
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
