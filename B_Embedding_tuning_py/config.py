from pathlib import Path

# 이 파일(B_Embedding_tuning_py/) 기준 프로젝트 루트(Project_team1/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CONFIG — Embedding 모델 담당 (B) — tuning
CONFIG = {
    "run_name": "B_embedding_tuning",
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0,
    "embedding_model": "nlpai-lab/KoE5",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "search_type": "similarity",
    "search_kwargs": {"k": 3},
    "prompt_template": (
        "다음 문맥을 근거로 질문에 답하세요. 문맥에 없는 내용은 모른다고 답하세요. "
        "반드시 한국어로만 답변하세요.\n"
        "[문맥]\n{context}\n\n[질문]\n{question}"
    ),
}

# ⚠️ TODO: 본인 PC에 있는 자동차관리법 조문 PDF 경로로 수정하세요.
PDF_PATH = str(PROJECT_ROOT / "data" / "자동차관리법.pdf")
EVAL_DIR = str(PROJECT_ROOT / "eval")

# 캐시 (재실행 시 중복 계산 방지)
CACHE_DIR = PROJECT_ROOT / "B_Embedding_tuning_py" / ".cache"
MD_CACHE_PATH = str(CACHE_DIR / "자동차관리법.md")  # PDF -> 마크다운 변환 결과 캐시
CHROMA_DIR = str(CACHE_DIR / "chroma_db")  # 임베딩 모델별 벡터스토어 캐시
