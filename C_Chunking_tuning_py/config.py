from pathlib import Path

# 이 파일(C_Chunking_tuning_py/) 기준 프로젝트 루트(Project_team1/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CONFIG — Chunking 담당 (C) — tuning
# 실험 변수: 청킹 전략. baseline character_500(F1 0.6039) 포함 4가지 비교 결과
# character_800(chunk_size=800/overlap=100)이 최고(F1 0.6423) — 최종 채택 (compare_chunking.py 결과 참고)
# LLM/임베딩/top_k는 baseline과 동일하게 유지
CONFIG = {
    "run_name": "C_chunking_tuning",
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0,
    "embedding_model": "jhgan/ko-sroberta-multitask",  # 팀 baseline 임베딩 (B의 실험 변수이므로 여기선 고정)
    "chunk_size": 800,
    "chunk_overlap": 100,
    "chunking_strategy": "character",  # "character" | "article"
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
CACHE_DIR = PROJECT_ROOT / "C_Chunking_tuning_py" / ".cache"
MD_CACHE_PATH = str(CACHE_DIR / "자동차관리법.md")
CHROMA_DIR = str(CACHE_DIR / "chroma_db")
