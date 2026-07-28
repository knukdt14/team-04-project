from pathlib import Path

# 이 파일(A_LLM_tuning_py/) 기준 프로젝트 루트(Project_team1/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CONFIG — LLM 담당 (A) — tuning
# 실험 변수: LLM 모델. baseline Qwen/Qwen2.5-7B-Instruct(F1 0.6039) 대비
# 5개 후보 비교 결과 microsoft/Phi-4-mini-instruct가 최고(F1 0.6462) — 최종 채택 (compare_llms.py 결과 참고)
# 임베딩/chunk_size/top_k는 baseline과 동일하게 유지
CONFIG = {
    "run_name": "A_llm_tuning",
    "llm_model": "microsoft/Phi-4-mini-instruct",
    "temperature": 0,
    "embedding_model": "jhgan/ko-sroberta-multitask",  # 팀 baseline 임베딩 (B의 실험 변수이므로 여기선 고정)
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

BASELINE_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ⚠️ TODO: 본인 PC에 있는 자동차관리법 조문 PDF 경로로 수정하세요.
PDF_PATH = str(PROJECT_ROOT / "data" / "자동차관리법.pdf")
EVAL_DIR = str(PROJECT_ROOT / "eval")

# 캐시 (재실행 시 중복 계산 방지)
CACHE_DIR = PROJECT_ROOT / "A_LLM_tuning_py" / ".cache"
MD_CACHE_PATH = str(CACHE_DIR / "자동차관리법.md")
CHROMA_DIR = str(CACHE_DIR / "chroma_db")
