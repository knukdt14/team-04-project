from pathlib import Path

# 이 파일(A_LLM_tuning_py/) 기준 프로젝트 루트(team-04-project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# CONFIG — LLM 모델 담당 (A) — tuning
# embedding_model / chunk_size / chunk_overlap / top_k는 내 담당이 아닌 축이므로 팀 BASELINE 값으로 고정
CONFIG = {
    "run_name": "A_llm_tuning",
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",  # 팀 BASELINE (GPU 4bit 양자화로 구동)
    "temperature": 0,
    "embedding_model": "jhgan/ko-sroberta-multitask",  # 고정 (담당 아님)
    "chunk_size": 500,  # 고정 (담당 아님)
    "chunk_overlap": 50,  # 고정 (담당 아님)
    "search_type": "similarity",  # 고정 (담당 아님)
    "search_kwargs": {"k": 3},  # 고정 (담당 아님)
    "prompt_template": (
        "다음 문맥을 근거로 질문에 답하세요. 문맥에 없는 내용은 모른다고 답하세요.\n"
        "[문맥]\n{context}\n\n[질문]\n{question}"
    ),
}

PDF_PATH = str(PROJECT_ROOT / "data" / "자동차관리법.pdf")
EVAL_DIR = str(PROJECT_ROOT / "models" / "eval")
