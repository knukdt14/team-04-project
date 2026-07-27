from pathlib import Path

# 이 파일(C_Chunking_tuning_py/) 기준 프로젝트 루트(Project_team1/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CONFIG — 문서 분할(청킹) 담당 (C) — tuning
CONFIG = {
    "run_name": "C_chunking_tuning",
    # 팀 baseline과 동일 — 로컬 GPU에서 4bit 양자화로 직접 실행 (B의 rag_chain.py와 동일 방식)
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0,
    "embedding_model": "jhgan/ko-sroberta-multitask",  # 팀 공통 baseline (B 담당 축 — 고정, 건드리지 않음)
    "splitter": "semantic",       # recursive / character / token / semantic — C 담당 변수
    "chunk_size": 500,
    "chunk_overlap": 50,
    "semantic_breakpoint_type": "percentile",     # semantic일 때만 사용
    "semantic_breakpoint_amount": 95,             # semantic일 때만 사용
    "search_type": "similarity",
    "search_kwargs": {"k": 3},
    "prompt_template": (
        "다음 문맥을 근거로 질문에 답하세요. 문맥에 없는 내용은 모른다고 답하세요.\n"
        "[문맥]\n{context}\n\n[질문]\n{question}"
    ),
}

# ⚠️ TODO: 본인 PC에 있는 자동차관리법 조문 PDF 경로로 수정하세요.
PDF_PATH = str(PROJECT_ROOT / "data" / "자동차관리법.pdf")
EVAL_DIR = str(PROJECT_ROOT / "eval")
