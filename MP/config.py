from pathlib import Path

# 이 파일(D_Retrieval_tuning_py/) 기준 프로젝트 루트(Project_team1/)
PROJECT_ROOT = Path(__file__).resolve().parent

# CONFIG — 검색/유사도(Retrieval) 담당 (D) — baseline
# 임베딩 모델은 팀 공통으로 하나 고정 (B 담당의 튜닝 결과와 동일하게 맞춤).
# 이 축에서는 search_type / search_kwargs(top_k) / distance_metric 만 바꿔가며 비교함.
CONFIG = {
    "run_name": "D_retrieval_baseline",
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0,
    "embedding_model": "BAAI/bge-m3",  # ⚠️ 고정값. B 담당과 상의해서 팀 공통 임베딩으로 통일하세요.
    "chunk_size": 500,
    "chunk_overlap": 50,
    "search_type": "similarity",       # similarity / mmr / similarity_score_threshold
    "search_kwargs": {"k": 3},
    "distance_metric": "cosine",       # cosine / l2 / ip  (Chroma hnsw:space)
    "prompt_template": (
        "다음 문맥을 근거로 질문에 답하세요. 문맥에 없는 내용은 모른다고 답하세요.\n"
        "반드시 한국어로만 답변하세요.\n"
        "문맥에 관련 조항 번호나 구체적인 수치(기간, 금액 등)가 있다면 반드시 포함해서 답변하세요.\n"
        "[문맥]\n{context}\n\n[질문]\n{question}"
    ),
}

# ⚠️ TODO: 본인 PC에 있는 자동차관리법 조문 PDF 경로로 수정하세요.
PDF_PATH = str(PROJECT_ROOT / "data" / "자동차관리법.pdf")
EVAL_DIR = str(PROJECT_ROOT / "eval")
