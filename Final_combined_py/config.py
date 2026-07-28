from pathlib import Path

# 이 파일(Final_combined_py/) 기준 프로젝트 루트(Project_team1/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CONFIG — A/B/C/D 각 축의 개별 승자를 그대로 합쳤더니(F1 0.5952, 상호작용으로 오히려 하락)
# 실패해서, greedy_search.py로 한 번에 하나씩만 바꿔가며 재검증한 조합(F1 0.6815)을 RAG 튜닝
# 최종으로 채택. 이후 Finetuning_py에서 실제 파인튜닝 진행: 임베딩(KoE5) LoRA는 F1 0.6755로
# 오히려 하락해 폐기, LLM(Qwen2.5-7B) QLoRA는 F1 0.7020으로 개선되어 채택(lora_adapter_path).
# KoE5(B) + chunk_800/100(C) + Qwen2.5-7B-Instruct+QLoRA(파인튜닝) + hybrid 0.3/0.7(D)
CONFIG = {
    "run_name": "final_combined",
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "lora_adapter_path": str(PROJECT_ROOT / "Finetuning_py" / ".cache" / "finetuned_qwen_lora"),
    "temperature": 0,
    "embedding_model": "nlpai-lab/KoE5",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "search_type": "similarity",
    "search_kwargs": {"k": 3},
    "retrieval_strategy": "hybrid",  # "dense" | "hybrid" | "dense_rerank"
    "hybrid_weights": [0.3, 0.7],  # [BM25 가중치, Dense 가중치]
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
CACHE_DIR = PROJECT_ROOT / "Final_combined_py" / ".cache"
MD_CACHE_PATH = str(CACHE_DIR / "자동차관리법.md")
CHROMA_DIR = str(CACHE_DIR / "chroma_db")
