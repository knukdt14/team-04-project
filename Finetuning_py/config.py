from pathlib import Path

# 이 파일(Finetuning_py/) 기준 프로젝트 루트(Project_team1/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RAG 튜닝(A/B/C/D)에서 찾은 최종 채택 조합(Final_combined_py, F1 0.6815)을 그대로 baseline으로 삼고,
# 그 위에 임베딩(KoE5) -> LLM(Qwen2.5-7B) 순서로 실제 파인튜닝을 적용해 성능 변화를 측정한다.
CONFIG = {
    "run_name": "finetuning",
    "base_llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "base_embedding_model": "nlpai-lab/KoE5",
    "qa_gen_llm_model": "Qwen/Qwen2.5-7B-Instruct",  # 합성 QA 데이터 생성에 사용할 LLM (baseline과 동일 모델)
    "temperature": 0,
    "chunk_size": 800,
    "chunk_overlap": 100,
    "search_type": "similarity",
    "search_kwargs": {"k": 3},
    "retrieval_strategy": "hybrid",
    "hybrid_weights": [0.3, 0.7],
    "prompt_template": (
        "다음 문맥을 근거로 질문에 답하세요. 문맥에 없는 내용은 모른다고 답하세요. "
        "반드시 한국어로만 답변하세요.\n"
        "[문맥]\n{context}\n\n[질문]\n{question}"
    ),
    # 임베딩 파인튜닝 하이퍼파라미터
    # KoE5(XLM-RoBERTa-large 기반, 560M) 전체 파라미터 fine-tuning은 Adam 옵티마이저 상태만으로도
    # 8GB VRAM을 초과해 OOM 발생 (CUBLAS_STATUS_INTERNAL_ERROR로 나타남) -> LoRA로 전환
    "embedding_epochs": 4,
    "embedding_batch_size": 16,
    "embedding_lora_r": 16,
    "embedding_lora_alpha": 32,
    "embedding_lora_dropout": 0.1,
    "embedding_lora_target_modules": ["query", "key", "value"],
    # LLM QLoRA 파인튜닝 하이퍼파라미터
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llm_ft_epochs": 3,
    "llm_ft_lr": 2e-4,
    "llm_ft_max_length": 1024,
}

# ⚠️ TODO: 본인 PC에 있는 자동차관리법 조문 PDF 경로로 수정하세요.
PDF_PATH = str(PROJECT_ROOT / "data" / "자동차관리법.pdf")
EVAL_DIR = str(PROJECT_ROOT / "eval")

# 캐시 및 산출물 (재실행 시 중복 계산 방지)
CACHE_DIR = PROJECT_ROOT / "Finetuning_py" / ".cache"
MD_CACHE_PATH = str(CACHE_DIR / "자동차관리법.md")
CHROMA_DIR = str(CACHE_DIR / "chroma_db")
QA_DATA_PATH = str(Path(EVAL_DIR) / "synthetic_qa_dataset.csv")
EMBEDDING_OUTPUT_DIR = str(CACHE_DIR / "finetuned_koe5")
LORA_OUTPUT_DIR = str(CACHE_DIR / "finetuned_qwen_lora")
