import os
import time

import pandas as pd
from bert_score import score as bertscore
from dotenv import load_dotenv

from config import BASELINE_LLM_MODEL, CONFIG, EVAL_DIR
from data_prep import load_and_split_markdown
from eval_data import EVAL_DATA
from models import build_llm, build_retriever, build_vectorstore
from rag_chain import build_rag_chain

# 비교해볼 LLM 후보 — 자유롭게 추가/삭제하세요
# 임베딩/chunk_size/top_k는 고정, LLM만 바꿔가며 비교
LLM_CANDIDATES = [
    BASELINE_LLM_MODEL,  # "Qwen/Qwen2.5-7B-Instruct" (팀 baseline)
    "Qwen/Qwen2.5-3B-Instruct",  # 1차 tuning 시도 — baseline보다 낮았음(F1 0.5621)
    # "Qwen/Qwen2.5-14B-Instruct",  # 8GB GPU에 4bit로도 안 들어감(임베딩 CPU로 빼도 안 됨) — 제외
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",  # 한국어 특화 Llama-3 파인튜닝
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",  # 한국어 특화, 소형
    "microsoft/Phi-4-mini-instruct",  # Microsoft, 최신 소형 모델
]


def evaluate_one_llm(llm_model_name, retriever):
    print(f"\n{'=' * 60}\nLLM 모델: {llm_model_name}\n{'=' * 60}")

    llm = build_llm(llm_model_name)
    rag_chain = build_rag_chain(retriever, llm)

    rows = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        rows.append({
            "llm_model": llm_model_name,
            "category": item["category"],
            "question": q,
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "response_time_sec": round(elapsed, 3),
        })
        print(f"[{elapsed:.2f}s] Q: {q}\n -> A: {answer[:80]}...\n")

    df = pd.DataFrame(rows)

    P, R, F1 = bertscore(
        df["answer"].tolist(),
        df["ground_truth"].tolist(),
        model_type="klue/bert-base",
        num_layers=12,
        lang="ko",
        verbose=False,
    )
    df["bertscore_precision"] = P.tolist()
    df["bertscore_recall"] = R.tolist()
    df["bertscore_f1"] = F1.tolist()

    print(
        f"[{llm_model_name}] 평균 BERTScore F1: {df['bertscore_f1'].mean():.4f} "
        f"/ 평균 응답시간: {df['response_time_sec'].mean():.2f}초"
    )
    return df


def main():
    load_dotenv()
    print("환경 변수 로드 완료")
    print(f"비교 대상 LLM {len(LLM_CANDIDATES)}개: {LLM_CANDIDATES}")

    splits = load_and_split_markdown()
    vectorstore = build_vectorstore(splits)  # 임베딩 고정 — 후보 전체가 재사용
    retriever = build_retriever(vectorstore)

    all_dfs = []
    failed = []
    for name in LLM_CANDIDATES:
        try:
            all_dfs.append(evaluate_one_llm(name, retriever))
        except Exception as e:
            print(f"\n[{name}] 실패 — 건너뛰고 계속 진행: {type(e).__name__}: {e}\n")
            failed.append(name)
    if failed:
        print(f"\n실패해서 제외된 LLM: {failed}")
    df_all = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(EVAL_DIR, exist_ok=True)

    detail_path = os.path.join(EVAL_DIR, "results_A_llm_comparison_all.csv")
    df_all.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 상세 결과 저장 완료: {detail_path}")

    summary = (
        df_all.groupby("llm_model")
        .agg(
            mean_bertscore_f1=("bertscore_f1", "mean"),
            mean_response_time_sec=("response_time_sec", "mean"),
        )
        .sort_values("mean_bertscore_f1", ascending=False)
        .reset_index()
    )
    summary_path = os.path.join(EVAL_DIR, "results_A_llm_comparison_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== LLM 비교 결과 (F1 높은 순) ===")
    print(summary.to_string(index=False))
    print(f"\n요약 결과 저장 완료: {summary_path}")

    best = summary.iloc[0]
    print(f"\n최고 F1 LLM: {best['llm_model']} (F1={best['mean_bertscore_f1']:.4f})")

    return df_all, summary


if __name__ == "__main__":
    main()
