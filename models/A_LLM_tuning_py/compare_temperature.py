import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import time

import pandas as pd
from bert_score import score as bertscore
from dotenv import load_dotenv

from config import CONFIG, EVAL_DIR
from data_prep import load_and_split
from eval_data import EVAL_DATA
from models import build_llm, build_retriever, build_vectorstore
from rag_chain import build_rag_chain

# temperature 스윕 — LLM 모델/임베딩/청킹/검색 파라미터는 BASELINE으로 고정, temperature만 변경
TEMPERATURE_CANDIDATES = [0.0, 0.2, 0.4, 0.7, 1.0]


def evaluate_one_temperature(temperature, vectorstore):
    label = f"temp_{temperature}"
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

    retriever = build_retriever(vectorstore)
    llm = build_llm(llm_model=CONFIG["llm_model"], temperature=temperature)
    rag_chain = build_rag_chain(retriever, llm)

    rows = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        rows.append({
            "temperature": temperature,
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
        f"[{label}] 평균 BERTScore F1: {df['bertscore_f1'].mean():.4f} "
        f"/ 평균 응답시간: {df['response_time_sec'].mean():.2f}초"
    )
    return df


def main():
    load_dotenv()
    print("환경 변수 로드 완료")
    print(f"temperature 스윕 대상: {TEMPERATURE_CANDIDATES} (모델 고정: {CONFIG['llm_model']})")

    splits = load_and_split()
    vectorstore = build_vectorstore(splits, collection_name="cmp_temperature_shared")

    all_dfs = [evaluate_one_temperature(t, vectorstore) for t in TEMPERATURE_CANDIDATES]
    df_all = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(EVAL_DIR, exist_ok=True)

    detail_path = os.path.join(EVAL_DIR, "results_A_temperature_sweep_all.csv")
    df_all.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 상세 결과 저장 완료: {detail_path}")

    summary = (
        df_all.groupby("temperature")
        .agg(
            mean_bertscore_f1=("bertscore_f1", "mean"),
            mean_response_time_sec=("response_time_sec", "mean"),
        )
        .sort_values("temperature")
        .reset_index()
    )
    summary_path = os.path.join(EVAL_DIR, "results_A_temperature_sweep_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== temperature 스윕 결과 ===")
    print(summary.to_string(index=False))
    print(f"\n요약 결과 저장 완료: {summary_path}")

    return df_all, summary


if __name__ == "__main__":
    main()
