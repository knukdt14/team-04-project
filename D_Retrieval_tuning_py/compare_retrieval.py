import os
import time

import pandas as pd
from bert_score import score as bertscore
from dotenv import load_dotenv

from config import CONFIG, EVAL_DIR
from data_prep import load_and_split_markdown
from eval_data import EVAL_DATA
from models import build_llm, build_retriever, build_vectorstore
from rag_chain import build_rag_chain

# 비교해볼 검색 전략 — 자유롭게 추가/삭제하세요
STRATEGIES = [
    "dense",  # baseline
    "hybrid_0.5_0.5",  # BM25 0.5 / Dense 0.5
    "hybrid_0.3_0.7",  # BM25 0.3 / Dense 0.7
    "hybrid_0.7_0.3",  # BM25 0.7 / Dense 0.3
    "dense_rerank",  # Dense top-10 -> Cross-Encoder(bge-reranker-v2-m3)로 top-3 재정렬
]


def evaluate_one_strategy(strategy, vectorstore, splits, llm):
    print(f"\n{'=' * 60}\n검색 전략: {strategy}\n{'=' * 60}")

    retriever = build_retriever(vectorstore, splits, strategy=strategy)
    rag_chain = build_rag_chain(retriever, llm)

    rows = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        rows.append({
            "retrieval_strategy": strategy,
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
        f"[{strategy}] 평균 BERTScore F1: {df['bertscore_f1'].mean():.4f} "
        f"/ 평균 응답시간: {df['response_time_sec'].mean():.2f}초"
    )
    return df


def main():
    load_dotenv()
    print("환경 변수 로드 완료")

    splits = load_and_split_markdown()
    vectorstore = build_vectorstore(splits)  # 임베딩/청킹 고정 — 두 전략이 재사용
    llm = build_llm()  # LLM도 한 번만 로드해서 재사용

    all_dfs = []
    failed = []
    for s in STRATEGIES:
        try:
            all_dfs.append(evaluate_one_strategy(s, vectorstore, splits, llm))
        except Exception as e:
            print(f"\n[{s}] 실패 — 건너뛰고 계속 진행: {type(e).__name__}: {e}\n")
            failed.append(s)
    if failed:
        print(f"\n실패해서 제외된 전략: {failed}")
    df_all = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(EVAL_DIR, exist_ok=True)

    detail_path = os.path.join(EVAL_DIR, "results_D_retrieval_comparison_all.csv")
    df_all.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 상세 결과 저장 완료: {detail_path}")

    summary = (
        df_all.groupby("retrieval_strategy")
        .agg(
            mean_bertscore_f1=("bertscore_f1", "mean"),
            mean_response_time_sec=("response_time_sec", "mean"),
        )
        .sort_values("mean_bertscore_f1", ascending=False)
        .reset_index()
    )
    summary_path = os.path.join(EVAL_DIR, "results_D_retrieval_comparison_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== 검색 전략 비교 결과 (F1 높은 순) ===")
    print(summary.to_string(index=False))
    print(f"\n요약 결과 저장 완료: {summary_path}")

    best = summary.iloc[0]
    print(f"\n최고 F1 검색 전략: {best['retrieval_strategy']} (F1={best['mean_bertscore_f1']:.4f})")

    return df_all, summary


if __name__ == "__main__":
    main()
