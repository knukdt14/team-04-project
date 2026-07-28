import os
import time
from functools import partial

import pandas as pd
from bert_score import score as bertscore
from dotenv import load_dotenv

from config import CONFIG, EVAL_DIR
from data_prep import load_and_split_character, load_and_split_article
from eval_data import EVAL_DATA
from models import build_llm, build_retriever, build_vectorstore
from rag_chain import build_rag_chain

# 비교해볼 청킹 전략 — (전략 이름, 분할 함수). 자유롭게 추가/삭제하세요
STRATEGIES = [
    ("character_300", partial(load_and_split_character, chunk_size=300, chunk_overlap=50)),
    ("character_500 (baseline)", partial(load_and_split_character, chunk_size=500, chunk_overlap=50)),
    ("character_800", partial(load_and_split_character, chunk_size=800, chunk_overlap=100)),
    ("article (tuning)", load_and_split_article),
]


def evaluate_one_strategy(strategy_name, split_fn, llm):
    print(f"\n{'=' * 60}\n청킹 전략: {strategy_name}\n{'=' * 60}")

    splits = split_fn()
    collection_name = (
        CONFIG["run_name"] + "_" + strategy_name.split()[0] + "_" +
        CONFIG["embedding_model"].replace("/", "_").replace("-", "_")
    )
    vectorstore = build_vectorstore(splits, collection_name=collection_name)
    retriever = build_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever, llm)

    rows = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        rows.append({
            "chunking_strategy": strategy_name,
            "num_chunks": len(splits),
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
        f"[{strategy_name}] 평균 BERTScore F1: {df['bertscore_f1'].mean():.4f} "
        f"/ 평균 응답시간: {df['response_time_sec'].mean():.2f}초 / 청크 수: {len(splits)}"
    )
    return df


def main():
    load_dotenv()
    print("환경 변수 로드 완료")

    llm = build_llm()  # LLM은 한 번만 로드해서 모든 전략에 재사용

    all_dfs = []
    failed = []
    for name, fn in STRATEGIES:
        try:
            all_dfs.append(evaluate_one_strategy(name, fn, llm))
        except Exception as e:
            print(f"\n[{name}] 실패 — 건너뛰고 계속 진행: {type(e).__name__}: {e}\n")
            failed.append(name)
    if failed:
        print(f"\n실패해서 제외된 전략: {failed}")
    df_all = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(EVAL_DIR, exist_ok=True)

    detail_path = os.path.join(EVAL_DIR, "results_C_chunking_comparison_all.csv")
    df_all.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 상세 결과 저장 완료: {detail_path}")

    summary = (
        df_all.groupby("chunking_strategy")
        .agg(
            mean_bertscore_f1=("bertscore_f1", "mean"),
            mean_response_time_sec=("response_time_sec", "mean"),
            num_chunks=("num_chunks", "first"),
        )
        .sort_values("mean_bertscore_f1", ascending=False)
        .reset_index()
    )
    summary_path = os.path.join(EVAL_DIR, "results_C_chunking_comparison_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== 청킹 전략 비교 결과 (F1 높은 순) ===")
    print(summary.to_string(index=False))
    print(f"\n요약 결과 저장 완료: {summary_path}")

    best = summary.iloc[0]
    print(f"\n최고 F1 청킹 전략: {best['chunking_strategy']} (F1={best['mean_bertscore_f1']:.4f})")

    return df_all, summary


if __name__ == "__main__":
    main()
