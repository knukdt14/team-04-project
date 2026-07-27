import os
import re
import time

import pandas as pd
from bert_score import score as bertscore
from dotenv import load_dotenv

from config import CONFIG, EVAL_DIR
from data_prep import load_and_split
from eval_data import EVAL_DATA
from models import build_embeddings, build_llm, build_retriever, build_vectorstore
from rag_chain import build_rag_chain

# 비교해볼 청킹 설정 후보 — 자유롭게 추가/삭제하세요
CHUNKING_CANDIDATES = [
    {"label": "baseline_recursive_500", "splitter": "recursive", "chunk_size": 500, "chunk_overlap": 50},
    {"label": "character_500",          "splitter": "character", "chunk_size": 500, "chunk_overlap": 50},
    {"label": "token_500",              "splitter": "token",     "chunk_size": 500, "chunk_overlap": 50},
    {"label": "semantic_pct95",         "splitter": "semantic",  "semantic_breakpoint_type": "percentile", "semantic_breakpoint_amount": 95},
    {"label": "semantic_pct90",         "splitter": "semantic",  "semantic_breakpoint_type": "percentile", "semantic_breakpoint_amount": 90},
    {"label": "recursive_700_70",       "splitter": "recursive", "chunk_size": 700, "chunk_overlap": 70},
    {"label": "recursive_1000_100",     "splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
]


def _collection_name(label):
    return "cmp_" + re.sub(r"[^0-9a-zA-Z]+", "_", label)


def evaluate_one_chunking_config(candidate, embeddings, llm):
    cfg = dict(CONFIG)
    cfg.update(candidate)
    label = candidate["label"]
    print(
        f"\n{'=' * 60}\n청킹 설정: {label} "
        f"(splitter={cfg['splitter']}, size={cfg.get('chunk_size')}, overlap={cfg.get('chunk_overlap')})"
        f"\n{'=' * 60}"
    )

    # semantic은 문서마다 청크 수가 chunk_size와 무관하게 달라지므로 매번 새로 분할
    splits = load_and_split(cfg, embeddings=embeddings)
    vectorstore = build_vectorstore(
        splits, embeddings=embeddings, collection_name=_collection_name(label), cfg=cfg
    )
    retriever = build_retriever(vectorstore, cfg=cfg)
    rag_chain = build_rag_chain(retriever, llm, cfg=cfg)

    rows = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        rows.append({
            "label": label,
            "splitter": cfg["splitter"],
            "chunk_size": cfg.get("chunk_size"),
            "chunk_overlap": cfg.get("chunk_overlap"),
            "category": item["category"],
            "question": q,
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "response_time_sec": round(elapsed, 3),
        })
        print(f"[{elapsed:.2f}s] Q: {q}\n -> A: {answer[:80]}...\n")

    df = pd.DataFrame(rows)

    # 한국어는 klue/bert-base 사용. bert_score 내장 레이어 매핑에 없어 num_layers 직접 지정
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
    print(f"비교 대상 청킹 설정 {len(CHUNKING_CANDIDATES)}개")

    embeddings = build_embeddings()  # 팀 공통 임베딩 — 실험 내내 재사용 (semantic 분할에도 재사용)
    llm = build_llm()  # LLM은 한 번만 로드해서 모든 후보에 재사용 (반복 재로드로 인한 지연 방지)

    all_dfs = [evaluate_one_chunking_config(c, embeddings, llm) for c in CHUNKING_CANDIDATES]
    df_all = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(EVAL_DIR, exist_ok=True)

    detail_path = os.path.join(EVAL_DIR, "results_C_chunking_comparison_all.csv")
    df_all.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 상세 결과 저장 완료: {detail_path}")

    summary = (
        df_all.groupby("label")
        .agg(
            splitter=("splitter", "first"),
            chunk_size=("chunk_size", "first"),
            chunk_overlap=("chunk_overlap", "first"),
            mean_bertscore_f1=("bertscore_f1", "mean"),
            mean_response_time_sec=("response_time_sec", "mean"),
        )
        .sort_values("mean_bertscore_f1", ascending=False)
        .reset_index()
    )
    summary_path = os.path.join(EVAL_DIR, "results_C_chunking_comparison_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== 청킹 설정 비교 결과 (F1 높은 순) ===")
    print(summary.to_string(index=False))
    print(f"\n요약 결과 저장 완료: {summary_path}")

    best = summary.iloc[0]
    print(f"\n최고 F1 청킹 설정: {best['label']} (F1={best['mean_bertscore_f1']:.4f})")

    return df_all, summary


if __name__ == "__main__":
    main()
