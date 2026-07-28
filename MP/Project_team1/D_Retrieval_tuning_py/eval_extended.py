"""
최종 5개 지표를 한 번에 뽑는 통합 스크립트.

    1. BERTScore F1   - 기존 (results_D_retrieval_comparison_all.csv에 이미 있음)
    2. 응답시간        - 기존 (위와 동일)
    3. Fact Match      - 신규: ground_truth 속 조항번호/수치가 answer에 그대로 있는가
                         (article_match + number_match를 하나로 통합)
    4. ROUGE-L         - 신규: 어휘 겹침
    5. MRR             - 신규: 검색기가 정답 근거 조항을 상위 몇 등에 올렸는가

동작 순서:
    A) 오프라인 단계 — 이미 저장된 results_D_retrieval_comparison_all.csv만 읽어서
       Fact Match / ROUGE-L 계산. LLM/검색기 재실행 없음.
    B) 검색 단계 — MRR 계산을 위해 retriever만 다시 호출 (LLM은 호출 안 함 -> GPU 불필요).
       models.py / data_prep.py / config.py 등 기존 코드는 그대로 재사용.
    C) A와 B를 (retrieval_strategy, question) 기준으로 합쳐 최종 5개 지표 요약표 저장.

사용법:
    cd D_Retrieval_tuning_py
    python eval_extended.py
"""

import os
import re

import pandas as pd
from rouge_score import rouge_scorer

from config import CONFIG, EVAL_DIR
from data_prep import load_and_split_markdown
from eval_data import EVAL_DATA
from models import build_retriever, build_vectorstore

STRATEGIES = ["dense", "hybrid_0.5_0.5", "hybrid_0.3_0.7", "hybrid_0.7_0.3", "dense_rerank"]

ARTICLE_PATTERN = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?")
NUMBER_PATTERN = re.compile(r"\d[\d,]*\s*(?:만원|원|일|개월|년|주|시간|km|kg|cc|%)")
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def _facts(text):
    """조항번호 + 수치를 한 번에 뽑아 '정답 근거 사실' 집합으로 취급"""
    if not isinstance(text, str):
        return set()
    norm = lambda m: re.sub(r"\s+", "", m)
    return {norm(m) for m in ARTICLE_PATTERN.findall(text)} | {norm(m) for m in NUMBER_PATTERN.findall(text)}


# ---------------- A) 오프라인: Fact Match + ROUGE-L ----------------
def add_offline_metrics(df):
    def fact_match(row):
        gt_facts = _facts(row["ground_truth"])
        if not gt_facts:
            return None  # 채점할 근거 자체가 없는 문항은 제외
        ans_facts = _facts(row["answer"])
        return gt_facts.issubset(ans_facts)

    def rouge_l(row):
        if not isinstance(row["answer"], str) or not isinstance(row["ground_truth"], str):
            return None
        gt = " ".join(row["ground_truth"])
        ans = " ".join(row["answer"])
        return _rouge.score(gt, ans)["rougeL"].fmeasure

    df["fact_match"] = df.apply(fact_match, axis=1)
    df["rouge_l"] = df.apply(rouge_l, axis=1)
    return df


# ---------------- B) 검색 단계: MRR ----------------
def compute_mrr(vectorstore, splits):
    rows = []
    for strategy in STRATEGIES:
        try:
            retriever = build_retriever(vectorstore, splits, strategy=strategy)
        except Exception as e:
            print(f"[{strategy}] retriever 생성 실패 — 건너뜀: {type(e).__name__}: {e}")
            continue
        print(f"[{strategy}] MRR 계산 중...")

        for item in EVAL_DATA:
            gt_facts = _facts(item["ground_truth"])
            question = item["question"]
            if not gt_facts:
                rows.append({"retrieval_strategy": strategy, "question": question, "reciprocal_rank": None})
                continue

            docs = retriever.invoke(question)
            rank_found = None
            for rank, doc in enumerate(docs, start=1):
                if gt_facts & _facts(doc.page_content):
                    rank_found = rank
                    break
            rows.append({
                "retrieval_strategy": strategy,
                "question": question,
                "reciprocal_rank": (1.0 / rank_found) if rank_found else 0.0,
            })
    return pd.DataFrame(rows)


def main():
    # A) 오프라인 지표
    gen_path = os.path.join(EVAL_DIR, "results_D_retrieval_comparison_all.csv")
    df_gen = pd.read_csv(gen_path, encoding="utf-8-sig")
    df_gen = add_offline_metrics(df_gen)

    # B) 검색 단계 지표
    splits = load_and_split_markdown()
    vectorstore = build_vectorstore(splits)
    df_mrr = compute_mrr(vectorstore, splits)

    # C) 병합 + 5개 지표 요약
    merged = df_gen.merge(df_mrr, on=["retrieval_strategy", "question"], how="left")
    detail_path = os.path.join(EVAL_DIR, "results_D_retrieval_5metrics_all.csv")
    merged.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n상세 결과 저장: {detail_path}")

    summary = merged.groupby("retrieval_strategy").agg(
        bertscore_f1=("bertscore_f1", "mean"),
        response_time_sec=("response_time_sec", "mean"),
        fact_match=("fact_match", "mean"),
        rouge_l=("rouge_l", "mean"),
        MRR=("reciprocal_rank", "mean"),
    ).sort_values("bertscore_f1", ascending=False)

    summary_path = os.path.join(EVAL_DIR, "results_D_retrieval_5metrics_summary.csv")
    summary.to_csv(summary_path, encoding="utf-8-sig")

    print("\n=== 최종 5개 지표 요약 (전략별) ===")
    print(summary.to_string())
    print(f"\n요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
