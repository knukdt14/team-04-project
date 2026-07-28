import os
import time

import pandas as pd
from bert_score import score as bertscore
from dotenv import load_dotenv

from config import CONFIG, EVAL_DIR
from data_prep import load_and_split
from eval_data import EVAL_DATA
from models import build_llm, build_multi_query_retriever, build_retriever, build_vectorstore
from rag_chain import build_rag_chain

# 비교해볼 검색/유사도 설정 후보 — 자유롭게 추가/삭제하세요
# embedding_model은 CONFIG 값으로 고정, 아래 축(search_type, top_k, distance_metric)만 바꿔가며 비교
SIMILARITY_CANDIDATES = [
    {
        "label": "baseline_similarity_cosine_k3",  # 팀 baseline
        "search_type": "similarity",
        "search_kwargs": {"k": 3},
        "distance_metric": "cosine",
    },
    {
        "label": "mmr_k3",  # 관련성 + 다양성을 함께 고려하는 검색
        "search_type": "mmr",
        "search_kwargs": {"k": 3},
        "distance_metric": "cosine",
    },
    {
        "label": "score_threshold_k3",  # 일정 유사도 이상만 채택
        "search_type": "similarity_score_threshold",
        "search_kwargs": {"k": 3},
        "distance_metric": "cosine",
    },
    {
        "label": "similarity_cosine_k5",  # top_k를 늘렸을 때 효과
        "search_type": "similarity",
        "search_kwargs": {"k": 5},
        "distance_metric": "cosine",
    },
    {
        "label": "similarity_l2_k3",  # 유사도 계산 방식(거리 척도) 자체를 변경
        "search_type": "similarity",
        "search_kwargs": {"k": 3},
        "distance_metric": "l2",
    },
]


def _collection_name(label):
    return f"cmp_{label}"


def evaluate_one_candidate(candidate, splits, llm, query_llm=None):
    label = candidate["label"]
    print(f"\n{'=' * 60}\n검색 설정: {label}\n{'=' * 60}")

    vectorstore = build_vectorstore(
        splits,
        embedding_model_name=CONFIG["embedding_model"],  # 고정
        distance_metric=candidate["distance_metric"],
        collection_name=_collection_name(label),
    )
    retriever = build_retriever(
        vectorstore,
        search_type=candidate["search_type"],
        search_kwargs=candidate["search_kwargs"],
    )
    # 멀티쿼리(build_multi_query_retriever)는 CPU 임베딩 호출이 3배로 늘어나
    # 응답 시간이 10초대→40초+로 튀어서 되돌림. GPU 여유가 생기면 재검토.
    rag_chain = build_rag_chain(retriever, llm)

    rows = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        rows.append({
            "label": label,
            "search_type": candidate["search_type"],
            "top_k": candidate["search_kwargs"].get("k"),
            "distance_metric": candidate["distance_metric"],
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
    print(f"임베딩 모델(고정): {CONFIG['embedding_model']}")
    print(f"비교 대상 검색 설정 {len(SIMILARITY_CANDIDATES)}개: {[c['label'] for c in SIMILARITY_CANDIDATES]}")

    splits = load_and_split()
    llm, query_llm = build_llm()  # LLM은 한 번만 로드해서 모든 후보에 재사용 (반복 재로드로 인한 지연 방지)

    all_dfs = [evaluate_one_candidate(c, splits, llm, query_llm) for c in SIMILARITY_CANDIDATES]
    df_all = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(EVAL_DIR, exist_ok=True)

    detail_path = os.path.join(EVAL_DIR, "results_D_similarity_comparison_all.csv")
    df_all.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 상세 결과 저장 완료: {detail_path}")

    summary = (
        df_all.groupby("label")
        .agg(
            mean_bertscore_f1=("bertscore_f1", "mean"),
            mean_response_time_sec=("response_time_sec", "mean"),
        )
        .sort_values("mean_bertscore_f1", ascending=False)
        .reset_index()
    )
    summary_path = os.path.join(EVAL_DIR, "results_D_similarity_comparison_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== 검색/유사도 설정 비교 결과 (F1 높은 순) ===")
    print(summary.to_string(index=False))
    print(f"\n요약 결과 저장 완료: {summary_path}")

    best = summary.iloc[0]
    print(f"\n최고 F1 검색 설정: {best['label']} (F1={best['mean_bertscore_f1']:.4f})")

    return df_all, summary


if __name__ == "__main__":
    main()
