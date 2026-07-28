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

# 비교해볼 임베딩 모델 후보 — 자유롭게 추가/삭제하세요
# 전부 마크다운 로더(load_and_split_markdown) 기준으로 통일해서 공정 비교
EMBEDDING_CANDIDATES = [
    "jhgan/ko-sroberta-multitask",  # 팀 baseline 임베딩
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",  # PDF로더 기준 1차 비교 1위
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "nlpai-lab/KURE-v1",  # 한국어 특화 (BGE-M3 기반 파인튜닝)
    "nlpai-lab/KoE5",  # 한국어 특화 (multilingual-e5 파인튜닝)
    "dragonkue/BGE-m3-ko",  # 한국어 특화
    "upskyy/bge-m3-korean",  # 한국어 특화
]


def evaluate_one_embedding(embedding_model_name, splits, llm):
    print(f"\n{'=' * 60}\n임베딩 모델: {embedding_model_name}\n{'=' * 60}")

    # collection_name을 안 넘기면 models.py의 기본 규칙(임베딩 모델명 기반)을 써서
    # evaluate.py와 벡터스토어 캐시를 공유함
    vectorstore = build_vectorstore(splits, embedding_model_name=embedding_model_name)
    retriever = build_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever, llm)

    rows = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        rows.append({
            "embedding_model": embedding_model_name,
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
        f"[{embedding_model_name}] 평균 BERTScore F1: {df['bertscore_f1'].mean():.4f} "
        f"/ 평균 응답시간: {df['response_time_sec'].mean():.2f}초"
    )
    return df


def main():
    load_dotenv()
    print("환경 변수 로드 완료")
    print(f"비교 대상 임베딩 모델 {len(EMBEDDING_CANDIDATES)}개: {EMBEDDING_CANDIDATES}")

    splits = load_and_split_markdown()
    llm = build_llm()  # LLM은 한 번만 로드해서 모든 임베딩 후보에 재사용 (반복 재로드로 인한 지연 방지)

    all_dfs = [evaluate_one_embedding(name, splits, llm) for name in EMBEDDING_CANDIDATES]
    df_all = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(EVAL_DIR, exist_ok=True)

    detail_path = os.path.join(EVAL_DIR, "results_B_embedding_comparison_all.csv")
    df_all.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 상세 결과 저장 완료: {detail_path}")

    summary = (
        df_all.groupby("embedding_model")
        .agg(
            mean_bertscore_f1=("bertscore_f1", "mean"),
            mean_response_time_sec=("response_time_sec", "mean"),
        )
        .sort_values("mean_bertscore_f1", ascending=False)
        .reset_index()
    )
    summary_path = os.path.join(EVAL_DIR, "results_B_embedding_comparison_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== 임베딩 모델 비교 결과 (F1 높은 순) ===")
    print(summary.to_string(index=False))
    print(f"\n요약 결과 저장 완료: {summary_path}")

    best = summary.iloc[0]
    print(f"\n최고 F1 임베딩 모델: {best['embedding_model']} (F1={best['mean_bertscore_f1']:.4f})")

    return df_all, summary


if __name__ == "__main__":
    main()
