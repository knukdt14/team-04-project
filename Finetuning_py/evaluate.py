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


def run_eval(tag, embedding_model_name=None, llm_model_name=None, lora_adapter_path=None):
    """tag: 결과 csv 파일명과 콘솔 요약에 붙는 식별자 (예: 'baseline', 'ft_embedding', 'ft_llm', 'ft_both')"""
    splits = load_and_split_markdown()
    vectorstore = build_vectorstore(
        splits,
        embedding_model_name=embedding_model_name,
        collection_name=f"finetuning_{tag}",
    )
    retriever = build_retriever(vectorstore, splits)
    llm = build_llm(llm_model_name, lora_adapter_path=lora_adapter_path)
    rag_chain = build_rag_chain(retriever, llm)

    results = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        results.append({
            "category": item["category"],
            "question": q,
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "response_time_sec": round(elapsed, 3),
        })
        print(f"[{elapsed:.2f}s] Q: {q}\n -> A: {answer[:80]}...\n")

    df_results = pd.DataFrame(results)

    P, R, F1 = bertscore(
        df_results["answer"].tolist(),
        df_results["ground_truth"].tolist(),
        model_type="klue/bert-base",
        num_layers=12,
        lang="ko",
        verbose=False,
    )
    df_results["bertscore_precision"] = P.tolist()
    df_results["bertscore_recall"] = R.tolist()
    df_results["bertscore_f1"] = F1.tolist()

    mean_f1 = df_results["bertscore_f1"].mean()
    mean_time = df_results["response_time_sec"].mean()
    print(f"[{tag}] 평균 BERTScore F1: {mean_f1:.4f} / 평균 응답시간: {mean_time:.2f}초")

    os.makedirs(EVAL_DIR, exist_ok=True)
    save_path = os.path.join(EVAL_DIR, f"results_finetuning_{tag}.csv")
    df_results.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"결과 저장 완료: {save_path}")

    return mean_f1, mean_time, df_results


def main():
    load_dotenv()
    for k, v in CONFIG.items():
        print(f"{k}: {v}")
    # 기본 실행: baseline(파인튜닝 전) 조합 그대로 재검증
    run_eval("baseline")


if __name__ == "__main__":
    main()
