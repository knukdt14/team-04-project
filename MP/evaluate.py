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


def main():
    load_dotenv()  # .env 파일에서 API 키 로드 (팀원 각자 자신의 .env 사용)
    print("환경 변수 로드 완료")

    for k, v in CONFIG.items():
        print(f"{k}: {v}")

    splits = load_and_split()
    vectorstore = build_vectorstore(splits)
    retriever = build_retriever(vectorstore)
    llm, _query_llm = build_llm()
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

    # BERTScore 평가 (필수 지표). 한국어는 klue/bert-base 사용 권장
    # bert_score의 내장 레이어 매핑에 klue/bert-base가 없어 num_layers를 직접 지정 (표준 BERT-base = 12)
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

    print(f"평균 BERTScore F1: {df_results['bertscore_f1'].mean():.4f}")
    print(f"평균 응답시간: {df_results['response_time_sec'].mean():.2f}초")

    # ⚠️ TODO: answer를 실제 조문 내용과 비교해서 사실과 다른 부분이 있으면 True로 표시하세요.
    df_results["hallucination"] = False

    os.makedirs(EVAL_DIR, exist_ok=True)
    save_path = os.path.join(EVAL_DIR, f"results_{CONFIG['run_name']}.csv")
    df_results.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"결과 저장 완료: {save_path}")

    print("\n=== 요약 ===")
    print(f"run_name        : {CONFIG['run_name']}")
    print(f"LLM 모델        : {CONFIG['llm_model']}")
    print(f"Embedding 모델  : {CONFIG['embedding_model']} (고정)")
    print(f"search_type     : {CONFIG['search_type']} / kwargs: {CONFIG['search_kwargs']}")
    print(f"distance_metric : {CONFIG['distance_metric']}")
    print(f"평균 BERTScore F1 : {df_results['bertscore_f1'].mean():.4f}")
    print(f"평균 응답시간     : {df_results['response_time_sec'].mean():.2f}초")

    return df_results


if __name__ == "__main__":
    main()
