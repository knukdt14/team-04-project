import sys

# Windows 콘솔 기본 코드페이지(cp949)로는 LLM이 생성하는 일부 유니코드 문자를 출력할 수 없어
# print()에서 UnicodeEncodeError로 죽는 경우가 있음 → stdout을 UTF-8로 강제 전환
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import time

import pandas as pd
from bert_score import score as bertscore
from dotenv import load_dotenv

from config import CONFIG, EVAL_DIR
from data_prep import load_and_split
from eval_data import EVAL_DATA
from models import build_llm, build_retriever, build_vectorstore, clear_llm_cache
from rag_chain import build_rag_chain

# 비교할 LLM 모델 후보 — GPU(RTX 4070, 8GB VRAM) + 4bit 양자화로 실행.
# 아래는 사전에 개별 검증한 결과 반영 (GPU/CPU와 무관한 코드 자체 호환성 문제라 여전히 제외):
#   - EXAONE-3.5: 리포지토리 커스텀 코드가 최신 transformers와 충돌(RopeParameters) → 제외 (대신 네이티브 지원되는 EXAONE-4.0 사용)
#   - Gemma-2-*-it, naver-hyperclovax/*: gated 모델이라 접근 권한 없음(401) → 제외
#   - beomi/gemma-ko-2b: instruct 튜닝이 안 된 base 모델이라 chat_template 없음 → 제외
LLM_CANDIDATES = [
    "Qwen/Qwen2.5-7B-Instruct",                       # 팀 BASELINE (GPU 4bit로 이제 실제 구동 가능)
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct",
    "LGAI-EXAONE/EXAONE-4.0-1.2B",                    # EXAONE 계열 (4.0은 transformers 네이티브 지원)
    "upstage/SOLAR-10.7B-Instruct-v1.0",               # Solar 계열
    "beomi/Llama-3-Open-Ko-8B-Instruct-preview",       # Llama-3 한국어 파인튜닝
    "kakaocorp/kanana-nano-2.1b-instruct",             # 카카오 Kanana
    "mistralai/Mistral-7B-Instruct-v0.3",              # Mistral 계열
]


def _collection_name(llm_model_name):
    return "cmp_llm_" + re.sub(r"[^0-9a-zA-Z]+", "_", llm_model_name)


def _detail_path(llm_model_name):
    safe_name = re.sub(r"[^0-9a-zA-Z]+", "_", llm_model_name)
    return os.path.join(EVAL_DIR, f"detail_A_llm_{safe_name}.csv")


def evaluate_one_llm(llm_model_name, splits, vectorstore=None):
    # 이미 이 모델로 성공한 결과가 저장되어 있으면 재실행하지 않고 그대로 재사용
    # (모델 하나 돌리는 데 몇 분씩 걸리는데, 뒤 모델에서 에러 나면 처음부터 다시 돌리는 걸 방지)
    detail_path = _detail_path(llm_model_name)
    if os.path.exists(detail_path):
        print(f"\n=== 스킵(캐시 사용): {llm_model_name} ===")
        df = pd.read_csv(detail_path)
        print(
            f"[{llm_model_name}] 평균 BERTScore F1: {df['bertscore_f1'].mean():.4f} "
            f"/ 평균 응답시간: {df['response_time_sec'].mean():.2f}초"
        )
        return df

    print(f"\n{'=' * 60}\nLLM 모델: {llm_model_name}\n{'=' * 60}")

    # 임베딩/청킹/검색 파라미터는 담당 축이 아니므로 고정 → 벡터스토어는 한 번만 만들어 재사용
    if vectorstore is None:
        vectorstore = build_vectorstore(splits, collection_name=_collection_name(llm_model_name))
    retriever = build_retriever(vectorstore)
    llm = build_llm(llm_model=llm_model_name, temperature=CONFIG["temperature"])
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

    # 이후 모델에서 실패해도 이 모델 결과는 남도록 즉시 저장
    os.makedirs(EVAL_DIR, exist_ok=True)
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"모델별 상세 결과 저장 완료: {detail_path}")

    # 다음 후보 모델을 위해 GPU 메모리 확보 (8GB VRAM에 여러 모델을 동시에 못 올려둠).
    # llm/rag_chain이 모델을 참조하고 있는 동안은 empty_cache()가 소용없으므로 먼저 참조를 끊음.
    del llm, rag_chain, retriever
    clear_llm_cache()
    return df


def main():
    load_dotenv()
    print("환경 변수 로드 완료")
    print(f"비교 대상 LLM 모델 {len(LLM_CANDIDATES)}개: {LLM_CANDIDATES}")

    splits = load_and_split()
    # 임베딩은 고정 축이므로 벡터스토어를 한 번만 만들어 모든 LLM 후보가 재사용
    vectorstore = build_vectorstore(splits, collection_name="cmp_llm_shared")

    all_dfs = [evaluate_one_llm(name, splits, vectorstore=vectorstore) for name in LLM_CANDIDATES]
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

    print("\n=== LLM 모델 비교 결과 (F1 내림차순) ===")
    print(summary.to_string(index=False))
    print(f"\n요약 결과 저장 완료: {summary_path}")

    best = summary.iloc[0]
    print(f"\n최고 F1 LLM 모델: {best['llm_model']} (F1={best['mean_bertscore_f1']:.4f})")

    return df_all, summary


if __name__ == "__main__":
    main()
