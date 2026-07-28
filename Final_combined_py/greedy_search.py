"""
A/B/C/D 각 축에서 독립적으로 찾은 최고 조합을 한꺼번에 합쳤더니(F1 0.5952) 오히려
개별 최고 성능(0.64~0.67)보다 낮아지는 상호작용이 확인됨. 그래서 이미 알려진 최고점
(KoE5 + chunk_500 + Qwen2.5-7B + dense = F1 0.6715, B_Embedding_tuning에서 실측)을
출발점 삼아, 한 번에 변수 하나씩만 바꿔가며(coordinate ascent) 개선 여부를 확인한다.
"""
import time

import pandas as pd
from bert_score import score as bertscore
from dotenv import load_dotenv

from data_prep import load_and_split_markdown
from eval_data import EVAL_DATA
from models import build_llm, build_retriever, build_vectorstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

PROMPT_TEMPLATE = (
    "다음 문맥을 근거로 질문에 답하세요. 문맥에 없는 내용은 모른다고 답하세요. "
    "반드시 한국어로만 답변하세요.\n"
    "[문맥]\n{context}\n\n[질문]\n{question}"
)

# 이미 B_Embedding_tuning 비교에서 실측된 값 (재실행 불필요)
KNOWN_BEST = {
    "name": "KoE5 + chunk_500 + Qwen2.5-7B-Instruct + dense",
    "f1": 0.6715,
}


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def run_once(name, splits, embedding_model, llm_model_name, retrieval_strategy, hybrid_weights=None):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    collection_name = "greedy_" + name.lower().replace(" ", "_").replace("+", "").replace(".", "_").replace("-", "_").replace("/", "_")
    vectorstore = build_vectorstore(splits, embedding_model_name=embedding_model, collection_name=collection_name)

    if retrieval_strategy == "hybrid":
        import models
        retriever = models.build_hybrid_retriever(vectorstore, splits, weights=hybrid_weights)
    else:
        retriever = build_retriever(vectorstore, splits, strategy="dense")

    llm = build_llm(llm_model_name)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    rows = []
    for item in EVAL_DATA:
        q = item["question"]
        start = time.time()
        answer = rag_chain.invoke(q)
        elapsed = time.time() - start
        rows.append({"question": q, "ground_truth": item["ground_truth"], "answer": answer,
                      "response_time_sec": round(elapsed, 3)})
    df = pd.DataFrame(rows)
    P, R, F1 = bertscore(df["answer"].tolist(), df["ground_truth"].tolist(),
                          model_type="klue/bert-base", num_layers=12, lang="ko", verbose=False)
    df["bertscore_f1"] = F1.tolist()
    mean_f1 = df["bertscore_f1"].mean()
    print(f"[{name}] F1={mean_f1:.4f} / 평균 응답시간={df['response_time_sec'].mean():.2f}초")
    return mean_f1


def main():
    load_dotenv()
    results = {KNOWN_BEST["name"]: KNOWN_BEST["f1"]}
    print(f"출발점(기지값, 재실행 안 함): {KNOWN_BEST['name']} = {KNOWN_BEST['f1']:.4f}")

    splits_500 = load_and_split_markdown(chunk_size=500, chunk_overlap=50)

    # STEP 1: LLM 교체 (KoE5 + chunk_500 + dense 고정) — Qwen2.5-7B(기지값) vs Phi-4-mini-instruct
    step1_name = "KoE5 + chunk_500 + Phi-4-mini-instruct + dense"
    f1_step1 = run_once(step1_name, splits_500, "nlpai-lab/KoE5", "microsoft/Phi-4-mini-instruct", "dense")
    results[step1_name] = f1_step1

    best_llm = "microsoft/Phi-4-mini-instruct" if f1_step1 > KNOWN_BEST["f1"] else "Qwen/Qwen2.5-7B-Instruct"
    best_after_step1 = max(f1_step1, KNOWN_BEST["f1"])
    print(f"\n>>> STEP1 승자: {best_llm} (F1={best_after_step1:.4f})\n")

    # STEP 2: chunk_size 교체 (best_llm + KoE5 + dense 고정) — 500(이미 알거나 방금 측정) vs 800
    splits_800 = load_and_split_markdown(chunk_size=800, chunk_overlap=100)
    step2_name = f"KoE5 + chunk_800 + {best_llm.split('/')[-1]} + dense"
    f1_step2 = run_once(step2_name, splits_800, "nlpai-lab/KoE5", best_llm, "dense")
    results[step2_name] = f1_step2

    best_chunk_splits, best_chunk_label = (splits_800, "chunk_800") if f1_step2 > best_after_step1 else (splits_500, "chunk_500")
    best_after_step2 = max(f1_step2, best_after_step1)
    print(f"\n>>> STEP2 승자: {best_chunk_label} (F1={best_after_step2:.4f})\n")

    # STEP 3: 검색 전략 교체 (best_llm + best_chunk + KoE5 고정) — dense(이미 측정) vs hybrid_0.3_0.7
    step3_name = f"KoE5 + {best_chunk_label} + {best_llm.split('/')[-1]} + hybrid_0.3_0.7"
    f1_step3 = run_once(step3_name, best_chunk_splits, "nlpai-lab/KoE5", best_llm, "hybrid", hybrid_weights=[0.3, 0.7])
    results[step3_name] = f1_step3

    best_retrieval = "hybrid_0.3_0.7" if f1_step3 > best_after_step2 else "dense"
    best_final = max(f1_step3, best_after_step2)

    print("\n\n=== GREEDY SEARCH 요약 ===")
    for name, f1 in results.items():
        print(f"{name}: {f1:.4f}")
    print(f"\n최종 최고 조합: KoE5 + {best_chunk_label} + {best_llm} + {best_retrieval} (F1={best_final:.4f})")


if __name__ == "__main__":
    main()
