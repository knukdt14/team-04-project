"""자동차관리법 조문(article) 각각을 근거로 LLM이 (질문, 답변) 쌍을 자동 생성.
- 임베딩 파인튜닝: (question, context) 쌍으로 대조학습(MultipleNegativesRankingLoss)
- LLM 파인튜닝: (context+question -> answer) 형태로 QLoRA instruction tuning
평가용 eval_data.py의 10문항과는 별개의, 훨씬 큰 규모의 학습 전용 데이터셋을 만드는 것이 목적.
"""
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv

from config import CONFIG, QA_DATA_PATH
from data_prep import load_and_split_article
from models import build_llm

QA_GEN_PROMPT = (
    "다음은 대한민국 자동차관리법의 한 조문입니다. 이 조문 내용만을 근거로 답할 수 있는 "
    "질문 1개와 정확한 답변 1개를 만드세요. 질문은 실제 사용자가 궁금해할 법한 자연스러운 문장으로, "
    "답변은 조문에 실제로 있는 사실만 근거로 구체적으로 작성하세요. "
    "다른 설명 없이 아래 JSON 형식으로만 답하세요.\n"
    '{{"question": "...", "answer": "..."}}\n\n'
    "[조문]\n{context}"
)

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
MIN_CONTEXT_LEN = 40


def _parse_qa(raw_text):
    match = _JSON_PATTERN.search(raw_text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    q, a = obj.get("question"), obj.get("answer")
    if not q or not a:
        return None
    return q.strip(), a.strip()


def main():
    load_dotenv()
    splits = load_and_split_article()
    llm = build_llm(CONFIG["qa_gen_llm_model"])

    rows = []
    failed = 0
    skipped_short = 0
    start_all = time.time()

    for i, doc in enumerate(splits):
        context = doc.page_content.strip()
        if len(context) < MIN_CONTEXT_LEN:
            skipped_short += 1
            continue

        prompt = QA_GEN_PROMPT.format(context=context)
        start = time.time()
        try:
            raw = llm.invoke(prompt).content
        except Exception as e:
            print(f"[{i}/{len(splits)}] LLM 호출 실패, 건너뜀: {type(e).__name__}: {e}")
            failed += 1
            continue

        parsed = _parse_qa(raw)
        if parsed is None:
            print(f"[{i}/{len(splits)}] JSON 파싱 실패, 건너뜀. 원본 출력 앞부분: {raw[:120]!r}")
            failed += 1
            continue

        question, answer = parsed
        elapsed = time.time() - start
        rows.append({"chunk_id": i, "context": context, "question": question, "answer": answer})
        print(f"[{i}/{len(splits)}] ({elapsed:.1f}s) Q: {question[:70]}")

    total_elapsed = time.time() - start_all
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(QA_DATA_PATH), exist_ok=True)
    df.to_csv(QA_DATA_PATH, index=False, encoding="utf-8-sig")

    print("\n=== 합성 QA 데이터 생성 요약 ===")
    print(f"전체 조문 수      : {len(splits)}")
    print(f"짧아서 건너뜀     : {skipped_short}")
    print(f"생성 실패         : {failed}")
    print(f"생성 성공         : {len(df)}")
    print(f"총 소요 시간      : {total_elapsed / 60:.1f}분")
    print(f"저장 경로         : {QA_DATA_PATH}")


if __name__ == "__main__":
    main()
