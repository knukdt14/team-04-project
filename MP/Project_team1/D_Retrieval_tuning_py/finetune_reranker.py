"""
Reranker(BAAI/bge-reranker-v2-m3) 파인튜닝 스크립트.

기존 파이프라인(models.py, config.py, data_prep.py)은 전혀 수정하지 않습니다.
이 스크립트로 만든 결과물(로컬 폴더)을 models.py의 _RERANKER_MODEL 값만
바꿔서 그대로 꽂아 쓰는 구조입니다.

왜 리랭커인가:
    - 5개 전략 비교에서 dense_rerank가 MRR(검색 정확도)은 1등인데
      Fact Match(팩트 정확도)는 꼴찌였음
      -> "의미상 가까운 후보를 상위로 올리지만, 정작 정답 근거는 놓치는" 경향
    - 리랭커만 도메인 데이터로 다시 학습시키면 이 문제를 직접 겨냥해서 고칠 수 있음
    - 임베딩 전체를 흔드는 것보다 학습 범위가 좁아 (이미 뽑힌 후보들의 순서만
      재조정) 리스크가 낮음 (참고: 임베딩 파인튜닝은 팀에서 이미 시도했다가
      성능 하락으로 폐기된 전례가 있음)

데이터: synthetic_qa_dataset.csv (조문 391개 중 387개에 대해 자동 생성된
        (question, context, answer) 쌍) 를 학습 데이터로 재활용합니다.
        -> D_Retrieval_tuning_py/ 폴더에 이 CSV를 넣고 실행하세요.

negative(오답) 샘플링:
    같은 질문에 대해, 정답 조문이 아닌 다른 무작위 청크를 negative로 씀.
    단, "제34조" 같은 조항번호가 같은 청크는 실제로 관련 있을 수 있어
    negative에서 제외합니다 (오라벨 방지).

사용법:
    cd D_Retrieval_tuning_py
    pip install sentence-transformers
    python finetune_reranker.py
"""

import random
import re

import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

from data_prep import load_and_split_markdown

SEED = 42
QA_CSV_PATH = "synthetic_qa_dataset.csv"      # eval/synthetic_qa_dataset.csv 를 이 폴더로 복사해서 사용
BASE_MODEL = "BAAI/bge-reranker-v2-m3"
OUTPUT_DIR = "./finetuned_reranker"           # 완료 후 이 경로를 models.py의 _RERANKER_MODEL에 넣으면 됨

NUM_NEGATIVES_PER_QUERY = 2      # 질문 1개당 오답 청크 몇 개 붙일지
VAL_RATIO = 0.1
BATCH_SIZE = 8                    # 8GB VRAM 기준. OOM 나면 4로 낮추세요
EPOCHS = 3

ARTICLE_PATTERN = re.compile(r"제\s*\d+\s*조(?:\s*의\s*\d+)?")


def _articles(text):
    if not isinstance(text, str):
        return set()
    return {re.sub(r"\s+", "", m) for m in ARTICLE_PATTERN.findall(text)}


def build_training_examples():
    random.seed(SEED)

    qa = pd.read_csv(QA_CSV_PATH, encoding="utf-8-sig")
    qa = qa.dropna(subset=["question", "context"]).reset_index(drop=True)
    print(f"QA 쌍 {len(qa)}개 로드 완료")

    # negative 후보 풀: 전체 청크(이미 만들어둔 load_and_split_markdown 재사용, 코드 변경 없음)
    splits = load_and_split_markdown()
    all_chunks = [d.page_content for d in splits]
    print(f"negative 샘플링용 전체 청크 {len(all_chunks)}개")

    examples = []
    skipped = 0
    for _, row in qa.iterrows():
        question = row["question"]
        positive = row["context"]
        pos_articles = _articles(positive)

        examples.append(InputExample(texts=[question, positive], label=1.0))

        # 같은 조항번호를 담은 청크는 negative에서 제외 (실질적으로 관련 있을 수 있어서)
        candidates = [
            c for c in all_chunks
            if c != positive and not (pos_articles & _articles(c))
        ]
        if len(candidates) < NUM_NEGATIVES_PER_QUERY:
            skipped += 1
            continue
        negatives = random.sample(candidates, NUM_NEGATIVES_PER_QUERY)
        for neg in negatives:
            examples.append(InputExample(texts=[question, neg], label=0.0))

    print(f"학습 예제 총 {len(examples)}개 생성 (negative 부족으로 건너뛴 질문 {skipped}개)")
    return examples


def main():
    examples = build_training_examples()
    random.shuffle(examples)

    n_val = int(len(examples) * VAL_RATIO)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]
    print(f"train={len(train_examples)}, val={len(val_examples)}")

    print(f"\n베이스 모델 로드: {BASE_MODEL}")
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=512)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    print(f"\n파인튜닝 시작 (epochs={EPOCHS}, batch_size={BATCH_SIZE}) ...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=OUTPUT_DIR,
        show_progress_bar=True,
    )

    print(f"\n파인튜닝 완료 -> {OUTPUT_DIR}")
    print("이제 models.py의 다음 줄만 바꾸면 그대로 적용됩니다:")
    print(f'    _RERANKER_MODEL = "{BASE_MODEL}"')
    print(f'    ->')
    print(f'    _RERANKER_MODEL = "{OUTPUT_DIR}"')
    print("\n그 다음 config.py에서 retrieval_strategy를 \"dense_rerank\"로 두고")
    print("evaluate.py / eval_extended.py를 다시 돌려서 파인튜닝 전후 성능을 비교하세요.")


if __name__ == "__main__":
    main()
