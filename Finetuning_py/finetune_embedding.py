"""KoE5(base_embedding_model)를 자동 생성된 (질문, 조문) 쌍으로 대조학습(contrastive fine-tuning).
MultipleNegativesRankingLoss: 배치 내 정답 조문을 positive, 나머지 조문들을 in-batch negative로 사용 —
질문 임베딩과 정답 조문 임베딩이 서로 가까워지도록 학습하여 RAG 검색(retrieval) 품질을 높이는 것이 목적.

KoE5는 XLM-RoBERTa-large 기반(560M 파라미터)이라 전체 파라미터를 그대로 fine-tuning하면
Adam 옵티마이저 상태(파라미터의 약 2배)만으로 8GB VRAM을 초과해 OOM이 발생한다.
-> LoRA(peft)로 attention의 query/key/value만 학습하고, 학습 후 병합(merge)해서
   평가 시에는 peft 없이 일반 SentenceTransformer 모델처럼 바로 로드할 수 있게 저장한다.
"""
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from config import CONFIG, EMBEDDING_OUTPUT_DIR, QA_DATA_PATH


def main():
    df = pd.read_csv(QA_DATA_PATH)
    print(f"학습 데이터 로드: {len(df)}개 (질문-조문) 쌍 ({QA_DATA_PATH})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        CONFIG["base_embedding_model"], device=device,
        model_kwargs={"torch_dtype": torch.float16},
    )
    print(f"베이스 임베딩 모델 로드 완료: {CONFIG['base_embedding_model']} (device={device}, fp16)")

    lora_config = LoraConfig(
        r=CONFIG["embedding_lora_r"],
        lora_alpha=CONFIG["embedding_lora_alpha"],
        lora_dropout=CONFIG["embedding_lora_dropout"],
        target_modules=CONFIG["embedding_lora_target_modules"],
        bias="none",
    )
    # Transformer.auto_model은 setter 없는 property(getter가 self.model을 반환)라서
    # model[0].auto_model = ... 로 대입하면 실제로는 별개의 "auto_model" 서브모듈이 새로 생성되어
    # 무시된다. 진짜 저장 위치인 self.model에 직접 대입해야 함.
    model[0].model = get_peft_model(model[0].auto_model, lora_config)
    model[0].auto_model.print_trainable_parameters()

    train_examples = [
        InputExample(texts=[row["question"], row["context"]])
        for _, row in df.iterrows()
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=CONFIG["embedding_batch_size"])
    train_loss = losses.MultipleNegativesRankingLoss(model)

    epochs = CONFIG["embedding_epochs"]
    warmup_steps = max(1, int(len(train_dataloader) * epochs * 0.1))
    print(f"학습 시작: epochs={epochs}, batch_size={CONFIG['embedding_batch_size']}, warmup_steps={warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    # LoRA 어댑터를 베이스 가중치에 병합 -> 평가 시 peft 없이 일반 모델처럼 바로 로드 가능
    model[0].model = model[0].auto_model.merge_and_unload()
    model.save(EMBEDDING_OUTPUT_DIR)
    print(f"파인튜닝된 임베딩 모델 저장 완료(LoRA 병합됨): {EMBEDDING_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
