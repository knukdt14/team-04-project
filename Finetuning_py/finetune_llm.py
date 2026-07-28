"""Qwen2.5-7B-Instruct(base_llm_model)를 자동 생성된 (조문+질문 -> 답변) 데이터로 QLoRA 파인튜닝.
4bit 양자화 베이스 모델 위에 LoRA 어댑터만 학습 (8GB VRAM 제약 때문에 전체 파라미터 학습은 불가능).
정답(answer) 토큰에 대해서만 loss를 계산하도록 prompt 부분은 label을 -100으로 마스킹.
"""
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from config import CONFIG, LORA_OUTPUT_DIR, QA_DATA_PATH
from models import load_base_model_and_tokenizer


class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.examples = []
        skipped = 0
        for _, row in df.iterrows():
            prompt_text = CONFIG["prompt_template"].format(context=row["context"], question=row["question"])
            # transformers 5.x에서 tokenize=True는 list가 아니라 BatchEncoding을 반환하므로 ["input_ids"]로 추출
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=True, add_generation_prompt=True,
            )["input_ids"]
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}, {"role": "assistant", "content": row["answer"]}],
                tokenize=True, add_generation_prompt=False,
            )["input_ids"]
            if len(full_ids) > max_length:
                skipped += 1
                continue
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = list(full_ids)
            for i in range(prompt_len):
                labels[i] = -100
            self.examples.append({"input_ids": full_ids, "labels": labels})
        print(f"학습 샘플 {len(self.examples)}개 준비 완료 (max_length={max_length} 초과로 {skipped}개 제외)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attention_mask = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    return collate_fn


def main():
    df = pd.read_csv(QA_DATA_PATH)
    print(f"학습 데이터 로드: {len(df)}개 (조문+질문 -> 답변) 샘플 ({QA_DATA_PATH})")

    model, tokenizer = load_base_model_and_tokenizer(CONFIG["base_llm_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 체크포인팅을 껐더니 backward 시 동결된 4bit 가중치의 역양자화 결과가 그대로 남아
    # reserved 메모리가 16GB+(물리 VRAM 8GB 초과)까지 치솟아 시스템 RAM으로 스필오버 -> 마이크로배치당 58초.
    # 체크포인팅을 켜되 use_reentrant=True(기본값)도 4bit+PEFT 조합에서 비효율적이라 여전히 느림.
    # use_reentrant=False로 명시하니 마이크로배치당 5.7초로 10배 개선됨.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False

    dataset = QADataset(df, tokenizer, max_length=CONFIG["llm_ft_max_length"])

    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR + "_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=CONFIG["llm_ft_epochs"],
        learning_rate=CONFIG["llm_ft_lr"],
        # fp16=True(Trainer의 autocast+GradScaler)를 켜면 4bit(bnb) 내부 fp16 연산과 이중으로 겹쳐서
        # iteration당 5.7s -> 25s로 4.4배 느려짐(체크포인팅 재계산과 맞물려 특히 악화).
        # bnb_4bit_compute_dtype=torch.float16으로 이미 fp16 연산 중이라 굳이 필요 없어서 끔.
        fp16=False,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"QLoRA 어댑터 저장 완료: {LORA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
