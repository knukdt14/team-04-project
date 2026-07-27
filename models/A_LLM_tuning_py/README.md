# A 담당 — LLM 모델 튜닝

담당 축: **LLM 모델 종류 + temperature**
고정 축(담당 아님): embedding 모델, chunk_size/overlap, top_k, search_type — 전부 팀 BASELINE 값 사용

## 실행 방법

```
cd models/A_LLM_tuning_py

# 단일 실행 (config.py의 기본 llm_model로 1회 평가)
python evaluate.py

# 실험 1: LLM 모델 비교 (4개)
python compare_llms.py

# 실험 2: temperature 스윕 (0.0/0.2/0.4/0.7/1.0)
python compare_temperature.py
```

결과는 `models/eval/` 아래 CSV로 저장됩니다.

## 왜 로컬 CPU인가

- 원래 BASELINE LLM(Qwen2.5-7B-Instruct)을 HF Inference API(원격)로 호출하려 했으나
  계정의 월간 무료 크레딧이 소진되어 402 Payment Required 에러 발생.
- 이 PC에 GPU가 없어(CUDA 미탑재) 7B급 모델을 로컬에서 돌리기엔 너무 느림.
- 그래서 `HuggingFacePipeline`으로 로컬 CPU에서 직접 추론하되, 모델 크기를 1.5~3B급으로
  낮춰서(BASELINE의 축소 버전) 현실적인 시간 안에 여러 모델/온도를 비교할 수 있게 함.

## LLM_CANDIDATES 선정 과정에서 제외한 모델과 이유

| 모델 | 제외 사유 |
|---|---|
| EXAONE-3.5-*B-Instruct | 리포지토리 커스텀 코드가 최신 transformers와 충돌 (`ImportError: RopeParameters`) |
| google/gemma-2-*-it | gated 모델 — 이 계정에 접근 권한 없음 (401) |
| beomi/gemma-ko-2b | instruct 튜닝이 안 된 base 모델이라 `tokenizer.chat_template` 없음 |
| microsoft/Phi-3.5-mini-instruct | (한때 제외했다가 재포함) `trust_remote_code=True`를 강제로 켜뒀을 때만 실패했고,
  기본값(False)으로 두면 transformers 내장 네이티브 구현을 사용해 정상 동작함 |

## 알려진 이슈와 해결

- **bert_score KeyError('klue/bert-base')**: `bert_score`의 내장 `model2layers` 목록에
  `klue/bert-base`가 없어서 자동으로 레이어 수를 못 찾음 → `num_layers=12`를 직접 지정해서 해결
  (klue/bert-base는 12-layer BERT-base 구조).
- **답변에 프롬프트 전체가 그대로 echo되는 문제**: `HuggingFacePipeline`의 기본 동작은
  입력 프롬프트를 포함해서 반환하므로, `pipeline_kwargs`에 `return_full_text=False`를 반드시 넣어야 함.
