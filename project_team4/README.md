# 자동차관리법 기반 RAG — 문서 분할(청킹) 전략 비교

경북대학교 AI·빅데이터 전문가 양성과정 미니 프로젝트. **자동차관리법**(법률 제21817호,
2026. 6. 16. 일부개정) 조문을 대상으로 RAG 질의응답 파이프라인을 구성하고,
**문서 분할(청킹) 전략과 그 파라미터**를 여러 종류로 바꿔가며 어떤 조합이 이 법률
문서에 가장 적합한지 비교·분석한다.

> ℹ️ LLM은 **gpt-4o-mini**(OpenAI), 임베딩은 **jhgan/ko-sroberta-multitask** 하나로
> 고정하고, **문서 분할 전략·chunk_size·chunk_overlap만 바꿔가며** 비교하는 데 집중한
> 구조다.
> ⚠️ `llm_model`은 팀 공통 BASELINE 항목이라 다른 3명(A/B/D) 노트북과 반드시 동일한
> 값이어야 결과 비교가 성립합니다. 현재 값이 팀 합의된 값인지 재확인이 필요합니다.

---

## 1. 비교 대상 청킹 전략 및 파라미터

**(1) 분할 전략 4종**

| 전략 | 방식 | 특징 |
|---|---|---|
| `recursive` (baseline) | 문자 수 기준, 구분자 우선순위대로 재귀 분할 | LangChain 기본값, 안정적 |
| `character` | 단일 구분자(`\n`) 기준 분할 | 구조 단순, 조문 경계 무시 가능성 |
| `token` | 토큰 수 기준 분할 | LLM 토큰 한도에 맞춰 정밀 |
| `semantic` | 임베딩 유사도로 의미 단위 분할 (`langchain_experimental`) | 문맥 응집도 높음, 속도 느림 |

**(2) chunk_size 스윕**: 200 / 300 / 500(baseline) / 700 / 1000
**(3) chunk_overlap 스윕**: 0 / 50(baseline) / 100 / 150
**(4) semantic 세부 파라미터 튜닝**: `breakpoint_threshold_type` × `amount`
(percentile 90/95, standard_deviation 1.0, interquartile 1.5)

## 2. 고정값 (baseline)

청킹 축만 순수하게 변수로 분리하기 위해 나머지는 전부 팀 공통 BASELINE으로 고정한다.

- LLM: `gpt-4o-mini` (temperature=0)
- 임베딩: `jhgan/ko-sroberta-multitask` (normalize=True)
- 검색: similarity, top_k=3
- VectorDB: Chroma (distance_metric=cosine)

## 3. 평가 지표

`eval_data`(질문 10개, 자동차관리법 조문 기반 Q&A)를 기준으로:

- **bertscore_f1**: `klue/bert-base` 기준 답변-정답 의미 유사도 (필수 지표, 최종 순위 기준)
- **RAGAS**: faithfulness, answer_relevancy, context_precision, context_recall
- **response_time_sec**: 질문당 평균 응답 시간
- **hallucination_count**: 근거 없는 답변 생성 횟수

## 4. 폴더 구조

```
Project_team1/
├── README.md
├── requirements.txt
├── data/
│   └── 자동차관리법.pdf              # 대상 법령 원문
├── notebooks/
│   └── C_experiment.ipynb           # PDF 로드~청킹~검색~LLM~평가를 모두 포함
├── eval/
│   ├── summary_C_model_compare.csv  # 실험1: 분할 전략 비교 결과
│   ├── summary_C_param_sweep.csv    # 실험2: chunk_size 스윕 결과
│   ├── summary_C_overlap_sweep.csv  # 실험3: chunk_overlap 스윕 결과
│   ├── summary_C_semantic_tune.csv  # 실험4: semantic 세부 파라미터 튜닝 결과
│   └── best_C.csv                   # 4개 실험 통틀어 최적 조합
├── experiments/
│   └── chunking_comparison/
│       ├── plot_C_model_compare.png
│       ├── plot_C_param_sweep.png
│       ├── plot_C_overlap_sweep.png
│       └── plot_C_semantic_tune.png
├── report/
└── slides/
```

## 5. 실행 방법

```bash
pip install -r requirements.txt
jupyter notebook notebooks/C_experiment.ipynb
```

노트북 셀을 위에서부터 순서대로 실행하면 4개 실험이 차례로 돌아가며, 결과는
`eval/summary_*.csv`와 `eval/best_C.csv`, 그래프는 `experiments/chunking_comparison/`에
저장된다.

## 6. 알려진 제약

- `semantic` 분할 전략이 의존하는 `langchain_experimental`은 유지보수가 종료(sunset)
  예정 패키지로, 향후 대체 라이브러리 검토가 필요하다.
- Chroma의 근사 최근접 이웃 검색(HNSW) 특성상, **동일한 설정으로 재실행해도 지표가
  다소 달라질 수 있다.** 실제로 동일 config(recursive, chunk_size=500)를 두 실험에서
  각각 실행했을 때 BERTScore F1이 0.53 / 0.64로 차이가 났다 — 최종 결론 도출 전
  상위 후보는 2~3회 반복 실행 후 평균값으로 비교할 것을 권장한다.
- `bert_score`가 `klue/bert-base`의 레이어 수를 자동으로 인식하지 못해
  `num_layers=12`를 직접 지정해야 한다.
