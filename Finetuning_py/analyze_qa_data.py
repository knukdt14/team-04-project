"""생성된 합성 QA 데이터셋(eval/synthetic_qa_dataset.csv)의 통계 요약과 길이 분포 시각화를 저장.
TROUBLESHOOTING.md에 첨부할 표/그래프 산출용."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EVAL_DIR, QA_DATA_PATH

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

GEN_TOTAL_ARTICLES = 391
GEN_FAILED = 4
GEN_ELAPSED_MIN = 51.7


def main():
    df = pd.read_csv(QA_DATA_PATH)

    q_len = df["question"].str.len()
    a_len = df["answer"].str.len()
    c_len = df["context"].str.len()

    summary = pd.DataFrame([{
        "전체 조문 수": GEN_TOTAL_ARTICLES,
        "생성 성공": len(df),
        "생성 실패": GEN_FAILED,
        "생성 소요시간(분)": GEN_ELAPSED_MIN,
        "질문 평균 길이": round(q_len.mean(), 1),
        "답변 평균 길이": round(a_len.mean(), 1),
        "조문(context) 평균 길이": round(c_len.mean(), 1),
    }])
    summary_path = f"{EVAL_DIR}/synthetic_qa_generation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"요약 통계 저장 완료: {summary_path}")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, series, title, color in zip(
        axes,
        [q_len, a_len, c_len],
        ["질문 길이 분포", "답변 길이 분포", "조문(context) 길이 분포"],
        ["#4C72B0", "#DD8452", "#55A868"],
    ):
        ax.hist(series, bins=20, color=color, edgecolor="white")
        ax.set_title(f"{title}\n(평균 {series.mean():.0f}자, n={len(series)})", fontsize=10)
        ax.set_xlabel("글자 수")
        ax.set_ylabel("개수")

    fig.suptitle("자동차관리법 합성 QA 데이터셋(387개) 길이 분포", fontsize=13)
    fig.tight_layout()
    chart_path = f"{EVAL_DIR}/synthetic_qa_length_distribution.png"
    fig.savefig(chart_path, dpi=150)
    print(f"길이 분포 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    main()
