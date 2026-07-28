import os

import matplotlib.pyplot as plt
import pandas as pd

from config import EVAL_DIR

# Windows 기본 한글 폰트 (없으면 한글이 네모(□)로 깨짐)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DETAIL_PATH = os.path.join(EVAL_DIR, "results_D_retrieval_comparison_all.csv")
SUMMARY_PATH = os.path.join(EVAL_DIR, "results_D_retrieval_comparison_summary.csv")
OUTPUT_PATH = os.path.join(EVAL_DIR, "retrieval_comparison.png")


def main():
    df_all = pd.read_csv(DETAIL_PATH)
    summary = pd.read_csv(SUMMARY_PATH).sort_values("mean_bertscore_f1", ascending=False)

    labels = summary["retrieval_strategy"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    ax = axes[0]
    bars = ax.bar(labels, summary["mean_bertscore_f1"], color="#4C72B0")
    ax.set_title("검색 전략별 평균 BERTScore F1")
    ax.set_ylabel("BERTScore F1")
    ax.set_ylim(0, max(summary["mean_bertscore_f1"]) * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    for bar, value in zip(bars, summary["mean_bertscore_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                 ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bars = ax.bar(labels, summary["mean_response_time_sec"], color="#DD8452")
    ax.set_title("검색 전략별 평균 응답시간 (초)")
    ax.set_ylabel("초")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    for bar, value in zip(bars, summary["mean_response_time_sec"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}s",
                 ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    order = summary["retrieval_strategy"].tolist()
    data = [df_all.loc[df_all["retrieval_strategy"] == s, "bertscore_f1"].values for s in order]
    ax.boxplot(data, tick_labels=order)
    ax.set_title("검색 전략별 질문 단위 F1 분포")
    ax.set_ylabel("BERTScore F1")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    fig.suptitle("검색 전략 비교 (자동차관리법 RAG)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"그래프 저장 완료: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
