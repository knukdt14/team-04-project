"""파인튜닝 단계(임베딩/LLM) 결과를 RAG 튜닝 baseline과 비교하는 표/그래프 생성."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EVAL_DIR

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

rows = [
    ("RAG 튜닝 최종\n(baseline)", 0.6815, "baseline"),
    ("임베딩(KoE5)\nLoRA 파인튜닝", 0.6755, "파인튜닝(하락→폐기)"),
    ("LLM(Qwen2.5-7B)\nQLoRA 파인튜닝", 0.7020, "파인튜닝(개선→채택)"),
]
df = pd.DataFrame(rows, columns=["label", "f1", "group"])
df.to_csv(f"{EVAL_DIR}/results_finetuning_summary.csv", index=False, encoding="utf-8-sig")

colors = {"baseline": "#8C8C8C", "파인튜닝(하락→폐기)": "#C44E52", "파인튜닝(개선→채택)": "#55A868"}
bar_colors = [colors[g] for g in df["group"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df["label"], df["f1"], color=bar_colors, width=0.5)
ax.set_ylabel("BERTScore F1")
ax.set_ylim(0, max(df["f1"]) * 1.2)
ax.axhline(0.6815, color="#8C8C8C", linestyle="--", linewidth=1, alpha=0.6)
ax.set_title("RAG 튜닝 baseline vs 파인튜닝(임베딩/LLM) 결과", fontsize=13)

for bar, value in zip(bars, df["f1"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}",
             ha="center", va="bottom", fontsize=10)

from matplotlib.patches import Patch
legend_handles = [Patch(color=c, label=g) for g, c in colors.items()]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(f"{EVAL_DIR}/finetuning_comparison.png", dpi=150)
print(f"저장 완료: {EVAL_DIR}/finetuning_comparison.png, {EVAL_DIR}/results_finetuning_summary.csv")
