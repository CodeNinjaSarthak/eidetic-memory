"""Figure 3 — Ablation progression bar chart."""
import matplotlib.pyplot as plt
import numpy as np

V1_COLOR = "#55A868"
V2_COLOR = "#C44E52"
GRAY = "#999999"

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LABELS = [
    "No isolation\nno rerank",
    "+ Isolation\nonly",
    "+ Isolation\n+ RR",
    "+ RR\n+ cross-enc.",
    "+ Score-based\n+ cross-enc.",
    "+ NER\n+ two-pass",
    "Full pipeline\n(all 10 convs)",
]
SCORES = [36.5, 26.6, 25.8, 55.8, 55.8, 56.6, 56.3]
COLORS = [GRAY, GRAY, GRAY, V1_COLOR, V1_COLOR, V1_COLOR, V2_COLOR]

x = np.arange(len(LABELS))
fig, ax = plt.subplots(figsize=(8, 4))

bars = ax.bar(x, SCORES, color=COLORS, width=0.6, edgecolor="white", linewidth=0.5)

for bar, score in zip(bars, SCORES):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
        f"{score}",
        ha="center", va="bottom", fontsize=8, fontweight="bold",
    )

# Double-headed arrow between bar index 2 (25.8) and bar index 3 (55.8)
y_arr = 65
ax.annotate(
    "",
    xy=(x[3], y_arr),
    xytext=(x[2], y_arr),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
)
ax.text(
    (x[2] + x[3]) / 2, y_arr + 0.8,
    "+29.2 pp\n(reranker)",
    ha="center", va="bottom", fontsize=8,
)

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=8)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 80)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

fig.text(
    0.5, -0.04,
    "All bars 2–7 include per-speaker isolation. RR = round-robin merge.",
    ha="center", va="bottom", fontsize=8, color="#555555",
)

fig.tight_layout()
fig.savefig("figures/ablation_progression.pdf", format="pdf", bbox_inches="tight")
print("Saved figures/ablation_progression.pdf")
