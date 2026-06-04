"""Figure 2 — Per-category grouped bar chart comparing all four systems."""
import matplotlib.pyplot as plt
import numpy as np

RAG_COLOR = "#4C72B0"
PIPELINE_COLOR = "#DD8452"
V1_COLOR = "#55A868"
V2_COLOR = "#C44E52"

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CATEGORIES = ["Single-hop", "Temporal", "Multi-hop", "Open-domain"]
RAG = [31.9, 24.9, 22.9, 58.4]
PIPE_V2 = [38.7, 57.3, 28.1, 47.3]
EID_V1 = [40.1, 64.2, 40.6, 60.5]
EID_V2 = [50.4, 75.1, 51.0, 70.5]

WIDTH = 0.18
SPACING = 0.8

x = np.arange(len(CATEGORIES))
group_pos = x * SPACING
offsets = [-1.5 * WIDTH, -0.5 * WIDTH, 0.5 * WIDTH, 1.5 * WIDTH]

fig, ax = plt.subplots(figsize=(7, 4))

b1 = ax.bar(group_pos + offsets[0], RAG,     WIDTH, color=RAG_COLOR,      label="RAG Baseline")
b2 = ax.bar(group_pos + offsets[1], PIPE_V2, WIDTH, color=PIPELINE_COLOR, label="Pipeline v2")
b3 = ax.bar(group_pos + offsets[2], EID_V1,  WIDTH, color=V1_COLOR,       label="Eidetic Memory v1")
b4 = ax.bar(group_pos + offsets[3], EID_V2,  WIDTH, color=V2_COLOR,       label="Eidetic Memory v2 (ours)")

for bars, values, bold, fs, col in [
    (b1, RAG,     False, 7, "#555555"),
    (b2, PIPE_V2, False, 7, "#555555"),
    (b3, EID_V1,  False, 7, "#555555"),
    (b4, EID_V2,  True,  9, "black"),
]:
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=fs,
            fontweight="bold" if bold else "normal", color=col,
        )

# Double-headed bracket on Temporal group (+50.2 pp over RAG)
temporal_idx = 1
x_rag = group_pos[temporal_idx] + offsets[0]
x_v2  = group_pos[temporal_idx] + offsets[3]
y_bracket = 83
ax.annotate(
    "",
    xy=(x_rag + WIDTH / 2, y_bracket),
    xytext=(x_v2 + WIDTH / 2, y_bracket),
    arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.2},
)
ax.text(
    (x_rag + x_v2 + WIDTH) / 2, y_bracket + 1,
    "+50.2 pp over RAG",
    ha="center", va="bottom", fontsize=8,
)

ax.set_xticks(group_pos)
ax.set_xticklabels(CATEGORIES)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 90)
ax.set_yticks([0, 20, 40, 60, 80])
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("figures/category_comparison.pdf", format="pdf", bbox_inches="tight")
print("Saved figures/category_comparison.pdf")
