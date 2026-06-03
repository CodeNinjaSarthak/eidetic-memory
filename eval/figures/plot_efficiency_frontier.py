"""Figure 5 — Efficiency frontier: LLM calls vs accuracy."""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RAG_COLOR      = "#4C72B0"
PIPELINE_COLOR = "#DD8452"
V1_COLOR       = "#55A868"
V2_COLOR       = "#C44E52"
MEM0_COLOR     = "#8172B2"
GRAY           = "#999999"

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# (name, avg_llm_calls, accuracy, marker, color, calls_known)
SYSTEMS = [
    ("RAG baseline",      1.0,  44.4, "o", RAG_COLOR,      True),
    ("Pipeline v2",       1.0,  46.6, "o", PIPELINE_COLOR, True),
    ("Eidetic Memory v2", 1.9,  66.6, "o", V2_COLOR,       True),
    ("Mem0",              2.0,  66.9, "s", MEM0_COLOR,     True),
    ("Memobase",          None, 75.8, "s", GRAY,           False),
    ("Hindsight-20B",     None, 83.2, "s", GRAY,           False),
    ("Hindsight-120B",    None, 85.7, "s", GRAY,           False),
]

# Explicit (dx, dy) offsets in data coordinates for each label
LABEL_OFFSETS = {
    "RAG baseline":      (+0.05, -4.0),
    "Pipeline v2":       (+0.05, +2.0),
    "Eidetic Memory v2": (-0.30, -4.0),
    "Mem0":              (+0.05, +2.0),
    "Memobase":          (-0.40, +1.0),
    "Hindsight-20B":     (-0.50, +1.0),
    "Hindsight-120B":    (-0.55, -3.0),
}

X_UNKNOWN = 2.7

fig, ax = plt.subplots(figsize=(6, 5))

# Shaded band for systems with unknown call counts
ax.axvspan(2.5, 3.5, alpha=0.10, color=GRAY)
ax.text(3.0, 90.5, "call count\nnot reported", ha="center", va="top", fontsize=8, color="#666666")

# Reference line at Eidetic v2 accuracy
ax.axhline(66.6, color=V2_COLOR, linestyle="--", lw=1.0, alpha=0.65)
ax.text(0.53, 67.3, "Eidetic Memory v2 (66.6%)", fontsize=8, color=V2_COLOR)

for name, calls, acc, marker, color, known in SYSTEMS:
    x = calls if known else X_UNKNOWN
    size = 200 if name == "Eidetic Memory v2" else 100
    ax.scatter(x, acc, marker=marker, color=color, s=size, zorder=5)
    dx, dy = LABEL_OFFSETS.get(name, (+0.05, +1.2))
    ax.annotate(
        name,
        xy=(x, acc),
        xytext=(x + dx, acc + dy),
        fontsize=8,
        fontweight="bold" if name == "Eidetic Memory v2" else "normal",
        ha="left" if dx >= 0 else "right",
    )

# Arrow pointing to Eidetic Memory v2
ax.annotate(
    "",
    xy=(1.9, 66.6),
    xytext=(1.55, 60.5),
    arrowprops=dict(arrowstyle="->", color=V2_COLOR, lw=1.5),
)

# Legend
this_work = mlines.Line2D([], [], marker="o", color="w", markerfacecolor=V2_COLOR,
                          markersize=8, label="This work (filled circles)")
prior_work = mlines.Line2D([], [], marker="s", color="w", markerfacecolor=GRAY,
                           markersize=8, label="Prior work (filled squares)")
ax.legend(handles=[this_work, prior_work], fontsize=8, loc="upper left", frameon=False)

ax.set_xlabel("Average LLM calls per query")
ax.set_ylabel("Overall accuracy on LoCoMo (%)")
ax.set_xlim(0.5, 3.5)
ax.set_ylim(40, 92)
ax.yaxis.grid(True, linestyle="--", alpha=0.45)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("figures/efficiency_frontier.pdf", format="pdf", bbox_inches="tight")
print("Saved figures/efficiency_frontier.pdf")
