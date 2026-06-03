"""Figure 4 — Per-conversation, per-category accuracy delta heatmap (Eidetic v2 − RAG)."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CONV_IDS = [
    "conv-26", "conv-30", "conv-41", "conv-42", "conv-43",
    "conv-44", "conv-47", "conv-48", "conv-49", "conv-50",
]
CONV_LABELS = [
    "conv-26*", "conv-30*", "conv-41", "conv-42", "conv-43",
    "conv-44",  "conv-47",  "conv-48", "conv-49", "conv-50",
]
CATS = [1, 2, 3, 4]
CAT_LABELS = ["Single-hop", "Temporal", "Multi-hop", "Open-domain"]

FALLBACK_DELTAS = [
    [+3,  +32, +15, +11],
    [-9,  +50,  -7,  -7],
    [+39, +41, +25,  +5],
    [+3,  +48,  +9, +13],
    [-6,  +42, +21,  -5],
    [+3,  +12, -14, -13],
    [+10, +56,  +8,  +6],
    [-5,  +36, +20,  -3],
    [+14, +27, +31,   0],
    [+16, +44, +43,  +7],
]


def _build_acc(pairs):
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for p in pairs:
        correct = 1 if p.get("label", "") == "CORRECT" else 0
        counts[p["conv_id"]][p["category"]][1] += 1
        counts[p["conv_id"]][p["category"]][0] += correct
    acc = {}
    for cid in CONV_IDS:
        acc[cid] = {}
        for cat in CATS:
            c, t = counts[cid][cat]
            acc[cid][cat] = 100.0 * c / t if t else None
    return acc


def compute_deltas():
    v1_path  = Path("eval/results/prompt_fix_full_v1.json")
    rag_path = Path("eval/results/rag_baseline_all10.json")
    if not v1_path.exists() or not rag_path.exists():
        return None, True

    v1_acc  = _build_acc(json.loads(v1_path.read_text())["per_pair"])
    rag_acc = _build_acc(json.loads(rag_path.read_text())["per_pair"])

    rows = []
    for cid in CONV_IDS:
        row = []
        for cat in CATS:
            v = v1_acc[cid][cat]
            r = rag_acc[cid][cat]
            row.append(v - r if (v is not None and r is not None) else np.nan)
        rows.append(row)
    return rows, False


deltas, used_fallback = compute_deltas()
if deltas is None:
    deltas, used_fallback = FALLBACK_DELTAS, True

data = np.array([[float(v) for v in row] for row in deltas])

fig, ax = plt.subplots(figsize=(6, 5))

cmap = plt.cm.RdYlGn
norm = mcolors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=60)
im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Δ accuracy (pp), Eidetic v2 − RAG", fontsize=9)

ax.set_xticks(range(len(CAT_LABELS)))
ax.set_xticklabels(CAT_LABELS, fontsize=9)
ax.set_yticks(range(len(CONV_LABELS)))
ax.set_yticklabels(CONV_LABELS, fontsize=9)

# Cell annotations — Temporal column (index 1) gets bold text
TEMPORAL_COL = 1
for i in range(len(CONV_IDS)):
    for j in range(len(CATS)):
        val = data[i, j]
        if np.isnan(val):
            text, color = "—", "black"
        else:
            text = f"{val:+.0f}" if j == TEMPORAL_COL else f"{val:.0f}"
            bg_rgb = cmap(norm(val))[:3]
            luminance = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
            color = "black" if luminance > 0.5 else "white"
        ax.text(
            j, i, text,
            ha="center", va="center",
            fontsize=8, color=color,
            fontweight="bold" if j == TEMPORAL_COL else "normal",
        )

# Thick border highlighting the Temporal column
for i in range(len(CONV_IDS)):
    rect = plt.Rectangle(
        (TEMPORAL_COL - 0.5, i - 0.5), 1, 1,
        fill=False, edgecolor="black", lw=2,
    )
    ax.add_patch(rect)

suffix = " (v1 fallback)" if used_fallback else ""
ax.set_title(f"Per-conversation accuracy delta{suffix}", fontsize=11)
fig.text(0.12, -0.02, "* Development set; remaining held-out", fontsize=8, color="#555555")

fig.tight_layout()
fig.savefig("figures/per_conv_heatmap.pdf", format="pdf", bbox_inches="tight")
print("Saved figures/per_conv_heatmap.pdf")
