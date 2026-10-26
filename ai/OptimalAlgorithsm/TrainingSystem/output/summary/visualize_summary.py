"""Render summary.json: python visualize_summary.py [path/to/summary.json]."""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/driver-guardian-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np

SOURCE = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("summary.json")
)
OUTPUT = SOURCE.resolve().parent
DATA = json.loads(SOURCE.read_text())
ALGORITHMS = list(DATA)
METHODS = list(dict.fromkeys(method for results in DATA.values() for method in results))
NAMES = {method: method.replace("_", " ").title() for method in METHODS}
COLORS = {method: plt.get_cmap("tab10")(i % 10) for i, method in enumerate(METHODS)}
METRICS = {
    "f1": "F1 score",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "false_positive_rate": "False positive rate",
}
ROWS = [
    (algorithm, method, scores)
    for algorithm, results in DATA.items()
    for method, scores in results.items()
]
TOTALS = sorted(
    {sum(scores[k] for k in ("tp", "tn", "fp", "fn")) for _, _, scores in ROWS}
)
NOTE = (
    f"Source: {SOURCE.name} | Evaluation sizes: "
    + ", ".join(f"{n:,}" for n in TOTALS)
    + " samples."
    + (" Different supports limit direct comparisons." if len(TOTALS) > 1 else "")
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#F8FAFC",
        "axes.facecolor": "white",
        "savefig.facecolor": "#F8FAFC",
        "svg.fonttype": "none",
    }
)


def save(fig, name):
    fig.text(0.025, 0.018, NOTE, fontsize=10, color="#526174")
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Shared 0-100% axes keep differences in scale honest.
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for ax, (metric, label) in zip(axes.flat, list(METRICS.items())[:6]):
    positions = np.arange(len(ALGORITHMS))
    width = 0.8 / len(METHODS)
    for j, method in enumerate(METHODS):
        values = [DATA[a].get(method, {}).get(metric, np.nan) for a in ALGORITHMS]
        bars = ax.bar(
            positions + (j - (len(METHODS) - 1) / 2) * width,
            values,
            width,
            color=COLORS[method],
            label=NAMES[method],
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.1%}" if np.isfinite(v) else "" for v in values],
            fontsize=8,
            padding=3,
            rotation=90,
        )
    ax.set(
        title=label,
        ylim=(0, 1.18),
        yticks=np.linspace(0, 1, 6),
        xticks=positions,
        xticklabels=[a.upper() for a in ALGORITHMS],
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
fig.suptitle(
    "Evaluation metrics by algorithm and aggregation method",
    fontsize=20,
    weight="bold",
    y=0.98,
)
fig.legend(
    *axes.flat[0].get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.935),
    ncol=len(METHODS),
    frameon=False,
)
fig.subplots_adjust(top=0.85, bottom=0.10, hspace=0.32, wspace=0.22)
save(fig, "01_metrics_comparison")

fig, ax = plt.subplots(figsize=(16, max(10, len(ROWS) * 0.55)))
values = np.array([[scores[key] for key in METRICS] for _, _, scores in ROWS])
# Invert FPR only for color: the annotated number remains the original FPR.
quality = values.copy()
quality[:, -1] = 1 - quality[:, -1]
im = ax.imshow(quality, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(
    range(len(METRICS)),
    [
        label.replace(" ", "\n") + (" ↓" if key == "false_positive_rate" else " ↑")
        for key, label in METRICS.items()
    ],
)
ax.set_yticks(range(len(ROWS)), [f"{a.upper()} / {NAMES[m]}" for a, m, _ in ROWS])
for row in range(len(ROWS)):
    for col in range(len(METRICS)):
        ax.text(
            col,
            row,
            f"{values[row, col]:.2%}",
            ha="center",
            va="center",
            color="white" if quality[row, col] > 0.6 else "#152238",
            fontsize=11,
        )
for row in range(1, len(ROWS)):
    if ROWS[row][0] != ROWS[row - 1][0]:
        ax.axhline(row - 0.5, color="white", linewidth=3)
fig.colorbar(
    im,
    ax=ax,
    pad=0.025,
    fraction=0.03,
    label="Higher is better (FPR color uses 1 − FPR)",
)
ax.set_title("All configurations | darker cells indicate better performance", pad=24)
fig.subplots_adjust(left=0.26, right=0.93, top=0.90, bottom=0.13)
save(fig, "02_metrics_heatmap")

fig, axes = plt.subplots(
    len(ALGORITHMS), len(METHODS),
    figsize=(4.5 * len(METHODS), 3.75 * len(ALGORITHMS)), squeeze=False,
)
for i, algorithm in enumerate(ALGORITHMS):
    for j, method in enumerate(METHODS):
        ax = axes[i, j]
        scores = DATA[algorithm].get(method)
        if scores is None:
            ax.set_visible(False)
            continue
        counts = np.array([[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]])
        support = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts, support, out=np.zeros_like(counts, dtype=float), where=support != 0
        )
        ax.imshow(normalized, cmap="Blues", norm=Normalize(0, 1))
        for r in range(2):
            for c in range(2):
                label = (("TN", "FP"), ("FN", "TP"))[r][c]
                ax.text(
                    c,
                    r,
                    f"{label} = {counts[r, c]:,}\n{normalized[r, c]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if normalized[r, c] > 0.55 else "#152238",
                )
        ax.set(
            title=f"{algorithm.upper()} / {NAMES[method]}\nn = {counts.sum():,}",
            xticks=[0, 1],
            xticklabels=["Negative", "Positive"],
            yticks=[0, 1],
            yticklabels=["Negative", "Positive"],
        )
        if j == 0:
            ax.set_ylabel("Actual class")
        if i == len(ALGORITHMS) - 1:
            ax.set_xlabel("Predicted class")
fig.suptitle(
    "Confusion matrices | counts and percentages within each actual class",
    fontsize=18,
    weight="bold",
    y=0.98,
)
fig.subplots_adjust(
    left=0.07, right=0.91, top=0.92, bottom=0.08, hspace=0.42, wspace=0.18
)
cax = fig.add_axes([0.93, 0.22, 0.016, 0.56])
fig.colorbar(
    plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="Blues"),
    cax=cax,
    format=PercentFormatter(1),
    label="Share of actual class",
)
save(fig, "03_confusion_matrices")

fig, axes = plt.subplots(1, 2, figsize=(20, max(9, len(ROWS) * 0.55)))
for ax, metric in zip(axes, ("f1", "balanced_accuracy")):
    ranked = sorted(ROWS, key=lambda row: row[2][metric], reverse=True)
    positions = np.arange(len(ranked))
    bars = ax.barh(
        positions,
        [s[metric] for _, _, s in ranked],
        color=[COLORS[m] for _, m, _ in ranked],
        height=0.65,
    )
    ax.bar_label(
        bars, labels=[f"{s[metric]:.3%}" for _, _, s in ranked], padding=5, fontsize=10
    )
    ax.set(
        yticks=positions,
        yticklabels=[f"{a.upper()} / {NAMES[m]}" for a, m, _ in ranked],
        xlim=(0, 1),
        title=f"Ranked by {METRICS[metric]}",
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
fig.suptitle(
    "Configuration rankings | single reported evaluation per configuration",
    fontsize=19,
    weight="bold",
    y=0.98,
)
fig.subplots_adjust(left=0.23, right=0.98, top=0.90, bottom=0.10, wspace=1.05)
save(fig, "04_configuration_rankings")

print(f"Saved 4 charts as PNG + SVG in {OUTPUT}")
