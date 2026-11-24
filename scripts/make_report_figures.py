"""Result figures (figures/*.png) from the saved results.

Run from the repo root:  python scripts/make_report_figures.py
Writes figures/fig1_architecture.png ... fig4_retrieval_encoders.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BLUE, BLUE_DARK = "#2a78d6", "#1c5aa6"
GRAY, GRAY_LIGHT = "#8a8984", "#d9d8d4"
INK, INK_2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRAY_LIGHT, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "savefig.dpi": 200,
})


def ci95(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


# ---- data
vit = [json.loads(l) for l in open(ROOT / "results" / "phase1_multiseed_results.jsonl")]
vit = {r["seed"]: r for r in vit if r["seed"] != 0} | {0: next(r for r in vit if r.get("run") == "rerun_in_aggregate")}
vit = [vit[s] for s in sorted(vit)]
d9 = json.load(open(ROOT / "results" / "d9_results.json"))["per_seed"]


# ---- fig 1: architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 49); ax.axis("off")

    def box(x, y, w, h, title, sub, strong=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="#eaf2fc" if strong else "white", ec=BLUE if strong else GRAY, lw=1))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7, color=INK_2)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=INK_2, lw=0.9))

    box(0, 13, 11, 10, "Query", "X-ray")
    box(15, 13, 15, 10, "Encoder", "frozen, 768-d")
    box(35, 24, 18, 10, "Retrieval", "cosine top-5, train\nown study excluded")
    box(35, 2, 18, 10, "Evidence tokens", "5 image + 5 label\n+ type, position")
    box(58, 13, 19, 10, "Cross-attention", "query over evidence", strong=True)
    box(81, 13, 19, 10, "Residual fusion", "query + context\n-> 14 sigmoids", strong=True)
    arrow((11.4, 18), (14.6, 18))
    arrow((30.4, 20), (34.6, 28))
    arrow((44, 23.6), (44, 12.4))
    arrow((53.4, 7), (57.6, 16))
    arrow((30.4, 17), (57.6, 18))
    arrow((77.4, 18), (80.6, 18))
    ax.annotate("", xy=(90.5, 23.6), xytext=(22.5, 23.6), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.8,
                connectionstyle="arc3,rad=-0.5", mutation_scale=9))
    ax.text(56.5, 46, "residual path: the query embedding is added back", fontsize=7, color=INK_2, ha="center")
    fig.savefig(OUT / "fig1_architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---- fig 2: test AUROC per design
def fig_auroc():
    rows = [
        ("ViT: baseline (no retrieval)", [r["arm1_test"] for r in vit], False),
        ("ViT: CXR-Recall", [r["arm3cf_test"] for r in vit], False),
        ("RAD-DINO: baseline (probe)", [r["probe_test"] for r in d9], True),
        ("RAD-DINO: CXR-Recall", [r["retrieval_test"] for r in d9], True),
        ("RAD-DINO: random neighbours", [r["random_control_test"] for r in d9], True),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    for i, (name, xs, rad) in enumerate(rows[::-1]):
        c = BLUE if rad else GRAY
        ax.scatter(xs, [i] * len(xs), s=22, color=c, alpha=0.45, edgecolor="white", linewidth=0.8, zorder=3)
        m, lo, hi = ci95(xs)
        ax.plot([lo, hi], [i, i], color=c, lw=2, zorder=2, solid_capstyle="round")
        ax.scatter([m], [i], s=46, color=BLUE_DARK if rad else INK_2, marker="D", zorder=4, edgecolor="white")
        ax.text(0.776, i, f"{m:.4f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlim(0.67, 0.785)
    ax.set_xlabel("Test macro AUROC (743 images)   dots = seeds, diamond = mean, bar = 95% CI")
    ax.grid(axis="x", color=GRAY_LIGHT, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Changing the encoder moved accuracy; adding retrieval did not", loc="left")
    fig.savefig(OUT / "fig2_test_auroc.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---- fig 3: paired margins
def fig_margins():
    rows = [
        ("ViT: CXR-Recall − baseline (3 seeds)", [r["margin_test"] for r in vit]),
        ("RAD-DINO: CXR-Recall − baseline (5)", [r["margin_retr_vs_probe"] for r in d9]),
        ("RAD-DINO: CXR-Recall − random (5)", [r["margin_retr_vs_random"] for r in d9]),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    ax.axvline(0, color=INK_2, lw=0.9, zorder=1)
    for i, (name, xs) in enumerate(rows[::-1]):
        ax.scatter(xs, [i] * len(xs), s=22, color=BLUE, alpha=0.45, edgecolor="white", linewidth=0.8, zorder=3)
        m, lo, hi = ci95(xs)
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2, zorder=2, solid_capstyle="round")
        ax.scatter([m], [i], s=46, color=BLUE_DARK, marker="D", zorder=4, edgecolor="white")
        p = stats.ttest_1samp(xs, 0).pvalue
        ax.text(0.0235, i, f"{m:+.4f}  p={p:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlim(-0.022, 0.034)
    ax.set_xlabel("Paired test AUROC difference (same 743 images, per training seed)")
    ax.grid(axis="x", color=GRAY_LIGHT, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Every retrieval confidence interval includes zero", loc="left")
    fig.savefig(OUT / "fig3_paired_margins.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---- fig 4: retrieval encoders (test, abnormal queries; from the encoder comparison run, RESULTS.md)
def fig_encoders():
    rows = [("ImageNet ViT, not fine-tuned", 0.1657), ("TorchXRayVision DenseNet", 0.1728),
            ("RAD-DINO, patch mean", 0.1838), ("Fine-tuned ViT (original)", 0.1942),
            ("Fine-tuned ViT, patch mean", 0.2058), ("RAD-DINO, CLS", 0.2725)]
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    for i, (name, v) in enumerate(rows):
        c = BLUE if name == "RAD-DINO, CLS" else (INK_2 if "original" in name else GRAY_LIGHT)
        ax.barh(i, v, height=0.62, color=c)
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.1313, color=INK_2, lw=0.9, ls=(0, (3, 2)))
    ax.set_ylim(-0.5, len(rows) + 0.1)
    ax.text(0.1313, len(rows) - 0.4, " random 0.131", fontsize=7.5, color=INK_2, va="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 0.32)
    ax.set_xlabel("Test P@5, abnormal queries (share of top-5 sharing a finding; ceiling 1.0)")
    ax.grid(axis="x", color=GRAY_LIGHT, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("RAD-DINO retrieves abnormal cases ~40% better", loc="left")
    fig.savefig(OUT / "fig4_retrieval_encoders.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


for f in (fig_architecture, fig_auroc, fig_margins, fig_encoders):
    f()
print("figures written to", OUT)
