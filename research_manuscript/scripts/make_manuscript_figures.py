"""Publication figures from final_analysis.json (matplotlib, colorblind-safe)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "results/research_final_v4/final_analysis.json"
FIG_DIR = REPO / "research_manuscript/figures"

# Okabe-Ito
COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]


def fig_primary_bar(data: dict) -> None:
    shared = data["paired_conditional_comparisons_shared_valid"]
    labels = ["Bounded", "Masked mean", "Forward fill", "Interval mid.", "Oracle*"]
    keys = ["bounded_deployable", "masked_mean", "forward_fill", "interval_midpoint", "oracle_diagnostic"]
    means = [shared[k]["mean"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=COLORS[: len(labels)], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("RMSE on missing links (vehicles)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_title(f"Shared-valid comparison (n={shared['masked_mean']['n_finite']} optimal bounded rows)")
    ymax = max(means[1:3] + [means[0]]) * 1.15
    ax.set_ylim(0, ymax)
    for b, m in zip(bars, means, strict=True):
        if m <= ymax:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    ax.annotate("*Oracle uses generative truth; not deployable.", xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_primary_shared_valid.png", dpi=200)
    fig.savefig(FIG_DIR / "fig_primary_shared_valid.pdf")
    plt.close(fig)


def fig_infeasible_by_variant(data: dict) -> None:
    variants = data["variant_denominators"]
    names = [v["stress_variant"] for v in variants]
    infeas = [v["n_infeasible_polytope"] for v in variants]
    valid = [v["n_bounded_point_valid"] for v in variants]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, valid, w, label="Bounded optimal", color=COLORS[2])
    ax.bar(x + w / 2, infeas, w, label="Infeasible polytope", color=COLORS[6])
    ax.set_ylabel("Scenarios per variant (of 10 seeds)")
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylim(0, 11)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Failure fraction by stress variant (final holdout v4)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_infeasible_by_variant.png", dpi=200)
    fig.savefig(FIG_DIR / "fig_infeasible_by_variant.pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    fig_primary_bar(data)
    fig_infeasible_by_variant(data)
    meta = {
        "source_analysis": str(ANALYSIS.relative_to(REPO)),
        "figures": ["fig_primary_shared_valid.png", "fig_infeasible_by_variant.png"],
    }
    (FIG_DIR / "figure_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("Wrote figures to", FIG_DIR)


if __name__ == "__main__":
    main()
