#!/usr/bin/env python3
"""Visualization — strictly follows ST5230 Project Proposal Section 5.

Proposal §5: "Distributional shifts (e.g., mean shift, variance inflation,
or tail deterioration) will be visualized using violin plots overlaid with
boxplots to reveal fine-grained model fragility across different wording
variations."

Generates two types of figures:
  1. Raw logprob: baseline vs paraphrase side-by-side (violin + boxplot)
  2. ΔLogprob: per-item confidence degradation distribution (violin + boxplot)
     directly corresponds to Δt metric (Proposal §4.2)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.analysis import build_paired_data

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASET_LABELS = {
    "commonsense_qa": "CommonsenseQA",
    "arc_challenge": "ARC-Challenge",
    "mmlu": "MMLU",
}
CONDITION_LABELS = {
    "lexical": "Lexical",
    "syntactic": "Syntactic",
    "style": "Style",
    "context": "Context",
    "translation": "Translation",
}
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "qwen3.5-27b": "Qwen-3.5-27B",
    "kimi-k2": "Kimi-K2",
}

COLORS = ["#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ── Figure 1: Raw Logprob (baseline vs paraphrase) ──────────────

def plot_logprob_violin_boxplot(model_name, dataset_name):
    """Violin + boxplot of raw logprob: baseline vs each paraphrase type."""
    conditions = config.PARAPHRASE_TYPES
    all_base = []
    all_para = []
    valid_conditions = []

    for cond in conditions:
        pairs = build_paired_data(model_name, dataset_name, cond)
        if not pairs:
            continue
        base_lp = [p["baseline_logprob"] for p in pairs
                    if p["baseline_logprob"] is not None]
        para_lp = [p["paraphrase_logprob"] for p in pairs
                    if p["paraphrase_logprob"] is not None]
        if base_lp and para_lp:
            all_base.append(base_lp)
            all_para.append(para_lp)
            valid_conditions.append(cond)

    if not valid_conditions:
        return

    n_cond = len(valid_conditions)
    fig, ax = plt.subplots(figsize=(2.5 * n_cond + 1.5, 5))

    positions_base = []
    positions_para = []
    for i in range(n_cond):
        positions_base.append(i * 3)
        positions_para.append(i * 3 + 1)

    # Violin plots
    vp_base = ax.violinplot(all_base, positions=positions_base, widths=0.8,
                             showmeans=False, showmedians=False, showextrema=False)
    vp_para = ax.violinplot(all_para, positions=positions_para, widths=0.8,
                             showmeans=False, showmedians=False, showextrema=False)

    for body in vp_base["bodies"]:
        body.set_facecolor("#4C72B0")
        body.set_alpha(0.6)
    for body in vp_para["bodies"]:
        body.set_facecolor("#DD8452")
        body.set_alpha(0.6)

    # Boxplots overlaid
    ax.boxplot(all_base, positions=positions_base, widths=0.3,
               patch_artist=True, showfliers=False,
               medianprops=dict(color="black", linewidth=1.5),
               boxprops=dict(facecolor="#4C72B0", alpha=0.8),
               whiskerprops=dict(color="#4C72B0"),
               capprops=dict(color="#4C72B0"))
    ax.boxplot(all_para, positions=positions_para, widths=0.3,
               patch_artist=True, showfliers=False,
               medianprops=dict(color="black", linewidth=1.5),
               boxprops=dict(facecolor="#DD8452", alpha=0.8),
               whiskerprops=dict(color="#DD8452"),
               capprops=dict(color="#DD8452"))

    # Mean markers
    for i in range(n_cond):
        ax.scatter(positions_base[i], np.mean(all_base[i]),
                   color="white", edgecolors="#4C72B0", s=30, zorder=5, marker="D")
        ax.scatter(positions_para[i], np.mean(all_para[i]),
                   color="white", edgecolors="#DD8452", s=30, zorder=5, marker="D")

    tick_positions = [(positions_base[i] + positions_para[i]) / 2
                      for i in range(n_cond)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in valid_conditions],
                       fontsize=10)
    ax.set_ylabel("Ground-Truth Logprob", fontsize=11)
    ax.set_ylim(bottom=-10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", alpha=0.7, label="Baseline"),
        Patch(facecolor="#DD8452", alpha=0.7, label="Paraphrased"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    ds_label = DATASET_LABELS.get(dataset_name, dataset_name)
    model_label = MODEL_LABELS.get(model_name, model_name)
    ax.set_title(f"{model_label} — {ds_label}", fontsize=12, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR,
                        f"violin_{model_name}_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: ΔLogprob Distribution ─────────────────────────────

def plot_delta_logprob(model_name, dataset_name):
    """Violin + boxplot of per-item ΔLogprob for each paraphrase type.

    ΔLogprob = Logprob_paraphrase − Logprob_baseline (per item).
    0 = no change; negative = confidence degradation.
    Directly visualizes Δt (Proposal §4.2).
    """
    conditions = config.PARAPHRASE_TYPES
    all_deltas = []
    valid_conditions = []

    for cond in conditions:
        pairs = build_paired_data(model_name, dataset_name, cond)
        if not pairs:
            continue
        # Only include items where at least one logprob is non-zero
        # (items where both are 0 carry no information about shift)
        deltas = [p["paraphrase_logprob"] - p["baseline_logprob"]
                  for p in pairs
                  if p["baseline_logprob"] is not None
                  and p["paraphrase_logprob"] is not None
                  and not (p["baseline_logprob"] == 0
                           and p["paraphrase_logprob"] == 0)]
        if deltas:
            all_deltas.append(deltas)
            valid_conditions.append(cond)

    if not valid_conditions:
        return

    n_cond = len(valid_conditions)
    fig, ax = plt.subplots(figsize=(1.8 * n_cond + 1.5, 5))

    positions = list(range(n_cond))

    # Violin
    vp = ax.violinplot(all_deltas, positions=positions, widths=0.7,
                        showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(COLORS[i % len(COLORS)])
        body.set_alpha(0.5)

    # Boxplot overlaid
    bp = ax.boxplot(all_deltas, positions=positions, widths=0.25,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", linewidth=1.5),
                     whiskerprops=dict(color="gray"),
                     capprops=dict(color="gray"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.8)

    # Mean markers
    for i in range(n_cond):
        mean_val = np.mean(all_deltas[i])
        ax.scatter(i, mean_val, color="white", edgecolors="black",
                   s=40, zorder=5, marker="D", linewidths=1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in valid_conditions],
                       fontsize=10)
    ax.set_ylabel("ΔLogprob (paraphrase − baseline)", fontsize=11)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-8, 8)

    ds_label = DATASET_LABELS.get(dataset_name, dataset_name)
    model_label = MODEL_LABELS.get(model_name, model_name)
    ax.set_title(f"{model_label} — {ds_label}\nConfidence Degradation (Δt) Distribution",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR,
                        f"delta_logprob_{model_name}_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ─────────────────────────────────────────────────────────

def generate_all_figures(analysis_results=None):
    """Generate all figures (one per model × dataset × figure type)."""
    print("\nGenerating figures (Proposal §5: violin + boxplot)...")

    for model_name in config.EVAL_MODELS:
        has_data = False
        for ds in config.DATASETS:
            path = os.path.join(config.RESULTS_DIR,
                                f"{model_name}_{ds}_baseline.json")
            if os.path.exists(path):
                has_data = True
                break
        if not has_data:
            continue

        for dataset_name in config.DATASETS:
            plot_logprob_violin_boxplot(model_name, dataset_name)
            plot_delta_logprob(model_name, dataset_name)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all_figures()
