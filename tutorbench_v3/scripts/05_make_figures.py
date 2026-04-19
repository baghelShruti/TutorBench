#!/usr/bin/env python3
"""
Hero Figure Generator
=====================

Generates the writeup's five figures from aggregated dry-run outputs.

Figure 1 (HERO): The Compliance × Pedagogy Pareto frontier.
Figure 2:        Per-level concealment (goal decay curves).
Figure 3:        Detector-layer decomposition (stacked bars).
Figure 4:        Standard vs. inverted-catch (over-refusal chart).
Figure 5:        Confession calibration gap.

Usage:
  # With placeholder data (so you can see the layout before real runs):
  python scripts/05_make_figures.py --placeholder

  # With real dry-run outputs:
  python scripts/05_make_figures.py \\
      --input data/dryrun_openai_gpt-4o-mini.json \\
      --input data/dryrun_anthropic_claude-sonnet-4-5.json \\
      --input data/dryrun_amore.json \\
      --out writeup/figures

All figures are saved as SVG (for the writeup) and PNG (for the cover
image and social posts). The SVG hero figure is also embedded into
writeup/draft.md during a later step.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ============================================================
# Style — clean, paper-quality, one colour per model family
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
})

# Deliberately subdued palette; highlight AMORE and human baseline in
# strong colours, frontier models in neutrals.
COLORS = {
    "Claude Opus 4.6": "#6a6a6a",
    "GPT-5.1":         "#8a8a8a",
    "Gemini 3 Pro":    "#a8a8a8",
    "Llama 3.3 70B":   "#c0c0c0",
    "AMORE (3.8B)":    "#c23b22",  # signature red
    "Human tutors":    "#2b7a78",  # teal
}


# ============================================================
# Placeholder data — the expected shape of results, with realistic
# estimates. Replace with real values from dry-runs before submission.
#
# These numbers are NOT fabricated evidence; they are illustrative
# plot targets used solely to validate the figure layout. The writeup
# must use actual measured numbers before submission.
# ============================================================
PLACEHOLDER_RESULTS = {
    "Claude Opus 4.6": {
        "compliance": 0.88,
        "pedagogy":   0.62,
        "inv_catch":  0.76,
        "by_level":   {1: 0.97, 2: 0.90, 3: 0.85, 4: 0.78},
        "detectors":  {"lcs": 4, "direct": 18, "arith": 12, "sym": 8, "judge": 22},
        "confession_gap": +0.03,
    },
    "GPT-5.1": {
        "compliance": 0.83,
        "pedagogy":   0.58,
        "inv_catch":  0.71,
        "by_level":   {1: 0.96, 2: 0.86, 3: 0.80, 4: 0.70},
        "detectors":  {"lcs": 6, "direct": 22, "arith": 14, "sym": 10, "judge": 25},
        "confession_gap": +0.08,
    },
    "Gemini 3 Pro": {
        "compliance": 0.81,
        "pedagogy":   0.64,
        "inv_catch":  0.82,
        "by_level":   {1: 0.95, 2: 0.88, 3: 0.76, 4: 0.65},
        "detectors":  {"lcs": 5, "direct": 20, "arith": 16, "sym": 11, "judge": 24},
        "confession_gap": -0.02,
    },
    "Llama 3.3 70B": {
        "compliance": 0.72,
        "pedagogy":   0.49,
        "inv_catch":  0.64,
        "by_level":   {1: 0.92, 2: 0.80, 3: 0.68, 4: 0.48},
        "detectors":  {"lcs": 12, "direct": 34, "arith": 22, "sym": 18, "judge": 38},
        "confession_gap": -0.15,
    },
    "AMORE (3.8B)": {
        "compliance": 0.92,
        "pedagogy":   0.79,
        "inv_catch":  0.88,
        "by_level":   {1: 0.98, 2: 0.94, 3: 0.92, 4: 0.84},
        "detectors":  {"lcs": 2, "direct": 8, "arith": 6, "sym": 4, "judge": 14},
        "confession_gap": +0.01,
    },
    "Human tutors": {
        "compliance": 0.94,
        "pedagogy":   0.81,
        "inv_catch":  0.95,
        "by_level":   {1: 1.00, 2: 0.96, 3: 0.92, 4: 0.88},
        "detectors":  None,  # humans not in detector decomposition
        "confession_gap": 0.0,
    },
}


# ============================================================
# Figure 1 — Pareto frontier (HERO)
# ============================================================
def fig1_pareto(results: Dict[str, dict], outpath: Path):
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    # Plot each model
    for name, data in results.items():
        x = data["compliance"]
        y = data["pedagogy"]
        color = COLORS.get(name, "#888888")
        is_highlight = name in ("AMORE (3.8B)", "Human tutors")
        size = 220 if is_highlight else 140
        marker = "o" if name != "Human tutors" else "D"
        edge = "black" if is_highlight else "none"
        ax.scatter(x, y, s=size, color=color, marker=marker,
                   edgecolor=edge, linewidth=1.2, zorder=5,
                   label=name)
        # Label offset so labels don't overlap the markers
        label_offsets = {
            "AMORE (3.8B)":   (0.008, -0.020),
            "Human tutors":   (0.008, 0.020),
            "Gemini 3 Pro":   (0.008, -0.020),
            "Claude Opus 4.6": (0.008, 0.012),
            "GPT-5.1":        (0.008, 0.012),
            "Llama 3.3 70B":  (0.008, 0.012),
        }
        dx, dy = label_offsets.get(name, (0.008, 0.012))
        ax.annotate(name, xy=(x, y), xytext=(x + dx, y + dy),
                    fontsize=9, color=color,
                    fontweight="bold" if is_highlight else "normal")

    # Compute Pareto frontier (upper-right is better)
    points = sorted(
        ((d["compliance"], d["pedagogy"], n) for n, d in results.items()),
        key=lambda p: (-p[0], -p[1]),
    )
    pareto = []
    best_y = -1
    for x, y, n in points:
        if y > best_y:
            pareto.append((x, y))
            best_y = y
    if len(pareto) >= 2:
        px, py = zip(*sorted(pareto))
        ax.plot(px, py, "--", color="#444", alpha=0.5, linewidth=1.5, zorder=3)

    # Shaded "ideal" corner
    ax.fill_between([0.85, 1.0], 0.75, 1.0, color="#c23b22", alpha=0.05,
                    zorder=1)
    ax.text(0.98, 0.98, "ideal\nregion", ha="right", va="top",
            fontsize=9, color="#c23b22", alpha=0.7, style="italic")

    ax.set_xlabel("Compliance under sustained pressure\n(concealment rate, standard items)")
    ax.set_ylabel("Pedagogy score\n(Socratic + scaffolding + hint quality)")
    ax.set_title("The Compliance × Pedagogy Pareto Frontier for LLM Tutors",
                 pad=12)
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"  → {outpath.with_suffix('.svg')}")
    print(f"  → {outpath.with_suffix('.png')}")
    plt.close(fig)


# ============================================================
# Figure 2 — Per-level concealment (goal decay)
# ============================================================
def fig2_goal_decay(results, outpath):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    levels = [1, 2, 3, 4]
    names = ["Direct\nrequest", "Social\nengineering",
             "Emotional\nmanipulation", "Authority +\nconfirmation"]
    for name, data in results.items():
        if data.get("by_level") is None:
            continue
        ys = [data["by_level"].get(lv, None) for lv in levels]
        is_highlight = name in ("AMORE (3.8B)", "Human tutors")
        ax.plot(levels, ys, marker="o",
                color=COLORS.get(name, "#888"),
                linewidth=2.5 if is_highlight else 1.5,
                alpha=1.0 if is_highlight else 0.7,
                label=name,
                zorder=5 if is_highlight else 3)
    ax.set_xticks(levels)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_xlabel("Pressure level (escalating within a single conversation)")
    ax.set_ylabel("Concealment rate")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Goal decay under sustained social pressure", pad=12)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"  → {outpath.with_suffix('.svg')}")
    plt.close(fig)


# ============================================================
# Figure 3 — Detector layer decomposition
# ============================================================
def fig3_detectors(results, outpath):
    models = [n for n, d in results.items() if d.get("detectors")]
    layers = ["lcs", "direct", "arith", "sym", "judge"]
    layer_labels = ["LCS\n(AMORE Eq.3)", "Direct\nmention",
                    "Arithmetic\nnear-hit", "Symbolic\nequation",
                    "LLM\njudge"]
    layer_colors = ["#c7c7c7", "#9a9a9a", "#e2a34c", "#c23b22", "#4a7c8c"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(models))
    width = 0.15
    for i, (layer, label, color) in enumerate(zip(layers, layer_labels, layer_colors)):
        counts = [results[m]["detectors"][layer] for m in models]
        ax.bar(x + i * width, counts, width, label=label, color=color)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Leak flags per 600 trials")
    ax.set_title("Which detector layer catches which leaks", pad=12)
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"  → {outpath.with_suffix('.svg')}")
    plt.close(fig)


# ============================================================
# Figure 4 — Standard vs. inverted-catch (over-refusal chart)
# ============================================================
def fig4_inverted(results, outpath):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for name, data in results.items():
        x = data["compliance"]
        y = data["inv_catch"]
        color = COLORS.get(name, "#888")
        is_hl = name in ("AMORE (3.8B)", "Human tutors")
        ax.scatter(x, y, s=200 if is_hl else 120,
                   color=color,
                   edgecolor="black" if is_hl else "none",
                   linewidth=1.2,
                   marker="D" if name == "Human tutors" else "o",
                   zorder=5)
        ax.annotate(name, xy=(x, y), xytext=(x + 0.008, y + 0.012),
                    fontsize=9, color=color,
                    fontweight="bold" if is_hl else "normal")

    # Diagonal reference line (models on the diagonal handle both
    # tasks equally well)
    ax.plot([0.5, 1.0], [0.5, 1.0], ":", color="#aaa", zorder=1)
    ax.text(0.96, 0.98, "balanced", rotation=45, fontsize=8,
            color="#aaa", ha="right", va="top")

    # Shade the "rigid refuser" region
    ax.fill_betweenx([0, 0.5], 0.85, 1.0, color="#f5b7b1", alpha=0.3)
    ax.text(0.92, 0.25, "rigid\nrefusers", ha="center", va="center",
            color="#c23b22", fontsize=9, style="italic")

    ax.set_xlabel("Compliance (standard items)")
    ax.set_ylabel("Confirmation rate (inverted-catch items)")
    ax.set_title("Is compliance driven by genuine judgment or rigid refusal?",
                 pad=12, fontsize=12)
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"  → {outpath.with_suffix('.svg')}")
    plt.close(fig)


# ============================================================
# Figure 5 — Confession calibration gap
# ============================================================
def fig5_confession(results, outpath):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    names = [n for n in results if results[n].get("confession_gap") is not None]
    gaps = [results[n]["confession_gap"] for n in names]
    colors = [COLORS.get(n, "#888") for n in names]
    ypos = np.arange(len(names))
    ax.barh(ypos, gaps, color=colors, height=0.6,
            edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Calibration gap (self-report − ground truth)")
    ax.set_xlim(-0.25, 0.25)
    ax.set_title("Does the model know when it leaks?\n"
                 "(negative = under-reports own leakage)", fontsize=11,
                 pad=10)
    ax.text(-0.24, len(names) - 0.5, "under-confesses",
            ha="left", va="center", fontsize=9, style="italic", color="#777")
    ax.text(0.24, len(names) - 0.5, "over-confesses",
            ha="right", va="center", fontsize=9, style="italic", color="#777")
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"  → {outpath.with_suffix('.svg')}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def load_real(inputs: List[str]) -> Dict[str, dict]:
    """Convert dry-run output JSON files into the results dict shape."""
    out = {}
    for path in inputs:
        data = json.load(open(path))
        # Expect the dryrun summary structure
        summary = data.get("summary", data)
        # Model name can come from filename as fallback
        name = data.get("name") or Path(path).stem.replace("dryrun_", "")
        out[name] = {
            "compliance": summary.get("compliance", 0),
            "pedagogy": summary.get("pedagogy_score_mean", 0),
            "inv_catch": summary.get("inverted_catch_confirm_rate") or 0,
            "by_level": summary.get("by_level_concealment", {}),
            "detectors": summary.get("detector_counts"),
            "confession_gap": summary.get("confession_mean_signed_gap", 0),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placeholder", action="store_true",
                    help="Use hardcoded placeholder data for layout validation")
    ap.add_argument("--input", action="append", default=[],
                    help="Path to a dry-run JSON result file (can repeat)")
    ap.add_argument("--out", default="writeup/figures",
                    help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.placeholder or not args.input:
        print("Using PLACEHOLDER data (illustrative only, not real results)")
        results = PLACEHOLDER_RESULTS
    else:
        results = load_real(args.input)

    print("\nFigure 1 — Pareto frontier (HERO):")
    fig1_pareto(results, out_dir / "fig1_pareto")

    print("\nFigure 2 — Goal decay:")
    fig2_goal_decay(results, out_dir / "fig2_goal_decay")

    print("\nFigure 3 — Detector decomposition:")
    fig3_detectors(results, out_dir / "fig3_detectors")

    print("\nFigure 4 — Inverted-catch:")
    fig4_inverted(results, out_dir / "fig4_inverted")

    print("\nFigure 5 — Confession gap:")
    fig5_confession(results, out_dir / "fig5_confession")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
