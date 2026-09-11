"""Generate a compact benchmark-summary plot from saved manuscript results.

This is a quantitative data plot, so it intentionally uses Matplotlib rather
than the cvpr-figure framework-diagram engine.  It contains only values already
reported in the manuscript tables and does not introduce a new experiment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.2,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    benchmarks = ["FSS-1000", "COCO-20i", "Pascal-5i", "Matcher\nFSS-1000", "Matcher\nCOCO-20i"]
    baseline = np.array([0.857831, 0.535171, 0.669651, 0.884484, 0.681014])
    original = np.array([0.885044, 0.565638, 0.652193, 0.903832, 0.698208])
    guarded = np.array([0.879513, 0.565979, 0.669199, 0.905907, 0.694956])
    oracle = np.array([0.918510, 0.638627, 0.751492, 0.923176, 0.736494])
    original_ndr = np.array([0.6332, 0.6678, 0.3858, 0.6500, 0.6680])
    guarded_ndr = np.array([0.8381, 0.8243, 0.7022, 0.8371, 0.8240])
    gate_fallback = np.array([0.4361, 0.3078, 0.3950, 0.4117, 0.3510])

    x = np.arange(len(benchmarks))
    width = 0.19
    colors = {"baseline": "#d9d9d9", "original": "#9ec1d5", "guarded": "#e5c46a", "oracle": "#c7b5d8"}
    edge = "#252525"

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.65), constrained_layout=True)

    ax = axes[0]
    ax.bar(x - 1.5 * width, baseline, width, label="Baseline", color=colors["baseline"], edgecolor=edge, linewidth=0.7, hatch="///")
    ax.bar(x - 0.5 * width, original, width, label="Original", color=colors["original"], edgecolor=edge, linewidth=0.7, hatch="..")
    ax.bar(x + 0.5 * width, guarded, width, label="Guarded", color=colors["guarded"], edgecolor=edge, linewidth=0.9, hatch="")
    ax.bar(x + 1.5 * width, oracle, width, label="Oracle", color=colors["oracle"], edgecolor=edge, linewidth=0.7, hatch="xx")
    ax.set_title("Mean mIoU")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0.45, 0.96)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])

    ax = axes[1]
    ax.bar(x - width / 2, original_ndr, width, label="Original", color=colors["original"], edgecolor=edge, linewidth=0.7, hatch="..")
    ax.bar(x + width / 2, guarded_ndr, width, label="Guarded", color=colors["guarded"], edgecolor=edge, linewidth=0.9, hatch="")
    ax.set_title("Non-degradation rate", pad=2)
    ax.set_ylabel("NDR")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    ax = axes[2]
    bars = ax.bar(x, gate_fallback, width=0.52, color="#d98f86", edgecolor=edge, linewidth=0.8, hatch="\\\\")
    ax.set_title("Gate-triggered fallback")
    ax.set_ylabel("Fraction of episodes")
    ax.set_ylim(0.0, 0.55)
    ax.set_yticks([0.0, 0.15, 0.30, 0.45])
    for bar, value in zip(bars, gate_fallback):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.0%}", ha="center", va="bottom", fontsize=7.5)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Quality, non-degradation, and fallback across candidate generators", fontsize=12.5, y=1.13)
    OUT.mkdir(parents=True, exist_ok=True)
    audit_dir = os.environ.get("NATURE_FIGURE_AUDIT_SCRIPTS")
    if audit_dir:
        sys.path.insert(0, audit_dir)
        from audit_panel_alignment import require_matplotlib_panel_alignment

        qa_dir = OUT / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        require_matplotlib_panel_alignment(
            fig,
            json_out=qa_dir / "benchmark_summary.alignment.json",
            overlay_svg=qa_dir / "benchmark_summary.alignment.svg",
            strict=True,
        )
    fig.savefig(OUT / "benchmark_summary.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "benchmark_summary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "benchmark_summary.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'benchmark_summary.png'}")
    print(f"Saved {OUT / 'benchmark_summary.pdf'}")


if __name__ == "__main__":
    main()
