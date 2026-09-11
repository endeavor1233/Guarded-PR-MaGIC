"""Generate paper visualizations for PR-MaGIC's three reported advantages.

Outputs:

  figures/generated/advantage_heatmap.png/.pdf
      Three aligned heatmaps for mIoU gain, NDR, and gate-triggered fallback.

  figures/generated/candidate_signal_heatmap.png/.pdf
      Candidate-level heatmaps for representative high-regret episodes.  Rows
      are mIoU and the four internal signals; columns are candidate iterations.
      Values are normalized within each row relative to candidate 0.  This is
      an analysis visualization, not a spatial feature map.

  figures/generated/selection_transition_heatmap.png/.pdf
      Per-benchmark transition heatmap from original selector to support_fg:
      rescued, preserved non-degraded, unchanged degraded, and newly degraded.

All values come from the saved CSV files.  No model inference or mask image is
needed.  Gate fallback is shown as a neutral risk-control quantity: a larger
value means a more conservative gate, not automatically a better result.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multicriteria"
OUT = ROOT / "figures" / "generated"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit_dir = os.environ.get("NATURE_FIGURE_AUDIT_SCRIPTS")
    if audit_dir:
        sys.path.insert(0, audit_dir)
        from audit_panel_alignment import require_matplotlib_panel_alignment

        qa_dir = OUT / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        require_matplotlib_panel_alignment(
            fig,
            json_out=qa_dir / f"{stem}.alignment.json",
            overlay_svg=qa_dir / f"{stem}.alignment.svg",
            strict=True,
        )
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_heatmap(ax, matrix, row_labels, col_labels, *, cmap, vmin, vmax,
                 title, value_format, center=None):
    matrix = np.asarray(matrix, dtype=float)
    if center is None:
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    else:
        image = ax.imshow(
            matrix,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax),
            aspect="auto",
        )
    ax.set_title(title, pad=8)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.45 else "#222222"
            ax.text(j, i, value_format(value), ha="center", va="center", fontsize=8.0, color=color)
    return image


def load_advantage_rows():
    rows = []
    benchmark_csv = RESULTS / "confidence_gate_benchmark_summary.csv"
    for row in read_csv(benchmark_csv):
        if row["method"] != "support_fg":
            continue
        rows.append(
            {
                "label": row["benchmark"],
                "gain": float(row["gain_over_baseline"]) * 100.0,
                "ndr": float(row["non_degradation_rate"]) * 100.0,
                "fallback": float(row["fallback_rate"]) * 100.0,
            }
        )

    operating = {
        (row["generator"], row["benchmark"]): row
        for row in read_csv(RESULTS / "risk_coverage_operating_points.csv")
    }
    for generator, benchmark, label in (
        ("Matcher", "FSS-1000", "Matcher / FSS-1000"),
        ("Matcher", "COCO-20i", "Matcher / COCO-20i"),
    ):
        row = operating[(generator, benchmark)]
        rows.append(
            {
                "label": label,
                "gain": float(row["overall_gain_over_baseline"]) * 100.0,
                "ndr": float(row["overall_ndr"]) * 100.0,
                "fallback": float(row["gate_fallback_rate"]) * 100.0,
            }
        )
    return rows


def draw_advantage_heatmap() -> None:
    rows = load_advantage_rows()
    labels = [row["label"] for row in rows]
    columns = ["FSS-1000", "COCO-20i", "Pascal-5i", "Matcher / FSS-1000", "Matcher / COCO-20i"]
    # Reorder to make the PerSAM-F-compatible three benchmarks appear first.
    row_map = {row["label"]: row for row in rows}
    row_data = [row_map.get(label, row_map.get(label.replace(" / ", "-"))) for label in columns]
    gain = np.array([[row["gain"] for row in row_data]])
    ndr = np.array([[row["ndr"] for row in row_data]])
    fallback = np.array([[row["fallback"] for row in row_data]])

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 4.9))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.82, hspace=0.92)
    xlabels = ["FSS-1000", "COCO-20i", "Pascal-5i", "Matcher\nFSS-1000", "Matcher\nCOCO-20i"]
    draw_heatmap(
        axes[0], gain, ["Gain vs. baseline"], xlabels,
        cmap="YlGn", vmin=-2.0, vmax=4.0,
        title="Quality improvement (higher is better)", value_format=lambda x: f"{x:+.2f} pp",
    )
    draw_heatmap(
        axes[1], ndr, ["NDR"], xlabels,
        cmap="Greys", vmin=30.0, vmax=90.0,
        title="Non-degradation rate (structurally increased by fallback)", value_format=lambda x: f"{x:.1f}%",
    )
    draw_heatmap(
        axes[2], fallback, ["Gate fallback"], xlabels,
        cmap="Oranges", vmin=0.0, vmax=50.0,
        title="Gate-triggered fallback (conservatism, not a quality score)", value_format=lambda x: f"{x:.1f}%",
    )
    fig.suptitle("Three reported properties of the guarded selector", fontsize=12.5, y=0.985)
    save(fig, "advantage_heatmap")


def proposal_for_group(group: list[dict[str, str]]) -> tuple[int, int]:
    group = sorted(group, key=lambda row: int(row["iteration"]))
    scores = [
        float(row["support_query"])
        + 0.25 * float(row["decoder_confidence"])
        + 0.10 * float(row["foreground_background_separation"])
        for row in group
    ]
    proposal = max(range(len(group)), key=lambda i: scores[i])
    oracle = max(range(len(group)), key=lambda i: float(group[i]["miou"]))
    return proposal, oracle


def choose_high_regret(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["obj_name"]].append(row)
    scored = []
    for name, group in groups.items():
        group = sorted(group, key=lambda row: int(row["iteration"]))
        proposal, oracle = proposal_for_group(group)
        baseline = float(group[0]["miou"])
        selected = float(group[proposal]["miou"])
        oracle_miou = float(group[oracle]["miou"])
        gate_accepts = proposal == 0 or (
            float(group[proposal]["support_query"]) >= float(group[0]["support_query"])
            and float(group[proposal]["foreground_background_separation"])
            >= float(group[0]["foreground_background_separation"])
        )
        regret = oracle_miou - selected
        degraded = selected < baseline - 1e-12
        scored.append((int(degraded and not gate_accepts), regret, group, proposal, oracle))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def draw_candidate_signal_heatmap() -> None:
    specs = [
        ("FSS-1000", RESULTS / "fss_f0_s42_full/0/csvs/selector_original_candidates.csv"),
        ("COCO-20i", RESULTS / "coco_f0_s45_heldout1000/0/csvs/selector_original_candidates.csv"),
        ("Pascal-5i", RESULTS / "pascal_f0_s45_full1000_fixed2/0/csvs/selector_original_candidates.csv"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.76, wspace=0.34)
    signal_names = [
        ("miou", "candidate mIoU"),
        ("support_query", "support-query"),
        ("decoder_confidence", "decoder confidence"),
        ("temporal_stability", "temporal stability"),
        ("foreground_background_separation", "foreground-background"),
    ]
    cmap = "RdBu_r"
    for ax, (benchmark, path) in zip(axes, specs):
        group = choose_high_regret(path)
        group = sorted(group, key=lambda row: int(row["iteration"]))
        proposal, oracle = proposal_for_group(group)
        matrix = []
        for key, _ in signal_names:
            values = np.array([float(row[key]) for row in group], dtype=float)
            delta = values - values[0]
            scale = max(float(np.max(np.abs(delta))), 1e-12)
            matrix.append(delta / scale)
        matrix = np.asarray(matrix)
        image = ax.imshow(matrix, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_title(benchmark)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels([str(int(row["iteration"])) for row in group])
        ax.set_yticks(range(len(signal_names)))
        ax.set_yticklabels([name for _, name in signal_names])
        ax.set_xlabel("Candidate iteration")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(proposal, color="#b2182b", linestyle="--", linewidth=1.4, label="original proposal")
        ax.axvline(oracle, color="#2166ac", linestyle="-.", linewidth=1.4, label="oracle")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=7.0,
                        color="white" if abs(matrix[i, j]) > 0.55 else "#222222")
    axes[0].set_ylabel("Signal relative to candidate 0")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.90), frameon=False)
    fig.suptitle("Baseline-relative candidate diagnostics", fontsize=12.5, y=0.985)
    save(fig, "candidate_signal_heatmap")


def draw_transition_heatmap() -> None:
    rows = read_csv(RESULTS / "confidence_gate_per_sample.csv")
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["benchmark"], row["seed"], row["fold"], row["obj_name"])
        grouped[key][row["method"]] = row

    categories = ["rescued", "preserved non-degraded", "unchanged degraded", "newly degraded"]
    counts = {}
    for benchmark in ("FSS-1000", "COCO-20i", "Pascal-5i"):
        counter = dict.fromkeys(categories, 0)
        total = 0
        for key, methods in grouped.items():
            if key[0] != benchmark or "original" not in methods or "support_fg" not in methods:
                continue
            original_ok = int(methods["original"]["non_degraded"]) == 1
            guarded_ok = int(methods["support_fg"]["non_degraded"]) == 1
            total += 1
            if not original_ok and guarded_ok:
                counter["rescued"] += 1
            elif original_ok and guarded_ok:
                counter["preserved non-degraded"] += 1
            elif not original_ok and not guarded_ok:
                counter["unchanged degraded"] += 1
            else:
                counter["newly degraded"] += 1
        counts[benchmark] = [100.0 * counter[name] / total if total else 0.0 for name in categories]

    matrix = np.asarray([counts[b] for b in ("FSS-1000", "COCO-20i", "Pascal-5i")])
    fig, ax = plt.subplots(figsize=(8.6, 2.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["FSS-1000", "COCO-20i", "Pascal-5i"])
    ax.set_title("Per-episode transitions after applying the support-foreground gate")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 55 else "#222222"
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=8.0, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Fraction of episodes")
    save(fig, "selection_transition_heatmap")


def main() -> None:
    style()
    draw_advantage_heatmap()
    draw_candidate_signal_heatmap()
    draw_transition_heatmap()
    print(f"Saved visualizations to {OUT}")


if __name__ == "__main__":
    main()
