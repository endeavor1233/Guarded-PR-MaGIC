"""Generate risk--coverage curves from saved candidate-level CSV files.

The script performs an offline replay of the original selector and the
baseline-relative support--foreground gate.  It does not load SAM or run
model inference.  For every episode, the original selector first proposes

    argmax_t support_query(t) + 0.25 * decoder_confidence(t)
            + 0.10 * foreground_background_separation(t).

For a non-baseline proposal, the risk score used to order selective decisions
is the baseline-relative margin

    m = min(support_query(t*) - support_query(0),
            foreground_background_separation(t*)
            - foreground_background_separation(0)).

At threshold lambda, the proposal is accepted when m >= lambda.  Coverage is the
fraction of all episodes for which a non-baseline proposal is accepted, and
conditional degradation risk is the fraction of accepted proposals whose
ground-truth mIoU is below the baseline mIoU.  Ground truth is used only to
evaluate the curve, never to select a proposal.

Outputs are written to results/multicriteria by default:

  risk_coverage_persam.csv
  risk_coverage_matcher.csv
  risk_coverage_per_sample.csv
  risk_coverage_operating_points.csv
  risk_coverage_persam.png/.pdf
  risk_coverage_matcher.png/.pdf
  risk_coverage_all_generators.png/.pdf

Figures are written to figures/generated so that the manuscript does not
depend on any pre-existing image directory.

The PerSAM-F-compatible files are the full saved evaluation sets used by the
manuscript.  The Matcher files are the full FSS-1000 and COCO-20i transfer
experiments.  Candidate files named ``guarded`` are still valid inputs here:
the candidate pool and diagnostics are replayed, while the original proposal
is recomputed from the columns rather than trusting the ``selected`` flag.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt


ORIGINAL_SCORE_ALPHA = 0.25
ORIGINAL_SCORE_BETA = 0.10
MIoU_TOLERANCE = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/results/multicriteria)",
    )
    return parser.parse_args()


def source_specs(repo: Path) -> list[dict[str, object]]:
    """Return the exact full-evaluation candidate files used by the paper."""

    specs: list[dict[str, object]] = []

    for seed in (42, 43, 44):
        specs.append(
            {
                "generator": "PerSAM-F-compatible",
                "benchmark": "FSS-1000",
                "seed": seed,
                "fold": 0,
                "path": repo
                / "results"
                / "multicriteria"
                / f"fss_f0_s{seed}_full"
                / "0"
                / "csvs"
                / "selector_original_candidates.csv",
            }
        )

    for benchmark, prefix, seed45_name, seed46_name, seed47_name in (
        (
            "COCO-20i",
            "coco",
            "heldout1000",
            "full1000_guarded",
            "full1000_guarded",
        ),
        (
            "Pascal-5i",
            "pascal",
            "full1000_fixed2",
            "full1000_guarded",
            "full1000_guarded",
        ),
    ):
        for fold in range(4):
            for seed, run_name, file_name in (
                (45, seed45_name, "selector_original_candidates.csv"),
                (46, seed46_name, "selector_guarded_candidates.csv"),
                (47, seed47_name, "selector_guarded_candidates.csv"),
            ):
                specs.append(
                    {
                        "generator": "PerSAM-F-compatible",
                        "benchmark": benchmark,
                        "seed": seed,
                        "fold": fold,
                        "path": repo
                        / "results"
                        / "multicriteria"
                        / f"{prefix}_f{fold}_s{seed}_{run_name}"
                        / str(fold)
                        / "csvs"
                        / file_name,
                    }
                )

    specs.extend(
        [
            {
                "generator": "Matcher",
                "benchmark": "FSS-1000",
                "seed": 42,
                "fold": 0,
                "path": repo
                / "results"
                / "multicriteria"
                / "matcher_fss_f0_s42_original_full_v6"
                / "csvs"
                / "selector_original_candidates.csv",
            },
            {
                "generator": "Matcher",
                "benchmark": "COCO-20i",
                "seed": 42,
                "fold": 0,
                "path": repo
                / "results"
                / "multicriteria"
                / "matcher_coco_f0_s42_original_full_v6"
                / "csvs"
                / "selector_original_candidates.csv",
            },
        ]
    )
    return specs


def read_candidate_file(path: Path) -> list[dict[str, object]]:
    required = {
        "obj_name",
        "iteration",
        "miou",
        "support_query",
        "decoder_confidence",
        "foreground_background_separation",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        rows: list[dict[str, object]] = []
        for raw in reader:
            rows.append(
                {
                    "obj_name": raw["obj_name"],
                    "iteration": int(raw["iteration"]),
                    "miou": float(raw["miou"]),
                    "support_query": float(raw["support_query"]),
                    "decoder_confidence": float(raw["decoder_confidence"]),
                    "foreground_background_separation": float(
                        raw["foreground_background_separation"]
                    ),
                }
            )
    return rows


def stable_argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


def replay_source(spec: dict[str, object]) -> list[dict[str, object]]:
    path = spec["path"]
    assert isinstance(path, Path)
    rows = read_candidate_file(path)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["obj_name"])].append(row)

    output: list[dict[str, object]] = []
    for obj_name, group in groups.items():
        group.sort(key=lambda row: int(row["iteration"]))
        if int(group[0]["iteration"]) != 0:
            raise ValueError(f"No iteration-0 baseline for {path}: {obj_name}")

        scores = [
            float(row["support_query"])
            + ORIGINAL_SCORE_ALPHA * float(row["decoder_confidence"])
            + ORIGINAL_SCORE_BETA
            * float(row["foreground_background_separation"])
            for row in group
        ]
        proposal_index = stable_argmax(scores)
        proposal = group[proposal_index]
        baseline = group[0]
        proposal_iter = int(proposal["iteration"])
        proposal_nonbaseline = proposal_iter != 0
        proposal_miou = float(proposal["miou"])
        baseline_miou = float(baseline["miou"])

        support_margin = (
            float(proposal["support_query"])
            - float(baseline["support_query"])
            if proposal_nonbaseline
            else math.nan
        )
        foreground_margin = (
            float(proposal["foreground_background_separation"])
            - float(baseline["foreground_background_separation"])
            if proposal_nonbaseline
            else math.nan
        )
        gate_margin = (
            min(support_margin, foreground_margin)
            if proposal_nonbaseline
            else math.nan
        )

        output.append(
            {
                "generator": spec["generator"],
                "benchmark": spec["benchmark"],
                "seed": spec["seed"],
                "fold": spec["fold"],
                "obj_name": obj_name,
                "proposal_iter": proposal_iter,
                "proposal_nonbaseline": int(proposal_nonbaseline),
                "baseline_miou": baseline_miou,
                "proposal_miou": proposal_miou,
                "proposal_delta_miou": proposal_miou - baseline_miou,
                "support_margin": support_margin,
                "foreground_margin": foreground_margin,
                "gate_margin": gate_margin,
                "proposal_degraded": int(
                    proposal_nonbaseline
                    and proposal_miou < baseline_miou - MIoU_TOLERANCE
                ),
                "source_file": str(path),
            }
        )
    return output


def curve_for_group(
    generator: str, benchmark: str, episodes: list[dict[str, object]]
) -> list[dict[str, object]]:
    total = len(episodes)
    nonbaseline_count = sum(
        int(row["proposal_nonbaseline"]) for row in episodes
    )
    mean_baseline = (
        sum(float(row["baseline_miou"]) for row in episodes) / total
        if total
        else math.nan
    )
    proposals = sorted(
        (
            row
            for row in episodes
            if int(row["proposal_nonbaseline"]) == 1
            and math.isfinite(float(row["gate_margin"]))
        ),
        key=lambda row: float(row["gate_margin"]),
        reverse=True,
    )

    curve: list[dict[str, object]] = []
    accepted_count = 0
    degraded_count = 0
    cumulative_loss = 0.0
    cumulative_delta = 0.0

    def append_point(threshold: float) -> None:
        coverage = accepted_count / total if total else math.nan
        risk = degraded_count / accepted_count if accepted_count else 0.0
        overall_gain = cumulative_delta / total if total else math.nan
        curve.append(
            {
                "generator": generator,
                "benchmark": benchmark,
                "num_episodes": total,
                "threshold": threshold,
                "accepted_count": accepted_count,
                "coverage": coverage,
                "degradation_count": degraded_count,
                "conditional_degradation_risk": risk,
                "conditional_excess_miou_loss": (
                    cumulative_loss / accepted_count if accepted_count else 0.0
                ),
                "conditional_mean_delta_miou": (
                    cumulative_delta / accepted_count if accepted_count else 0.0
                ),
                "mean_baseline_miou": mean_baseline,
                "overall_selected_miou": mean_baseline + overall_gain,
                "overall_gain_over_baseline": overall_gain,
                "overall_ndr": (
                    1.0 - degraded_count / total if total else math.nan
                ),
                "baseline_output_rate": (
                    1.0 - coverage if total else math.nan
                ),
                "gate_fallback_rate": (
                    (nonbaseline_count - accepted_count) / total
                    if total
                    else math.nan
                ),
                "maximum_refinement_coverage": (
                    nonbaseline_count / total if total else math.nan
                ),
                "is_default_tau": int(threshold == 0.0),
                "is_original_endpoint": int(accepted_count == nonbaseline_count),
            }
        )

    # All-abstain endpoint.  We then process proposals once in descending
    # margin order, making the empirical curve O(N log N) rather than
    # rescanning all episodes for every threshold.
    append_point(math.inf)
    zero_written = False
    for margin, tied_rows_iter in groupby(
        proposals, key=lambda row: float(row["gate_margin"])
    ):
        if margin < 0.0 and not zero_written:
            append_point(0.0)
            zero_written = True

        tied_rows = list(tied_rows_iter)
        accepted_count += len(tied_rows)
        degraded_count += sum(int(row["proposal_degraded"]) for row in tied_rows)
        cumulative_loss += sum(
            max(-float(row["proposal_delta_miou"]), 0.0) for row in tied_rows
        )
        cumulative_delta += sum(
            float(row["proposal_delta_miou"]) for row in tied_rows
        )

        if margin == 0.0:
            append_point(0.0)
            zero_written = True
        else:
            append_point(margin)

    if not zero_written:
        append_point(0.0)
    return curve


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact_label(generator: str, benchmark: str) -> str:
    if generator == "PerSAM-F-compatible":
        return f"PerSAM-F / {benchmark}"
    return f"Matcher / {benchmark}"


def plot_curves(
    curves: dict[tuple[str, str], list[dict[str, object]]],
    output_path: Path,
    title: str,
    panels: list[tuple[str, str]],
    show_legend: bool = True,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes[0]
    colors = {"PerSAM-F-compatible": "#1f77b4", "Matcher": "#d62728"}
    linestyles = {"PerSAM-F-compatible": "-", "Matcher": "--"}

    for ax, (generator, benchmark) in zip(axes_flat, panels):
        data = curves[(generator, benchmark)]
        plot_data = [row for row in data if math.isfinite(float(row["threshold"]))]
        # Sort from low to high coverage for the usual risk--coverage view.
        plot_data.sort(key=lambda row: float(row["coverage"]))
        x = [float(row["coverage"]) for row in plot_data]
        y = [float(row["conditional_degradation_risk"]) for row in plot_data]
        if x:
            ax.plot(
                [0.0] + x,
                [0.0] + y,
                color=colors[generator],
                linestyle=linestyles[generator],
                linewidth=1.8,
                label=compact_label(generator, benchmark),
            )

        default_rows = [row for row in data if int(row["is_default_tau"]) == 1]
        if default_rows:
            row = default_rows[0]
            ax.scatter(
                [float(row["coverage"])],
                [float(row["conditional_degradation_risk"])],
                color=colors[generator],
                edgecolor="white",
                linewidth=0.7,
                s=42,
                zorder=5,
                label=("support_fg operating point" if show_legend else None),
            )

        endpoint_rows = [
            row for row in data if int(row["is_original_endpoint"]) == 1
        ]
        if endpoint_rows:
            row = endpoint_rows[-1]
            ax.scatter(
                [float(row["coverage"])],
                [float(row["conditional_degradation_risk"])],
                facecolors="none",
                edgecolors=colors[generator],
                marker="s",
                linewidth=1.2,
                s=42,
                zorder=5,
                label=("original selector endpoint" if show_legend else None),
            )

        ax.set_title(benchmark)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Accepted refinement coverage")
        ax.set_ylabel("Conditional degradation risk")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        if show_legend:
            ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo / "results" / "multicriteria"
    )
    image_dir = repo / "figures" / "generated"

    specs = source_specs(repo)
    missing = [str(spec["path"]) for spec in specs if not Path(spec["path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            "The following expected full-evaluation candidate files are missing:\n"
            + "\n".join(missing)
        )

    all_per_sample: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for spec in specs:
        replayed = replay_source(spec)
        all_per_sample.extend(replayed)
        grouped[(str(spec["generator"]), str(spec["benchmark"]))].extend(replayed)

    curves: dict[tuple[str, str], list[dict[str, object]]] = {}
    all_curve_rows: list[dict[str, object]] = []
    for key, episodes in grouped.items():
        curve = curve_for_group(key[0], key[1], episodes)
        curves[key] = curve
        all_curve_rows.extend(curve)

    persam_curves = [
        row for row in all_curve_rows if row["generator"] == "PerSAM-F-compatible"
    ]
    matcher_curves = [
        row for row in all_curve_rows if row["generator"] == "Matcher"
    ]
    write_csv(output_dir / "risk_coverage_persam.csv", persam_curves)
    write_csv(output_dir / "risk_coverage_matcher.csv", matcher_curves)
    write_csv(output_dir / "risk_coverage_per_sample.csv", all_per_sample)

    operating_rows = [
        row for row in all_curve_rows if int(row["is_default_tau"]) == 1
    ]
    write_csv(output_dir / "risk_coverage_operating_points.csv", operating_rows)

    persam_panels = [
        ("PerSAM-F-compatible", "FSS-1000"),
        ("PerSAM-F-compatible", "COCO-20i"),
        ("PerSAM-F-compatible", "Pascal-5i"),
    ]
    matcher_panels = [
        ("Matcher", "FSS-1000"),
        ("Matcher", "COCO-20i"),
    ]
    plot_curves(
        curves,
        image_dir / "risk_coverage_persam.png",
        "Risk--coverage of PerSAM-F-compatible candidate selection",
        persam_panels,
    )
    plot_curves(
        curves,
        image_dir / "risk_coverage_matcher.png",
        "Risk--coverage of Matcher candidate selection",
        matcher_panels,
    )

    # The combined figure uses one panel per benchmark.  It overlays Matcher
    # only where a Matcher transfer experiment exists.
    combined_panels = [
        ("PerSAM-F-compatible", "FSS-1000"),
        ("PerSAM-F-compatible", "COCO-20i"),
        ("PerSAM-F-compatible", "Pascal-5i"),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(15.0, 4.0), squeeze=False, constrained_layout=True
    )
    axes_flat = axes[0]
    colors = {"PerSAM-F-compatible": "#1f77b4", "Matcher": "#d62728"}
    linestyles = {"PerSAM-F-compatible": "-", "Matcher": "--"}
    for ax, (_, benchmark) in zip(axes_flat, combined_panels):
        available = [("PerSAM-F-compatible", benchmark)]
        if ("Matcher", benchmark) in curves:
            available.append(("Matcher", benchmark))
        for generator, current_benchmark in available:
            data = [
                row
                for row in curves[(generator, current_benchmark)]
                if math.isfinite(float(row["threshold"]))
            ]
            data.sort(key=lambda row: float(row["coverage"]))
            x = [float(row["coverage"]) for row in data]
            y = [float(row["conditional_degradation_risk"]) for row in data]
            ax.plot(
                [0.0] + x,
                [0.0] + y,
                color=colors[generator],
                linestyle=linestyles[generator],
                linewidth=1.8,
                label=(
                    "PerSAM-F-compatible"
                    if generator == "PerSAM-F-compatible"
                    else "Matcher"
                ),
            )
            default_rows = [
                row
                for row in curves[(generator, current_benchmark)]
                if int(row["is_default_tau"]) == 1
            ]
            if default_rows:
                row = default_rows[0]
                ax.scatter(
                    [float(row["coverage"])],
                    [float(row["conditional_degradation_risk"])],
                    color=colors[generator],
                    edgecolor="white",
                    linewidth=0.7,
                    s=40,
                    zorder=5,
                )
            endpoint_rows = [
                row
                for row in curves[(generator, current_benchmark)]
                if int(row["is_original_endpoint"]) == 1
            ]
            if endpoint_rows:
                row = endpoint_rows[-1]
                ax.scatter(
                    [float(row["coverage"])],
                    [float(row["conditional_degradation_risk"])],
                    facecolors="none",
                    edgecolors=colors[generator],
                    marker="s",
                    linewidth=1.2,
                    s=40,
                    zorder=5,
                )
        ax.set_title(benchmark)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Accepted refinement coverage")
        ax.set_ylabel("Conditional degradation risk")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.suptitle("Risk--coverage across candidate generators", fontsize=12)
    image_dir.mkdir(parents=True, exist_ok=True)
    combined_path = image_dir / "risk_coverage_all_generators.png"
    fig.savefig(combined_path, dpi=600, bbox_inches="tight")
    fig.savefig(combined_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Episodes replayed: {len(all_per_sample)}")
    for key in sorted(curves):
        operating = next(
            row for row in curves[key] if int(row["is_default_tau"]) == 1
        )
        print(
            f"{key[0]} | {key[1]} | N={operating['num_episodes']} | "
            f"lambda=0 coverage={float(operating['coverage']):.6f} | "
            f"risk={float(operating['conditional_degradation_risk']):.6f} | "
            f"NDR={float(operating['overall_ndr']):.6f} | "
            f"baseline-output={float(operating['baseline_output_rate']):.6f} | "
            f"gate-fallback={float(operating['gate_fallback_rate']):.6f} | "
            f"mIoU={float(operating['overall_selected_miou']):.6f}"
        )
    print(f"Curve CSVs: {output_dir}")
    print(f"Figures: {image_dir}")


if __name__ == "__main__":
    main()
