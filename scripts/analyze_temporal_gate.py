"""Offline ablation for adding temporal stability to the support--foreground gate.

The candidate CSVs already contain all four internal diagnostics and query IoU.
This script therefore replays candidate selection without running SAM again.

The original proposal is reconstructed with the manuscript score

    support_query + 0.25 * decoder_confidence
                  + 0.10 * foreground_background_separation.

The support_fg_tmp policy accepts that proposal only if support and foreground
do not fall below candidate 0 and

    temporal_stability(proposal) >= 1 - tau_tmp.

Candidate 0 has temporal stability 1 by construction.  Consequently,
tau_tmp=0 is the literal baseline-relative temporal gate and is expected to be
extremely conservative.  The script reports this diagnostic rather than hiding
it, scans tau_tmp, performs deterministic episode-disjoint calibration, and
compares each selected policy with matched-rate random fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceGroup:
    generator: str
    benchmark: str
    seed: int
    fold: int
    path: Path


def calibration_bucket(benchmark: str, obj_name: str) -> int:
    """Match the split used by evaluate_split_calibration.py exactly."""
    key = f"guarded-episode-split-v1::{benchmark}::{obj_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 5


def discover_sources(root: Path) -> list[SourceGroup]:
    base = root / "results" / "multicriteria"
    groups: list[SourceGroup] = []

    for seed in (42, 43, 44):
        groups.append(
            SourceGroup(
                "PerSAM-F-compatible",
                "FSS-1000",
                seed,
                0,
                base
                / f"fss_f0_s{seed}_full"
                / "0"
                / "csvs"
                / "selector_original_candidates.csv",
            )
        )

    for seed in (45, 46, 47):
        for fold in range(4):
            if seed == 45:
                directory = f"coco_f{fold}_s45_heldout1000"
                filename = "selector_original_candidates.csv"
            else:
                directory = f"coco_f{fold}_s{seed}_full1000_guarded"
                filename = "selector_guarded_candidates.csv"
            groups.append(
                SourceGroup(
                    "PerSAM-F-compatible",
                    "COCO-20i",
                    seed,
                    fold,
                    base / directory / str(fold) / "csvs" / filename,
                )
            )

    for seed in (45, 46, 47):
        for fold in range(4):
            if seed == 45:
                directory = f"pascal_f{fold}_s45_full1000_fixed2"
                filename = "selector_original_candidates.csv"
            else:
                directory = f"pascal_f{fold}_s{seed}_full1000_guarded"
                filename = "selector_guarded_candidates.csv"
            groups.append(
                SourceGroup(
                    "PerSAM-F-compatible",
                    "Pascal-5i",
                    seed,
                    fold,
                    base / directory / str(fold) / "csvs" / filename,
                )
            )

    groups.extend(
        [
            SourceGroup(
                "Matcher",
                "FSS-1000",
                42,
                0,
                base
                / "matcher_fss_f0_s42_original_full_v6"
                / "csvs"
                / "selector_original_candidates.csv",
            ),
            SourceGroup(
                "Matcher",
                "COCO-20i",
                42,
                0,
                base
                / "matcher_coco_f0_s42_original_full_v6"
                / "csvs"
                / "selector_original_candidates.csv",
            ),
        ]
    )

    missing = [str(group.path) for group in groups if not group.path.is_file()]
    if missing:
        raise FileNotFoundError("Missing candidate CSVs:\n" + "\n".join(missing))
    return groups


def reconstruct_episodes(group: SourceGroup) -> pd.DataFrame:
    candidates = pd.read_csv(group.path)
    required = {
        "obj_name",
        "iteration",
        "miou",
        "support_query",
        "decoder_confidence",
        "temporal_stability",
        "foreground_background_separation",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"{group.path} lacks columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for obj_name, episode in candidates.groupby("obj_name", sort=False):
        episode = episode.sort_values("iteration", kind="stable").reset_index(drop=True)
        iterations = episode["iteration"].to_numpy(dtype=int)
        if iterations.size == 0 or iterations[0] != 0:
            raise ValueError(f"Episode {obj_name} in {group.path} lacks candidate 0")

        original_score = (
            episode["support_query"].to_numpy(dtype=float)
            + 0.25 * episode["decoder_confidence"].to_numpy(dtype=float)
            + 0.10
            * episode["foreground_background_separation"].to_numpy(dtype=float)
        )
        proposal_pos = int(np.argmax(original_score))
        baseline = episode.iloc[0]
        proposal = episode.iloc[proposal_pos]
        proposal_iter = int(proposal["iteration"])
        baseline_miou = float(baseline["miou"])
        proposal_miou = float(proposal["miou"])
        support_margin = float(proposal["support_query"] - baseline["support_query"])
        foreground_margin = float(
            proposal["foreground_background_separation"]
            - baseline["foreground_background_separation"]
        )
        temporal_value = float(proposal["temporal_stability"])
        temporal_baseline = float(baseline["temporal_stability"])

        rows.append(
            {
                "generator": group.generator,
                "benchmark": group.benchmark,
                "seed": group.seed,
                "fold": group.fold,
                "obj_name": str(obj_name),
                "episode_id": f"{group.benchmark}|fold{group.fold}|{obj_name}",
                "proposal_iter": proposal_iter,
                "proposal_nonbaseline": int(proposal_iter != 0),
                "baseline_miou": baseline_miou,
                "proposal_miou": proposal_miou,
                "proposal_delta_miou": proposal_miou - baseline_miou,
                "proposal_degraded": int(proposal_miou < baseline_miou - 1e-12),
                "support_margin": support_margin,
                "foreground_margin": foreground_margin,
                "temporal_stability": temporal_value,
                "temporal_baseline": temporal_baseline,
                "temporal_deficit": temporal_baseline - temporal_value,
                "support_fg_pass": int(
                    proposal_iter == 0
                    or (support_margin >= 0.0 and foreground_margin >= 0.0)
                ),
                "source_file": str(group.path),
            }
        )
    return pd.DataFrame(rows)


def load_all_episodes(root: Path) -> pd.DataFrame:
    frames = [reconstruct_episodes(group) for group in discover_sources(root)]
    episodes = pd.concat(frames, ignore_index=True)
    duplicated = episodes.duplicated(
        subset=["generator", "benchmark", "seed", "fold", "obj_name"]
    )
    if duplicated.any():
        raise ValueError("Duplicate episode keys found in source files")
    return episodes


def policy_accepts(frame: pd.DataFrame, tau_tmp: float) -> np.ndarray:
    nonbaseline = frame["proposal_nonbaseline"].to_numpy(dtype=bool)
    support_fg = frame["support_fg_pass"].to_numpy(dtype=bool)
    temporal_ok = frame["temporal_deficit"].to_numpy(dtype=float) <= tau_tmp + 1e-12
    return nonbaseline & support_fg & temporal_ok


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = average_ranks(scores)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, labels.size + 1)
    return float(precision[sorted_labels].sum() / positives)


def bootstrap_discrimination(
    labels: np.ndarray,
    scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    if repetitions <= 0:
        return {
            "bootstrap_repetitions": 0,
            "bootstrap_seed": seed,
            "auroc_ci_low": math.nan,
            "auroc_ci_high": math.nan,
            "average_precision_ci_low": math.nan,
            "average_precision_ci_high": math.nan,
        }
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    ap_values: list[float] = []
    while len(auc_values) < repetitions:
        indices = rng.integers(0, labels.size, size=labels.size)
        sample_labels = labels[indices]
        if sample_labels.min() == sample_labels.max():
            continue
        sample_scores = scores[indices]
        auc_values.append(binary_auroc(sample_labels, sample_scores))
        ap_values.append(binary_average_precision(sample_labels, sample_scores))
    return {
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "auroc_ci_low": float(np.quantile(auc_values, 0.025)),
        "auroc_ci_high": float(np.quantile(auc_values, 0.975)),
        "average_precision_ci_low": float(np.quantile(ap_values, 0.025)),
        "average_precision_ci_high": float(np.quantile(ap_values, 0.975)),
    }


def temporal_discrimination(
    frame: pd.DataFrame,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    eligible = frame.loc[
        (frame["proposal_nonbaseline"] == 1) & (frame["support_fg_pass"] == 1)
    ]
    labels = eligible["proposal_degraded"].to_numpy(dtype=bool)
    scores = eligible["temporal_deficit"].to_numpy(dtype=float)
    point_estimates = {
        "eligible_proposals": len(eligible),
        "degraded_proposals": int(labels.sum()),
        "degradation_prevalence": float(labels.mean()) if len(labels) else math.nan,
        "temporal_deficit_auroc": binary_auroc(labels, scores),
        "temporal_deficit_average_precision": binary_average_precision(labels, scores),
        "mean_temporal_deficit": float(scores.mean()) if len(scores) else math.nan,
        "median_temporal_deficit": float(np.median(scores)) if len(scores) else math.nan,
    }
    return {
        **point_estimates,
        **bootstrap_discrimination(
            labels, scores, bootstrap_repetitions, bootstrap_seed
        ),
    }


def summarize(frame: pd.DataFrame, tau_tmp: float) -> dict[str, float | int]:
    n = len(frame)
    accept = policy_accepts(frame, tau_tmp)
    nonbaseline = frame["proposal_nonbaseline"].to_numpy(dtype=bool)
    baseline = frame["baseline_miou"].to_numpy(dtype=float)
    proposal = frame["proposal_miou"].to_numpy(dtype=float)
    degraded = frame["proposal_degraded"].to_numpy(dtype=bool)
    selected = np.where(accept, proposal, baseline)
    accepted_count = int(accept.sum())
    rejected_nonbaseline = int((nonbaseline & ~accept).sum())
    degraded_accepted = int((accept & degraded).sum())
    return {
        "episodes": n,
        "tau_tmp": tau_tmp,
        "accepted_count": accepted_count,
        "coverage": accepted_count / n if n else math.nan,
        "gate_fallback_rate": rejected_nonbaseline / n if n else math.nan,
        "conditional_degradation_risk": (
            degraded_accepted / accepted_count if accepted_count else 0.0
        ),
        "selected_miou": float(selected.mean()) if n else math.nan,
        "baseline_miou": float(baseline.mean()) if n else math.nan,
        "gain_over_baseline": float((selected - baseline).mean()) if n else math.nan,
        "ndr": float((selected >= baseline - 1e-12).mean()) if n else math.nan,
    }


def tau_grid(frame: pd.DataFrame, max_points: int = 401) -> np.ndarray:
    eligible = frame.loc[
        (frame["proposal_nonbaseline"] == 1) & (frame["support_fg_pass"] == 1),
        "temporal_deficit",
    ].to_numpy(dtype=float)
    eligible = eligible[np.isfinite(eligible)]
    if eligible.size == 0:
        return np.array([0.0])
    values = np.unique(np.clip(eligible, 0.0, None))
    if values.size > max_points:
        quantiles = np.linspace(0.0, 1.0, max_points)
        values = np.unique(np.quantile(values, quantiles))
    return np.unique(np.concatenate(([0.0], values, [float(values.max()) + 1e-12])))


def matched_random_control(
    frame: pd.DataFrame,
    tau_tmp: float,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    accept = policy_accepts(frame, tau_tmp)
    nonbaseline_positions = np.flatnonzero(
        frame["proposal_nonbaseline"].to_numpy(dtype=bool)
    )
    accepted_count = int(accept.sum())
    baseline = frame["baseline_miou"].to_numpy(dtype=float)
    proposal = frame["proposal_miou"].to_numpy(dtype=float)
    guarded = np.where(accept, proposal, baseline)
    guarded_mean = float(guarded.mean())

    rng = np.random.default_rng(seed)
    random_means = np.empty(repetitions, dtype=float)
    for rep in range(repetitions):
        random_accept = np.zeros(len(frame), dtype=bool)
        if accepted_count:
            chosen = rng.choice(
                nonbaseline_positions, size=accepted_count, replace=False
            )
            random_accept[chosen] = True
        random_selected = np.where(random_accept, proposal, baseline)
        random_means[rep] = random_selected.mean()

    difference = guarded_mean - random_means
    return {
        "random_repetitions": repetitions,
        "random_seed": seed,
        "random_miou_mean": float(random_means.mean()),
        "random_miou_ci_low": float(np.quantile(random_means, 0.025)),
        "random_miou_ci_high": float(np.quantile(random_means, 0.975)),
        "guarded_minus_random_pp": float(100.0 * difference.mean()),
        "difference_ci_low_pp": float(100.0 * np.quantile(difference, 0.025)),
        "difference_ci_high_pp": float(100.0 * np.quantile(difference, 0.975)),
    }


def calibrate_tau(frame: pd.DataFrame, epsilon: float) -> tuple[float, bool]:
    candidates = tau_grid(frame, max_points=10_000)
    feasible: list[tuple[float, float]] = []
    for tau in candidates:
        metrics = summarize(frame, float(tau))
        risk = float(metrics["conditional_degradation_risk"])
        coverage = float(metrics["coverage"])
        if risk <= epsilon + 1e-12:
            feasible.append((coverage, float(tau)))
    if not feasible:
        return math.nan, False
    feasible.sort(key=lambda item: (item[0], item[1]))
    selected_coverage, selected_tau = feasible[-1]
    return selected_tau, selected_coverage > 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epsilon", type=float, default=0.40)
    parser.add_argument("--random-repetitions", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260908)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/multicriteria"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_all_episodes(root)
    episodes.to_csv(output_dir / "temporal_gate_per_sample.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    discrimination_rows: list[dict[str, object]] = []

    group_columns = ["generator", "benchmark"]
    for group_index, (group_key, frame) in enumerate(
        episodes.groupby(group_columns, sort=False)
    ):
        generator, benchmark = group_key
        frame = frame.reset_index(drop=True)
        discrimination_rows.append(
            {
                "generator": generator,
                "benchmark": benchmark,
                **temporal_discrimination(
                    frame,
                    args.bootstrap_repetitions,
                    args.bootstrap_seed + group_index,
                ),
            }
        )

        eligible_deficits = frame.loc[
            (frame["proposal_nonbaseline"] == 1)
            & (frame["support_fg_pass"] == 1),
            "temporal_deficit",
        ]
        max_tau = float(max(0.0, eligible_deficits.max())) if len(eligible_deficits) else 0.0
        policies = [("literal_tau0", 0.0), ("support_fg_equivalent", max_tau + 1e-12)]
        for policy, tau in policies:
            metrics = summarize(frame, tau)
            random_metrics = matched_random_control(
                frame,
                tau,
                repetitions=args.random_repetitions,
                seed=args.random_seed,
            )
            summary_rows.append(
                {
                    "generator": generator,
                    "benchmark": benchmark,
                    "policy": policy,
                    **metrics,
                    **random_metrics,
                }
            )

        for tau in tau_grid(frame):
            sweep_rows.append(
                {
                    "generator": generator,
                    "benchmark": benchmark,
                    **summarize(frame, float(tau)),
                }
            )

        seeds = sorted(frame["seed"].unique())
        calibration_seed = int(seeds[0])
        split_bucket = frame.apply(
            lambda row: calibration_bucket(
                str(row["benchmark"]), str(row["obj_name"])
            ),
            axis=1,
        )
        calibration_mask = (frame["seed"] == calibration_seed) & (split_bucket == 0)
        calibration = frame.loc[calibration_mask].reset_index(drop=True)
        tau_cal, calibration_feasible = calibrate_tau(calibration, args.epsilon)

        calibration_rows.append(
            {
                "generator": generator,
                "benchmark": benchmark,
                "split": "calibration",
                "seed": calibration_seed,
                "epsilon": args.epsilon,
                "calibration_feasible": int(calibration_feasible),
                **summarize(calibration, tau_cal),
            }
        )
        for seed in seeds:
            evaluation = frame.loc[
                (frame["seed"] == seed) & (split_bucket != 0)
            ].reset_index(drop=True)
            if evaluation.empty:
                continue
            metrics = summarize(evaluation, tau_cal)
            random_metrics = matched_random_control(
                evaluation,
                tau_cal,
                repetitions=args.random_repetitions,
                seed=args.random_seed + int(seed),
            )
            calibration_rows.append(
                {
                    "generator": generator,
                    "benchmark": benchmark,
                    "split": "held_out",
                    "seed": int(seed),
                    "epsilon": args.epsilon,
                    "calibration_feasible": int(calibration_feasible),
                    **metrics,
                    **random_metrics,
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "temporal_gate_summary.csv", index=False
    )
    pd.DataFrame(sweep_rows).to_csv(
        output_dir / "temporal_gate_tolerance_sweep.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        output_dir / "temporal_gate_episode_split_calibration.csv", index=False
    )
    pd.DataFrame(discrimination_rows).to_csv(
        output_dir / "temporal_gate_discrimination.csv", index=False
    )

    summary = pd.DataFrame(summary_rows)
    print(
        summary[
            [
                "generator",
                "benchmark",
                "policy",
                "episodes",
                "tau_tmp",
                "coverage",
                "conditional_degradation_risk",
                "selected_miou",
                "ndr",
                "guarded_minus_random_pp",
            ]
        ].to_string(index=False)
    )
    print("\nEpisode-disjoint temporal calibration")
    calibration_table = pd.DataFrame(calibration_rows)
    print(
        calibration_table[
            [
                "generator",
                "benchmark",
                "split",
                "seed",
                "episodes",
                "tau_tmp",
                "coverage",
                "conditional_degradation_risk",
                "selected_miou",
                "ndr",
            ]
        ].to_string(index=False)
    )
    print("\nTemporal-deficit discrimination after the support--foreground gate")
    print(pd.DataFrame(discrimination_rows).to_string(index=False))


if __name__ == "__main__":
    main()
