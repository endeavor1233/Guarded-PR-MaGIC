#!/usr/bin/env python3
"""Evaluate whether support_fg rejects harmful proposals better than chance.

The analysis reuses saved episode-level outputs and never reruns a segmentation
model.  It compares support_fg with a random fallback policy that rejects
exactly the same number of non-baseline proposals within each
generator/benchmark group.  It also measures how well the negative gate margin
ranks proposals that are worse than the baseline.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "multicriteria" / "risk_coverage_per_sample.csv"
OUT_RANDOM = ROOT / "results" / "multicriteria" / "gate_matched_random_control.csv"
OUT_DISC = ROOT / "results" / "multicriteria" / "gate_discrimination_metrics.csv"

SEED = 20260907
RANDOM_REPETITIONS = 10_000
BOOTSTRAP_REPETITIONS = 2_000

GROUP_ORDER = [
    ("PerSAM-F-compatible", "FSS-1000"),
    ("PerSAM-F-compatible", "COCO-20i"),
    ("PerSAM-F-compatible", "Pascal-5i"),
    ("Matcher", "FSS-1000"),
    ("Matcher", "COCO-20i"),
]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""
    y = y.astype(np.int8, copy=False)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan

    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    end_ranks = np.cumsum(counts, dtype=float)
    average_ranks = end_ranks - 0.5 * (counts - 1)
    ranks = average_ranks[inverse]
    rank_sum_pos = float(ranks[y == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision with degraded proposals as the positive class."""
    y = y.astype(np.int8, copy=False)
    n_pos = int(y.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    true_positive = np.cumsum(y_sorted)
    precision = true_positive / np.arange(1, y.size + 1)
    return float(precision[y_sorted == 1].sum() / n_pos)


def bootstrap_discrimination(
    y: np.ndarray, score: np.ndarray, rng: np.random.Generator
) -> tuple[tuple[float, float], tuple[float, float]]:
    auc_values = []
    ap_values = []
    n = y.size
    while len(auc_values) < BOOTSTRAP_REPETITIONS:
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.min() == y_b.max():
            continue
        score_b = score[idx]
        auc_values.append(auroc(y_b, score_b))
        ap_values.append(average_precision(y_b, score_b))
    return (
        percentile_interval(np.asarray(auc_values)),
        percentile_interval(np.asarray(ap_values)),
    )


def load_groups() -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with INPUT.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups[(row["generator"], row["benchmark"])].append(row)
    return groups


def analyze_random_control(
    key: tuple[str, str], rows: list[dict[str, str]], rng: np.random.Generator
) -> dict[str, object]:
    baseline = np.asarray([float(row["baseline_miou"]) for row in rows])
    proposal = np.asarray([float(row["proposal_miou"]) for row in rows])
    nonbaseline = np.asarray([parse_bool(row["proposal_nonbaseline"]) for row in rows])
    degraded = np.asarray([parse_bool(row["proposal_degraded"]) for row in rows])
    margin = np.asarray([float(row["gate_margin"]) for row in rows])

    proposal_idx = np.flatnonzero(nonbaseline)
    rejected = nonbaseline & np.isfinite(margin) & (margin < 0)
    accepted = nonbaseline & np.isfinite(margin) & (margin >= 0)
    rejected_idx = np.flatnonzero(rejected)
    accepted_idx = np.flatnonzero(accepted)
    n = len(rows)
    k = rejected_idx.size

    original_output = proposal
    guarded_output = proposal.copy()
    guarded_output[rejected] = baseline[rejected]
    original_ndr = float((~degraded).mean())
    guarded_ndr = float((~(accepted & degraded)).mean())

    delta_if_rejected = baseline[proposal_idx] - proposal[proposal_idx]
    degraded_proposal = degraded[proposal_idx]
    original_mean = float(original_output.mean())
    original_degraded_count = int(degraded.sum())
    accepted_count = int(proposal_idx.size - k)

    random_miou = np.empty(RANDOM_REPETITIONS, dtype=float)
    random_ndr = np.empty(RANDOM_REPETITIONS, dtype=float)
    random_conditional_risk = np.empty(RANDOM_REPETITIONS, dtype=float)
    for rep in range(RANDOM_REPETITIONS):
        selected_local = rng.choice(proposal_idx.size, size=k, replace=False)
        rejected_degraded = int(degraded_proposal[selected_local].sum())
        random_miou[rep] = original_mean + float(delta_if_rejected[selected_local].sum()) / n
        random_ndr[rep] = original_ndr + rejected_degraded / n
        random_conditional_risk[rep] = (
            (original_degraded_count - rejected_degraded) / accepted_count
            if accepted_count
            else math.nan
        )

    miou_ci = percentile_interval(random_miou)
    ndr_ci = percentile_interval(random_ndr)
    risk_ci = percentile_interval(random_conditional_risk)
    gate_minus_random = float(guarded_output.mean()) - random_miou
    gate_diff_ci = percentile_interval(gate_minus_random)

    return {
        "generator": key[0],
        "benchmark": key[1],
        "episodes": n,
        "nonbaseline_proposals": int(proposal_idx.size),
        "matched_rejections": k,
        "gate_fallback_rate": k / n,
        "refinement_coverage": accepted_count / n,
        "baseline_miou": float(baseline.mean()),
        "original_miou": original_mean,
        "random_miou_mean": float(random_miou.mean()),
        "random_miou_ci_low": miou_ci[0],
        "random_miou_ci_high": miou_ci[1],
        "guarded_miou": float(guarded_output.mean()),
        "guarded_minus_random_miou_mean": float(gate_minus_random.mean()),
        "guarded_minus_random_miou_ci_low": gate_diff_ci[0],
        "guarded_minus_random_miou_ci_high": gate_diff_ci[1],
        "original_ndr": original_ndr,
        "random_ndr_mean": float(random_ndr.mean()),
        "random_ndr_ci_low": ndr_ci[0],
        "random_ndr_ci_high": ndr_ci[1],
        "guarded_ndr": guarded_ndr,
        "random_conditional_risk_mean": float(random_conditional_risk.mean()),
        "random_conditional_risk_ci_low": risk_ci[0],
        "random_conditional_risk_ci_high": risk_ci[1],
        "guarded_conditional_risk": (
            float(degraded[accepted_idx].mean()) if accepted_idx.size else math.nan
        ),
        "random_repetitions": RANDOM_REPETITIONS,
        "random_seed": SEED,
    }


def analyze_discrimination(
    key: tuple[str, str], rows: list[dict[str, str]], rng: np.random.Generator
) -> dict[str, object]:
    nonbaseline_rows = [row for row in rows if parse_bool(row["proposal_nonbaseline"])]
    valid_rows = [
        row for row in nonbaseline_rows if math.isfinite(float(row["gate_margin"]))
    ]
    margin = np.asarray([float(row["gate_margin"]) for row in valid_rows])
    degraded = np.asarray(
        [parse_bool(row["proposal_degraded"]) for row in valid_rows], dtype=np.int8
    )
    risk_score = -margin
    auc_value = auroc(degraded, risk_score)
    ap_value = average_precision(degraded, risk_score)
    auc_ci, ap_ci = bootstrap_discrimination(degraded, risk_score, rng)

    rejected = margin < 0
    accepted = margin >= 0
    degraded_count = int(degraded.sum())
    rejected_degraded = int((rejected & (degraded == 1)).sum())

    return {
        "generator": key[0],
        "benchmark": key[1],
        "nonbaseline_proposals": len(nonbaseline_rows),
        "valid_margins": len(valid_rows),
        "degraded_proposals": degraded_count,
        "degradation_prevalence": float(degraded.mean()),
        "auroc": auc_value,
        "auroc_ci_low": auc_ci[0],
        "auroc_ci_high": auc_ci[1],
        "average_precision": ap_value,
        "average_precision_ci_low": ap_ci[0],
        "average_precision_ci_high": ap_ci[1],
        "rejection_precision": float(degraded[rejected].mean()),
        "false_rejection_fraction": float(1.0 - degraded[rejected].mean()),
        "accepted_conditional_risk": float(degraded[accepted].mean()),
        "accepted_non_degradation": float(1.0 - degraded[accepted].mean()),
        "rescue_fraction_of_degraded": rejected_degraded / degraded_count,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": SEED,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    groups = load_groups()
    missing = [key for key in GROUP_ORDER if key not in groups]
    if missing:
        raise RuntimeError(f"Missing expected groups: {missing}")

    random_rng = np.random.default_rng(SEED)
    bootstrap_rng = np.random.default_rng(SEED + 1)
    random_rows = [
        analyze_random_control(key, groups[key], random_rng) for key in GROUP_ORDER
    ]
    discrimination_rows = [
        analyze_discrimination(key, groups[key], bootstrap_rng) for key in GROUP_ORDER
    ]
    write_csv(OUT_RANDOM, random_rows)
    write_csv(OUT_DISC, discrimination_rows)

    print(f"Wrote {OUT_RANDOM.relative_to(ROOT)}")
    print(f"Wrote {OUT_DISC.relative_to(ROOT)}")
    for row in random_rows:
        print(
            f"{row['generator']} / {row['benchmark']}: "
            f"random mIoU={row['random_miou_mean']:.6f}, "
            f"guarded-random={row['guarded_minus_random_miou_mean']:+.6f}"
        )
    for row in discrimination_rows:
        print(
            f"{row['generator']} / {row['benchmark']}: "
            f"AUROC={row['auroc']:.3f}, AP={row['average_precision']:.3f}, "
            f"rejection precision={row['rejection_precision']:.3f}"
        )


if __name__ == "__main__":
    main()
