#!/usr/bin/env python3
"""Evaluate episode-disjoint calibration and cross-seed threshold transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path


DEFAULT_EPSILON = 0.40
INPUT_NAME = "risk_coverage_per_sample.csv"
OUTPUT_NAME = "episode_split_calibration.csv"
GENERATOR = "PerSAM-F-compatible"
BENCHMARK_SEEDS = {
    "FSS-1000": (42, 43, 44),
    "COCO-20i": (45, 46, 47),
    "Pascal-5i": (45, 46, 47),
}
CALIBRATION_BUCKET = 0
CALIBRATION_BUCKETS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_episodes(repo: Path) -> dict[tuple[str, int], list[dict[str, object]]]:
    input_path = repo / "results" / "multicriteria" / INPUT_NAME
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["generator"] != GENERATOR:
                continue
            row: dict[str, object] = {
                "benchmark": raw["benchmark"],
                "seed": int(raw["seed"]),
                "fold": int(raw["fold"]),
                "obj_name": raw["obj_name"],
                "proposal_nonbaseline": int(raw["proposal_nonbaseline"]),
                "baseline_miou": float(raw["baseline_miou"]),
                "proposal_miou": float(raw["proposal_miou"]),
                "proposal_degraded": int(raw["proposal_degraded"]),
                "gate_margin": float(raw["gate_margin"]),
            }
            groups.setdefault((str(row["benchmark"]), int(row["seed"])), []).append(row)
    return groups


def calibration_bucket(benchmark: str, obj_name: str) -> int:
    key = f"guarded-episode-split-v1::{benchmark}::{obj_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % CALIBRATION_BUCKETS


def accepted(row: dict[str, object], threshold: float) -> bool:
    margin = float(row["gate_margin"])
    return int(row["proposal_nonbaseline"]) == 1 and math.isfinite(margin) and margin >= threshold


def evaluate(episodes: list[dict[str, object]], threshold: float) -> dict[str, object]:
    n = len(episodes)
    accepted_rows = [row for row in episodes if accepted(row, threshold)]
    degraded_count = sum(int(row["proposal_degraded"]) for row in accepted_rows)
    baseline_miou = sum(float(row["baseline_miou"]) for row in episodes) / n
    selected_miou = sum(
        float(row["proposal_miou"] if accepted(row, threshold) else row["baseline_miou"])
        for row in episodes
    ) / n
    accepted_count = len(accepted_rows)
    return {
        "episodes": n,
        "accepted_count": accepted_count,
        "coverage": accepted_count / n if n else math.nan,
        "conditional_degradation_risk": degraded_count / accepted_count if accepted_count else 0.0,
        "baseline_miou": baseline_miou,
        "selected_miou": selected_miou,
        "gain_over_baseline": selected_miou - baseline_miou,
        "ndr": 1.0 - degraded_count / n if n else math.nan,
    }


def candidate_thresholds(episodes: list[dict[str, object]]) -> list[float]:
    values = {
        float(row["gate_margin"])
        for row in episodes
        if int(row["proposal_nonbaseline"]) == 1 and math.isfinite(float(row["gate_margin"]))
    }
    values.add(math.inf)
    return sorted(values, key=lambda value: (math.isinf(value), value))


def calibrate(episodes: list[dict[str, object]], epsilon: float) -> tuple[float, dict[str, object]]:
    feasible: list[tuple[float, float, dict[str, object]]] = []
    for threshold in candidate_thresholds(episodes):
        summary = evaluate(episodes, threshold)
        if float(summary["conditional_degradation_risk"]) <= epsilon + 1.0e-12:
            feasible.append((float(summary["coverage"]), threshold, summary))
    if not feasible:
        raise RuntimeError(f"No feasible threshold for epsilon={epsilon}")
    feasible.sort(key=lambda item: (-item[0], item[1]))
    _, threshold, summary = feasible[0]
    return threshold, summary


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    repo = args.repo.resolve()
    output = args.output.resolve() if args.output is not None else repo / "results" / "multicriteria" / OUTPUT_NAME
    groups = load_episodes(repo)
    rows: list[dict[str, object]] = []

    for benchmark, seeds in BENCHMARK_SEEDS.items():
        calibration_seed = seeds[0]
        calibration_all = groups[(benchmark, calibration_seed)]
        calibration = [row for row in calibration_all if calibration_bucket(benchmark, str(row["obj_name"])) == CALIBRATION_BUCKET]
        heldout_same_seed = [row for row in calibration_all if calibration_bucket(benchmark, str(row["obj_name"])) != CALIBRATION_BUCKET]
        threshold, calibration_summary = calibrate(calibration, args.epsilon)
        evaluations = [("calibration", calibration_seed, calibration), ("heldout_same_seed", calibration_seed, heldout_same_seed)]
        for seed in seeds[1:]:
            heldout = [row for row in groups[(benchmark, seed)] if calibration_bucket(benchmark, str(row["obj_name"])) != CALIBRATION_BUCKET]
            evaluations.append(("heldout_seed", seed, heldout))
        for split, seed, episodes in evaluations:
            summary = calibration_summary if split == "calibration" else evaluate(episodes, threshold)
            default_summary = evaluate(episodes, 0.0)
            original_summary = evaluate(episodes, -math.inf)
            rows.append(
                {
                    "benchmark": benchmark,
                    "calibration_seed": calibration_seed,
                    "evaluation_seed": seed,
                    "split": split,
                    "calibration_fraction": 1.0 / CALIBRATION_BUCKETS,
                    "epsilon": args.epsilon,
                    "threshold": threshold,
                    **summary,
                    "default_gain_over_baseline": default_summary["gain_over_baseline"],
                    "original_gain_over_baseline": original_summary["gain_over_baseline"],
                }
            )
        print(f"{benchmark}: calibration seed={calibration_seed}, episodes={len(calibration)}, lambda={threshold:.8f}, risk={float(calibration_summary['conditional_degradation_risk']):.4f}, coverage={float(calibration_summary['coverage']):.4f}")
        for split, seed, episodes in evaluations[1:]:
            summary = evaluate(episodes, threshold)
            print(f"  {split} seed={seed}: episodes={len(episodes)}, risk={float(summary['conditional_degradation_risk']):.4f}, coverage={float(summary['coverage']):.4f}, gain={float(summary['gain_over_baseline']):+.6f}, NDR={float(summary['ndr']):.4f}")

    write_rows(output, rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
