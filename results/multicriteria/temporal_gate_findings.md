# Temporal-stability gate ablation

## Question

Can the saved temporal-stability signal improve the existing support--foreground
gate without rerunning SAM?

## Protocol

- Reconstruct the original proposal from every candidate CSV using
  `support_query + 0.25 * decoder_confidence + 0.10 * foreground_background_separation`.
- Keep the proposal fixed, apply the existing support--foreground gate, and add
  `temporal_stability(proposal) >= 1 - tau_tmp` as a third acceptance condition.
- Evaluate all three PerSAM-F-compatible benchmarks and both Matcher transfer
  benchmarks.
- Report mIoU, accepted-refinement coverage, conditional degradation risk, NDR,
  and matched-rate random fallback over 10,000 assignments.
- Evaluate temporal deficit as a degradation score with 2,000 episode bootstrap
  resamples.
- Use the same deterministic 20% episode split as the manuscript calibration
  experiment.

## Main finding

Temporal stability should not be added to the default gate in its current form.
Candidate 0 has temporal stability 1 by construction, so the literal
baseline-relative rule (`tau_tmp = 0`) accepts no refinements on FSS-1000 and
COCO-20i and only one of 12,000 Pascal-5i refinements. It is therefore an
almost-all-fallback policy rather than an informative three-signal gate.

Among proposals already accepted by the support--foreground gate, temporal
deficit ranks degraded proposals worse than chance in every evaluation:

| Generator | Benchmark | AUROC (95% bootstrap interval) | AP (95% bootstrap interval) |
|---|---|---:|---:|
| PerSAM-F-compatible | FSS-1000 | 0.456 (0.435--0.476) | 0.365 (0.345--0.385) |
| PerSAM-F-compatible | COCO-20i | 0.431 (0.416--0.446) | 0.296 (0.281--0.311) |
| PerSAM-F-compatible | Pascal-5i | 0.269 (0.255--0.283) | 0.550 (0.535--0.567) |
| Matcher | FSS-1000 | 0.388 (0.355--0.425) | 0.314 (0.285--0.347) |
| Matcher | COCO-20i | 0.445 (0.390--0.497) | 0.352 (0.301--0.410) |

The episode-disjoint tolerance experiment reaches the same conclusion:

- FSS-1000 selects `tau_tmp = 0.00377` on the calibration subset, but held-out
  conditional risk is 40.49--48.40%. At the same accepted count, the temporal
  gate is 0.23--0.40 mIoU percentage points below random acceptance for the
  three held-out seeds (one interval includes zero).
- COCO-20i selects `tau_tmp = 1`, which removes the temporal condition and
  reproduces the existing support--foreground gate.
- Pascal-5i and Matcher FSS-1000 can satisfy the nominal risk target only with
  zero or near-zero coverage under this temporal rule.
- Matcher COCO-20i selects a permissive tolerance, but its held-out advantage
  over matched random acceptance is not conclusive.

## Interpretation

The current signal measures IoU with the immediately preceding candidate, not
quality relative to candidate 0. High local mask stability is therefore not a
reliable indicator that the proposal is safer than the baseline. These results
are a useful negative ablation, but they do not justify changing the main gate.

## Outputs

- `scripts/analyze_temporal_gate.py`
- `results/multicriteria/temporal_gate_per_sample.csv`
- `results/multicriteria/temporal_gate_summary.csv`
- `results/multicriteria/temporal_gate_tolerance_sweep.csv`
- `results/multicriteria/temporal_gate_discrimination.csv`
- `results/multicriteria/temporal_gate_episode_split_calibration.csv`

