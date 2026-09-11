"""Analyze PR-MaGIC candidate CSVs without loading a GPU model.

The CSV is produced by Personalize-SAM/pr_magic_for_persam.py.  This script
reuses exactly the same selector implementation and compares several
selectors on one fixed set of candidate masks.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Keep invocation independent of the caller's working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Personalize-SAM')))
from multi_criteria_selector import CRITERIA, select_candidate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-csv', required=True,
                        help='Absolute path to selector_*_candidates.csv')
    parser.add_argument('--output-csv', default='',
                        help='Absolute output path for the summary CSV')
    parser.add_argument('--no-fallback', action='store_true')
    return parser.parse_args()


def choose_single(values, criterion):
    values = np.asarray(values, dtype=np.float64)
    return int(np.argmax(values))


def main():
    args = parse_args()
    candidate_csv = os.path.abspath(args.candidate_csv)
    if not os.path.isfile(candidate_csv):
        raise FileNotFoundError(candidate_csv)

    df = pd.read_csv(candidate_csv)
    required = {'obj_name', 'iteration', 'miou', *CRITERIA}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f'Missing columns in {candidate_csv}: {missing}')

    methods = [
        'original',
        'support_top1',
        'single_decoder_confidence',
        'single_temporal_stability',
        'single_foreground_background_separation',
        'rank_sum',
        'median_rank',
        'consensus',
    ]
    rows = []
    for obj_name, group in df.groupby('obj_name', sort=False):
        group = group.sort_values('iteration')
        criteria = {
            key: group[key].to_numpy(dtype=np.float64)
            for key in CRITERIA
        }
        mious = group['miou'].to_numpy(dtype=np.float64)
        baseline = float(mious[0])
        oracle = float(np.max(mious))

        for method in methods:
            if method.startswith('single_'):
                criterion = method[len('single_'):]
                selected = choose_single(criteria[criterion], criterion)
                pre_fallback = selected
                fallback_applied = False
            else:
                selected, diag = select_candidate(
                    criteria,
                    method=method,
                    fallback=not args.no_fallback,
                )
                pre_fallback = int(diag['pre_fallback_index'])
                fallback_applied = bool(diag['fallback_applied'])

            selected_miou = float(mious[selected])
            rows.append({
                'obj_name': obj_name,
                'method': method,
                'selected_iter': int(group.iloc[selected]['iteration']),
                'pre_fallback_iter': int(group.iloc[pre_fallback]['iteration']),
                'fallback_applied': fallback_applied,
                'selected_miou': selected_miou,
                'baseline_miou': baseline,
                'oracle_miou': oracle,
                'non_degraded': int(selected_miou >= baseline - 1e-12),
                'oracle_gap_closure': (
                    (selected_miou - baseline) / (oracle - baseline)
                    if oracle > baseline + 1e-12 else np.nan
                ),
            })

    result = pd.DataFrame(rows)
    summary_rows = []
    for method, group in result.groupby('method', sort=False):
        mean_selected = group['selected_miou'].mean()
        mean_baseline = group['baseline_miou'].mean()
        mean_oracle = group['oracle_miou'].mean()
        summary_rows.append({
            'method': method,
            'num_samples': len(group),
            'mean_selected_miou': group['selected_miou'].mean(),
            'mean_baseline_miou': group['baseline_miou'].mean(),
            'mean_oracle_miou': group['oracle_miou'].mean(),
            'gain_over_baseline': mean_selected - mean_baseline,
            'non_degradation_rate': group['non_degraded'].mean(),
            # Aggregate before dividing. Averaging per-sample ratios is
            # unstable when oracle and baseline are nearly identical.
            'mean_oracle_gap_closure': (
                (mean_selected - mean_baseline) / (mean_oracle - mean_baseline)
                if mean_oracle > mean_baseline + 1e-12 else np.nan
            ),
            'mean_selected_iter': group['selected_iter'].mean(),
            'fallback_rate': group['fallback_applied'].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.6f}'))

    if args.output_csv:
        output_csv = os.path.abspath(args.output_csv)
    else:
        stem, _ = os.path.splitext(candidate_csv)
        output_csv = stem + '_analysis.csv'
    summary.to_csv(output_csv, index=False)
    print(f'Saved: {output_csv}')


if __name__ == '__main__':
    main()
