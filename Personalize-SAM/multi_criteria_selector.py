"""Training-free candidate-mask selectors for the PR-MaGIC prototype.

This module intentionally contains only NumPy code so that the selector can
be tested independently from SAM.  Every score is assumed to be larger when
the candidate is better.
"""

import numpy as np


CRITERIA = (
    "support_query",
    "decoder_confidence",
    "temporal_stability",
    "foreground_background_separation",
)


def _as_score_array(values, n):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != n:
        raise ValueError(f"Expected {n} scores, got {values.size}")
    values = np.nan_to_num(values, nan=-1e9, posinf=1e9, neginf=-1e9)
    return values


def _descending_rank(values):
    """Return zero-based ranks; ties receive the same best rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    if values.size > 1:
        return 1.0 - ranks / float(values.size - 1)
    return np.ones_like(values)


def _rank_matrix(criteria):
    n = len(next(iter(criteria.values())))
    return np.stack([_descending_rank(_as_score_array(criteria[k], n)) for k in CRITERIA], axis=0)


def select_candidate(criteria, method="rank_sum", fallback=True):
    """Select a candidate without using query ground truth.

    Args:
        criteria: dict mapping every name in ``CRITERIA`` to an array of K
            candidate scores. Candidate 0 must be the unrefined baseline.
        method: ``original`` reproduces the existing PR-MaGIC score
            ``support_query + 0.25 * decoder_confidence + 0.10 * separation``;
            ``guarded`` first computes the original selection and then accepts
            it only when support-query consistency and foreground-background
            separation are no worse than candidate 0.  Otherwise it returns
            candidate 0.  The guarded rule is the baseline-relative
            support_fg safety gate used in the manuscript;
            ``support_top1`` uses support-query similarity only;
            ``rank_sum`` sums per-criterion ranks;
            ``median_rank`` selects the candidate with the best median rank;
            ``consensus`` uses a majority-first rule followed by median rank.
        fallback: if true, a non-baseline candidate must be no worse than the
            baseline on at least two criteria by rank; otherwise candidate 0
            is selected.

    Returns:
        selected index and a diagnostic dictionary.
    """
    missing = [key for key in CRITERIA if key not in criteria]
    if missing:
        raise ValueError(f"Missing criteria: {missing}")

    n = len(np.asarray(criteria[CRITERIA[0]]).reshape(-1))
    if n == 0:
        raise ValueError("At least one candidate is required")

    raw = {key: _as_score_array(criteria[key], n) for key in CRITERIA}
    ranks = _rank_matrix(raw)

    if method in ("original", "guarded"):
        # Match the original PR-MaGIC implementation in this repository:
        # base + score_alpha * sam_score + score_beta * margin, with the
        # default score_alpha=0.25 and score_beta=0.10.
        legacy_score = (
            raw["support_query"]
            + 0.25 * raw["decoder_confidence"]
            + 0.10 * raw["foreground_background_separation"]
        )
        selected = int(np.argmax(legacy_score))
        aggregate = legacy_score
    elif method == "support_top1":
        selected = int(np.argmax(raw["support_query"]))
        aggregate = raw["support_query"]
    elif method == "rank_sum":
        aggregate = ranks.mean(axis=0)
        selected = int(np.argmax(aggregate))
    elif method == "median_rank":
        median_rank = np.median(1.0 - ranks, axis=0)
        selected = int(np.argmin(median_rank))
        aggregate = 1.0 - median_rank
    elif method == "consensus":
        first_choices = np.argmax(ranks, axis=1)
        counts = np.bincount(first_choices, minlength=n)
        majority = int(np.argmax(counts))
        if counts[majority] >= 3:
            selected = majority
        else:
            median_rank = np.median(1.0 - ranks, axis=0)
            selected = int(np.argmin(median_rank))
        aggregate = ranks.mean(axis=0)
    else:
        raise ValueError(f"Unknown selector method: {method}")

    # ``original`` is the exact support-query Top-1 reference and must not be
    # altered by the optional multi-criteria fallback gate.  ``guarded`` is a
    # separate, explicit baseline-relative safety policy: it checks the two
    # support_fg criteria against candidate 0 and falls back when either one
    # is worse.  This gate is deliberately independent of the optional
    # rank-based fallback below.
    pre_fallback = selected
    supports_baseline = True
    if method == "guarded" and n > 1 and selected != 0:
        supports_baseline = (
            raw["support_query"][selected] >= raw["support_query"][0]
            and raw["foreground_background_separation"][selected]
            >= raw["foreground_background_separation"][0]
        )
        if not supports_baseline:
            selected = 0
    elif fallback and method not in ("original", "support_top1", "guarded") and n > 1 and selected != 0:
        # A conservative, scale-free gate.  This is not a formal guarantee;
        # it only prevents a candidate that loses to baseline on most signals
        # from being selected.
        baseline_rank = ranks[:, 0]
        selected_rank = ranks[:, selected]
        supports_baseline = int(np.sum(selected_rank >= baseline_rank)) >= 2
        if not supports_baseline:
            selected = 0

    return int(selected), {
        "pre_fallback_index": int(pre_fallback),
        "fallback_applied": bool(selected == 0 and pre_fallback != 0),
        "supports_baseline": bool(supports_baseline),
        "aggregate": aggregate.tolist(),
        "rank_matrix": ranks.tolist(),
        "raw": {key: values.tolist() for key, values in raw.items()},
    }


def iou(mask_a, mask_b):
    """Binary mask IoU used for temporal stability."""
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)
