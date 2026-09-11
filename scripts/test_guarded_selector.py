"""Dependency-light regression test for the guarded selector."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Personalize-SAM"))

from multi_criteria_selector import select_candidate  # noqa: E402


def main():
    accepted_case = {
        "support_query": [0.80, 0.90, 0.70],
        "decoder_confidence": [0.80, 0.95, 0.99],
        "temporal_stability": [1.00, 0.70, 0.80],
        "foreground_background_separation": [0.80, 0.90, 0.95],
    }
    selected, diagnostics = select_candidate(
        accepted_case,
        method="guarded",
    )
    assert selected == 1
    assert diagnostics["fallback_applied"] is False

    fallback_case = {
        "support_query": [0.80, 0.75, 0.70],
        "decoder_confidence": [0.80, 1.20, 0.81],
        "temporal_stability": [1.00, 0.70, 0.80],
        "foreground_background_separation": [0.80, 0.70, 0.81],
    }
    selected, diagnostics = select_candidate(
        fallback_case,
        method="guarded",
    )
    assert diagnostics["pre_fallback_index"] == 1
    assert selected == 0
    assert diagnostics["fallback_applied"] is True

    print("guarded selector regression test: PASS")


if __name__ == "__main__":
    main()
