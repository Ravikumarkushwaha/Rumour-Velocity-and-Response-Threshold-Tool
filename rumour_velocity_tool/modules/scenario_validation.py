"""Regression checks for the project's authored fictional scenarios.

These expected outcomes are design fixtures, not independently validated labels
or evidence that the thresholds perform well on real-world claims.
"""
from __future__ import annotations

import pandas as pd

from config import (REC_INSUFFICIENT, REC_MONITOR, REC_REVIEW, REC_SILENT,
                    SYNTHETIC_CSV, default_config)
from modules.decision_engine import ClaimContext
from modules.pipeline import run_assessment

EXPECTED_RECOMMENDATIONS = {
    "SYN-001": REC_SILENT,
    "SYN-002": REC_MONITOR,
    "SYN-003": REC_REVIEW,
    "SYN-004": REC_MONITOR,
    "SYN-005": REC_REVIEW,
    "SYN-006": REC_INSUFFICIENT,
    "SYN-007": REC_MONITOR,
    "SYN-008": REC_MONITOR,
    "SYN-009": REC_MONITOR,
    "SYN-010": REC_SILENT,
}


def evaluate_synthetic_scenarios() -> pd.DataFrame:
    """Compare current recommendations with the authored scenario fixtures."""
    data = pd.read_csv(SYNTHETIC_CSV)
    scenarios = data.drop_duplicates("claim_id").set_index("claim_id")["scenario"]
    rows = []
    for claim_id, scenario in scenarios.items():
        claim_rows = data[data["claim_id"] == claim_id]
        claim = claim_rows.iloc[0]
        context = ClaimContext(
            claim_id=claim_id,
            description=claim["description"],
            category=claim["category"],
            verification_status=claim["verification_status"],
            public_impact=claim["public_impact"],
            public_safety_flag=bool(claim["public_safety_flag"]),
            credible_allegation_flag=bool(claim["credible_allegation_flag"]),
        )
        bundle = run_assessment(context, claim_rows, default_config(), claim["first_observed"])
        actual = bundle.decision.recommendation
        rows.append({
            "Claim": claim_id,
            "Scenario": scenario,
            "Expected": EXPECTED_RECOMMENDATIONS.get(claim_id, "UNSPECIFIED"),
            "Actual": actual,
            "Match": actual == EXPECTED_RECOMMENDATIONS.get(claim_id),
            "RPS": bundle.decision.rps_score,
            "Reach confidence": bundle.reach.confidence,
        })
    return pd.DataFrame(rows)
