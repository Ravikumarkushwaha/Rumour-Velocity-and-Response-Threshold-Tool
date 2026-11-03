import copy

import pytest

from config import (REC_INSUFFICIENT, REC_MONITOR, REC_REVIEW, REC_SILENT, UNVERIFIED, VERIFIED_FALSE,
                    VERIFIED_TRUE, default_config)
from modules.decision_engine import ClaimContext, decide, escalation_reasons
from modules.scenario_validation import evaluate_synthetic_scenarios
from tests.helpers import make_obs, run_assessment, synthetic_bundle


# =========================== fictional regression fixtures ===================
def test_authored_scenario_regression_fixtures_match():
    results = evaluate_synthetic_scenarios()
    assert len(results) == 10
    assert results["Match"].all()


def bundle_from(mentions, reach=None, step=1.0, **ctx):
    kw = dict(claim_id="T", description="d", category="Other", verification_status=UNVERIFIED, public_impact="Low")
    kw.update(ctx)
    return run_assessment(ClaimContext(**kw), make_obs(mentions, step, reach), default_config())


# =========================== the eight required fictional scenarios =============
def test_scenario_1_limited_low_velocity_high_amplification_remains_silent():
    b = synthetic_bundle("SYN-001")
    assert b.trend.label == "Low velocity" and b.amp.risk == "High"
    assert b.decision.recommendation == REC_SILENT
    assert b.decision.silence_justification is not None

def test_scenario_2_moderate_uncertain_is_monitor():
    b = synthetic_bundle("SYN-002")
    assert b.ctx.verification_status == "Conflicting evidence"
    assert b.decision.recommendation == REC_MONITOR

def test_scenario_3_rapid_high_impact_is_response_review():
    b = synthetic_bundle("SYN-003")
    assert b.trend.label == "Rapidly accelerating"
    assert b.decision.recommendation == REC_REVIEW and not b.decision.escalation   # threshold-driven, not safeguard
    assert "human review" in b.decision.human_review_status.lower()

def test_scenario_4_high_historical_reach_declining_does_not_auto_respond():
    b = synthetic_bundle("SYN-004")
    assert b.reach.current_reach > 100_000 and b.trend.label == "Declining"
    assert b.decision.recommendation != REC_REVIEW
    assert b.decision.recommendation == REC_MONITOR          # evaluated on current conditions

def test_scenario_5_credible_public_safety_allegation_escalates_despite_low_circulation():
    b = synthetic_bundle("SYN-005")
    assert b.decision.rps_score < 0.1                         # numerically tiny
    assert b.decision.escalation and b.decision.recommendation == REC_REVIEW
    assert "ESCALATED" in b.decision.human_review_status

def test_scenario_6_insufficient_observations():
    b = synthetic_bundle("SYN-006")
    assert b.decision.recommendation == REC_INSUFFICIENT and not b.validation.sufficient
    assert b.decision.silence_justification is None

def test_scenario_7_low_velocity_rising_reach_is_not_silent():
    b = synthetic_bundle("SYN-007")
    assert b.trend.label == "Low velocity"
    assert b.reach.last_interval_growth_pct >= 25
    assert b.decision.recommendation == REC_MONITOR
    assert any("reach" in x.lower() for x in b.decision.blockers)

def test_scenario_8_rapid_uncertain_high_amplification_low_impact_is_monitor():
    b = synthetic_bundle("SYN-008")
    assert b.trend.label == "Rapidly accelerating" and b.amp.risk == "High"
    assert b.decision.recommendation == REC_MONITOR
    assert b.decision.rps_score >= b.decision.thresholds.response_min     # pressure is at response level, but held

def test_scenario_8_becomes_review_when_impact_is_high():
    b = synthetic_bundle("SYN-008", public_impact="High")
    assert b.decision.recommendation == REC_REVIEW and not b.decision.escalation


# =========================== safeguards ===========================================
def test_high_amplification_never_suppresses_safeguard_escalation():
    b = synthetic_bundle("SYN-005")
    assert b.amp.risk == "High" and b.decision.escalation

def test_verified_false_high_impact_escalates():
    ctx = ClaimContext("T", "d", "Other", VERIFIED_FALSE, "High")
    assert escalation_reasons(ctx, default_config())

def test_verified_false_low_impact_does_not_escalate():
    assert not escalation_reasons(ClaimContext("T", "d", "Other", VERIFIED_FALSE, "Low"), default_config())

def test_escalation_overrides_insufficient_data():
    b = bundle_from([1, 2], public_safety_flag=True)
    assert b.decision.recommendation == REC_REVIEW and b.decision.escalation

def test_escalation_independent_of_score():
    quiet = bundle_from([1, 1, 1, 1, 2], reach=[10, 10, 10, 10, 11], credible_allegation_flag=True)
    assert quiet.decision.escalation and quiet.decision.recommendation == REC_REVIEW


# =========================== insufficient data / unknown reach =====================
def test_insufficient_data_is_not_forced_into_confident_recommendation():
    b = bundle_from([1, 2, 3])
    assert b.decision.recommendation == REC_INSUFFICIENT
    assert "HUMAN VERIFICATION" in b.decision.human_review_status

def test_no_observations_at_all():
    b = bundle_from([])
    assert b.decision.recommendation == REC_INSUFFICIENT

def test_unknown_reach_prevents_silence():
    b = bundle_from([5, 6, 7, 8, 9, 10])               # tiny, slow, but no reach data
    assert not b.reach.available and b.reach.message == "INSUFFICIENT REACH DATA"
    assert b.decision.recommendation == REC_MONITOR
    assert any("Reach data unavailable" in x for x in b.decision.blockers)

def test_verified_true_is_never_silenced_by_default():
    b = synthetic_bundle("SYN-009")
    assert b.decision.recommendation == REC_MONITOR and "inappropriate" in b.decision.response_note

def test_high_public_impact_blocks_silence():
    b = bundle_from([4, 6, 9, 11, 13, 16], reach=[250, 280, 320, 360, 410, 450], public_impact="High")
    assert b.decision.recommendation != REC_SILENT


# =========================== never publishes ======================================
def test_system_never_issues_publication_action():
    for cid in [f"SYN-00{i}" for i in range(1, 10)] + ["SYN-010"]:
        b = synthetic_bundle(cid)
        assert b.decision.recommendation in {REC_SILENT, REC_MONITOR, REC_REVIEW, REC_INSUFFICIENT}
        assert not hasattr(b.decision, "publish")
        if b.decision.recommendation == REC_REVIEW:
            assert "human review" in b.decision.human_review_status.lower()


# =========================== silence justification =================================
def test_silence_justification_has_ten_sections_and_uses_real_numbers():
    b = synthetic_bundle("SYN-001")
    sj = b.decision.silence_justification
    assert len(sj.sections) == 10
    text = sj.as_text()
    assert f"{b.trend.current_velocity:.2f}" in text                       # velocity
    assert f"{b.reach.current_reach:,.0f}" in text                         # reach
    assert f"{b.decision.rps_score:.2f}" in text                           # RPS
    assert "Public intervention is not recommended at this stage" in sj.summary
    assert "Silence is not the absence of action" in text

def test_silence_justification_is_dynamic_not_fixed():
    a, b = synthetic_bundle("SYN-001"), synthetic_bundle("SYN-010")
    assert a.decision.recommendation == b.decision.recommendation == REC_SILENT
    assert a.decision.silence_justification.summary != b.decision.silence_justification.summary
    assert a.decision.silence_justification.as_text() != b.decision.silence_justification.as_text()

def test_silence_justification_wording_changes_with_amplification_level():
    cfg = default_config()
    hi = synthetic_bundle("SYN-001", cfg)
    cfg_low = default_config(); cfg_low.amplification.response_exposure = 10          # makes ratio tiny -> Low
    lo = synthetic_bundle("SYN-001", cfg_low)
    assert hi.amp.risk == "High" and lo.amp.risk == "Low"
    assert "could introduce the claim to additional audiences" in hi.decision.silence_justification.summary
    assert "could introduce the claim to additional audiences" not in lo.decision.silence_justification.summary

def test_silence_justification_lists_monitoring_and_change_conditions():
    t = synthetic_bundle("SYN-001").decision.silence_justification.as_text()
    assert "What continues to be monitored" in t and "Conditions that would change the recommendation" in t
    assert "Reassess at the next observation" in t

def test_no_silence_justification_for_other_recommendations():
    for cid in ("SYN-002", "SYN-003", "SYN-005", "SYN-006"):
        assert synthetic_bundle(cid).decision.silence_justification is None


# =========================== configurability ======================================
def test_thresholds_are_configurable_and_change_decisions():
    cfg = default_config()
    cfg.threshold.silence_max = 0.0001            # nothing is below the ceiling any more
    b = synthetic_bundle("SYN-001", cfg)
    assert b.decision.recommendation == REC_MONITOR

def test_lowering_response_threshold_triggers_review():
    cfg = default_config()
    cfg.threshold.silence_max = 0.1
    cfg.threshold.response_min = 0.2
    assert synthetic_bundle("SYN-002", cfg).decision.recommendation == REC_REVIEW

def test_every_recommendation_carries_reassessment_conditions_and_rationale():
    for cid in [f"SYN-00{i}" for i in range(1, 10)] + ["SYN-010"]:
        d = synthetic_bundle(cid).decision
        assert d.reassessment_conditions and d.rationale and d.monitoring_items
