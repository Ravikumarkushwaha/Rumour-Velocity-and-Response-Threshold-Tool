import pytest

from config import AmplificationConfig
from modules.amplification_engine import (NOT_ASSESSABLE, amplification_ratio, assess_amplification,
                                          classify_ratio, potential_additional_exposure, projected_reach)

A = AmplificationConfig()

def test_additional_exposure_formula():
    assert potential_additional_exposure(50_000, 0.5) == pytest.approx(25_000)

def test_projected_reach_formula():
    assert projected_reach(1_000, 25_000) == 26_000

def test_ratio_formula():
    assert amplification_ratio(25_000, 1_000) == pytest.approx(25.0)

def test_ratio_zero_reach_uses_max_one():
    assert amplification_ratio(500, 0) == pytest.approx(500.0)

def test_negative_inputs_clamped():
    assert potential_additional_exposure(-5, 0.5) == 0 and potential_additional_exposure(100, -1) == 0

def test_classification_boundaries():
    assert classify_ratio(0.24, A) == "Low"
    assert classify_ratio(0.25, A) == "Moderate"
    assert classify_ratio(0.99, A) == "Moderate"
    assert classify_ratio(1.0, A) == "High"

def test_assess_small_reach_is_high_risk():
    r = assess_amplification(520, A)
    assert r.assessable and r.risk == "High"
    assert r.additional_exposure == pytest.approx(25_000)
    assert r.projected == pytest.approx(25_520)

def test_assess_large_reach_is_low_risk():
    assert assess_amplification(400_000, A).risk == "Low"

def test_assess_three_scenarios_present_and_ordered():
    r = assess_amplification(10_000, A)
    lo, me, hi = (r.scenarios[k]["additional"] for k in ("low", "medium", "high"))
    assert lo < me < hi

def test_assess_unavailable_reach_is_not_manufactured():
    r = assess_amplification(None, A)
    assert not r.assessable and r.risk == NOT_ASSESSABLE and r.ratio is None and r.projected is None

def test_assumptions_and_limitations_always_documented():
    r = assess_amplification(1000, A)
    assert r.assumptions and any("not predictions" in l.lower() or "not predict" in l.lower() for l in r.limitations)

def test_classification_scenario_is_configurable():
    cfg = AmplificationConfig(classification_scenario="low")
    assert assess_amplification(50_000, cfg).risk == "Low"          # 10,000/50,000 = 0.20
    assert assess_amplification(50_000, AmplificationConfig(classification_scenario="high")).risk == "High"
