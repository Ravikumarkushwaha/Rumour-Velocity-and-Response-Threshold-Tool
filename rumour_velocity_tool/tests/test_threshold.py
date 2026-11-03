import pytest

from config import PRESETS, ThresholdConfig
from modules import threshold_engine as te
from modules.threshold_engine import RPSInputs, compute_rps, effective_thresholds, normalise

T = ThresholdConfig()

def test_default_weights_sum_to_one():
    assert te.weights_valid(T) and te.weights_sum(T) == pytest.approx(1.0)

def test_invalid_weights_detected():
    assert not te.weights_valid(ThresholdConfig(w_velocity=0.5))

def test_normalise_weights_rescales():
    w = te.normalise_weights({"a": 2, "b": 2})
    assert w == {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError):
        te.normalise_weights({"a": 0, "b": 0})

def test_normalise_clips_and_handles_none():
    assert normalise(20, 40) == 0.5
    assert normalise(80, 40) == 1.0
    assert normalise(-5, 40) == 0.0
    assert normalise(None, 40) is None

def test_rps_known_value_all_half():
    r = compute_rps(RPSInputs(20, 50, 2.5, 50_000), T)
    assert r.score == pytest.approx(0.5)

def test_rps_known_weighted_value():
    # V=40 -> 1.0, G=0 -> 0, A=0 -> 0, R=0 -> 0  => RPS = w1
    assert compute_rps(RPSInputs(40, 0, 0, 0), T).score == pytest.approx(T.w_velocity)

def test_rps_maximum_is_one_and_minimum_zero():
    assert compute_rps(RPSInputs(1e6, 1e6, 1e6, 1e9), T).score == pytest.approx(1.0)
    assert compute_rps(RPSInputs(0, 0, 0, 0), T).score == 0.0

def test_rps_negative_acceleration_adds_no_pressure():
    assert compute_rps(RPSInputs(0, 0, -50, 0), T).score == 0.0

def test_rps_missing_reach_excluded_and_renormalised_not_guessed():
    r = compute_rps(RPSInputs(40, 100, 5, None), T)
    assert "reach" in r.excluded and r.components["reach"]["used"] is False
    assert r.score == pytest.approx(1.0)               # remaining components all maxed, weights renormalised
    assert r.note

def test_rps_missing_growth_excluded():
    r = compute_rps(RPSInputs(20, None, 2.5, 50_000), T)
    assert "growth" in r.excluded and r.score == pytest.approx(0.5)

def test_rps_all_missing():
    assert compute_rps(RPSInputs(None, None, None, None), T).score is None

def test_effective_thresholds_amplification_raises_response_bar():
    base = effective_thresholds(T, "Low", "Low", "Unverified")
    high = effective_thresholds(T, "High", "Low", "Unverified")
    assert high.response_min > base.response_min and high.silence_max == base.silence_max

def test_effective_thresholds_high_impact_lowers_both():
    e = effective_thresholds(T, "Low", "High", "Unverified")
    assert e.silence_max < T.silence_max and e.response_min < T.response_min

def test_effective_thresholds_ordering_always_preserved():
    cfg = ThresholdConfig(silence_max=0.5, response_min=0.52)
    e = effective_thresholds(cfg, "Low", "High", "Verified false")
    assert e.response_min >= e.silence_max + cfg.min_gap - 1e-9

def test_bands():
    e = effective_thresholds(T, "Low", "Low", "Unverified")
    assert te.band(0.1, e) == te.BAND_BELOW
    assert te.band(0.45, e) == te.BAND_APPROACH
    assert te.band(0.7, e) == te.BAND_ABOVE
    assert te.band(None, e) == "Not computed"

def test_threshold_boundary_is_inclusive_for_response():
    e = effective_thresholds(T, "Low", "Low", "Unverified")
    assert te.band(e.response_min, e) == te.BAND_ABOVE
    assert te.band(e.silence_max, e) == te.BAND_APPROACH

def test_weight_sensitivity_shape_and_baseline():
    inp = RPSInputs(20, 50, 2.5, 50_000)
    e = effective_thresholds(T, "Low", "Low", "Unverified")
    ws = te.weight_sensitivity(inp, T, e)
    assert set(ws["weight_varied"]) == {"velocity", "growth", "acceleration", "reach"} and len(ws) == 28
    base_rows = ws[ws["shift"] == 0.0]
    assert all(abs(x - compute_rps(inp, T).score) < 1e-9 for x in base_rows["rps"])

def test_weight_sensitivity_rescaled_weights_sum_to_one():
    w = te._rescaled_weights(T.weights, "velocity", 0.6)
    assert sum(w.values()) == pytest.approx(1.0) and w["velocity"] == pytest.approx(0.6)

def test_preset_comparison_orders_thresholds():
    df = te.compare_presets(RPSInputs(20, 50, 2.5, 50_000), T, "Low", "Low", "Unverified")
    assert len(df) == len(PRESETS)
    cons, resp = df.iloc[0], df.iloc[2]
    assert cons["response_min"] > resp["response_min"]

def test_threshold_sensitivity_changes_band():
    e = effective_thresholds(T, "Low", "Low", "Unverified")
    df = te.threshold_sensitivity(0.55, e)
    assert df["band"].nunique() >= 2
