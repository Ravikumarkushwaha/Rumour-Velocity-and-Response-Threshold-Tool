"""
Response Threshold Model.

    RPS = w1*Vn + w2*Gn + w3*An + w4*Rn        (0 = no pressure, 1 = maximum)

Vn, Gn, An, Rn are velocity, growth, acceleration and reach, each normalised
to [0, 1] by dividing by a configurable cap and clipping. Weights sum to 1.

If a component is undefined (e.g. reach unavailable, or growth undefined from
a zero base) it is EXCLUDED and the remaining weights are re-normalised, and
this is reported explicitly. Undefined values are never replaced by guesses.

Context (amplification risk, public impact, evidence status) shifts the
thresholds, not the score, so every adjustment is visible and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import PRESETS, ThresholdConfig

BAND_BELOW = "Below silence ceiling"
BAND_APPROACH = "Approaching response threshold"
BAND_ABOVE = "At or above response threshold"

EXPLANATIONS = {
    "velocity": "Vn: how fast new mentions are arriving (mentions/hour), scaled by the velocity cap.",
    "growth": "Gn: percentage growth over the latest interval, scaled by the growth cap.",
    "acceleration": "An: is the rate itself rising (mentions/hour per hour); only positive acceleration adds pressure.",
    "reach": "Rn: estimated number of people reached, scaled by the reach cap. Excluded if reach is unavailable.",
}


def normalise(value: Optional[float], cap: float) -> Optional[float]:
    """Scale to [0,1]; negative values contribute 0; None stays None."""
    if value is None or cap is None or cap <= 0:
        return None
    return min(max(float(value) / float(cap), 0.0), 1.0)


def weights_sum(cfg: ThresholdConfig) -> float:
    return cfg.w_velocity + cfg.w_growth + cfg.w_accel + cfg.w_reach


def weights_valid(cfg: ThresholdConfig, tol: float = 1e-6) -> bool:
    w = list(cfg.weights.values())
    return all(x >= 0 for x in w) and abs(sum(w) - 1.0) <= tol


def normalise_weights(w: dict) -> dict:
    total = sum(max(v, 0.0) for v in w.values())
    if total <= 0:
        raise ValueError("Weights must contain at least one positive value.")
    return {k: max(v, 0.0) / total for k, v in w.items()}


@dataclass
class RPSInputs:
    velocity: Optional[float]
    growth: Optional[float]
    acceleration: Optional[float]
    reach: Optional[float]


@dataclass
class RPSResult:
    score: Optional[float]
    components: dict = field(default_factory=dict)
    excluded: list = field(default_factory=list)
    note: str = ""


def compute_rps(inputs: RPSInputs, cfg: Optional[ThresholdConfig] = None) -> RPSResult:
    cfg = cfg or ThresholdConfig()
    raw = {"velocity": inputs.velocity, "growth": inputs.growth,
           "acceleration": inputs.acceleration, "reach": inputs.reach}
    caps = {"velocity": cfg.velocity_cap, "growth": cfg.growth_cap,
            "acceleration": cfg.accel_cap, "reach": cfg.reach_cap}
    w = cfg.weights
    norm = {k: normalise(raw[k], caps[k]) for k in raw}
    used = [k for k in raw if norm[k] is not None]
    excluded = [k for k in raw if norm[k] is None]
    if not used:
        return RPSResult(None, {}, excluded, "No component could be computed.")
    wsum = sum(w[k] for k in used)
    if wsum <= 0:
        return RPSResult(None, {}, excluded, "All usable components have zero weight.")
    comps, score = {}, 0.0
    for k in raw:
        if k in used:
            eff = w[k] / wsum
            contrib = eff * norm[k]
            score += contrib
            comps[k] = {"raw": raw[k], "normalised": norm[k], "cap": caps[k], "weight": w[k],
                        "effective_weight": eff, "contribution": contrib, "used": True}
        else:
            comps[k] = {"raw": raw[k], "normalised": None, "cap": caps[k], "weight": w[k],
                        "effective_weight": 0.0, "contribution": 0.0, "used": False}
    note = ""
    if excluded:
        note = ("Component(s) excluded because undefined: " + ", ".join(excluded) +
                ". Remaining weights were re-normalised; nothing was estimated.")
    return RPSResult(score, comps, excluded, note)


@dataclass
class EffectiveThresholds:
    silence_max: float
    response_min: float
    base_silence_max: float
    base_response_min: float
    adjustments: list = field(default_factory=list)  # (reason, d_silence, d_response)


def effective_thresholds(cfg: ThresholdConfig, amplification_risk: str, impact: str,
                         verification: str) -> EffectiveThresholds:
    s, r = cfg.silence_max, cfg.response_min
    adj = []
    d = cfg.amp_adjust_response.get(amplification_risk, 0.0)
    if d:
        adj.append((f"Amplification risk {amplification_risk}: a response carries more exposure risk, "
                    f"so the response bar is raised", 0.0, d))
    ds, dr = cfg.impact_adjust_silence.get(impact, 0.0), cfg.impact_adjust_response.get(impact, 0.0)
    if ds or dr:
        adj.append((f"Public impact {impact}: both thresholds lowered so review and monitoring start earlier", ds, dr))
    de = cfg.evidence_adjust_response.get(verification, 0.0)
    if de:
        adj.append((f"Verification status '{verification}': response bar adjusted", 0.0, de))
    s_eff = s + sum(a[1] for a in adj)
    r_eff = r + sum(a[2] for a in adj)
    s_eff = min(max(s_eff, 0.02), 0.95)
    r_eff = min(max(r_eff, s_eff + cfg.min_gap), 0.99)
    return EffectiveThresholds(s_eff, r_eff, s, r, adj)


def band(score: Optional[float], thr: EffectiveThresholds) -> str:
    if score is None:
        return "Not computed"
    if score < thr.silence_max:
        return BAND_BELOW
    if score < thr.response_min:
        return BAND_APPROACH
    return BAND_ABOVE


# ---------------------------------------------------------------- sensitivity
def _rescaled_weights(w: dict, key: str, new_val: float) -> dict:
    new_val = min(max(new_val, 0.0), 1.0)
    others = {k: v for k, v in w.items() if k != key}
    total_o = sum(others.values())
    out = {key: new_val}
    for k, v in others.items():
        out[k] = (v / total_o) * (1 - new_val) if total_o > 0 else (1 - new_val) / len(others)
    return out


def _score_with_weights(inputs: RPSInputs, cfg: ThresholdConfig, weights: dict) -> Optional[float]:
    c = ThresholdConfig(**{**cfg.__dict__})
    c.w_velocity, c.w_growth = weights["velocity"], weights["growth"]
    c.w_accel, c.w_reach = weights["acceleration"], weights["reach"]
    return compute_rps(inputs, c).score


def weight_sensitivity(inputs: RPSInputs, cfg: ThresholdConfig, thr: EffectiveThresholds,
                       shifts=(-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)) -> pd.DataFrame:
    """One-way sensitivity: shift one weight, rescale the others proportionally."""
    rows = []
    base = cfg.weights
    for key in base:
        for sh in shifts:
            w = _rescaled_weights(base, key, base[key] + sh)
            sc = _score_with_weights(inputs, cfg, w)
            rows.append({"weight_varied": key, "shift": sh, "weight_value": w[key], "rps": sc,
                         "band": band(sc, thr)})
    return pd.DataFrame(rows)


def threshold_sensitivity(score: Optional[float], thr: EffectiveThresholds,
                          shifts=(-0.10, -0.05, 0.0, 0.05, 0.10)) -> pd.DataFrame:
    rows = []
    for sh in shifts:
        s, r = min(max(thr.silence_max + sh, 0.02), 0.95), min(max(thr.response_min + sh, 0.05), 0.99)
        r = max(r, s + 0.05)
        t = EffectiveThresholds(s, r, thr.base_silence_max, thr.base_response_min)
        rows.append({"threshold_shift": sh, "silence_max": s, "response_min": r, "rps": score, "band": band(score, t)})
    return pd.DataFrame(rows)


def compare_presets(inputs: RPSInputs, cfg: ThresholdConfig, amp_risk: str, impact: str,
                    verification: str) -> pd.DataFrame:
    rows = []
    for name, p in PRESETS.items():
        c = ThresholdConfig(**{**cfg.__dict__})
        c.w_velocity, c.w_growth, c.w_accel, c.w_reach = p["weights"]
        c.silence_max, c.response_min = p["silence_max"], p["response_min"]
        res = compute_rps(inputs, c)
        thr = effective_thresholds(c, amp_risk, impact, verification)
        rows.append({"preset": name, "weights (V/G/A/R)": " / ".join(f"{x:.2f}" for x in p["weights"]),
                     "rps": res.score, "silence_max": thr.silence_max, "response_min": thr.response_min,
                     "band": band(res.score, thr)})
    return pd.DataFrame(rows)
