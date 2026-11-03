"""
Amplification Risk Engine (hypothetical scenario model).

    Potential Additional Exposure = Assumed Response Exposure x Amplification Factor
    Projected Reach               = Current Reach + Potential Additional Exposure
    Amplification Ratio           = Potential Additional Exposure / max(Current Reach, 1)

Outputs are SCENARIO ESTIMATES, not predictions. The model has no access to
platform algorithms, private audiences or classified information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import AmplificationConfig

NOT_ASSESSABLE = "Not assessable"


def potential_additional_exposure(response_exposure: float, factor: float) -> float:
    return max(float(response_exposure), 0.0) * max(float(factor), 0.0)


def projected_reach(current_reach: float, additional_exposure: float) -> float:
    return float(current_reach) + float(additional_exposure)


def amplification_ratio(additional_exposure: float, current_reach: float) -> float:
    return float(additional_exposure) / max(float(current_reach), 1.0)


def classify_ratio(ratio: float, cfg: Optional[AmplificationConfig] = None) -> str:
    cfg = cfg or AmplificationConfig()
    if ratio >= cfg.ratio_high_min:
        return "High"
    if ratio >= cfg.ratio_moderate_min:
        return "Moderate"
    return "Low"


@dataclass
class AmplificationResult:
    assessable: bool
    current_reach: Optional[float]
    response_exposure: float
    scenarios: dict = field(default_factory=dict)
    classification_scenario: str = "medium"
    risk: str = NOT_ASSESSABLE
    ratio: Optional[float] = None
    additional_exposure: Optional[float] = None
    projected: Optional[float] = None
    assumptions: list = field(default_factory=list)
    confidence: str = "None"
    limitations: list = field(default_factory=list)


LIMITATIONS = [
    "Scenario estimates only; they are not predictions of real public behaviour.",
    "The amplification factors are analyst assumptions, not measured values.",
    "No access to platform recommendation algorithms, private audiences or classified information is assumed.",
    "Current reach is an analyst estimate and may be wrong in either direction.",
    "Audience overlap, timing, platform and media pick-up are not modelled.",
]


def assess_amplification(current_reach: Optional[float], cfg: Optional[AmplificationConfig] = None,
                         reach_confidence: str = "None") -> AmplificationResult:
    cfg = cfg or AmplificationConfig()
    assumptions = [
        f"Assumed audience an official response could reach: {cfg.response_exposure:,.0f} people.",
        f"Amplification factors (share of that audience newly exposed to the claim): "
        f"low {cfg.factor_low:.2f}, medium {cfg.factor_medium:.2f}, high {cfg.factor_high:.2f}.",
        f"Risk classification uses the '{cfg.classification_scenario}' scenario: "
        f"Low if ratio < {cfg.ratio_moderate_min:g}, Moderate if < {cfg.ratio_high_min:g}, otherwise High.",
    ]
    if current_reach is None:
        return AmplificationResult(False, None, cfg.response_exposure, {}, cfg.classification_scenario,
                                   NOT_ASSESSABLE, None, None, None,
                                   assumptions + ["Current reach is unavailable, so amplification cannot be assessed."],
                                   "None", LIMITATIONS)
    factors = {"low": cfg.factor_low, "medium": cfg.factor_medium, "high": cfg.factor_high}
    scenarios = {}
    for name, f in factors.items():
        add = potential_additional_exposure(cfg.response_exposure, f)
        ratio = amplification_ratio(add, current_reach)
        scenarios[name] = {"factor": f, "additional": add, "projected": projected_reach(current_reach, add),
                           "ratio": ratio, "risk": classify_ratio(ratio, cfg)}
    sel = scenarios.get(cfg.classification_scenario, scenarios["medium"])
    return AmplificationResult(True, float(current_reach), cfg.response_exposure, scenarios,
                               cfg.classification_scenario, sel["risk"], sel["ratio"], sel["additional"],
                               sel["projected"], assumptions, reach_confidence, LIMITATIONS)
