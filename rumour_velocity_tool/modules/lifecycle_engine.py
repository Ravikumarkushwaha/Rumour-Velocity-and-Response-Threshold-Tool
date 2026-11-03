"""
Rumour Lifecycle Analysis: Emergence, Growth, Acceleration, Stabilisation, Decline.

The stage is inferred ONLY from the observed time series. It describes how
circulation is behaving; it says nothing about whether the claim is true.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import LifecycleConfig, VelocityConfig
from modules.velocity_engine import TrendMetrics

DISCLAIMER = "Lifecycle stage describes circulation behaviour only. It does not establish whether a claim is true or false."


@dataclass
class LifecycleResult:
    stage: str
    rationale: str
    measurements: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER


def classify_lifecycle(series: pd.DataFrame, trend: TrendMetrics,
                       lcfg: Optional[LifecycleConfig] = None,
                       vcfg: Optional[VelocityConfig] = None) -> LifecycleResult:
    lcfg, vcfg = lcfg or LifecycleConfig(), vcfg or VelocityConfig()
    v, a, ma, peak = trend.current_velocity, trend.acceleration, trend.moving_avg, trend.peak_velocity
    meas = {"cumulative mentions": trend.total_mentions, "current velocity": v, "acceleration": a,
            "moving average": ma, "peak velocity": peak, "observations": trend.n_observations}
    if v is None or trend.n_observations < 3:
        return LifecycleResult("Insufficient data", "At least three observations are needed to place a claim in a lifecycle stage.", meas)

    if peak and peak > 0 and v <= lcfg.decline_peak_ratio * peak and trend.n_observations >= 4:
        return LifecycleResult("Decline", f"Current velocity ({v:.2f}/h) is at or below {lcfg.decline_peak_ratio:.0%} of the "
                               f"peak observed velocity ({peak:.2f}/h).", meas)
    if a is not None and a <= vcfg.accel_decline_max and ma is not None and v < ma:
        return LifecycleResult("Decline", f"Negative acceleration ({a:.2f}/h²) with velocity below its moving average.", meas)
    if a is not None and a >= vcfg.accel_rapid_min and v >= vcfg.low_max:
        return LifecycleResult("Acceleration", f"Velocity ({v:.2f}/h) is rising quickly (acceleration {a:+.2f}/h²).", meas)
    if trend.total_mentions < lcfg.emergence_total_max:
        return LifecycleResult("Emergence", f"Only {trend.total_mentions:.0f} cumulative mentions so far "
                               f"(below {lcfg.emergence_total_max}); circulation is at an early stage.", meas)
    flat_level = ma is not None and ma > 0 and abs(v - ma) / ma <= lcfg.stable_band
    flat_accel = a is not None and abs(a) < lcfg.stable_accel_abs
    if flat_level and flat_accel:
        return LifecycleResult("Stabilisation", f"Velocity ({v:.2f}/h) is within {lcfg.stable_band:.0%} of its moving "
                               f"average ({ma:.2f}/h) with near-zero acceleration ({a:+.2f}/h²).", meas)
    return LifecycleResult("Growth", f"Mentions are still increasing (velocity {v:.2f}/h) without the sharp rise in rate "
                           f"that defines acceleration.", meas)
