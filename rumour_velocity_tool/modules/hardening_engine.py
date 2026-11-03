"""
Hardening-risk engine: the COST OF SILENCE.

The amplification engine prices the risk of speaking (a denial carries the
rumour). This module prices the opposite risk: a rumour left alone past a
certain point hardens into accepted fact.

    Hardening index = Rn x Pn
      Rn = min(1, current reach / hardening reach point)
      Pn = mean(min(1, hours in circulation / persistence hours),
                min(1, independent sources / sources point))

Reach is REQUIRED: without a reach estimate the index is "Not assessable"
(it is never inferred). ETAs are straight-line extrapolations of observed
trends, not predictions of real public behaviour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from config import HardeningConfig
from modules.simulation_engine import suggest_growth_pct_per_hour

NOT_ASSESSABLE = "Not assessable"


@dataclass
class HardeningResult:
    assessable: bool
    index: Optional[float]
    level: str
    reach_ratio: Optional[float] = None
    persistence_ratio: Optional[float] = None
    sources_ratio: Optional[float] = None
    hours_in_circulation: float = 0.0
    eta_reach_hours: Optional[float] = None       # hours until reach hits the hardening point
    eta_velocity_hours: Optional[float] = None    # hours until velocity hits the High class
    notes: list = field(default_factory=list)


def classify(index: float, cfg: HardeningConfig) -> str:
    if index >= cfg.high_min:
        return "High"
    if index >= cfg.elevated_min:
        return "Elevated"
    return "Low"


def hours_to_reach_point(current: float, point: float, growth_pct_per_hour: Optional[float]) -> Optional[float]:
    """Compound-growth ETA. 0 if already there; None if growth is absent/non-positive."""
    if current >= point:
        return 0.0
    if growth_pct_per_hour is None or growth_pct_per_hour <= 0 or current <= 0:
        return None
    return math.log(point / current) / math.log(1.0 + growth_pct_per_hour / 100.0)


def hours_to_velocity(v: Optional[float], a: Optional[float], target: float) -> Optional[float]:
    """Linear ETA for velocity to reach `target` given constant acceleration."""
    if v is None:
        return None
    if v >= target:
        return 0.0
    if a is None or a <= 0:
        return None
    return (target - v) / a


def assess_hardening(obs, trend, reach, cfg: Optional[HardeningConfig] = None, first_observed=None,
                     high_velocity: Optional[float] = None) -> HardeningResult:
    cfg = cfg or HardeningConfig()
    notes: list = []
    hours = 0.0
    if obs is not None and len(obs):
        end = obs["timestamp"].iloc[-1]
        start = obs["timestamp"].iloc[0]
        if first_observed is not None:
            import pandas as pd
            fo = pd.to_datetime(first_observed, errors="coerce")
            if pd.notna(fo) and fo <= start:
                start = fo
        hours = max((end - start).total_seconds() / 3600.0, 0.0)

    eta_v = hours_to_velocity(trend.current_velocity, trend.acceleration, high_velocity) if high_velocity else None
    if not reach.available:
        notes.append("Hardening risk cannot be assessed without a reach estimate; none has been inferred.")
        return HardeningResult(False, None, NOT_ASSESSABLE, None, None, None, hours, None, eta_v, notes)

    rn = min(1.0, reach.current_reach / cfg.reach_point) if cfg.reach_point > 0 else 1.0
    pn_p = min(1.0, hours / cfg.persistence_hours) if cfg.persistence_hours > 0 else 1.0
    sn = None
    if reach.independent_sources is not None and cfg.sources_point > 0:
        sn = min(1.0, reach.independent_sources / cfg.sources_point)
    pn = pn_p if sn is None else (pn_p + sn) / 2.0
    if sn is None:
        notes.append("Independent-source count unavailable; persistence alone is used for corroboration.")
    idx = rn * pn

    growth = suggest_growth_pct_per_hour(reach.distribution)
    eta_r = hours_to_reach_point(reach.current_reach, cfg.reach_point, growth)
    if eta_r is not None and eta_r > cfg.eta_cap_hours:
        eta_r = None
        notes.append("At the observed reach growth the hardening reach point is not approached within 30 days.")
    notes.append("ETAs are straight-line extrapolations of observed trends, not predictions of public behaviour.")
    return HardeningResult(True, idx, classify(idx, cfg), rn, pn_p, sn, hours, eta_r, eta_v, notes)
