"""Orchestrates the full processing architecture for one claim:
INPUT -> validation -> velocity -> growth/acceleration -> reach -> amplification
-> threshold -> decision -> justification -> (storage / report)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from config import AppConfig
from modules.amplification_engine import AmplificationResult, assess_amplification
from modules.data_processing import ValidationReport, clean_observations
from modules.decision_engine import ClaimContext, DecisionResult, decide
from modules.hardening_engine import HardeningResult, assess_hardening
from modules.lifecycle_engine import LifecycleResult, classify_lifecycle
from modules.reach_engine import ReachAssessment, assess_reach
from modules.threshold_engine import RPSInputs
from modules.velocity_engine import TrendMetrics, compute_series, summarise


@dataclass
class AssessmentBundle:
    ctx: ClaimContext
    first_observed: Optional[str]
    obs: pd.DataFrame
    validation: ValidationReport
    series: pd.DataFrame
    trend: TrendMetrics
    reach: ReachAssessment
    amp: AmplificationResult
    lifecycle: LifecycleResult
    hardening: HardeningResult
    decision: DecisionResult
    cfg: AppConfig
    assessed_at: str

    @property
    def rps_inputs(self) -> RPSInputs:
        return RPSInputs(self.trend.current_velocity, self.trend.growth_pct, self.trend.acceleration,
                         self.reach.current_reach)


def run_assessment(ctx: ClaimContext, obs_raw: pd.DataFrame, cfg: AppConfig,
                   first_observed=None) -> AssessmentBundle:
    obs, rep = clean_observations(obs_raw, cfg.data, first_observed)
    series = compute_series(obs, cfg.velocity)
    trend = summarise(series, cfg.velocity)
    reach = assess_reach(obs, cfg.reach)
    amp = assess_amplification(reach.current_reach, cfg.amplification, reach.confidence)
    lifecycle = classify_lifecycle(series, trend, cfg.lifecycle, cfg.velocity)
    hardening = assess_hardening(obs, trend, reach, cfg.hardening, first_observed, cfg.velocity.high_min)
    decision = decide(ctx, trend, reach, amp, cfg, rep.sufficient, rep.insufficient_reason, hardening)
    return AssessmentBundle(ctx, None if first_observed is None else str(first_observed), obs, rep, series,
                            trend, reach, amp, lifecycle, hardening, decision, cfg,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
