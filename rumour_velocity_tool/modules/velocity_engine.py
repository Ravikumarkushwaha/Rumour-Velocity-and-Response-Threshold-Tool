"""
Rumour Velocity Engine - the central mathematical module.

A. Velocity       V(t) = [M(t) - M(t-1)] / dt            (mentions per hour)
B. Growth         G(t) = [(M(t) - M(t-1)) / M(t-1)] x 100 (percent)
C. Acceleration   A(t) = [V(t) - V(t-1)] / dt            (mentions per hour^2)
D. Moving average rolling mean of V over a configurable window
E. Trend label    configurable thresholds

Scalar functions return None when a quantity is mathematically undefined
(zero denominator, missing value, non-positive dt). Series functions use NaN.
Total mentions are NEVER used as a proxy for rate of spread.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import (TREND_DECLINING, TREND_HIGH, TREND_LOW, TREND_MODERATE,
                    TREND_NA, TREND_RAPID, VelocityConfig)


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def velocity(m_now, m_prev, dt_hours) -> Optional[float]:
    """V(t): additional mentions per hour. None if undefined."""
    if not (_ok(m_now) and _ok(m_prev) and _ok(dt_hours)) or dt_hours <= 0:
        return None
    return (float(m_now) - float(m_prev)) / float(dt_hours)


def percentage_growth(m_now, m_prev) -> Optional[float]:
    """G(t): percent growth. None if the previous count is zero or missing."""
    if not (_ok(m_now) and _ok(m_prev)) or m_prev <= 0:
        return None
    return (float(m_now) - float(m_prev)) / float(m_prev) * 100.0


def acceleration(v_now, v_prev, dt_hours) -> Optional[float]:
    """A(t): change of velocity per hour. None if undefined."""
    if not (_ok(v_now) and _ok(v_prev) and _ok(dt_hours)) or dt_hours <= 0:
        return None
    return (float(v_now) - float(v_prev)) / float(dt_hours)


def moving_average(values, window: int = 3) -> pd.Series:
    """Trailing rolling mean that ignores undefined (NaN) entries."""
    s = pd.Series(values, dtype="float64")
    return s.rolling(max(int(window), 1), min_periods=1).mean()


def compute_series(df: pd.DataFrame, cfg: Optional[VelocityConfig] = None) -> pd.DataFrame:
    """Per-observation velocity, growth, acceleration and moving average.

    Uses the ACTUAL elapsed hours between observations, so irregular intervals
    are handled correctly.
    """
    cfg = cfg or VelocityConfig()
    cols = ["timestamp", "mentions", "dt_hours", "new_mentions", "velocity",
            "growth_pct", "acceleration", "moving_avg_velocity"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)

    t = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    m = df["mention_count"].astype(float).reset_index(drop=True)
    n = len(df)
    dt = [np.nan] + [(t[i] - t[i - 1]).total_seconds() / 3600.0 for i in range(1, n)]
    new = [np.nan] + [m[i] - m[i - 1] for i in range(1, n)]

    vel, gro, acc = [np.nan] * n, [np.nan] * n, [np.nan] * n
    for i in range(1, n):
        v = velocity(m[i], m[i - 1], dt[i])
        g = percentage_growth(m[i], m[i - 1])
        vel[i] = np.nan if v is None else v
        gro[i] = np.nan if g is None else g
    for i in range(2, n):
        a = acceleration(vel[i], vel[i - 1], dt[i])
        acc[i] = np.nan if a is None else a

    out = pd.DataFrame({"timestamp": t, "mentions": m, "dt_hours": dt, "new_mentions": new,
                        "velocity": vel, "growth_pct": gro, "acceleration": acc})
    out["moving_avg_velocity"] = moving_average(out["velocity"], cfg.moving_avg_window).values
    return out


@dataclass
class TrendMetrics:
    n_observations: int
    total_mentions: float
    window_hours: float
    current_velocity: Optional[float]
    previous_velocity: Optional[float]
    growth_pct: Optional[float]
    acceleration: Optional[float]
    moving_avg: Optional[float]
    peak_velocity: Optional[float]
    median_dt_hours: Optional[float]
    label: str


def _clean(x) -> Optional[float]:
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def classify_trend(v, prev_v, accel, ma, cfg: Optional[VelocityConfig] = None) -> str:
    """Classify circulation using configurable thresholds.

    Order matters: a falling rate is labelled Declining first; a high rate that
    is still rising fast is Rapidly accelerating; otherwise the level decides.
    """
    cfg = cfg or VelocityConfig()
    if v is None:
        return TREND_NA
    falling = (accel is not None and accel <= cfg.accel_decline_max) and (ma is None or v < ma)
    if falling:
        return TREND_DECLINING
    if accel is not None and accel >= cfg.accel_rapid_min and v >= cfg.low_max:
        return TREND_RAPID
    if v >= cfg.high_min:
        return TREND_HIGH
    if v >= cfg.low_max:
        return TREND_MODERATE
    return TREND_LOW


def summarise(series: pd.DataFrame, cfg: Optional[VelocityConfig] = None) -> TrendMetrics:
    cfg = cfg or VelocityConfig()
    n = len(series)
    if n == 0:
        return TrendMetrics(0, 0.0, 0.0, None, None, None, None, None, None, None, TREND_NA)
    last = series.iloc[-1]
    prev_v = _clean(series["velocity"].iloc[-2]) if n >= 3 else None
    v, g, a = _clean(last["velocity"]), _clean(last["growth_pct"]), _clean(last["acceleration"])
    ma = _clean(last["moving_avg_velocity"])
    window = (series["timestamp"].iloc[-1] - series["timestamp"].iloc[0]).total_seconds() / 3600.0
    dts = series["dt_hours"].dropna()
    peak = _clean(series["velocity"].max()) if series["velocity"].notna().any() else None
    return TrendMetrics(
        n_observations=n, total_mentions=float(last["mentions"]), window_hours=float(window),
        current_velocity=v, previous_velocity=prev_v, growth_pct=g, acceleration=a, moving_avg=ma,
        peak_velocity=peak, median_dt_hours=_clean(float(dts.median())) if len(dts) else None,
        label=classify_trend(v, prev_v, a, ma, cfg))
