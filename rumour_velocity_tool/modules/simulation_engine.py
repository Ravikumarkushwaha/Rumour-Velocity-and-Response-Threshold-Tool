"""
What-if simulation: REMAIN SILENT versus ISSUE A PUBLIC DENIAL.

HYPOTHETICAL ONLY. Reach compounds at an analyst-chosen growth rate; a denial
adds (response exposure x amplification factor) people at t=0, who then follow
the post-response growth rate. These are scenario illustrations, not forecasts
of real public behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

HYPOTHETICAL_LABEL = ("HYPOTHETICAL SCENARIO - not a prediction of actual public behaviour. "
                      "All values follow from the assumptions entered.")
MAX_HORIZON_HOURS = 24 * 30


@dataclass
class SimulationResult:
    timeline: pd.DataFrame
    summary: pd.DataFrame
    label: str = HYPOTHETICAL_LABEL


def suggest_growth_pct_per_hour(distribution: pd.DataFrame) -> Optional[float]:
    """Observed compound hourly reach growth (percent), or None if not computable."""
    if distribution is None or len(distribution) == 0:
        return None
    r = distribution.dropna(subset=["estimated_reach"])
    if len(r) < 2:
        return None
    r0, r1 = float(r["estimated_reach"].iloc[0]), float(r["estimated_reach"].iloc[-1])
    hours = (r["timestamp"].iloc[-1] - r["timestamp"].iloc[0]).total_seconds() / 3600.0
    if r0 <= 0 or hours <= 0 or r1 <= 0:
        return None
    return float(((r1 / r0) ** (1.0 / hours) - 1.0) * 100.0)


def simulate(current_reach: float, response_exposure: float, factors: dict, horizon_hours: float,
             growth_pct_per_hour_silence: float, growth_pct_per_hour_response: Optional[float] = None,
             step_hours: float = 1.0, audience_ceiling: float = 100_000_000.0,
             hardening_reach: Optional[float] = None) -> SimulationResult:
    horizon = float(min(max(horizon_hours, 1.0), MAX_HORIZON_HOURS))
    step = max(float(step_hours), 0.25)
    g_a = np.clip(growth_pct_per_hour_silence, -50.0, 50.0) / 100.0
    g_b = g_a if growth_pct_per_hour_response is None else np.clip(growth_pct_per_hour_response, -50.0, 50.0) / 100.0
    r0 = max(float(current_reach), 0.0)
    t = np.arange(0.0, horizon + 1e-9, step)

    def grow(base, g):
        return np.minimum(base * np.power(1.0 + g, t), audience_ceiling)

    data = {"hours": t, "Remain silent": grow(r0, g_a)}
    rows = [{"scenario": "A: Remain silent", "amplification factor": 0.0, "additional exposure at t=0": 0.0,
             "reach at end": float(data["Remain silent"][-1]), "difference vs silence": 0.0}]
    for name in ("low", "medium", "high"):
        f = float(factors[name])
        add = max(response_exposure, 0.0) * max(f, 0.0)
        series = grow(r0 + add, g_b)
        label = f"B: Public denial ({name})"
        data[label] = series
        rows.append({"scenario": label, "amplification factor": f, "additional exposure at t=0": add,
                     "reach at end": float(series[-1]),
                     "difference vs silence": float(series[-1] - data["Remain silent"][-1])})
    timeline = pd.DataFrame(data)
    summary = pd.DataFrame(rows)
    if hardening_reach is not None and hardening_reach > 0:
        crossing = []
        for col in data:
            if col == "hours":
                continue
            hit = np.where(np.asarray(data[col]) >= hardening_reach)[0]
            crossing.append(None if len(hit) == 0 else float(t[hit[0]]))
        summary["hours to hardening point"] = crossing
    summary["difference vs silence (%)"] = np.where(
        summary["reach at end"].iloc[0] > 0,
        summary["difference vs silence"] / max(summary["reach at end"].iloc[0], 1.0) * 100.0, 0.0)
    return SimulationResult(timeline, summary)
