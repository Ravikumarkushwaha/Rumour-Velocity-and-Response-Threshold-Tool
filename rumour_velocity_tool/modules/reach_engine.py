"""
Public Reach Assessment.

Principles
----------
* Reach is read from analyst-supplied values only. It is NEVER inferred or
  manufactured. If no usable value exists the result is "INSUFFICIENT REACH DATA".
* Engagement (likes, shares, comments) is NOT unique audience and is shown
  separately and labelled as such.
* "Independent sources" are counted from an explicit column if supplied,
  otherwise from the number of distinct source CATEGORIES (a weaker proxy,
  and labelled as such).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import ReachConfig

INSUFFICIENT_REACH = "INSUFFICIENT REACH DATA"


@dataclass
class ReachAssessment:
    available: bool
    current_reach: Optional[float]
    basis: str                              # estimated / observed / mixed / none
    reach_points: int
    reach_growth_pct_total: Optional[float]
    last_interval_growth_pct: Optional[float]
    independent_sources: Optional[int]
    sources_basis: str
    total_engagement: Optional[float]
    confidence: str                         # None / Low / Moderate / High
    notes: list = field(default_factory=list)
    message: str = ""
    distribution: pd.DataFrame = field(default_factory=pd.DataFrame)


def assess_reach(obs: pd.DataFrame, cfg: Optional[ReachConfig] = None) -> ReachAssessment:
    cfg = cfg or ReachConfig()
    if obs is None or len(obs) == 0:
        return ReachAssessment(False, None, "none", 0, None, None, None, "none", None, "None",
                               ["No observations."], INSUFFICIENT_REACH, pd.DataFrame())

    dist = obs[["timestamp", "mention_count", "engagement_count", "estimated_reach",
                "independent_sources", "reach_basis"]].copy()
    r = obs.dropna(subset=["estimated_reach"]).copy()
    notes: list = []

    # engagement (not audience)
    eng = obs["engagement_count"].dropna()
    total_eng = float(eng.iloc[-1]) if len(eng) else None
    if total_eng is not None:
        notes.append("Engagement is interaction volume, not unique audience; it is not used as reach.")

    # independent sources
    src_series = obs["independent_sources"].dropna()
    if len(src_series):
        n_src, src_basis = int(src_series.max()), "explicit independent-source count"
    else:
        cats = obs["source_type"].dropna().nunique()
        n_src, src_basis = (int(cats), "distinct source categories (weak proxy)") if cats else (None, "none")

    if len(r) == 0:
        notes.append("No reach value has been supplied for this claim; none has been inferred.")
        return ReachAssessment(False, None, "none", 0, None, None, n_src, src_basis, total_eng,
                               "None", notes, INSUFFICIENT_REACH, dist)

    current = float(r["estimated_reach"].iloc[-1])
    first = float(r["estimated_reach"].iloc[0])
    total_growth = ((current - first) / first * 100.0) if (len(r) >= cfg.min_reach_points and first > 0) else None
    last_growth = None
    if len(r) >= 2 and float(r["estimated_reach"].iloc[-2]) > 0:
        last_growth = (current - float(r["estimated_reach"].iloc[-2])) / float(r["estimated_reach"].iloc[-2]) * 100.0

    if (r["estimated_reach"].diff().dropna() < 0).any():
        notes.append("Reach readings are not monotonic; cumulative reach should not fall. Review the source estimates.")

    bases = set(r["reach_basis"].dropna().unique())
    if not bases:
        basis = "estimated"
    elif bases == {"observed"}:
        basis = "observed"
    elif bases == {"estimated"}:
        basis = "estimated"
    else:
        basis = "mixed"
    if basis != "observed":
        notes.append("Reach is an analyst ESTIMATE, not a measured audience figure.")

    score = 0
    score += 1 if len(r) >= 3 else 0
    score += 1 if len(r) >= 5 else 0
    score += 1 if (n_src or 0) >= cfg.min_independent_sources else 0
    score += 1 if basis == "observed" else 0
    confidence = {0: "Low", 1: "Low", 2: "Moderate", 3: "Moderate", 4: "High"}[score]

    return ReachAssessment(True, current, basis, int(len(r)), total_growth, last_growth, n_src,
                           src_basis, total_eng, confidence, notes, "", dist)
