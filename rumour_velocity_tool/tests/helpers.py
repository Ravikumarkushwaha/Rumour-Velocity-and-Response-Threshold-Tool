"""Shared test helpers."""
from datetime import datetime, timedelta

import pandas as pd

from config import SYNTHETIC_CSV, default_config
from modules.decision_engine import ClaimContext
from modules.pipeline import run_assessment


def make_obs(mentions, step_hours=1.0, reach=None, start="2026-11-01 00:00", engagement=None, sources=None):
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M")
    n = len(mentions)
    return pd.DataFrame({
        "claim_id": ["T"] * n,
        "timestamp": [t0 + timedelta(hours=step_hours * i) for i in range(n)],
        "mention_count": mentions,
        "engagement_count": engagement if engagement is not None else [None] * n,
        "estimated_reach": reach if reach is not None else [None] * n,
        "source_type": ["Public forum"] * n,
        "independent_sources": sources if sources is not None else [None] * n,
        "reach_basis": ["estimated"] * n,
    })


def synthetic_bundle(claim_id, cfg=None, **ctx_overrides):
    df = pd.read_csv(SYNTHETIC_CSV)
    g = df[df.claim_id == claim_id]
    r = g.iloc[0]
    kw = dict(claim_id=claim_id, description=r.description, category=r.category,
              verification_status=r.verification_status, public_impact=r.public_impact,
              public_safety_flag=bool(r.public_safety_flag), credible_allegation_flag=bool(r.credible_allegation_flag))
    kw.update(ctx_overrides)
    return run_assessment(ClaimContext(**kw), g, cfg or default_config(), r.first_observed)
