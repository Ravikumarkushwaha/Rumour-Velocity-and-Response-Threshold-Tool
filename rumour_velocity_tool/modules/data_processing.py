"""
Data processing: cleaning, timestamp validation, duplicate detection,
missing-value handling, aggregation and sufficiency checks.

Design rules
------------
* Observations are CUMULATIVE mention counts at a timestamp.
* Nothing is inferred: missing reach stays missing, bad rows are dropped and
  reported, never silently "fixed".
* Duplicate observations (same claim and timestamp) are not independent
  evidence and are collapsed to one row.
* Timestamps are treated as naive UTC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import DataConfig, SOURCE_RELIABILITY_LEVELS

OBS_COLUMNS = [
    "claim_id", "timestamp", "mention_count", "engagement_count",
    "estimated_reach", "source_type", "independent_sources", "reach_basis",
    "source_reference", "source_reliability",
]
REQUIRED_COLUMNS = ["timestamp", "mention_count"]


@dataclass
class ValidationReport:
    rows_in: int = 0
    rows_out: int = 0
    issues: list = field(default_factory=list)      # (severity, message)
    dropped: dict = field(default_factory=dict)
    sufficient: bool = False
    insufficient_reason: Optional[str] = None
    median_interval_hours: Optional[float] = None
    irregular_intervals: bool = False

    def add(self, severity: str, message: str, key: Optional[str] = None, n: int = 0) -> None:
        self.issues.append((severity, message))
        if key:
            self.dropped[key] = self.dropped.get(key, 0) + n

    @property
    def errors(self) -> list:
        return [m for s, m in self.issues if s == "error"]

    @property
    def warnings(self) -> list:
        return [m for s, m in self.issues if s == "warning"]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "claim_id": pd.Series(dtype="object"),
        "timestamp": pd.Series(dtype="datetime64[ns]"),
        "mention_count": pd.Series(dtype="float64"),
        "engagement_count": pd.Series(dtype="float64"),
        "estimated_reach": pd.Series(dtype="float64"),
        "source_type": pd.Series(dtype="object"),
        "independent_sources": pd.Series(dtype="float64"),
        "reach_basis": pd.Series(dtype="object"),
        "source_reference": pd.Series(dtype="object"),
        "source_reliability": pd.Series(dtype="object"),
    })


def clean_observations(df: Optional[pd.DataFrame], data_cfg: Optional[DataConfig] = None,
                       first_observed=None) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean raw observations. Returns (clean_df, report)."""
    cfg = data_cfg or DataConfig()
    rep = ValidationReport()
    if df is None or len(df) == 0:
        rep.add("error", "No observations supplied.")
        rep.insufficient_reason = "No observations supplied."
        return _empty_frame(), rep

    rep.rows_in = len(df)
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in d.columns]
    if missing:
        rep.add("error", f"Required column(s) missing: {', '.join(missing)}.")
        rep.insufficient_reason = "Required columns missing."
        return _empty_frame(), rep

    for c in OBS_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    d = d[OBS_COLUMNS].copy()

    # --- timestamp validation ---------------------------------------------
    ts = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    d["timestamp"] = ts.dt.tz_localize(None)
    n_bad = int(d["timestamp"].isna().sum())
    if n_bad:
        rep.add("warning", f"{n_bad} row(s) dropped: unparseable or missing timestamp.", "bad_timestamp", n_bad)
        d = d[d["timestamp"].notna()]

    if first_observed is not None and len(d):
        fo = pd.to_datetime(first_observed, errors="coerce")
        if pd.notna(fo):
            if getattr(fo, "tzinfo", None) is not None:
                fo = fo.tz_convert("UTC").tz_localize(None)
            early = d["timestamp"] < fo
            if early.any():
                n = int(early.sum())
                rep.add("warning", f"{n} row(s) dropped: timestamp earlier than first-observed time.", "before_first_observed", n)
                d = d[~early]

    # --- numeric coercion --------------------------------------------------
    for c in ["mention_count", "engagement_count", "estimated_reach", "independent_sources"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    n_missing = int(d["mention_count"].isna().sum())
    if n_missing:
        rep.add("warning", f"{n_missing} row(s) dropped: missing or non-numeric mention count (not inferred).",
                "missing_mentions", n_missing)
        d = d[d["mention_count"].notna()]

    neg = d["mention_count"] < 0
    if neg.any():
        n = int(neg.sum())
        rep.add("warning", f"{n} row(s) dropped: negative mention count.", "negative_mentions", n)
        d = d[~neg]

    frac = (d["mention_count"] % 1 != 0)
    if frac.any():
        rep.add("warning", f"{int(frac.sum())} mention count(s) were non-integer and have been rounded.")
        d["mention_count"] = d["mention_count"].round()

    # Optional numeric fields: invalid values become missing (never inferred)
    for c, label in [("engagement_count", "engagement"), ("estimated_reach", "reach"), ("independent_sources", "source count")]:
        bad = d[c] < 0
        if bad.any():
            rep.add("warning", f"{int(bad.sum())} negative {label} value(s) treated as missing.")
            d.loc[bad, c] = np.nan
    n_reach_missing = int(d["estimated_reach"].isna().sum())
    if n_reach_missing and len(d):
        rep.add("info", f"{n_reach_missing} of {len(d)} observation(s) have no reach value. Reach is never inferred.")

    d["reach_basis"] = d["reach_basis"].astype("object").where(d["reach_basis"].notna(), None)
    d["reach_basis"] = d["reach_basis"].map(lambda x: str(x).strip().lower() if x is not None else None)
    bad_basis = d["reach_basis"].notna() & ~d["reach_basis"].isin(["estimated", "observed"])
    if bad_basis.any():
        d.loc[bad_basis, "reach_basis"] = "estimated"
        rep.add("info", "Unrecognised reach_basis values were labelled 'estimated' (the conservative label).")

    d["source_reference"] = d["source_reference"].map(
        lambda value: str(value).strip() if pd.notna(value) and str(value).strip() else None
    )
    d["source_reliability"] = d["source_reliability"].astype("object").where(
        d["source_reliability"].notna(), "Unassessed"
    ).astype(str).str.strip()
    bad_reliability = ~d["source_reliability"].isin(SOURCE_RELIABILITY_LEVELS)
    if bad_reliability.any():
        d.loc[bad_reliability, "source_reliability"] = "Unassessed"
        rep.add("warning", f"{int(bad_reliability.sum())} unrecognised source reliability value(s) set to Unassessed.")

    # --- duplicates --------------------------------------------------------
    d = d.reset_index(drop=True)
    n0 = len(d)
    d = d.drop_duplicates(keep="first")
    n_exact = n0 - len(d)
    if n_exact:
        rep.add("warning", f"{n_exact} exact duplicate row(s) removed (not independent evidence).", "exact_duplicates", n_exact)
    n1 = len(d)
    d = d.sort_values("timestamp", kind="stable").drop_duplicates(subset=["timestamp"], keep="last")
    n_ts = n1 - len(d)
    if n_ts:
        rep.add("warning", f"{n_ts} observation(s) shared a timestamp with another; the latest was kept.",
                "duplicate_timestamp", n_ts)

    # --- cumulative validation ---------------------------------------------
    d = d.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(d):
        running_max = d["mention_count"].cummax()
        bad = d["mention_count"] < running_max
        if bad.any():
            n = int(bad.sum())
            rep.add("warning", f"{n} observation(s) dropped: cumulative mention count decreased "
                               f"(inconsistent with a cumulative series).", "non_cumulative", n)
            d = d[~bad].reset_index(drop=True)

    # --- optional aggregation ----------------------------------------------
    if cfg.aggregation_hours and cfg.aggregation_hours > 0 and len(d) > 1:
        before = len(d)
        d = aggregate_to_interval(d, cfg.aggregation_hours)
        rep.add("info", f"Aggregated {before} observation(s) into {len(d)} interval(s) of {cfg.aggregation_hours:g} h "
                        f"(last observation per interval).")

    rep.rows_out = len(d)

    # --- interval regularity ------------------------------------------------
    if len(d) >= 3:
        dts = d["timestamp"].diff().dt.total_seconds().dropna() / 3600.0
        rep.median_interval_hours = float(dts.median())
        mean = float(dts.mean())
        if mean > 0 and float(dts.std(ddof=0)) / mean > cfg.irregular_cv:
            rep.irregular_intervals = True
            rep.add("info", "Observation intervals are irregular. Velocity uses the actual elapsed hours "
                            "between observations, so results remain valid.")
    elif len(d) == 2:
        rep.median_interval_hours = float((d["timestamp"].iloc[1] - d["timestamp"].iloc[0]).total_seconds() / 3600.0)

    ok, reason = assess_sufficiency(d, cfg)
    rep.sufficient, rep.insufficient_reason = ok, reason
    if not ok:
        rep.add("warning", f"INSUFFICIENT DATA: {reason}")
    return d, rep


def assess_sufficiency(df: pd.DataFrame, cfg: Optional[DataConfig] = None) -> tuple[bool, Optional[str]]:
    cfg = cfg or DataConfig()
    n = 0 if df is None else len(df)
    if n < cfg.min_observations:
        return False, f"{n} valid observation(s); at least {cfg.min_observations} are required."
    span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600.0
    if span < cfg.min_span_hours:
        return False, f"Observation window is {span:.1f} h; at least {cfg.min_span_hours:g} h are required."
    return True, None


def aggregate_to_interval(df: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Collapse observations to one per fixed interval (last cumulative value).

    The retained timestamp is the real timestamp of the last observation in the
    bin, so no artificial observation times are created.
    """
    if df.empty or hours <= 0:
        return df
    d = df.sort_values("timestamp").copy()
    for column in OBS_COLUMNS:
        if column not in d:
            d[column] = "Unassessed" if column == "source_reliability" else np.nan
    t0 = d["timestamp"].iloc[0]
    d["_bin"] = ((d["timestamp"] - t0).dt.total_seconds() // (hours * 3600)).astype(int)
    agg = d.groupby("_bin", sort=True).agg(
        claim_id=("claim_id", "last"), timestamp=("timestamp", "last"),
        mention_count=("mention_count", "max"), engagement_count=("engagement_count", "last"),
        estimated_reach=("estimated_reach", "last"), source_type=("source_type", "last"),
        independent_sources=("independent_sources", "max"), reach_basis=("reach_basis", "last"),
        source_reference=("source_reference", "last"), source_reliability=("source_reliability", "last"),
    ).reset_index(drop=True)
    return agg[OBS_COLUMNS]


def find_similar_descriptions(new_text: str, existing: dict, threshold: float = 0.8) -> list:
    """Return [(claim_id, similarity)] for existing descriptions similar to new_text.

    Uses TF-IDF cosine similarity (scikit-learn). Used to warn that a new claim
    may be a repost/variant of one already tracked, so it is not counted twice.
    """
    if not new_text or not existing:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        ids = list(existing.keys())
        corpus = [new_text] + [existing[i] for i in ids]
        mat = TfidfVectorizer(stop_words="english").fit_transform(corpus)
        sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
        out = [(i, float(s)) for i, s in zip(ids, sims) if s >= threshold]
        return sorted(out, key=lambda x: -x[1])
    except Exception:
        return []
