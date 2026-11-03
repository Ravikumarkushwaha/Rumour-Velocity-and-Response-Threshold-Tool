"""SQLite persistence layer with validation and referential integrity."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (ALL_RECOMMENDATIONS, CATEGORIES, DB_PATH, IMPACT_LEVELS, REACH_BASES, SCHEMA_PATH,
                    SOURCE_RELIABILITY_LEVELS, VERIFICATION_STATUSES)


def _none(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x


def _ts(x) -> str:
    t = pd.to_datetime(x, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"Invalid timestamp: {x!r}")
    if getattr(t, "tzinfo", None) is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path: Optional[Path | str] = None):
        self.path = str(path or DB_PATH)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    # ------------------------------------------------------------- plumbing
    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def init_schema(self) -> None:
        with self.conn() as c:
            c.executescript(Path(SCHEMA_PATH).read_text(encoding="utf-8"))
            # lightweight migration for databases created before the hardening columns existed
            cols = {r["name"] for r in c.execute("PRAGMA table_info(assessments)")}
            for name, typ in (("hardening_index", "REAL"), ("hardening_level", "TEXT")):
                if name not in cols:
                    c.execute(f"ALTER TABLE assessments ADD COLUMN {name} {typ}")
            claim_cols = {r["name"] for r in c.execute("PRAGMA table_info(claims)")}
            if "review_due_at" not in claim_cols:
                c.execute("ALTER TABLE claims ADD COLUMN review_due_at TEXT")
            observation_cols = {r["name"] for r in c.execute("PRAGMA table_info(observations)")}
            for name, typ in (("source_reference", "TEXT"),
                              ("source_reliability", "TEXT NOT NULL DEFAULT 'Unassessed'")):
                if name not in observation_cols:
                    c.execute(f"ALTER TABLE observations ADD COLUMN {name} {typ}")

    @staticmethod
    def new_claim_id() -> str:
        return f"CLM-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:4].upper()}"

    # --------------------------------------------------------------- claims
    def add_claim(self, description: str, category: str, first_observed, verification_status: str,
                  public_impact: str = "Low", public_safety_flag: bool = False,
                  credible_allegation_flag: bool = False, analyst_remarks: str = "",
                  claim_id: Optional[str] = None, is_synthetic: bool = False) -> str:
        if not description or not str(description).strip():
            raise ValueError("Claim description is required.")
        if category not in CATEGORIES:
            raise ValueError(f"Invalid category: {category!r}")
        if verification_status not in VERIFICATION_STATUSES:
            raise ValueError(f"Invalid verification status: {verification_status!r}")
        if public_impact not in IMPACT_LEVELS:
            raise ValueError(f"Invalid public impact: {public_impact!r}")
        cid = claim_id or self.new_claim_id()
        with self.conn() as c:
            try:
                c.execute("""INSERT INTO claims (claim_id, description, category, first_observed, verification_status,
                             created_at, public_impact, public_safety_flag, credible_allegation_flag, analyst_remarks,
                             is_synthetic) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                          (cid, description.strip(), category, _ts(first_observed), verification_status,
                           datetime.now().strftime("%Y-%m-%d %H:%M:%S"), public_impact, int(bool(public_safety_flag)),
                           int(bool(credible_allegation_flag)), analyst_remarks or "", int(bool(is_synthetic))))
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Could not store claim: {e}") from e
            self._write_audit(c, cid, "claim_created", "claim", cid,
                              ["description", "category", "first_observed", "verification_status",
                               "public_impact", "analyst_remarks"], "Claim created.")
        return cid

    def update_claim(self, claim_id: str, **fields) -> None:
        allowed = {"verification_status", "public_impact", "public_safety_flag", "credible_allegation_flag",
                   "analyst_remarks", "description", "category"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Fields cannot be updated: {sorted(bad)}")
        if not fields:
            return
        if "verification_status" in fields and fields["verification_status"] not in VERIFICATION_STATUSES:
            raise ValueError("Invalid verification status.")
        if "public_impact" in fields and fields["public_impact"] not in IMPACT_LEVELS:
            raise ValueError("Invalid public impact.")
        if "category" in fields and fields["category"] not in CATEGORIES:
            raise ValueError("Invalid category.")
        for k in ("public_safety_flag", "credible_allegation_flag"):
            if k in fields:
                fields[k] = int(bool(fields[k]))
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.conn() as c:
            current = c.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
            if current is None:
                raise ValueError(f"Unknown claim: {claim_id}")
            changed = [key for key, value in fields.items() if current[key] != value]
            if not changed:
                return
            cur = c.execute(f"UPDATE claims SET {sets} WHERE claim_id = ?", (*fields.values(), claim_id))
            if cur.rowcount == 0:
                raise ValueError(f"Unknown claim: {claim_id}")
            self._write_audit(c, claim_id, "claim_updated", "claim", claim_id, changed, "Claim details updated.")

    @staticmethod
    def _write_audit(c, claim_id: str, event_type: str, record_type: str, record_id,
                     field_names: list[str], detail: str) -> None:
        c.execute("""INSERT INTO audit_events
            (claim_id, event_type, record_type, record_id, field_names, detail, occurred_at)
            VALUES (?,?,?,?,?,?,?)""",
            (claim_id, event_type, record_type, None if record_id is None else str(record_id),
             json.dumps(field_names), detail, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def delete_claim(self, claim_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM claims WHERE claim_id = ?", (claim_id,))

    def get_claims(self) -> pd.DataFrame:
        with self.conn() as c:
            return pd.read_sql_query("SELECT * FROM claims ORDER BY created_at, claim_id", c)

    def get_claim(self, claim_id: str) -> Optional[dict]:
        with self.conn() as c:
            r = c.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        return dict(r) if r else None

    # --------------------------------------------------------- observations
    def add_observation(self, claim_id: str, timestamp, mention_count, engagement_count=None,
                        estimated_reach=None, source_type: Optional[str] = None,
                        independent_sources=None, reach_basis: Optional[str] = None,
                        source_reference: Optional[str] = None,
                        source_reliability: str = "Unassessed") -> bool:
        """Insert one observation. Returns False if it duplicates (claim, timestamp)."""
        mc = _none(mention_count)
        if mc is None or float(mc) < 0 or float(mc) != int(float(mc)):
            raise ValueError("Mention count must be a non-negative whole number.")
        for name, val in (("engagement", engagement_count), ("reach", estimated_reach), ("sources", independent_sources)):
            v = _none(val)
            if v is not None and float(v) < 0:
                raise ValueError(f"{name} cannot be negative.")
        rb = _none(reach_basis)
        if rb is not None and str(rb).lower() not in REACH_BASES:
            raise ValueError("reach_basis must be 'estimated' or 'observed'.")
        source_reliability = _none(source_reliability) or "Unassessed"
        if source_reliability not in SOURCE_RELIABILITY_LEVELS:
            raise ValueError(f"source_reliability must be one of {SOURCE_RELIABILITY_LEVELS}.")
        eng, rch, src = _none(engagement_count), _none(estimated_reach), _none(independent_sources)
        with self.conn() as c:
            try:
                cur = c.execute("""INSERT OR IGNORE INTO observations (claim_id, timestamp, mention_count,
                    engagement_count, estimated_reach, source_type, independent_sources, reach_basis,
                    source_reference, source_reliability)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (claim_id, _ts(timestamp), int(float(mc)), None if eng is None else int(float(eng)),
                     None if rch is None else float(rch), _none(source_type),
                     None if src is None else int(float(src)), None if rb is None else str(rb).lower(),
                     _none(source_reference), source_reliability))
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Could not store observation: {e}") from e
            if cur.rowcount > 0:
                self._write_audit(c, claim_id, "observation_added", "observation", cur.lastrowid,
                                  ["timestamp", "mention_count", "engagement_count", "estimated_reach",
                                   "source_type", "independent_sources", "reach_basis", "source_reference",
                                   "source_reliability"], "Observation added.")
            return cur.rowcount > 0

    def add_observations_df(self, claim_id: str, df: pd.DataFrame) -> dict:
        """Insert rows from a (cleaned) DataFrame. Returns counts."""
        added = skipped = 0
        for _, r in df.iterrows():
            ok = self.add_observation(claim_id, r["timestamp"], r["mention_count"], r.get("engagement_count"),
                                      r.get("estimated_reach"), r.get("source_type"),
                                      r.get("independent_sources"), r.get("reach_basis"),
                                      r.get("source_reference"), r.get("source_reliability", "Unassessed"))
            added += int(ok)
            skipped += int(not ok)
        return {"added": added, "skipped_duplicates": skipped}

    def get_observations(self, claim_id: str) -> pd.DataFrame:
        with self.conn() as c:
            df = pd.read_sql_query("""SELECT claim_id, timestamp, mention_count, engagement_count, estimated_reach,
                source_type, independent_sources, reach_basis, source_reference, source_reliability
                FROM observations WHERE claim_id = ?
                ORDER BY timestamp""", c, params=(claim_id,))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def set_review_due(self, claim_id: str, review_due_at) -> None:
        due = None if review_due_at is None else _ts(review_due_at)
        with self.conn() as c:
            cur = c.execute("UPDATE claims SET review_due_at = ? WHERE claim_id = ?", (due, claim_id))
            if cur.rowcount == 0:
                raise ValueError(f"Unknown claim: {claim_id}")
            self._write_audit(c, claim_id, "review_schedule_updated", "claim", claim_id,
                              ["review_due_at"], "Next review schedule updated.")

    def get_audit_events(self, claim_id: Optional[str] = None) -> pd.DataFrame:
        q, p = "SELECT * FROM audit_events", ()
        if claim_id:
            q, p = q + " WHERE claim_id = ?", (claim_id,)
        with self.conn() as c:
            return pd.read_sql_query(q + " ORDER BY occurred_at DESC, audit_id DESC", c, params=p)

    # ---------------------------------------------------------- assessments
    def get_assessments(self, claim_id: Optional[str] = None) -> pd.DataFrame:
        q, p = "SELECT * FROM assessments", ()
        if claim_id:
            q, p = q + " WHERE claim_id = ?", (claim_id,)
        with self.conn() as c:
            return pd.read_sql_query(q + " ORDER BY assessed_at, assessment_id", c, params=p)

    def latest_assessment(self, claim_id: str) -> Optional[dict]:
        with self.conn() as c:
            r = c.execute("SELECT * FROM assessments WHERE claim_id = ? ORDER BY assessment_id DESC LIMIT 1",
                          (claim_id,)).fetchone()
        return dict(r) if r else None

    @staticmethod
    def _reason_for_change(prev: Optional[dict], new: dict) -> str:
        if prev is None:
            return "Initial assessment."
        bits = []
        def delta(label, a, b, fmt="{:.2f}"):
            if a is not None and b is not None and not (pd.isna(a) or pd.isna(b)):
                bits.append(f"{label} {fmt.format(a)} -> {fmt.format(b)}")
        delta("velocity", prev["velocity"], new["velocity"])
        delta("RPS", prev["response_pressure_score"], new["response_pressure_score"])
        delta("reach", prev["reach"], new["reach"], "{:,.0f}")
        if prev["amplification_risk"] != new["amplification_risk"]:
            bits.append(f"amplification {prev['amplification_risk']} -> {new['amplification_risk']}")
        if prev["recommendation"] == new["recommendation"]:
            return "Recommendation unchanged. " + "; ".join(bits)
        return "Recommendation changed. " + "; ".join(bits)

    def save_assessment(self, bundle) -> int:
        """Persist an AssessmentBundle (duck-typed) and log recommendation changes."""
        d, t, r, a = bundle.decision, bundle.trend, bundle.reach, bundle.amp
        rec = d.recommendation
        if rec not in ALL_RECOMMENDATIONS:
            raise ValueError(f"Invalid recommendation {rec!r}")
        justification = d.silence_justification.as_text() if d.silence_justification else "\n".join(d.rationale)
        row = {
            "velocity": _none(t.current_velocity), "growth": _none(t.growth_pct), "acceleration": _none(t.acceleration),
            "reach": _none(r.current_reach), "amplification_risk": a.risk,
            "response_pressure_score": _none(d.rps_score), "recommendation": rec,
        }
        prev = self.latest_assessment(bundle.ctx.claim_id)
        with self.conn() as c:
            cur = c.execute("""INSERT INTO assessments (claim_id, velocity, growth, acceleration, reach, amplification_risk,
                response_pressure_score, recommendation, justification, assessed_at, trend_label, lifecycle_stage,
                escalation, human_review_status, observations_used, config_snapshot, hardening_index, hardening_level)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (bundle.ctx.claim_id, row["velocity"], row["growth"], row["acceleration"], row["reach"],
                 row["amplification_risk"], row["response_pressure_score"], rec, justification,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), t.label, bundle.lifecycle.stage,
                 int(d.escalation), d.human_review_status, int(len(bundle.obs)),
                 json.dumps(bundle.cfg.to_dict(), default=str), _none(bundle.hardening.index), bundle.hardening.level))
            aid = cur.lastrowid
            if prev is None or prev["recommendation"] != rec:
                c.execute("""INSERT INTO decision_history (claim_id, previous_recommendation, new_recommendation,
                    reason_for_change, decision_timestamp) VALUES (?,?,?,?,?)""",
                    (bundle.ctx.claim_id, None if prev is None else prev["recommendation"], rec,
                     self._reason_for_change(prev, {**row}), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self._write_audit(c, bundle.ctx.claim_id, "assessment_saved", "assessment", aid,
                              ["recommendation", "response_pressure_score", "velocity", "reach"],
                              f"Assessment saved: {rec}.")
        return aid

    def get_decision_history(self, claim_id: Optional[str] = None) -> pd.DataFrame:
        q, p = "SELECT * FROM decision_history", ()
        if claim_id:
            q, p = q + " WHERE claim_id = ?", (claim_id,)
        with self.conn() as c:
            return pd.read_sql_query(q + " ORDER BY decision_timestamp, decision_id", c, params=p)

    # --------------------------------------------------------------- import
    def import_long_csv(self, df: pd.DataFrame, is_synthetic: bool = False) -> dict:
        """Import a multi-claim CSV (claim + observation columns, one row per observation)."""
        d = df.copy()
        d.columns = [str(c).strip().lower() for c in d.columns]
        need = {"claim_id", "description", "category", "verification_status", "timestamp", "mention_count"}
        miss = need - set(d.columns)
        if miss:
            raise ValueError(f"CSV missing required column(s): {sorted(miss)}")
        existing = set(self.get_claims()["claim_id"])
        n_claims = n_obs = n_skipped = 0
        for cid, g in d.groupby("claim_id", sort=False):
            r = g.iloc[0]
            if cid not in existing:
                self.add_claim(
                    str(r["description"]), r["category"], r.get("first_observed", g["timestamp"].min()),
                    r["verification_status"], str(_none(r.get("public_impact")) or "Low"),
                    bool(_none(r.get("public_safety_flag")) or 0), bool(_none(r.get("credible_allegation_flag")) or 0),
                    str(_none(r.get("analyst_remarks")) or ""), claim_id=str(cid), is_synthetic=is_synthetic)
                n_claims += 1
            res = self.add_observations_df(str(cid), g)
            n_obs += res["added"]
            n_skipped += res["skipped_duplicates"]
        return {"claims_added": n_claims, "observations_added": n_obs, "duplicates_skipped": n_skipped}

    def seed_synthetic_if_empty(self, csv_path: Path | str) -> bool:
        if len(self.get_claims()) > 0 or not Path(csv_path).exists():
            return False
        self.import_long_csv(pd.read_csv(csv_path), is_synthetic=True)
        return True

    # ------------------------------------------------------------ integrity
    def integrity_report(self) -> dict:
        with self.conn() as c:
            fk = c.execute("PRAGMA foreign_key_check").fetchall()
            orphan_obs = c.execute("SELECT COUNT(*) FROM observations o LEFT JOIN claims c ON c.claim_id=o.claim_id "
                                   "WHERE c.claim_id IS NULL").fetchone()[0]
            counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("claims", "observations", "assessments", "decision_history", "audit_events")}
        return {"foreign_key_violations": len(fk), "orphan_observations": orphan_obs, "row_counts": counts}
