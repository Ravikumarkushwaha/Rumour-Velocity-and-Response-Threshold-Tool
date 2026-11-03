import os
import sqlite3
import tempfile

import pandas as pd
import pytest

from config import SYNTHETIC_CSV, default_config
from database.db_manager import Database
from modules.data_processing import find_similar_descriptions
from modules.lifecycle_engine import classify_lifecycle
from modules.reach_engine import INSUFFICIENT_REACH, assess_reach
from modules.simulation_engine import HYPOTHETICAL_LABEL, simulate, suggest_growth_pct_per_hour
from modules.velocity_engine import compute_series, summarise
from reports.report_generator import assessment_summary_df, build_pdf_report
from tests.helpers import make_obs, synthetic_bundle


# ----------------------------------------------------------------------- reach
def test_reach_unavailable_message_and_no_fabrication():
    r = assess_reach(make_obs([1, 2, 3, 4]))
    assert not r.available and r.current_reach is None and r.message == INSUFFICIENT_REACH

def test_reach_uses_latest_supplied_value_and_growth():
    r = assess_reach(make_obs([1, 2, 3, 4], reach=[100, 150, None, 300]))
    assert r.current_reach == 300 and r.reach_growth_pct_total == pytest.approx(200.0)
    assert r.last_interval_growth_pct == pytest.approx(100.0)

def test_engagement_is_not_treated_as_reach():
    r = assess_reach(make_obs([1, 2, 3], engagement=[10, 20, 9999]))
    assert not r.available and r.total_engagement == 9999

def test_reach_confidence_levels():
    low = assess_reach(make_obs([1, 2], reach=[100, 200]))
    assert low.confidence == "Low"
    hi = assess_reach(synthetic_bundle("SYN-009").obs)
    assert hi.basis == "observed" and hi.confidence in ("Moderate", "High")

def test_reach_non_monotonic_noted():
    r = assess_reach(make_obs([1, 2, 3], reach=[500, 300, 400]))
    assert any("monotonic" in n for n in r.notes)


# ----------------------------------------------------------------------- lifecycle
def _stage(mentions, step=1.0):
    s = compute_series(make_obs(mentions, step))
    return classify_lifecycle(s, summarise(s)).stage

def test_lifecycle_stages():
    assert _stage([2, 3, 4, 5, 6]) == "Emergence"
    assert _stage([100, 110, 122, 135, 149]) == "Growth"
    assert _stage([100, 120, 160, 260, 460]) == "Acceleration"
    assert _stage([1000, 1100, 1200, 1300, 1400, 1500]) == "Stabilisation"
    assert _stage([1000, 2000, 3000, 3300, 3400, 3450]) == "Decline"

def test_lifecycle_insufficient_and_disclaimer():
    s = compute_series(make_obs([1, 2]))
    r = classify_lifecycle(s, summarise(s))
    assert r.stage == "Insufficient data" and "does not establish" in r.disclaimer

def test_lifecycle_synthetic_claims():
    assert synthetic_bundle("SYN-003").lifecycle.stage == "Acceleration"
    assert synthetic_bundle("SYN-004").lifecycle.stage == "Decline"


# ----------------------------------------------------------------------- simulation
FACT = {"low": 0.2, "medium": 0.5, "high": 1.0}

def test_simulation_silence_vs_denial_exposure():
    sim = simulate(1000, 50_000, FACT, 24, 0.0, 0.0)
    assert sim.timeline["Remain silent"].iloc[-1] == pytest.approx(1000)
    assert sim.timeline["B: Public denial (medium)"].iloc[-1] == pytest.approx(26_000)
    assert sim.summary["difference vs silence"].iloc[2] == pytest.approx(25_000)
    assert HYPOTHETICAL_LABEL.startswith("HYPOTHETICAL")

def test_simulation_low_medium_high_ordered():
    s = simulate(1000, 50_000, FACT, 48, 1.0).summary
    vals = s["reach at end"].tolist()
    assert vals[0] < vals[1] < vals[2] < vals[3]

def test_simulation_growth_compounds():
    sim = simulate(1000, 0, FACT, 10, 10.0, step_hours=1)
    assert sim.timeline["Remain silent"].iloc[-1] == pytest.approx(1000 * 1.1 ** 10)

def test_simulation_denial_can_reduce_growth():
    sim = simulate(1000, 5_000, FACT, 72, 5.0, 0.0)
    assert sim.summary["reach at end"].iloc[2] < sim.summary["reach at end"].iloc[0]

def test_simulation_is_bounded():
    sim = simulate(1e6, 1e6, FACT, 10_000, 50.0)
    assert sim.timeline["hours"].max() <= 24 * 30 and sim.timeline.drop(columns="hours").max().max() <= 1e8 + 1

def test_suggest_growth_from_data():
    g = suggest_growth_pct_per_hour(make_obs([1, 2, 3], reach=[100, 110, 121]))
    assert g == pytest.approx(10.0, rel=1e-6) is not None
    assert suggest_growth_pct_per_hour(make_obs([1, 2, 3])) is None


# ----------------------------------------------------------------------- database
@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "t.db")

def test_db_seed_and_integrity(db):
    assert db.seed_synthetic_if_empty(SYNTHETIC_CSV)
    rep = db.integrity_report()
    assert rep["foreign_key_violations"] == 0 and rep["row_counts"]["claims"] == 10
    assert not db.seed_synthetic_if_empty(SYNTHETIC_CSV)       # idempotent

def test_db_claim_validation(db):
    with pytest.raises(ValueError):
        db.add_claim("", "Other", "2026-11-01", "Unverified")
    with pytest.raises(ValueError):
        db.add_claim("x", "Bad category", "2026-11-01", "Unverified")
    with pytest.raises(ValueError):
        db.add_claim("x", "Other", "2026-11-01", "Maybe")

def test_db_foreign_key_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.conn() as c:
            c.execute("INSERT INTO observations (claim_id, timestamp, mention_count) VALUES ('NOPE','2026-11-01 00:00:00',1)")

def test_db_observation_validation_and_duplicates(db):
    cid = db.add_claim("a claim", "Other", "2026-11-01 00:00", "Unverified")
    assert db.add_observation(cid, "2026-11-01 01:00", 5)
    assert not db.add_observation(cid, "2026-11-01 01:00", 6)          # duplicate timestamp ignored
    with pytest.raises(ValueError):
        db.add_observation(cid, "2026-11-01 02:00", -1)
    with pytest.raises(ValueError):
        db.add_observation(cid, "2026-11-01 02:00", 3, estimated_reach=-5)
    with pytest.raises(ValueError):
        db.add_observation(cid, "garbage", 3)
    assert len(db.get_observations(cid)) == 1

def test_db_observation_source_evidence_is_persisted_and_validated(db):
    cid = db.add_claim("public report claim", "Other", "2026-11-01", "Unverified")
    assert db.add_observation(
        cid, "2026-11-01 01:00", 5, source_reference="https://example.org/report",
        source_reliability="Moderate",
    )
    row = db.get_observations(cid).iloc[0]
    assert row["source_reference"] == "https://example.org/report"
    assert row["source_reliability"] == "Moderate"
    events = db.get_audit_events(cid)
    assert list(events["event_type"]) == ["observation_added", "claim_created"]
    with pytest.raises(ValueError):
        db.add_observation(cid, "2026-11-01 02:00", 8, source_reliability="Certain")

def test_db_cascade_delete(db):
    db.seed_synthetic_if_empty(SYNTHETIC_CSV)
    db.delete_claim("SYN-001")
    assert db.get_observations("SYN-001").empty and db.integrity_report()["foreign_key_violations"] == 0

def test_db_assessment_and_decision_history(db):
    db.seed_synthetic_if_empty(SYNTHETIC_CSV)
    cfg = default_config()
    b = synthetic_bundle("SYN-002", cfg)
    db.save_assessment(b)
    db.save_assessment(b)                                              # unchanged recommendation -> no new history row
    cfg2 = default_config(); cfg2.threshold.silence_max = 0.1; cfg2.threshold.response_min = 0.2
    db.save_assessment(synthetic_bundle("SYN-002", cfg2))               # changed -> history row
    h = db.get_decision_history("SYN-002")
    assert len(h) == 2 and pd.isna(h.iloc[0]["previous_recommendation"])
    assert h.iloc[1]["previous_recommendation"] == "MONITOR" and h.iloc[1]["new_recommendation"] == "RECOMMEND RESPONSE REVIEW"
    assert len(db.get_assessments("SYN-002")) == 3

def test_db_stored_silence_justification(db):
    db.seed_synthetic_if_empty(SYNTHETIC_CSV)
    db.save_assessment(synthetic_bundle("SYN-001"))
    assert "Public intervention is not recommended" in db.latest_assessment("SYN-001")["justification"]

def test_db_update_claim_and_validation(db):
    cid = db.add_claim("a claim", "Other", "2026-11-01", "Unverified")
    db.update_claim(cid, verification_status="Verified false", public_safety_flag=True)
    c = db.get_claim(cid)
    assert c["verification_status"] == "Verified false" and c["public_safety_flag"] == 1
    with pytest.raises(ValueError):
        db.update_claim(cid, verification_status="nonsense")
    with pytest.raises(ValueError):
        db.update_claim(cid, created_at="x")
    events = db.get_audit_events(cid)
    assert list(events["event_type"]) == ["claim_updated", "claim_created"]
    assert "verification_status" in events.iloc[0]["field_names"]

def test_db_review_schedule_is_persisted_and_audited(db):
    cid = db.add_claim("scheduled review claim", "Other", "2026-11-01", "Unverified")
    db.set_review_due(cid, "2026-11-03 15:30")
    assert db.get_claim(cid)["review_due_at"] == "2026-11-03 15:30:00"
    assert db.get_audit_events(cid).iloc[0]["event_type"] == "review_schedule_updated"
    db.set_review_due(cid, None)
    assert db.get_claim(cid)["review_due_at"] is None

def test_db_csv_import_skips_duplicates(db):
    df = pd.read_csv(SYNTHETIC_CSV)
    r1 = db.import_long_csv(df)
    r2 = db.import_long_csv(df)
    assert r1["claims_added"] == 10 and r2["claims_added"] == 0 and r2["observations_added"] == 0


# ----------------------------------------------------------------------- similarity
def test_similar_description_detection():
    ex = {"A": "Fictional claim that a recruitment rally has been postponed", "B": "Weather forecast for tomorrow"}
    out = find_similar_descriptions("Claim that the recruitment rally was postponed", ex, 0.4)
    assert out and out[0][0] == "A"
    assert find_similar_descriptions("zzz qqq", ex, 0.9) == []


# ----------------------------------------------------------------------- reports
@pytest.mark.parametrize("cid", ["SYN-001", "SYN-003", "SYN-005", "SYN-006"])
def test_pdf_report_generates_for_every_status(cid):
    pdf = build_pdf_report(synthetic_bundle(cid))
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000

def test_pdf_report_accepts_review_schedule_and_source_evidence():
    bundle = synthetic_bundle("SYN-001")
    bundle.obs.loc[bundle.obs.index[-1], "source_reference"] = "Public report title"
    bundle.obs.loc[bundle.obs.index[-1], "source_reliability"] = "Moderate"
    pdf = build_pdf_report(bundle, review_due_at="2026-11-04 10:00:00")
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000

def test_csv_summary_contains_recommendation():
    df = assessment_summary_df(synthetic_bundle("SYN-001"))
    assert df.loc[0, "recommendation"] == "REMAIN SILENT" and df.loc[0, "claim_id"] == "SYN-001"
