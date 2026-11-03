import math
import sqlite3

import pytest

from config import HardeningConfig, REC_MONITOR, REC_SILENT, default_config
from database.db_manager import Database
from modules import hardening_engine as he
from modules.simulation_engine import simulate
from tests.helpers import make_obs, run_assessment, synthetic_bundle, ClaimContext

H = HardeningConfig()


def test_classification_boundaries():
    assert he.classify(0.29, H) == "Low" and he.classify(0.30, H) == "Elevated" and he.classify(0.60, H) == "High"

def test_eta_compound_growth():
    # 1000 -> 2000 at 10%/h  => ln2 / ln1.1 hours
    assert he.hours_to_reach_point(1000, 2000, 10.0) == pytest.approx(math.log(2) / math.log(1.1))

def test_eta_edge_cases():
    assert he.hours_to_reach_point(5000, 2000, 10.0) == 0.0
    assert he.hours_to_reach_point(1000, 2000, 0.0) is None
    assert he.hours_to_reach_point(1000, 2000, None) is None
    assert he.hours_to_reach_point(0, 2000, 5.0) is None

def test_velocity_eta_linear():
    assert he.hours_to_velocity(10, 2, 20) == pytest.approx(5.0)
    assert he.hours_to_velocity(30, 2, 20) == 0.0
    assert he.hours_to_velocity(10, -1, 20) is None and he.hours_to_velocity(None, 2, 20) is None

def test_hardening_not_assessable_without_reach_and_not_inferred():
    b = run_assessment(ClaimContext("T", "d", "Other"), make_obs([1, 2, 3, 4, 5, 6]), default_config())
    assert not b.hardening.assessable and b.hardening.level == "Not assessable" and b.hardening.index is None

def test_hardening_index_formula():
    # reach 25,000 of 50,000 => Rn .5 ; 12h of 24h => .5 ; 5 sources of 5 => 1 ; Pn=.75 ; idx=.375
    obs = make_obs([1, 2, 3, 4, 5], step_hours=3, reach=[5000, 9000, 14000, 20000, 25000], sources=[1, 2, 3, 4, 5])
    b = run_assessment(ClaimContext("T", "d", "Other"), obs, default_config())
    assert b.hardening.index == pytest.approx(0.5 * 0.75, abs=1e-6) and b.hardening.level == "Elevated"

def test_small_reach_never_hardens_even_if_persistent():
    b = synthetic_bundle("SYN-001")
    assert b.hardening.level == "Low" and b.hardening.persistence_ratio == pytest.approx(1.0)

def test_widely_seen_persistent_claim_is_high_hardening_and_blocks_silence():
    b = synthetic_bundle("SYN-004")
    assert b.hardening.level == "High"
    cfg = default_config(); cfg.threshold.silence_max = 0.9; cfg.threshold.response_min = 0.95   # force the silence zone
    forced = synthetic_bundle("SYN-004", cfg)
    assert forced.decision.recommendation == REC_MONITOR
    assert any("Hardening risk is High" in x for x in forced.decision.blockers)

def test_hardening_block_can_be_switched_off():
    cfg = default_config(); cfg.threshold.silence_max = 0.9; cfg.threshold.response_min = 0.95
    cfg.hardening.high_blocks_silence = False
    d = synthetic_bundle("SYN-004", cfg).decision
    assert not any("Hardening" in x for x in d.blockers)

def test_silence_justification_mentions_cost_of_silence():
    t = synthetic_bundle("SYN-001").decision.silence_justification.as_text()
    assert "cost of staying silent" in t and "hardening index" in t and "not a prediction" in t

def test_both_costs_shown_in_rationale():
    r = " ".join(synthetic_bundle("SYN-002").decision.rationale)
    assert "Cost of silence" in r and "amplification risk" in r

def test_scenarios_unchanged_by_hardening():
    exp = {"SYN-001": "REMAIN SILENT", "SYN-002": "MONITOR", "SYN-004": "MONITOR", "SYN-007": "MONITOR", "SYN-010": "REMAIN SILENT"}
    for k, v in exp.items():
        assert synthetic_bundle(k).decision.recommendation == v

def test_simulation_reports_hours_to_hardening_point():
    f = {"low": 0.2, "medium": 0.5, "high": 1.0}
    s = simulate(1000, 50_000, f, 48, 2.0, 2.0, hardening_reach=40_000).summary
    silent, medium = s.iloc[0]["hours to hardening point"], s.iloc[2]["hours to hardening point"]
    import pandas as pd
    assert pd.notna(medium) and (pd.isna(silent) or medium < silent)

def test_simulation_hardening_none_when_never_reached():
    s = simulate(100, 0, {"low": 0, "medium": 0, "high": 0}, 24, 0.0, hardening_reach=1e6).summary
    assert s["hours to hardening point"].isna().all()

def test_db_stores_hardening_and_migrates_old_database(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)       # an "old" assessments table without the hardening columns
    con.executescript("""CREATE TABLE claims (claim_id TEXT PRIMARY KEY, description TEXT NOT NULL, category TEXT NOT NULL,
        first_observed TEXT NOT NULL, verification_status TEXT NOT NULL, created_at TEXT NOT NULL, public_impact TEXT NOT NULL DEFAULT 'Low',
        public_safety_flag INTEGER NOT NULL DEFAULT 0, credible_allegation_flag INTEGER NOT NULL DEFAULT 0, analyst_remarks TEXT, is_synthetic INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE assessments (assessment_id INTEGER PRIMARY KEY AUTOINCREMENT, claim_id TEXT, velocity REAL, growth REAL, acceleration REAL, reach REAL,
        amplification_risk TEXT, response_pressure_score REAL, recommendation TEXT NOT NULL, justification TEXT, assessed_at TEXT NOT NULL, trend_label TEXT,
        lifecycle_stage TEXT, escalation INTEGER NOT NULL DEFAULT 0, human_review_status TEXT, observations_used INTEGER, config_snapshot TEXT);""")
    con.commit(); con.close()
    db = Database(path)
    with db.conn() as c:
        claim_columns = {row["name"] for row in c.execute("PRAGMA table_info(claims)")}
        observation_columns = {row["name"] for row in c.execute("PRAGMA table_info(observations)")}
    assert "review_due_at" in claim_columns
    assert {"source_reference", "source_reliability"}.issubset(observation_columns)
    cid = db.add_claim("migration claim", "Other", "2026-11-01", "Unverified")
    for i, m in enumerate([5, 8, 12, 15]):
        db.add_observation(cid, f"2026-11-01 0{i*2}:00", m, estimated_reach=300 + i * 20)
    b = run_assessment(ClaimContext(cid, "migration claim", "Other"), db.get_observations(cid), default_config())
    db.save_assessment(b)
    row = db.latest_assessment(cid)
    assert row["hardening_level"] == b.hardening.level
