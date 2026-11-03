"""
Generate data/synthetic_rumours.csv - FICTIONAL demonstration data only.

No real units, people, incidents or Army data are used. Each scenario is
designed to exercise a particular branch of the decision logic (see README).
Run:  python data/generate_synthetic.py
"""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

N = None
SCENARIOS = [
    dict(id="SYN-001", scenario="1. Limited circulation, low velocity, high amplification risk",
         desc="Fictional claim that a minor training-day schedule at a placeholder location has been quietly cancelled.",
         cat="Unit-related claim", ver="Verified false", impact="Low", safety=0, cred=0,
         remarks="Small cluster of public posts; little pick-up beyond the original group.",
         start="2026-11-01 06:00", step=3, ver_basis="estimated",
         mentions=[4, 6, 9, 11, 13, 16, 17, 19, 21], reach=[250, 280, 320, 360, 410, 450, 480, 500, 520],
         eng=[6, 9, 14, 18, 22, 26, 28, 31, 34], src=[2, 2, 3, 3, 3, 4, 4, 4, 5]),
    dict(id="SYN-002", scenario="2. Moderate circulation, uncertain verification",
         desc="Fictional claim that a placeholder unit has been given short-notice movement orders.",
         cat="Unit-related claim", ver="Conflicting evidence", impact="Medium", safety=0, cred=0,
         remarks="Conflicting public posts; no authoritative confirmation either way.",
         start="2026-11-01 08:00", step=2, ver_basis="estimated",
         mentions=[60, 68, 78, 91, 107, 127, 151, 179, 213],
         reach=[4000, 5500, 7500, 10000, 12500, 15000, 17500, 19000, 20000],
         eng=[90, 120, 160, 210, 270, 340, 420, 520, 640], src=[4, 5, 6, 7, 8, 8, 9, 9, 10]),
    dict(id="SYN-003", scenario="3. Rapidly increasing circulation, significant public impact",
         desc="Fictional message circulating a fake recruitment notification with an unofficial registration link.",
         cat="Recruitment notification", ver="Conflicting evidence", impact="High", safety=0, cred=0,
         remarks="Potential financial harm to applicants; spreading across several public platforms.",
         start="2026-11-02 06:00", step=1, ver_basis="estimated",
         mentions=[20, 45, 85, 150, 250, 400, 620, 900, 1300, 1850],
         reach=[1500, 3000, 6500, 12000, 22000, 38000, 55000, 72000, 86000, 92000],
         eng=[40, 110, 240, 480, 900, 1700, 3000, 5200, 8800, 14000], src=[2, 3, 5, 7, 9, 12, 15, 18, 22, 26]),
    dict(id="SYN-004", scenario="4. High historical reach but declining velocity",
         desc="Fictional claim about a ceremonial event being cancelled, widely shared earlier but now fading.",
         cat="Other", ver="Verified false", impact="Medium", safety=0, cred=0,
         remarks="Peak circulation was two days ago; fresh mentions are now sparse.",
         start="2026-10-29 06:00", step=6, ver_basis="estimated",
         mentions=[800, 1900, 3200, 4300, 5000, 5400, 5600, 5700, 5750],
         reach=[60000, 140000, 230000, 300000, 350000, 380000, 395000, 400000, 402000],
         eng=[1500, 3800, 6900, 9800, 11800, 12900, 13400, 13600, 13700], src=[8, 14, 20, 24, 26, 27, 27, 28, 28]),
    dict(id="SYN-005", scenario="5. Serious credible public-safety allegation with low circulation",
         desc="Fictional unverified report alleging a hazardous incident at a placeholder training area affecting nearby civilians.",
         cat="Incident", ver="Unverified", impact="High", safety=1, cred=1,
         remarks="Low volume, but the allegation concerns possible harm to civilians; analyst rated it credible.",
         start="2026-11-02 09:00", step=3, ver_basis="estimated",
         mentions=[3, 5, 6, 8, 9, 11, 12, 14], reach=[120, 160, 200, 230, 260, 300, 320, 350],
         eng=[4, 7, 9, 12, 14, 18, 20, 24], src=[1, 2, 2, 2, 3, 3, 3, 3]),
    dict(id="SYN-006", scenario="6. Insufficient observations",
         desc="Fictional claim about a placeholder family welfare scheme being withdrawn.",
         cat="Family-related claim", ver="Unverified", impact="Low", safety=0, cred=0,
         remarks="Only two observations so far; monitoring started late.",
         start="2026-11-02 18:00", step=2, ver_basis="estimated",
         mentions=[5, 9], reach=[N, N], eng=[7, 12], src=[1, 2]),
    dict(id="SYN-007", scenario="7. Low velocity but increasing reach",
         desc="Fictional claim about altered selection criteria, repeated slowly but picked up by large public pages.",
         cat="Recruitment notification", ver="Unverified", impact="Medium", safety=0, cred=0,
         remarks="Few new mentions, but each is posted by accounts with large followings.",
         start="2026-10-31 06:00", step=4, ver_basis="estimated",
         mentions=[30, 33, 37, 40, 44, 47, 50, 53, 57],
         reach=[2000, 3500, 6000, 10000, 16000, 24000, 34000, 46000, 60000],
         eng=[50, 60, 80, 110, 160, 240, 360, 520, 760], src=[3, 4, 5, 6, 8, 10, 12, 13, 14]),
    dict(id="SYN-008", scenario="8. Rapid circulation, high uncertainty, high amplification risk",
         desc="Fictional rumour about an unnamed placeholder exercise, spreading fast with conflicting accounts.",
         cat="Incident", ver="Conflicting evidence", impact="Low", safety=0, cred=0,
         remarks="Fast spread, but nobody can establish what actually happened; reach is still modest.",
         start="2026-11-02 12:00", step=1, ver_basis="estimated",
         mentions=[40, 90, 170, 290, 450, 660, 950, 1400], reach=[800, 1500, 2800, 5000, 8500, 13000, 18500, 24000],
         eng=[60, 150, 330, 640, 1100, 1900, 3200, 5400], src=[2, 3, 4, 6, 8, 10, 13, 16]),
    dict(id="SYN-009", scenario="Extra A. Verified TRUE claim (a denial would be inappropriate)",
         desc="Fictional recruitment rally date extension circulating on public pages; the extension is real.",
         cat="Recruitment notification", ver="Verified true", impact="Low", safety=0, cred=0,
         remarks="Matches an official public notice; circulating accurately.",
         start="2026-10-31 06:00", step=4, ver_basis="observed",
         mentions=[100, 140, 185, 225, 262, 300, 335, 370], reach=[8000, 11000, 14000, 16500, 18500, 20500, 22000, 23500],
         eng=[200, 290, 390, 480, 560, 650, 730, 810], src=[4, 5, 6, 6, 7, 8, 8, 9]),
    dict(id="SYN-010", scenario="Extra B. Stable, tiny circulation, low impact",
         desc="Fictional family-related claim about a placeholder welfare helpline timing change.",
         cat="Family-related claim", ver="Unverified", impact="Low", safety=0, cred=0,
         remarks="Slow trickle of mentions in one public forum.",
         start="2026-10-30 08:00", step=6, ver_basis="estimated",
         mentions=[10, 14, 17, 19, 21, 22, 23, 24], reach=[400, 550, 650, 720, 780, 800, 820, 830],
         eng=[12, 17, 21, 24, 27, 28, 30, 31], src=[1, 1, 2, 2, 2, 2, 2, 2]),
]
SRC_TYPES = ["Public social media post", "Public forum", "News media", "Blog / website", "Video platform"]


def build() -> pd.DataFrame:
    rows = []
    for s in SCENARIOS:
        t0 = datetime.strptime(s["start"], "%Y-%m-%d %H:%M")
        for i, m in enumerate(s["mentions"]):
            rows.append({
                "claim_id": s["id"], "description": s["desc"], "category": s["cat"],
                "verification_status": s["ver"], "public_impact": s["impact"],
                "public_safety_flag": s["safety"], "credible_allegation_flag": s["cred"],
                "analyst_remarks": s["remarks"], "scenario": s["scenario"],
                "first_observed": t0.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": (t0 + timedelta(hours=s["step"] * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "mention_count": m, "engagement_count": s["eng"][i], "estimated_reach": s["reach"][i],
                "source_type": SRC_TYPES[(i + len(s["id"])) % len(SRC_TYPES)],
                "independent_sources": s["src"][i], "reach_basis": s["ver_basis"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "synthetic_rumours.csv"
    df = build()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows for {df.claim_id.nunique()} claims -> {out}")
