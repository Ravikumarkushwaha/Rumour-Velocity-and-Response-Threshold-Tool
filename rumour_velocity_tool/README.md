# Rumour Velocity and Response Threshold Tool

Decision-support prototype for the **Indian Army Internship Programme 2026-27**
(HQ Southern Command, Pune Camp - GS (Information Warfare) Branch - Strategic Communication and Digital Media).

> Independent academic and technical prototype. Open-source concepts and **synthetic, fictional data only**.
> No Army network, internal, classified or private data. Thresholds are **experimental and are not official Army policy**.
> The tool **never publishes anything**; every recommendation is advisory and requires authorised human review.

## The idea in one paragraph
Given a hypothetical Army-related claim and its observed public circulation, the tool measures how fast the claim is
**actually spreading** (velocity, not just volume), estimates its **reach**, models whether an **official denial could amplify**
it, compares everything with a **configurable threshold**, and recommends **REMAIN SILENT / MONITOR / RECOMMEND RESPONSE REVIEW**
(or **INSUFFICIENT DATA**). Its centrepiece is the **Silence Justification Engine**, which explains, from the actual numbers,
why not responding can be the right decision.

> *Silence is not the absence of action. It is a deliberate, evidence-based strategic communication decision supported by
> continuous monitoring and clearly defined intervention thresholds.*

## Quick start
```bash
cd rumour_velocity_tool
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Python 3.11+. Works fully offline, no API keys. On first launch the SQLite database (`data/rumour_tool.db`) is created and
seeded with the 10 synthetic claims in `data/synthetic_rumours.csv`.
Run the tests: `python -m pytest -q` (146 tests). Check the authored synthetic scenarios: `python run_scenarios.py`.
Regenerate the synthetic data: `python data/generate_synthetic.py`.

## Architecture (matches the specified pipeline)
```
Input -> Validation/Cleaning -> Velocity -> Growth & Acceleration -> Reach -> Amplification
      -> Threshold (RPS) -> Decision Engine -> Recommendation + Justification -> Monitoring/Reassessment -> Report
```
| File | Responsibility |
|---|---|
| `config.py` | **Every numerical assumption** (thresholds, weights, caps, amplification factors), controlled vocabularies |
| `modules/data_processing.py` | Timestamp validation, duplicate removal, cumulative-count validation, missing values, aggregation, sufficiency, TF-IDF similar-claim warning |
| `modules/velocity_engine.py` | V(t), G(t), A(t), moving average, trend classification |
| `modules/reach_engine.py` | Reach, independent sources, engagement (kept separate), confidence; "INSUFFICIENT REACH DATA" |
| `modules/amplification_engine.py` | Additional exposure, projected reach, amplification ratio, Low/Moderate/High |
| `modules/threshold_engine.py` | Normalisation, Response Pressure Score, effective thresholds, sensitivity, preset comparison |
| `modules/decision_engine.py` | Safeguards, four statuses, rule trace, reassessment conditions |
| `modules/silence_justification.py` | Dynamic 10-part silence explanation |
| `modules/lifecycle_engine.py` | Emergence / Growth / Acceleration / Stabilisation / Decline |
| `modules/simulation_engine.py` | Remain-silent vs public-denial what-if |
| `modules/pipeline.py` | Runs the whole chain for one claim |
| `database/` | `schema.sql` + `db_manager.py` (SQLite, foreign keys, CHECK constraints, decision history) |
| `reports/report_generator.py` | PDF report (ReportLab) and CSV helpers |
| `app.py` | Streamlit dashboard, 7 pages |
| `modules/hardening_engine.py` | **Cost of silence**: hardening index and time-to-hardening estimates |
| `tests/` | 142 pytest tests |

## Both sides of the decision
The brief names two opposing risks, and the tool prices both:
* **Cost of speaking - amplification risk:** a denial carries the rumour to people who have not seen it.
* **Cost of silence - hardening risk:** a rumour left alone past a certain point is accepted as fact.

`Hardening index = Rn x Pn`, where `Rn = min(1, reach / hardening reach point)` and `Pn` is the mean of time-in-circulation
and independent-source ratios (each saturating at a configurable value). Reach is required: with no reach estimate the index is
*Not assessable*, never guessed. Low < 0.30 <= Elevated < 0.60 <= High. The tool also gives an extrapolated **hours to the
hardening reach point** (a straight-line estimate, not a prediction). The Decision page shows the two costs side by side with a
plain-language reading, the what-if chart draws the hardening point and reports when silence vs a denial would cross it, and
**High hardening risk blocks a silence recommendation** (switchable). Example: SYN-001 has High amplification and Low hardening, so
silence is right; SYN-004 has Low amplification but High hardening, so silence is *not* recommended even though velocity is falling.

## Mathematics
* **Velocity** `V(t) = [M(t) - M(t-1)] / dt` (mentions/hour; dt = actual elapsed hours, so irregular intervals are correct)
* **Growth** `G(t) = [(M(t) - M(t-1)) / M(t-1)] x 100`; undefined (not zero) if `M(t-1) = 0`
* **Acceleration** `A(t) = [V(t) - V(t-1)] / dt`
* **Moving average** trailing mean of V (default window 3)
* **Amplification**: `Additional = Response exposure x Factor`; `Projected = Current + Additional`; `Ratio = Additional / max(Current, 1)`
* **Response Pressure Score** `RPS = w1*Vn + w2*Gn + w3*An + w4*Rn`, each `n` = value / cap clipped to [0,1]; weights sum to 1.
  If a component is undefined (e.g. no reach) it is **excluded, weights re-normalised and this is reported**; nothing is guessed.

## Documented assumptions (all in `config.py`, editable on the Decision page)
| Item | Default | Meaning |
|---|---|---|
| Velocity classes | Low < 5 <= Moderate < 20 <= High (mentions/h) | Illustrative |
| Rapidly accelerating | A >= +2 /h² at non-low velocity | |
| Declining | A <= -1 /h² and V below its moving average | |
| RPS weights | V 0.35, G 0.15, A 0.20, R 0.30 | Velocity weighted highest: rate, not volume, is the point |
| Caps | V 40/h, G 100 %, A 5 /h², R 100,000 people | Value at cap scores 1.0 |
| Silence ceiling / response threshold | RPS < 0.30 / RPS >= 0.60 | Base values |
| Amplification | Response audience 50,000; factors 0.2 / 0.5 / 1.0 | Share of audience newly exposed to the claim |
| Amplification risk | ratio < 0.25 Low, < 1.0 Moderate, else High | Classified on the medium scenario |
| Threshold context | High amp. +0.07 response bar; High impact -0.10 both; Medium -0.05; Verified false -0.05 response | Shown explicitly on screen |
| Reach-rise trigger | >= 25 % in one interval blocks silence | Low velocity does not mean low risk |
| Hardening point | 50,000 reach; 24 h persistence; 5 sources; High at index >= 0.60 | Illustrative; edit on Decision page |
| Minimum data | 4 valid observations spanning >= 3 h | Else INSUFFICIENT DATA |

All values are **assumptions for demonstration**, not calibrated or official figures.

## Decision logic (`decision_engine.py`)
1. **Safeguards first, regardless of score**: analyst-flagged public-safety issue, serious credible allegation, or *verified false at High impact*
   -> **RECOMMEND RESPONSE REVIEW (human escalation)**. High amplification risk can never suppress this. Applies even with insufficient data.
2. **Insufficient data** -> `INSUFFICIENT DATA / HUMAN VERIFICATION REQUIRED` (no confident output is forced).
3. **RPS >= response threshold** -> response review (referred to authorised humans). Exception: evidence uncertain + impact Low + amplification High -> MONITOR (held, with reasons).
4. **RPS < silence ceiling** -> **REMAIN SILENT** *unless a blocker applies*: reach unknown, reach rising fast, high velocity class, High impact,
   uncertain evidence with rising circulation, **High hardening risk**, or claim verified **true** (a denial would be inappropriate) -> MONITOR.
5. Otherwise -> **MONITOR**.

## Regression checks on authored fictional scenarios (`python run_scenarios.py`)
| Claim | Scenario | Result |
|---|---|---|
| SYN-001 | 1. Limited, low velocity, high amplification risk | **REMAIN SILENT** (RPS 0.02, amplification 48x) |
| SYN-002 | 2. Moderate circulation, uncertain verification | **MONITOR** |
| SYN-003 | 3. Rapid increase, significant impact | **RECOMMEND RESPONSE REVIEW** (RPS 0.89) |
| SYN-004 | 4. High historical reach, declining velocity | **MONITOR** (no automatic new response) |
| SYN-005 | 5. Credible public-safety allegation, low circulation | **RESPONSE REVIEW - HUMAN ESCALATION** despite RPS 0.04 |
| SYN-006 | 6. Insufficient observations | **INSUFFICIENT DATA** |
| SYN-007 | 7. Low velocity, rising reach | **MONITOR** (reach assessed separately; blocks silence) |
| SYN-008 | 8. Rapid, uncertain, high amplification | **MONITOR** (pressure at response level but held); becomes REVIEW if impact is High (tested) |
| SYN-009 | Extra: verified TRUE claim | **MONITOR** (denial inappropriate; human planning) |
| SYN-010 | Extra: tiny, stable | **REMAIN SILENT** (different wording from SYN-001) |

The command reports expected and actual recommendations and exits unsuccessfully if an authored scenario changes unexpectedly.
This is a regression check only: its expected outcomes are design fixtures, not independent labels, historical evidence,
or proof that the thresholds work in practice. A real generated silence justification: `docs/sample_silence_justification.txt`.

## Test results
`python -m pytest -q` -> **146 passed** (see `docs/TEST_RESULTS.txt`). Coverage: velocity, growth, acceleration, zero/missing values,
irregular intervals, moving average, trend classification, data cleaning and insufficiency, reach (no fabrication), amplification
formulas and boundaries, RPS arithmetic, weight validation, threshold adjustments, sensitivity/presets, all 8 scenarios, safeguards,
silence-justification dynamics, lifecycle, simulation, SQLite integrity/validation/history, PDF generation.
The Streamlit app was smoke-tested with `AppTest`: startup, switching the active claim, the Decision page, and the
claim-specific History/PDF page rendered without exceptions.

## Dashboard pages
1. **Executive Overview** - active-claim recommendation, metrics and charts; expandable portfolio view for all claims.
2. **Rumour Input** - new claim form (auto Claim ID), add observation, public source reference and analyst-rated reliability,
   CSV upload (single claim or multi-claim), data-quality report, claim management.
3. **Velocity Analysis** - mentions, velocity (+ moving average and thresholds), growth, acceleration, lifecycle stage.
4. **Amplification and Risk** - reach, sources, engagement, confidence, low/medium/high scenarios, assumptions, limitations.
5. **Response Threshold and Decision** - RPS vs thresholds, recommendation, rule trace, **silence justification**, evidence coverage
   (not a probability of truth), quick threshold-robustness summary and configurable input sensitivity, analyst-owned next-review schedule, human-review status, reassessment conditions,
   preset comparison, configuration editor.
6. **What-If Simulation** - remain silent vs denial with adjustable assumptions; clearly labelled hypothetical.
7. **Assessment History and Reports** - stored assessments, decision changes, comparison, local claim activity audit, **PDF report**, integrity check.

## CSV format
Single-claim upload needs `timestamp`, `mention_count` (cumulative) and optionally `engagement_count`, `estimated_reach`,
`source_type`, `independent_sources`, `reach_basis` (`estimated`/`observed`), `source_reference` (optional public citation or note),
and `source_reliability` (`Unassessed`/`Low`/`Moderate`/`High`, optional; defaults to Unassessed). Reliability is an analyst judgment,
not machine-verified. Templates are downloadable in the app. CSV downloads are not offered as assessment exports; the assessment
report is PDF.
Multi-claim format = `data/synthetic_rumours.csv` (adds `claim_id, description, category, verification_status, public_impact, ...`).

## Ethical and security design
Open-source/synthetic only; no scraping, no private messages, no personal profiling, no operational details, no Army data;
nothing is sent over a network; no automatic public communication; reach is never invented; scenario outputs are labelled hypothetical;
lifecycle stage is never treated as evidence of truth. Source-reference fields are for public citations only and must not contain
personal or private information. The recorded reliability level is advisory and does not verify a source. Any optional public-source
integration must respect site terms and privacy.

## Limitations
Mention counts must be de-duplicated at collection (the tool removes duplicate observations, not duplicate reposts inside a count).
Reach and amplification factors are analyst assumptions. Thresholds are untuned and illustrative; passing the fictional scenario
regression checks is not real-world validation. Public impact, verification status, and source reliability are analyst inputs.
Before operational use, thresholds and decision outcomes would need evaluation against authorized, appropriately anonymized,
historical or expert-reviewed cases, including false-silence and unnecessary-amplification tradeoffs. The tool does not currently
provide that empirical validation.
The local activity audit records timestamps, event types, record IDs, and changed field names, but not changed values or analyst identity;
it is not an immutable or authenticated audit log.

## Interview talking points
* Velocity, not volume: a claim with 10,000 old mentions can be less urgent than one with 60 mentions doubling hourly.
* The amplification ratio makes the cost of a denial explicit; it raises the bar for response but **never overrides safeguards**.
* Every number on screen traces to a formula and a config value; thresholds change the score's *interpretation*, so adjustments are auditable.
* Silence is blocked when the evidence for it is weak (no reach data, rising reach, high impact, high hardening risk, verified true).
* The tool weighs *both* failure modes in the brief: a denial that spreads the rumour, and silence that lets it harden into fact.
