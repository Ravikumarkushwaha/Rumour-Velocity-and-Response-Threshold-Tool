"""
Rumour Velocity and Response Threshold Tool - Streamlit dashboard.

Run:  streamlit run app.py
Decision-support prototype. Synthetic demonstration data by default.
It never publishes anything; all recommendations are advisory.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (ALL_RECOMMENDATIONS, APP_SUBTITLE, APP_TITLE, CATEGORIES, DB_PATH, DISCLAIMER,
                    IMPACT_LEVELS, REACH_BASES, REC_INSUFFICIENT, REC_MONITOR, REC_REVIEW, REC_SILENT,
                    SOURCE_RELIABILITY_LEVELS, SOURCE_TYPES, SYNTHETIC_CSV, TREND_DECLINING, TREND_HIGH, TREND_LOW, TREND_MODERATE,
                    TREND_NA, TREND_RAPID, VERIFICATION_STATUSES, AppConfig, default_config)
from database.db_manager import Database
from modules import threshold_engine as te
from modules.amplification_engine import NOT_ASSESSABLE
from modules.data_processing import clean_observations, find_similar_descriptions
from modules.decision_engine import ClaimContext
from modules.pipeline import AssessmentBundle, run_assessment
from modules.reach_engine import INSUFFICIENT_REACH
from modules.silence_justification import PRINCIPLE
from modules.simulation_engine import HYPOTHETICAL_LABEL, simulate, suggest_growth_pct_per_hour
from reports.report_generator import build_pdf_report, to_csv_bytes

NAVY, BLUE, GREY, LIGHT, PALE = "#14284B", "#5B7A9D", "#8A94A0", "#D9E1EC", "#EEF2F7"
REC_COLOURS = {REC_SILENT: BLUE, REC_MONITOR: "#7C8794", REC_REVIEW: NAVY, REC_INSUFFICIENT: "#B8C0CC"}
PAGES = ["1. Executive Overview", "2. Rumour Input", "3. Velocity Analysis", "4. Amplification and Risk",
         "5. Response Threshold and Decision", "6. What-If Simulation", "7. Assessment History and Reports"]

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

CSS = f"""
<style>
.stApp {{ background:#F7F9FC; }}
h1,h2,h3,h4 {{ color:{NAVY}; font-family: 'Segoe UI', Helvetica, Arial, sans-serif; letter-spacing:-0.01em; }}
[data-testid="stSidebar"] {{ background:{PALE}; border-right:1px solid {LIGHT}; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid {LIGHT}; border-radius:6px; padding:10px 14px; }}
[data-testid="stMetricLabel"] p {{ color:{GREY}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em; }}
[data-testid="stMetricValue"] {{ color:{NAVY}; }}
.rec-badge {{ display:inline-block; padding:8px 16px; border-radius:4px; color:#fff; font-weight:600;
              font-size:1.05rem; letter-spacing:0.02em; }}
.callout {{ background:#fff; border-left:4px solid {NAVY}; border-radius:4px; padding:12px 16px; margin:8px 0;
            border-top:1px solid {LIGHT}; border-right:1px solid {LIGHT}; border-bottom:1px solid {LIGHT}; }}
.callout.muted {{ border-left-color:{BLUE}; }}
.callout.grey {{ border-left-color:{GREY}; }}
.tag {{ display:inline-block; font-size:0.72rem; padding:2px 8px; border-radius:3px; background:{PALE};
        color:{NAVY}; border:1px solid {LIGHT}; margin-right:6px; text-transform:uppercase; letter-spacing:0.05em; }}
.small {{ color:{GREY}; font-size:0.82rem; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color:{NAVY}; }}
</style>
"""


# ----------------------------------------------------------------------------- helpers
@st.cache_resource
def get_db() -> Database:
    db = Database(DB_PATH)
    db.seed_synthetic_if_empty(SYNTHETIC_CSV)
    return db


def cfg() -> AppConfig:
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = default_config()
    return st.session_state["cfg"]


def show_df(df: pd.DataFrame, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_fig(fig: go.Figure, key: str | None = None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def style_fig(fig: go.Figure, title: str = "", height: int = 330) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=NAVY)), height=height, margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#2B3A4F", size=12),
        legend=dict(orientation="h", y=-0.22), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor=LIGHT, zeroline=False)
    fig.update_yaxes(gridcolor=PALE, linecolor=LIGHT, zeroline=False)
    return fig


def fnum(x, d=2, suffix="") -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}{suffix}"


def rec_badge(rec: str, escalated: bool = False) -> str:
    label = rec + ("  |  HUMAN ESCALATION" if escalated else "")
    fg = "#1F2937" if rec == REC_INSUFFICIENT else "#fff"
    return f'<span class="rec-badge" style="background:{REC_COLOURS[rec]};color:{fg}">{label}</span>'


def callout(html: str, kind: str = ""):
    st.markdown(f'<div class="callout {kind}">{html}</div>', unsafe_allow_html=True)


def load_context(claim_id: str):
    c = get_db().get_claim(claim_id)
    ctx = ClaimContext(c["claim_id"], c["description"], c["category"], c["verification_status"], c["public_impact"],
                       bool(c["public_safety_flag"]), bool(c["credible_allegation_flag"]))
    return ctx, c


def get_bundle(claim_id: str) -> AssessmentBundle:
    ctx, c = load_context(claim_id)
    return run_assessment(ctx, get_db().get_observations(claim_id), cfg(), c["first_observed"])


def claim_label(row) -> str:
    d = row["description"]
    return f"{row['claim_id']}  -  {d[:52] + '...' if len(d) > 52 else d}"


def selected_claim_id() -> str | None:
    return st.session_state.get("claim_id")


def flash(msg: str, kind: str = "success") -> None:
    """Queue a message to show after st.rerun() (otherwise it would vanish)."""
    st.session_state.setdefault("_flash", []).append((kind, msg))


def show_flash() -> None:
    for kind, msg in st.session_state.pop("_flash", []):
        {"success": st.success, "warning": st.warning, "error": st.error}.get(kind, st.info)(msg)


def need_claim() -> str | None:
    cid = selected_claim_id()
    if not cid:
        st.info("No claim selected. Add one on the Rumour Input page or load the synthetic demonstration data.")
    return cid


def parse_optional_number(text: str, name: str, integer: bool = False):
    t = (text or "").strip().replace(",", "")
    if t == "":
        return None, None
    try:
        v = float(t)
    except ValueError:
        return None, f"{name} must be a number (or left blank)."
    if v < 0:
        return None, f"{name} cannot be negative."
    if integer and v != int(v):
        return None, f"{name} must be a whole number."
    return (int(v) if integer else v), None


def header(title: str, sub: str = ""):
    st.markdown(f"## {title}")
    if sub:
        st.markdown(f'<div class="small">{sub}</div>', unsafe_allow_html=True)


def validation_banner(b: AssessmentBundle):
    if not b.validation.sufficient:
        st.error(f"INSUFFICIENT DATA: {b.validation.insufficient_reason} No confident recommendation will be issued; "
                 f"human verification is required.")
    for m in b.validation.warnings:
        if not m.startswith("INSUFFICIENT"):
            st.warning(m)


# ----------------------------------------------------------------------------- charts
def fig_mentions(s: pd.DataFrame) -> go.Figure:
    f = go.Figure(go.Scatter(x=s["timestamp"], y=s["mentions"], mode="lines+markers", line=dict(color=NAVY, width=2.4),
                             marker=dict(size=6), name="Cumulative mentions (observed)"))
    f.update_yaxes(title="Cumulative mentions")
    return style_fig(f, "Mention history (cumulative, observed)")


def fig_velocity(s: pd.DataFrame, v) -> go.Figure:
    f = go.Figure()
    f.add_bar(x=s["timestamp"], y=s["velocity"], name="Velocity V(t)", marker_color=BLUE)
    f.add_scatter(x=s["timestamp"], y=s["moving_avg_velocity"], name="Moving average", mode="lines",
                  line=dict(color=NAVY, width=2.6))
    f.add_hline(y=v.low_max, line_dash="dot", line_color=GREY, annotation_text=f"Low/Moderate {v.low_max:g}")
    f.add_hline(y=v.high_min, line_dash="dash", line_color=GREY, annotation_text=f"High {v.high_min:g}")
    f.update_yaxes(title="Mentions per hour")
    return style_fig(f, "Rumour velocity")


def fig_growth(s: pd.DataFrame) -> go.Figure:
    f = go.Figure(go.Bar(x=s["timestamp"], y=s["growth_pct"], marker_color=BLUE, name="Growth G(t)"))
    f.update_yaxes(title="% change per interval")
    return style_fig(f, "Percentage growth")


def fig_accel(s: pd.DataFrame) -> go.Figure:
    colours = [NAVY if (x is not None and not np.isnan(x) and x >= 0) else GREY for x in s["acceleration"]]
    f = go.Figure(go.Bar(x=s["timestamp"], y=s["acceleration"], marker_color=colours, name="Acceleration"))
    f.add_hline(y=0, line_color=NAVY, line_width=1)
    f.update_yaxes(title="Mentions per hour squared")
    return style_fig(f, "Acceleration (navy = accelerating, grey = decelerating)")


# ============================================================================= PAGE 1
def page_overview():
    cid = selected_claim_id()
    if cid:
        ctx, claim = load_context(cid)
        b = get_bundle(cid)
        header("Executive Overview", "Live assessment of the active claim under the current configuration.")
        st.markdown(f"### {cid}")
        st.write(ctx.description)
        st.caption(
            f"Category: {claim['category']}  |  Verification: {claim['verification_status']}  |  "
            f"Impact: {claim['public_impact']}"
        )

        metrics = st.columns(5)
        metrics[0].metric("Current velocity", f"{fnum(b.trend.current_velocity, 2)} mentions/h")
        metrics[1].metric(
            "Estimated reach",
            INSUFFICIENT_REACH if b.reach.current_reach is None else f"{b.reach.current_reach:,.0f}",
        )
        metrics[2].metric("Amplification risk", b.amp.risk)
        metrics[3].metric("Response pressure score", fnum(b.decision.rps_score, 3))
        metrics[4].metric("Recommendation", b.decision.recommendation)
        source_refs = int(b.obs["source_reference"].notna().sum())
        assessed_sources = int((b.obs["source_reliability"] != "Unassessed").sum())
        st.caption(
            f"Evidence coverage (not a probability): {len(b.obs)} valid observations; "
            f"{source_refs} with public source references; {assessed_sources} with an assessed source reliability; "
            f"reach confidence: {b.reach.confidence}."
        )
        validation_banner(b)

        c1, c2 = st.columns(2)
        with c1:
            show_fig(fig_mentions(b.series), key="active_claim_mentions")
        with c2:
            show_fig(fig_velocity(b.series, cfg().velocity), key="active_claim_velocity")

    with st.expander("Portfolio overview - all tracked claims", expanded=not bool(cid)):
        page_portfolio_overview()


def page_portfolio_overview():
    header("Portfolio Overview", "Live assessment of every tracked claim under the current configuration.")
    db = get_db()
    claims = db.get_claims()
    if claims.empty:
        st.info("No claims yet. Use the Rumour Input page to add one or load the synthetic demonstration data.")
        return
    rows = []
    for _, c in claims.iterrows():
        b = get_bundle(c["claim_id"])
        d, t = b.decision, b.trend
        rows.append({"Claim": c["claim_id"], "Category": c["category"], "Verification": c["verification_status"],
                     "Impact": c["public_impact"], "Trend": t.label, "Velocity (/h)": t.current_velocity,
                     "Reach": b.reach.current_reach, "Amplification": b.amp.risk, "RPS": d.rps_score,
                     "Recommendation": d.recommendation, "Escalated": "Yes" if d.escalation else "",
                     "Synthetic": "Yes" if c["is_synthetic"] else ""})
    tbl = pd.DataFrame(rows)
    counts = tbl["Recommendation"].value_counts()
    cols = st.columns(5)
    cols[0].metric("Claims assessed", len(tbl))
    cols[1].metric("Remain silent", int(counts.get(REC_SILENT, 0)))
    cols[2].metric("Monitor", int(counts.get(REC_MONITOR, 0)))
    cols[3].metric("Response review", int(counts.get(REC_REVIEW, 0)),
                   help="Includes mandatory human escalations triggered by safeguards.")
    cols[4].metric("Insufficient data", int(counts.get(REC_INSUFFICIENT, 0)))
    callout("Silence is not the absence of action. It is a deliberate, evidence-based strategic communication decision "
            "supported by continuous monitoring and clearly defined intervention thresholds.", "muted")

    c1, c2 = st.columns([3, 2])
    with c1:
        v = tbl.dropna(subset=["Velocity (/h)"]).copy()
        v["plot"] = v["Velocity (/h)"].clip(lower=0.01)
        v = v.sort_values("plot")
        f = go.Figure(go.Bar(x=v["plot"], y=v["Claim"], orientation="h",
                             marker_color=[REC_COLOURS[r] for r in v["Recommendation"]],
                             text=[fnum(x, 1) for x in v["Velocity (/h)"]], textposition="outside",
                             customdata=v["Recommendation"], hovertemplate="%{y}<br>%{x:.2f}/h<br>%{customdata}<extra></extra>"))
        f.add_vline(x=cfg().velocity.low_max, line_dash="dot", line_color=GREY)
        f.add_vline(x=cfg().velocity.high_min, line_dash="dash", line_color=GREY)
        f.update_xaxes(type="log", title="Current velocity, mentions/hour (log scale)")
        show_fig(style_fig(f, "Current velocity by claim (colour = recommendation)", 380), key="ov_vel")
    with c2:
        order = [TREND_LOW, TREND_MODERATE, TREND_HIGH, TREND_RAPID, TREND_DECLINING, TREND_NA]
        tc = tbl["Trend"].value_counts().reindex(order).fillna(0)
        f2 = go.Figure(go.Bar(x=tc.index, y=tc.values, marker_color=BLUE))
        f2.update_yaxes(title="Claims", dtick=1)
        show_fig(style_fig(f2, "Velocity classification distribution", 380), key="ov_dist")

    st.markdown("#### Claims and current recommendations")
    show_df(tbl.assign(**{"Velocity (/h)": tbl["Velocity (/h)"].round(2), "RPS": tbl["RPS"].round(3),
                          "Reach": tbl["Reach"].map(lambda x: INSUFFICIENT_REACH if pd.isna(x) else f"{x:,.0f}")}))

    bc1, bc2 = st.columns([1, 3])
    if bc1.button("Save assessments for all claims", help="Stores one assessment per claim in the history database."):
        n = 0
        for cid in claims["claim_id"]:
            db.save_assessment(get_bundle(cid))
            n += 1
        st.success(f"Saved {n} assessments to the history database.")
    st.markdown("#### Recent assessment history")
    h = db.get_assessments()
    if h.empty:
        st.caption("No assessments saved yet. Use the button above, or save from the Decision page.")
    else:
        h = h.sort_values("assessment_id", ascending=False).head(10)
        show_df(h[["assessed_at", "claim_id", "velocity", "response_pressure_score", "amplification_risk",
                   "recommendation"]].round(3))


# ============================================================================= PAGE 2
def _build_obs_row(d_obs, t_obs, mentions, eng_txt, reach_txt, basis, src_type, srcs_txt,
                   source_reference="", source_reliability="Unassessed"):
    errs = []
    eng, e1 = parse_optional_number(eng_txt, "Engagement", True)
    reach, e2 = parse_optional_number(reach_txt, "Estimated reach")
    srcs, e3 = parse_optional_number(srcs_txt, "Independent sources", True)
    errs += [e for e in (e1, e2, e3) if e]
    row = {"timestamp": datetime.combine(d_obs, t_obs), "mention_count": int(mentions), "engagement_count": eng,
           "estimated_reach": reach, "source_type": src_type, "independent_sources": srcs,
           "reach_basis": basis if reach is not None else None,
           "source_reference": source_reference.strip() or None,
           "source_reliability": source_reliability}
    return row, errs


def page_input():
    header("Rumour Input", "Enter a hypothetical claim and its observations, or upload a CSV. "
                           "OBSERVED values: mentions, engagement. ESTIMATED values: reach (analyst estimate, optional, never inferred).")
    db = get_db()
    tabs = st.tabs(["New claim", "Add observation", "CSV upload", "Manage claim and data quality"])

    # -------- New claim
    with tabs[0]:
        st.caption("A Claim ID (CLM-date-code) is generated automatically when the claim is saved.")
        now = datetime.now().replace(second=0, microsecond=0)
        with st.form("new_claim_form"):
            desc = st.text_area("Claim description (hypothetical / fictional)", height=90,
                                placeholder="e.g. Fictional claim that a placeholder recruitment rally has been postponed.")
            c1, c2, c3 = st.columns(3)
            cat = c1.selectbox("Claim category", CATEGORIES)
            ver = c2.selectbox("Verification status", VERIFICATION_STATUSES, index=3)
            impact = c3.selectbox("Public impact (analyst-rated)", IMPACT_LEVELS)
            c4, c5 = st.columns(2)
            fo_d = c4.date_input("First observed - date", value=(now - timedelta(hours=6)).date())
            fo_t = c5.time_input("First observed - time", value=(now - timedelta(hours=6)).time())
            f1, f2 = st.columns(2)
            safety = f1.checkbox("Urgent public-safety issue (forces human escalation)")
            cred = f2.checkbox("Serious credible allegation / potential harm (forces human escalation)")
            remarks = st.text_area("Analyst remarks", height=70)
            st.markdown("**First observation** (more can be added later)")
            o1, o2, o3 = st.columns(3)
            od = o1.date_input("Observation date", value=(now - timedelta(hours=6)).date(), key="nc_od")
            ot = o2.time_input("Observation time", value=(now - timedelta(hours=6)).time(), key="nc_ot")
            mentions = o3.number_input("Observed mention count (cumulative)", min_value=0, value=1, step=1)
            o4, o5, o6 = st.columns(3)
            eng_t = o4.text_input("Public engagement observations (blank if unknown)")
            reach_t = o5.text_input("Estimated reach (blank if unknown - never inferred)")
            basis = o6.selectbox("Reach basis", REACH_BASES)
            o7, o8 = st.columns(2)
            src_type = o7.selectbox("Source category", SOURCE_TYPES)
            srcs_t = o8.text_input("Independent public sources (blank if unknown)")
            o9, o10 = st.columns(2)
            source_ref = o9.text_input("Public source reference / evidence note (optional; no personal data)")
            source_reliability = o10.selectbox("Source reliability", SOURCE_RELIABILITY_LEVELS)
            override = st.checkbox("Create even if a similar claim already exists")
            ok = st.form_submit_button("Create claim")
        if ok:
            errs = []
            if not desc.strip():
                errs.append("Claim description is required.")
            fo = datetime.combine(fo_d, fo_t)
            row, e = _build_obs_row(od, ot, mentions, eng_t, reach_t, basis, src_type, srcs_t,
                                    source_ref, source_reliability)
            errs += e
            if row["timestamp"] < fo:
                errs.append("Observation time cannot be earlier than the first-observed time.")
            existing = db.get_claims()
            sims = find_similar_descriptions(desc, dict(zip(existing["claim_id"], existing["description"])),
                                             cfg().data.similarity_threshold) if len(existing) else []
            if sims and not override:
                errs.append("Similar claim(s) already tracked: " + ", ".join(f"{i} ({s:.0%})" for i, s in sims) +
                            ". Add observations to that claim instead, or tick 'Create even if similar'.")
            if errs:
                for m in errs:
                    st.error(m)
            else:
                try:
                    cid = db.add_claim(desc, cat, fo, ver, impact, safety, cred, remarks)
                    db.add_observation(cid, row["timestamp"], row["mention_count"], row["engagement_count"],
                                       row["estimated_reach"], row["source_type"], row["independent_sources"],
                                       row["reach_basis"], row["source_reference"], row["source_reliability"])
                    st.session_state["claim_id"] = cid
                    flash(f"Claim {cid} created. At least {cfg().data.min_observations} observations are needed "
                               f"before a recommendation is issued.")
                    st.rerun()
                except ValueError as ex:
                    st.error(str(ex))

    # -------- Add observation
    with tabs[1]:
        cid = selected_claim_id()
        if not cid:
            st.info("Select or create a claim first.")
        else:
            ctx, c = load_context(cid)
            obs = db.get_observations(cid)
            st.markdown(f"**Claim {cid}**  -  {ctx.description}")
            if len(obs):
                last = obs.iloc[-1]
                st.caption(f"Latest stored observation: {last['timestamp']:%Y-%m-%d %H:%M} - {int(last['mention_count'])} cumulative mentions.")
            base = (obs["timestamp"].iloc[-1] + timedelta(hours=2)) if len(obs) else datetime.now().replace(second=0, microsecond=0)
            with st.form(f"add_obs_{cid}"):
                a1, a2, a3 = st.columns(3)
                od = a1.date_input("Observation date", value=base.date())
                ot = a2.time_input("Observation time", value=base.time())
                mentions = a3.number_input("Observed mention count (cumulative)", min_value=0, step=1,
                                           value=int(obs["mention_count"].iloc[-1]) if len(obs) else 0)
                b1, b2, b3 = st.columns(3)
                eng_t = b1.text_input("Public engagement observations (blank if unknown)")
                reach_t = b2.text_input("Estimated reach (blank if unknown)")
                basis = b3.selectbox("Reach basis", REACH_BASES)
                d1, d2 = st.columns(2)
                src_type = d1.selectbox("Source category", SOURCE_TYPES)
                srcs_t = d2.text_input("Independent public sources (blank if unknown)")
                d3, d4 = st.columns(2)
                source_ref = d3.text_input("Public source reference / evidence note (optional; no personal data)")
                source_reliability = d4.selectbox("Source reliability", SOURCE_RELIABILITY_LEVELS)
                ok = st.form_submit_button("Add observation and reassess")
            if ok:
                row, errs = _build_obs_row(od, ot, mentions, eng_t, reach_t, basis, src_type, srcs_t,
                                           source_ref, source_reliability)
                if len(obs):
                    if row["timestamp"] <= obs["timestamp"].iloc[-1]:
                        errs.append("Timestamp must be later than the latest stored observation (duplicates are not independent evidence).")
                    if row["mention_count"] < obs["mention_count"].iloc[-1]:
                        errs.append("Cumulative mention count cannot be lower than the previous observation.")
                if row["timestamp"] < pd.to_datetime(c["first_observed"]):
                    errs.append("Timestamp is earlier than the first-observed time.")
                if errs:
                    for m in errs:
                        st.error(m)
                else:
                    db.add_observation(cid, row["timestamp"], row["mention_count"], row["engagement_count"],
                                       row["estimated_reach"], row["source_type"], row["independent_sources"],
                                       row["reach_basis"], row["source_reference"], row["source_reliability"])
                    b = get_bundle(cid)
                    db.save_assessment(b)
                    flash(f"Observation stored. Reassessed: {b.decision.recommendation} (saved to history).")
                    st.rerun()
            if len(obs):
                st.markdown("**Stored observations**")
                show_df(obs.drop(columns=["claim_id"]))

    # -------- CSV upload
    with tabs[2]:
        tmpl_obs = pd.DataFrame({"timestamp": ["2026-11-02 09:00:00", "2026-11-02 11:00:00"], "mention_count": [10, 14],
                                 "engagement_count": [15, 22], "estimated_reach": [400, 520], "source_type": ["News media", "Public forum"],
                                 "independent_sources": [2, 3], "reach_basis": ["estimated", "estimated"],
                                 "source_reference": ["https://example.org/public-report", "Public forum post title"],
                                 "source_reliability": ["Moderate", "Unassessed"]})
        t1, t2 = st.columns(2)
        t1.download_button("Download observation CSV template", to_csv_bytes(tmpl_obs), "observations_template.csv", "text/csv")
        if SYNTHETIC_CSV.exists():
            t2.download_button("Download multi-claim format (synthetic data)", SYNTHETIC_CSV.read_bytes(), "synthetic_rumours.csv", "text/csv")
        mode = st.radio("Upload mode", ["Add observations to the selected claim", "Import claims and observations (multi-claim format)"])
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as ex:
                st.error(f"Could not read CSV: {ex}")
                raw = None
            if raw is not None:
                st.caption(f"{len(raw)} rows, columns: {', '.join(raw.columns)}")
                show_df(raw.head(10))
                if mode.startswith("Add"):
                    cid = selected_claim_id()
                    if not cid:
                        st.info("Select a claim first.")
                    else:
                        ctx, c = load_context(cid)
                        clean, rep = clean_observations(raw, cfg().data, c["first_observed"])
                        for sev, m in rep.issues:
                            (st.error if sev == "error" else st.warning if sev == "warning" else st.info)(m)
                        st.write(f"{len(clean)} valid row(s) after cleaning.")
                        if len(clean) and st.button("Import validated observations"):
                            res = db.add_observations_df(cid, clean)
                            flash(f"Added {res['added']} observation(s); skipped {res['skipped_duplicates']} already stored.")
                            st.rerun()
                else:
                    if st.button("Import all claims and observations"):
                        try:
                            res = db.import_long_csv(raw)
                            flash(f"Imported {res['claims_added']} new claim(s), {res['observations_added']} observation(s); "
                                       f"{res['duplicates_skipped']} duplicate observation(s) skipped.")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Import failed: {ex}")
        st.divider()
        if st.button("Load synthetic demonstration data (skips claims already present)"):
            res = db.import_long_csv(pd.read_csv(SYNTHETIC_CSV), is_synthetic=True)
            flash(f"Loaded {res['claims_added']} claim(s), {res['observations_added']} observation(s).")
            st.rerun()

    # -------- Manage
    with tabs[3]:
        cid = selected_claim_id()
        if not cid:
            st.info("Select a claim first.")
        else:
            ctx, c = load_context(cid)
            st.markdown(f"**Claim {cid}**")
            with st.form(f"manage_{cid}"):
                m1, m2 = st.columns(2)
                ver = m1.selectbox("Verification status", VERIFICATION_STATUSES, index=VERIFICATION_STATUSES.index(c["verification_status"]))
                imp = m2.selectbox("Public impact", IMPACT_LEVELS, index=IMPACT_LEVELS.index(c["public_impact"]))
                g1, g2 = st.columns(2)
                sf = g1.checkbox("Urgent public-safety issue", value=bool(c["public_safety_flag"]))
                cf = g2.checkbox("Serious credible allegation", value=bool(c["credible_allegation_flag"]))
                rem = st.text_area("Analyst remarks", value=c["analyst_remarks"] or "")
                if st.form_submit_button("Update claim and reassess"):
                    db.update_claim(cid, verification_status=ver, public_impact=imp, public_safety_flag=sf,
                                    credible_allegation_flag=cf, analyst_remarks=rem)
                    db.save_assessment(get_bundle(cid))
                    flash("Claim updated and reassessment saved.")
                    st.rerun()
            st.markdown("**Data-quality report**")
            b = get_bundle(cid)
            st.write(f"Rows stored: {len(db.get_observations(cid))}  |  valid after cleaning: {len(b.obs)}  |  "
                     f"median interval: {fnum(b.validation.median_interval_hours, 1, ' h')}  |  "
                     f"regular intervals: {'No (irregular)' if b.validation.irregular_intervals else 'Yes'}")
            if not b.validation.issues:
                st.success("No data-quality issues detected.")
            for sev, m in b.validation.issues:
                (st.error if sev == "error" else st.warning if sev == "warning" else st.info)(m)
            with st.expander("Delete this claim"):
                sure = st.checkbox("I understand this permanently deletes the claim, its observations and assessments")
                if st.button("Delete claim", disabled=not sure):
                    db.delete_claim(cid)
                    st.session_state.pop("claim_id", None)
                    st.rerun()


# ============================================================================= PAGE 3
def page_velocity():
    header("Velocity Analysis", "How fast is the claim actually travelling? Total mentions are shown separately from velocity.")
    cid = need_claim()
    if not cid:
        return
    b = get_bundle(cid)
    t, s = b.trend, b.series
    st.markdown(f"**{cid}**  -  {b.ctx.description}")
    validation_banner(b)
    source_refs = int(b.obs["source_reference"].notna().sum())
    assessed_sources = int((b.obs["source_reliability"] != "Unassessed").sum())
    evidence_cols = st.columns(4)
    evidence_cols[0].metric("Valid observations", f"{len(b.obs)} / {b.validation.rows_in}")
    evidence_cols[1].metric("Public references", f"{source_refs} / {len(b.obs)}")
    evidence_cols[2].metric("Reliability assessed", f"{assessed_sources} / {len(b.obs)}")
    evidence_cols[3].metric("Reach confidence", b.reach.confidence)
    st.caption(
        "Evidence coverage is not a probability that the claim is true. Source reliability is an analyst assessment; "
        "the tool does not verify references or source credibility."
    )
    if len(s) < 2:
        st.warning("At least two observations are needed to compute velocity.")
        if len(s):
            show_df(s[["timestamp", "mentions"]])
        return

    m = st.columns(6)
    m[0].metric("Total mentions", fnum(t.total_mentions, 0), help="Observed cumulative volume. This is NOT the rate of spread.")
    m[1].metric("Current velocity", fnum(t.current_velocity, 2, " /h"))
    m[2].metric("Previous velocity", fnum(t.previous_velocity, 2, " /h"))
    m[3].metric("% growth", fnum(t.growth_pct, 1, " %"))
    m[4].metric("Acceleration", fnum(t.acceleration, 2, " /h²"))
    m[5].metric("Moving average", fnum(t.moving_avg, 2, " /h"), help=f"Window: {cfg().velocity.moving_avg_window} observations")
    d1, d2 = st.columns(2)
    d1.markdown(f'<span class="tag">Trend</span> **{t.label}**', unsafe_allow_html=True)
    d2.markdown(f'<span class="tag">Lifecycle</span> **{b.lifecycle.stage}** - {b.lifecycle.rationale}', unsafe_allow_html=True)
    st.caption(b.lifecycle.disclaimer)

    c1, c2 = st.columns(2)
    with c1:
        show_fig(fig_mentions(s), key="v_m")
        show_fig(fig_growth(s), key="v_g")
    with c2:
        show_fig(fig_velocity(s, cfg().velocity), key="v_v")
        show_fig(fig_accel(s), key="v_a")

    with st.expander("Formulas used"):
        st.markdown(
            "- **Velocity** V(t) = [M(t) − M(t−1)] / Δt, with Δt in hours between the two actual timestamps.\n"
            "- **Growth** G(t) = [(M(t) − M(t−1)) / M(t−1)] × 100; undefined (shown as n/a) when M(t−1) = 0.\n"
            "- **Acceleration** A(t) = [V(t) − V(t−1)] / Δt.\n"
            f"- **Moving average**: trailing mean of V over {cfg().velocity.moving_avg_window} observations.\n"
            f"- **Trend**: Low < {cfg().velocity.low_max:g}/h ≤ Moderate < {cfg().velocity.high_min:g}/h ≤ High; "
            f"Rapidly accelerating if A ≥ {cfg().velocity.accel_rapid_min:g}/h² at non-low velocity; "
            f"Declining if A ≤ {cfg().velocity.accel_decline_max:g}/h² and V is below its moving average.")
    st.markdown("#### Calculation table")
    tbl = s.copy()
    tbl["timestamp"] = tbl["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    show_df(tbl.round(3))
    st.caption(f"Interval regularity: median {fnum(b.validation.median_interval_hours, 1, ' h')}"
               f"{'; intervals are irregular, actual elapsed hours are used' if b.validation.irregular_intervals else ''}.")


# ============================================================================= PAGE 4
def page_amplification():
    header("Amplification and Risk", "Could an official denial introduce the claim to a larger audience? Scenario estimates, not predictions.")
    cid = need_claim()
    if not cid:
        return
    b = get_bundle(cid)
    r, a = b.reach, b.amp
    st.markdown(f"**{cid}**  -  {b.ctx.description}")
    validation_banner(b)

    st.markdown("#### Public reach")
    m = st.columns(5)
    m[0].metric("Estimated current reach", "INSUFFICIENT REACH DATA" if not r.available else f"{r.current_reach:,.0f}",
                help="Analyst estimate. Never inferred or manufactured by the tool.")
    m[1].metric("Independent sources", "n/a" if r.independent_sources is None else r.independent_sources, help=r.sources_basis)
    m[2].metric("Engagement (not audience)", fnum(r.total_engagement, 0), help="Interactions are not unique people reached.")
    m[3].metric("Reach growth (window)", fnum(r.reach_growth_pct_total, 1, " %"))
    m[4].metric("Data confidence", r.confidence)
    m2 = st.columns(3)
    m2[0].metric("Verification status", b.ctx.verification_status)
    m2[1].metric("Public impact (analyst)", b.ctx.public_impact)
    m2[2].metric("Reach basis", r.basis)
    for n in r.notes:
        st.caption("• " + n)
    if not r.available:
        st.error("INSUFFICIENT REACH DATA. The tool does not manufacture audience figures; amplification risk cannot be assessed. "
                 "Obtain a defensible reach estimate and add it as an observation.")

    d = r.distribution
    if len(d) >= 2:
        f = go.Figure()
        f.add_scatter(x=d["timestamp"], y=d["estimated_reach"], name="Estimated reach (estimate)", mode="lines+markers",
                      line=dict(color=NAVY, width=2.4), connectgaps=False)
        f.add_scatter(x=d["timestamp"], y=d["engagement_count"], name="Engagement (observed, not audience)", mode="lines",
                      line=dict(color=GREY, dash="dot"))
        f.update_yaxes(title="People / interactions")
        show_fig(style_fig(f, "Distribution over time: reach versus engagement"), key="a_dist")

    st.markdown("#### Amplification risk if an official denial were issued")
    if not a.assessable:
        st.warning(f"Amplification risk: {NOT_ASSESSABLE}. Current reach is unavailable.")
    else:
        mm = st.columns(4)
        mm[0].metric("Current reach", f"{a.current_reach:,.0f}")
        mm[1].metric("Potential additional exposure", f"{a.additional_exposure:,.0f}", help=f"{a.classification_scenario} scenario")
        mm[2].metric("Projected reach after response", f"{a.projected:,.0f}")
        mm[3].metric("Amplification ratio", f"{a.ratio:,.2f}x")
        callout(f"<b>Amplification risk: {a.risk}</b> ({a.classification_scenario} scenario). "
                f"A ratio of {a.ratio:,.2f} means the assumed response would add about {a.ratio:,.2f} times the claim's present audience.",
                "muted" if a.risk != "High" else "")
        rows = [{"Scenario": k.title(), "Amplification factor": v["factor"], "Potential additional exposure": round(v["additional"]),
                 "Projected reach": round(v["projected"]), "Amplification ratio": round(v["ratio"], 3), "Risk": v["risk"]}
                for k, v in a.scenarios.items()]
        show_df(pd.DataFrame(rows))
        f = go.Figure()
        f.add_bar(x=["Current reach"] + [k.title() for k in a.scenarios], y=[a.current_reach] + [v["projected"] for v in a.scenarios.values()],
                  marker_color=[NAVY, "#9DB1C9", BLUE, "#2F4B73"])
        f.update_yaxes(title="People (hypothetical)", type="log" if a.current_reach and max(v["projected"] for v in a.scenarios.values()) / max(a.current_reach, 1) > 50 else "linear")
        show_fig(style_fig(f, "Projected reach after a public response (hypothetical scenarios)"), key="a_proj")

    st.markdown("#### Assumptions, confidence and limitations")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Assumptions** (edit on the Decision page)")
        for x in a.assumptions:
            st.markdown("- " + x)
        st.markdown(f"**Confidence:** {a.confidence} (inherits reach data confidence)")
    with cc2:
        st.markdown("**Limitations**")
        for x in a.limitations:
            st.markdown("- " + x)
    balance_panel(b)


def balance_panel(b: AssessmentBundle):
    """Cost of speaking (amplification) set against cost of silence (hardening)."""
    a, h = b.amp, b.hardening
    st.markdown("#### Balance of risks: cost of speaking versus cost of staying silent")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cost of a public denial: amplification risk**")
        x = st.columns(2)
        x[0].metric("Amplification risk", a.risk)
        x[1].metric("Ratio", "n/a" if a.ratio is None else f"{a.ratio:,.2f}x",
                    help="Potential additional exposure divided by current reach (medium scenario).")
        st.caption("A denial carries the rumour to people who have not seen it.")
    with c2:
        st.markdown("**Cost of silence: hardening risk**")
        x = st.columns(3)
        x[0].metric("Hardening risk", h.level)
        x[1].metric("Index", fnum(h.index, 2), help="Reach ratio x persistence/source ratio. See README.")
        x[2].metric("Hours to hardening reach", "n/a" if h.eta_reach_hours is None else ("reached" if h.eta_reach_hours == 0 else f"{h.eta_reach_hours:,.0f} h"),
                    help="Straight-line extrapolation of observed reach growth; not a prediction.")
        st.caption("A rumour left alone past a point is accepted as fact. "
                   f"Hardening point (configurable): {cfg().hardening.reach_point:,.0f} people reached, "
                   f"{cfg().hardening.persistence_hours:g} h in circulation, {cfg().hardening.sources_point}+ independent sources.")
    if not a.assessable or not h.assessable:
        verdict, kind = "Reach is unknown, so the two costs cannot be weighed. Obtain a reach estimate before relying on silence or on a denial.", "grey"
    elif a.risk in ("High", "Moderate") and h.level == "Low":
        verdict, kind = "Speaking would cost more than staying silent right now: a denial could widen the audience while the claim is nowhere near hardening.", "muted"
    elif a.risk == "Low" and h.level == "High":
        verdict, kind = "Silence now costs more than speaking: the claim is already widely seen and persistent, so a denial adds little exposure.", ""
    elif a.risk in ("High", "Moderate") and h.level == "High":
        verdict, kind = "Both options carry a high cost. This needs authorised human judgement on whether, how and when to respond.", ""
    elif h.level == "Elevated":
        verdict, kind = "The claim is moving toward the hardening point. Keep monitoring closely and prepare for review.", "grey"
    else:
        verdict, kind = "Neither option is costly at present; no public response is warranted.", "muted"
    callout("<b>Reading the balance:</b> " + verdict + " Both indicators are scenario estimates built on analyst assumptions.", kind)


# ============================================================================= PAGE 5
def fig_rps(score, thr: te.EffectiveThresholds) -> go.Figure:
    f = go.Figure()
    zones = [(0, thr.silence_max, "#E3EAF3", "Silence zone"), (thr.silence_max, thr.response_min, "#ECEEF1", "Monitor zone"),
             (thr.response_min, 1, "#BCC9DC", "Response-review zone")]
    for x0, x1, col, name in zones:
        f.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=col, line_width=0, layer="below")
        f.add_annotation(x=(x0 + x1) / 2, y=0.15, text=name, showarrow=False, font=dict(size=11, color="#4A5568"))
    if score is not None:
        f.add_shape(type="line", x0=score, x1=score, y0=0, y1=1, line=dict(color=NAVY, width=4))
        f.add_annotation(x=score, y=0.82, text=f"RPS {score:.2f}", showarrow=False, bgcolor=NAVY, font=dict(color="white", size=12),
                         xanchor="left" if score < 0.85 else "right", xshift=6 if score < 0.85 else -6)
    f.update_xaxes(range=[0, 1], title="Response Pressure Score (0 = none, 1 = maximum)", dtick=0.1)
    f.update_yaxes(visible=False, range=[0, 1])
    return style_fig(f, "Response Pressure Score versus thresholds", 210)


def config_editor():
    c = cfg()
    t, v, a, dcfg = c.threshold, c.velocity, c.amplification, c.data
    st.markdown("Thresholds and weights are **experimental** configuration values. They are not official Army policy.")
    with st.form("cfg_form"):
        st.markdown("**Response Pressure Score weights** (must sum to 1)")
        w = st.columns(4)
        w1 = w[0].number_input("w1 Velocity", 0.0, 1.0, float(t.w_velocity), 0.05, format="%.2f")
        w2 = w[1].number_input("w2 Growth", 0.0, 1.0, float(t.w_growth), 0.05, format="%.2f")
        w3 = w[2].number_input("w3 Acceleration", 0.0, 1.0, float(t.w_accel), 0.05, format="%.2f")
        w4 = w[3].number_input("w4 Reach", 0.0, 1.0, float(t.w_reach), 0.05, format="%.2f")
        auto = st.checkbox("Auto-normalise weights so they sum to 1", value=True)
        st.markdown("**Normalisation caps** (a value at or above the cap scores 1.0)")
        k = st.columns(4)
        vc = k[0].number_input("Velocity cap (mentions/h)", 1.0, 100000.0, float(t.velocity_cap))
        gc = k[1].number_input("Growth cap (%)", 1.0, 10000.0, float(t.growth_cap))
        ac = k[2].number_input("Acceleration cap (/h²)", 0.1, 10000.0, float(t.accel_cap))
        rc = k[3].number_input("Reach cap (people)", 100.0, 1e9, float(t.reach_cap), step=1000.0)
        st.markdown("**Decision thresholds (0-1 scale)**")
        k = st.columns(3)
        sm = k[0].number_input("Silence ceiling (RPS below = silence zone)", 0.0, 1.0, float(t.silence_max), 0.01, format="%.2f")
        rm = k[1].number_input("Response threshold (RPS at/above = response review)", 0.0, 1.0, float(t.response_min), 0.01, format="%.2f")
        rg = k[2].number_input("Reach-rise monitor trigger (% per interval)", 0.0, 1000.0, float(t.reach_growth_monitor_pct))
        st.markdown("**Velocity classification** (mentions per hour)")
        k = st.columns(5)
        lo = k[0].number_input("Low max", 0.0, 10000.0, float(v.low_max))
        hi = k[1].number_input("High min", 0.0, 10000.0, float(v.high_min))
        ar = k[2].number_input("Rapid accel min", 0.0, 10000.0, float(v.accel_rapid_min))
        ad = k[3].number_input("Decline accel max", -10000.0, 0.0, float(v.accel_decline_max))
        mw = k[4].number_input("Moving-avg window", 1, 20, int(v.moving_avg_window))
        st.markdown("**Amplification assumptions** (hypothetical)")
        k = st.columns(4)
        rx = k[0].number_input("Assumed response exposure (people)", 0.0, 1e9, float(a.response_exposure), step=1000.0)
        fl = k[1].number_input("Factor - low", 0.0, 5.0, float(a.factor_low), 0.05, format="%.2f")
        fm = k[2].number_input("Factor - medium", 0.0, 5.0, float(a.factor_medium), 0.05, format="%.2f")
        fh = k[3].number_input("Factor - high", 0.0, 5.0, float(a.factor_high), 0.05, format="%.2f")
        k = st.columns(3)
        rmod = k[0].number_input("Ratio for Moderate risk", 0.0, 1000.0, float(a.ratio_moderate_min), 0.05)
        rhigh = k[1].number_input("Ratio for High risk", 0.0, 1000.0, float(a.ratio_high_min), 0.05)
        scen = k[2].selectbox("Scenario used to classify risk", ["low", "medium", "high"], index=["low", "medium", "high"].index(a.classification_scenario))
        st.markdown("**Hardening point** (when a claim left alone is treated as accepted fact)")
        k = st.columns(4)
        hrp = k[0].number_input("Hardening reach point (people)", 100.0, 1e9, float(c.hardening.reach_point), step=1000.0)
        hph = k[1].number_input("Persistence saturates (hours)", 1.0, 1000.0, float(c.hardening.persistence_hours))
        hsp = k[2].number_input("Sources saturate at", 1, 1000, int(c.hardening.sources_point))
        hhm = k[3].number_input("High hardening index", 0.05, 1.0, float(c.hardening.high_min), 0.05, format="%.2f")
        st.markdown("**Data sufficiency**")
        k = st.columns(2)
        mo = k[0].number_input("Minimum valid observations", 2, 100, int(dcfg.min_observations))
        ms = k[1].number_input("Minimum observation span (hours)", 0.0, 1000.0, float(dcfg.min_span_hours))
        apply = st.form_submit_button("Apply configuration")
    if apply:
        new = c.copy()
        ws = {"velocity": w1, "growth": w2, "acceleration": w3, "reach": w4}
        errs = []
        if abs(sum(ws.values()) - 1) > 1e-6:
            if auto and sum(ws.values()) > 0:
                ws = te.normalise_weights(ws)
            else:
                errs.append(f"Weights sum to {sum(ws.values()):.3f}; they must sum to 1 (or tick auto-normalise).")
        if sm >= rm:
            errs.append("Silence ceiling must be lower than the response threshold.")
        if lo >= hi:
            errs.append("Velocity 'Low max' must be lower than 'High min'.")
        if not (fl <= fm <= fh):
            errs.append("Amplification factors must satisfy low <= medium <= high.")
        if rmod >= rhigh:
            errs.append("Moderate-risk ratio must be lower than the High-risk ratio.")
        if errs:
            for e in errs:
                st.error(e)
        else:
            nt = new.threshold
            nt.w_velocity, nt.w_growth, nt.w_accel, nt.w_reach = ws["velocity"], ws["growth"], ws["acceleration"], ws["reach"]
            nt.velocity_cap, nt.growth_cap, nt.accel_cap, nt.reach_cap = vc, gc, ac, rc
            nt.silence_max, nt.response_min, nt.reach_growth_monitor_pct = sm, rm, rg
            new.velocity.low_max, new.velocity.high_min, new.velocity.accel_rapid_min = lo, hi, ar
            new.velocity.accel_decline_max, new.velocity.moving_avg_window = ad, int(mw)
            na = new.amplification
            na.response_exposure, na.factor_low, na.factor_medium, na.factor_high = rx, fl, fm, fh
            na.ratio_moderate_min, na.ratio_high_min, na.classification_scenario = rmod, rhigh, scen
            new.hardening.reach_point, new.hardening.persistence_hours = hrp, hph
            new.hardening.sources_point, new.hardening.high_min = int(hsp), hhm
            new.hardening.elevated_min = min(new.hardening.elevated_min, hhm - 0.05)
            new.data.min_observations, new.data.min_span_hours = int(mo), ms
            st.session_state["cfg"] = new
            flash("Configuration applied. All pages now use the new values.")
            st.rerun()
    if st.button("Reset all configuration to defaults"):
        st.session_state["cfg"] = default_config()
        st.rerun()


def page_decision():
    header("Response Threshold and Decision", "Advisory recommendation with full reasoning. The tool never publishes any statement.")
    cid = need_claim()
    if not cid:
        return
    b = get_bundle(cid)
    d = b.decision
    st.markdown(f"**{cid}**  -  {b.ctx.description}")
    validation_banner(b)
    source_refs = int(b.obs["source_reference"].notna().sum())
    assessed_sources = int((b.obs["source_reliability"] != "Unassessed").sum())
    evidence_cols = st.columns(4)
    evidence_cols[0].metric("Valid observations", f"{len(b.obs)} / {b.validation.rows_in}")
    evidence_cols[1].metric("Public references", f"{source_refs} / {len(b.obs)}")
    evidence_cols[2].metric("Reliability assessed", f"{assessed_sources} / {len(b.obs)}")
    evidence_cols[3].metric("Reach confidence", b.reach.confidence)
    st.caption(
        "Evidence coverage is not a probability that the claim is true. Source reliability is an analyst assessment; "
        "the tool does not verify references or source credibility."
    )

    top = st.columns([3, 1])
    with top[0]:
        st.markdown(rec_badge(d.recommendation, d.escalation), unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:8px'>{d.headline}</div>", unsafe_allow_html=True)
    if top[1].button("Save assessment to history", key="save_dec"):
        get_db().save_assessment(b)
        st.success("Assessment saved.")

    callout(f"<b>Human review status:</b> {d.human_review_status}", "" if d.recommendation == REC_REVIEW else "grey")
    if d.escalation:
        callout("<b>Mandatory escalation (safeguard)</b><br>" + "<br>".join("• " + r for r in d.escalation_reasons) +
                "<br>Escalation applies regardless of the numerical score or amplification risk.")

    if d.thresholds is not None:
        mm = st.columns(4)
        mm[0].metric("Response Pressure Score", fnum(d.rps_score, 3))
        mm[1].metric("Silence ceiling", f"{d.thresholds.silence_max:.2f}", help=f"Base {d.thresholds.base_silence_max:.2f}")
        mm[2].metric("Response threshold", f"{d.thresholds.response_min:.2f}", help=f"Base {d.thresholds.base_response_min:.2f}")
        mm[3].metric("Position", d.band)
        show_fig(fig_rps(d.rps_score, d.thresholds), key="rps_fig")
        if d.rps_score is not None:
            threshold_cases = te.threshold_sensitivity(d.rps_score, d.thresholds)
            bands = threshold_cases["band"].drop_duplicates().tolist()
            message = (
                f"An illustrative ±0.10 shift to both thresholds keeps the score in {bands[0]}."
                if len(bands) == 1 else
                "An illustrative ±0.10 threshold shift moves this score across decision bands: "
                + ", ".join(bands) + "."
            )
            st.info("Quick robustness check: " + message)
            st.caption("This varies thresholds only, not the underlying evidence. It is not a probability or final recommendation; safeguards still apply.")

    balance_panel(b)

    tabs = st.tabs(["Explanation", "Silence justification", "Score components", "Reassessment", "Sensitivity analysis", "Configuration"])

    with tabs[0]:
        st.markdown("**Reasoning (rule trace)**")
        for r in d.rationale:
            st.markdown("- " + r)
        if d.blockers:
            st.markdown("**Why silence is not recommended for this claim**")
            for x in d.blockers:
                st.markdown("- " + x)
        st.markdown("**Note on response type**")
        st.markdown(d.response_note)
        if d.recommendation == REC_MONITOR:
            st.markdown("**Why monitor rather than respond**: the evidence or the circulation does not yet justify a public statement, "
                        "and a premature denial carries amplification risk. Reassessment conditions are on the Reassessment tab.")
        if d.recommendation == REC_REVIEW and not d.escalation:
            st.markdown("**Why a response review**: circulation pressure is at the response threshold. Leaving it unanswered risks the claim "
                        "being accepted as fact. Amplification risk is a design input for the human reviewers, not a reason to suppress review.")

    with tabs[1]:
        sj = d.silence_justification
        if sj is None:
            st.info(f"The current recommendation is **{d.recommendation}**, so no silence justification is generated. "
                    f"A silence justification is produced automatically whenever the recommendation is REMAIN SILENT.")
            st.markdown("**What would be required for REMAIN SILENT**")
            if d.thresholds is not None:
                st.markdown(f"- Response Pressure Score below the silence ceiling ({d.thresholds.silence_max:.2f}); currently {fnum(d.rps_score, 3)}.")
            st.markdown("- Reach data available, no sharply rising reach, no urgent impact or safeguard flag, claim not verified true.")
        else:
            callout(f"<b>Silence justification</b><br>{sj.summary}", "muted")
            for i, (title, body) in enumerate(sj.sections, 1):
                with st.expander(f"{i}. {title}", expanded=i in (1, 2, 6, 8)):
                    st.markdown(body)
            callout(f"<i>{PRINCIPLE}</i>", "muted")

    with tabs[2]:
        if d.rps is None or d.rps.score is None:
            st.warning("Response Pressure Score could not be computed (insufficient data).")
        else:
            rows = []
            for k, c in d.rps.components.items():
                rows.append({"Component": k.title(), "Raw value": c["raw"], "Cap": c["cap"], "Normalised (0-1)": c["normalised"],
                             "Weight": c["weight"], "Effective weight": c["effective_weight"], "Contribution": c["contribution"],
                             "What it means": te.EXPLANATIONS[k]})
            show_df(pd.DataFrame(rows).round(4))
            st.markdown(f"**RPS = Σ weight × normalised value = {d.rps.score:.4f}**")
            if d.rps.note:
                st.warning(d.rps.note)
            st.markdown("**Threshold adjustments applied** (context shifts the thresholds, never the score)")
            if d.thresholds.adjustments:
                for r, ds, dr in d.thresholds.adjustments:
                    st.markdown(f"- {r}  (silence {ds:+.2f}, response {dr:+.2f})")
            else:
                st.markdown("- None.")

    with tabs[3]:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Reassessment conditions**")
            for x in d.reassessment_conditions:
                st.markdown("- " + x)
        with r2:
            st.markdown("**What continues to be monitored**")
            for x in d.monitoring_items:
                st.markdown("- " + x)
        claim = get_db().get_claim(cid)
        stored_due = pd.to_datetime(claim["review_due_at"], errors="coerce")
        suggested_due = datetime.now().replace(second=0, microsecond=0) + timedelta(hours=4)
        if pd.notna(stored_due):
            suggested_due = stored_due.to_pydatetime()
            st.info(f"Next review is scheduled for {suggested_due:%Y-%m-%d %H:%M}.")
            if suggested_due < datetime.now():
                st.warning("The scheduled review time has passed.")
        st.caption("Set an analyst-owned review time. The 4-hour default is only an editable convenience, not a policy or automatic alert.")
        s1, s2 = st.columns(2)
        review_date = s1.date_input("Next review date", value=suggested_due.date(), key=f"review_date_{cid}")
        review_time = s2.time_input("Next review time", value=suggested_due.time(), key=f"review_time_{cid}")
        rbtn1, rbtn2 = st.columns(2)
        if rbtn1.button("Save review schedule", key=f"save_review_{cid}"):
            get_db().set_review_due(cid, datetime.combine(review_date, review_time))
            st.success("Next review time saved.")
            st.rerun()
        if pd.notna(stored_due) and rbtn2.button("Clear review schedule", key=f"clear_review_{cid}"):
            get_db().set_review_due(cid, None)
            st.success("Review schedule cleared.")
            st.rerun()

    with tabs[4]:
        if d.rps is None or d.rps.score is None:
            st.info("Sensitivity analysis needs a computed score.")
        else:
            thr = d.thresholds
            st.markdown("**Input sensitivity: one measured component at a time**")
            st.caption(
                "Explore how an analyst-selected percentage change to the current inputs changes the RPS band. "
                "This is a scenario, not an error bar or probability; decision safeguards and blockers still apply."
            )
            variation = st.slider("Input variation (%)", 0, 100, 25, 5, key=f"input_variation_{cid}")
            base_values = {
                "Velocity": b.rps_inputs.velocity,
                "Growth": b.rps_inputs.growth,
                "Acceleration": b.rps_inputs.acceleration,
                "Reach": b.rps_inputs.reach,
            }
            input_rows = []
            for label, field in (("Velocity", "velocity"), ("Growth", "growth"),
                                 ("Acceleration", "acceleration"), ("Reach", "reach")):
                base_value = base_values[label]
                if base_value is None or pd.isna(base_value):
                    input_rows.append({"Input": label, "Change": "n/a", "Value": None, "RPS": None, "Band": "Unavailable"})
                    continue
                for delta in (-variation, 0, variation):
                    scaled = dict(velocity=b.rps_inputs.velocity, growth=b.rps_inputs.growth,
                                  acceleration=b.rps_inputs.acceleration, reach=b.rps_inputs.reach)
                    scaled[field] = float(base_value) * (1 + delta / 100)
                    result = te.compute_rps(te.RPSInputs(**scaled), cfg().threshold)
                    input_rows.append({
                        "Input": label,
                        "Change": f"{delta:+d}%" if delta else "Baseline",
                        "Value": scaled[field],
                        "RPS": result.score,
                        "Band": te.band(result.score, thr),
                    })
            show_df(pd.DataFrame(input_rows).round(3))
            ws = te.weight_sensitivity(b.rps_inputs, cfg().threshold, thr)
            f = go.Figure()
            shades = [NAVY, BLUE, GREY, "#9DB1C9"]
            for col, (k, g) in zip(shades, ws.groupby("weight_varied", sort=False)):
                f.add_scatter(x=g["weight_value"], y=g["rps"], mode="lines+markers", name=f"vary {k}", line=dict(color=col))
            f.add_hline(y=thr.silence_max, line_dash="dot", line_color=GREY, annotation_text="Silence ceiling")
            f.add_hline(y=thr.response_min, line_dash="dash", line_color=NAVY, annotation_text="Response threshold")
            f.update_xaxes(title="Value of the varied weight (others rescaled proportionally)")
            f.update_yaxes(title="RPS", range=[0, 1])
            show_fig(style_fig(f, "One-way weight sensitivity"), key="sens_w")
            changed = ws["band"].nunique() > 1
            st.markdown("Recommendation band is **" + ("sensitive" if changed else "stable") +
                        "** to ±0.15 changes in any single weight.")
            with st.expander("Weight sensitivity table"):
                show_df(ws.round(4))
            st.markdown("**Threshold sensitivity**")
            show_df(te.threshold_sensitivity(d.rps_score, thr).round(3))
            st.markdown("**Comparison of threshold settings (presets)**")
            cmp_ = te.compare_presets(b.rps_inputs, cfg().threshold, b.amp.risk if b.amp.assessable else "Unknown",
                                      b.ctx.public_impact, b.ctx.verification_status)
            show_df(cmp_.round(3))
            st.caption("Presets are illustrative experiments. Safeguard escalations are unaffected by any setting.")

    with tabs[5]:
        st.markdown("**Explanation of each variable**")
        for k, e in te.EXPLANATIONS.items():
            st.markdown(f"- {e}")
        config_editor()


# ============================================================================= PAGE 6
def page_whatif():
    header("What-If Simulation", "Remain silent versus issue a public denial. Hypothetical scenarios only.")
    st.error(HYPOTHETICAL_LABEL)
    cid = need_claim()
    if not cid:
        return
    b = get_bundle(cid)
    a_cfg = cfg().amplification
    st.markdown(f"**{cid}**  -  {b.ctx.description}")
    suggested = suggest_growth_pct_per_hour(b.reach.distribution)
    default_reach = float(b.reach.current_reach) if b.reach.available else 1000.0
    if not b.reach.available:
        st.warning("INSUFFICIENT REACH DATA for this claim: the reach below is a purely hypothetical analyst input, not an observation.")
    k = f"{cid}"
    c1, c2, c3 = st.columns(3)
    reach0 = c1.number_input("Current reach (people)", 0.0, 1e9, default_reach, step=100.0, key=f"wi_r_{k}")
    resp = c2.number_input("Assumed additional exposure from a response (people)", 0.0, 1e9, float(a_cfg.response_exposure), step=1000.0, key=f"wi_e_{k}")
    horizon = c3.slider("Observation period (hours)", 6, 336, 72, key=f"wi_h_{k}")
    f1, f2, f3 = st.columns(3)
    fl = f1.slider("Amplification factor - low", 0.0, 2.0, float(a_cfg.factor_low), 0.05, key=f"wi_fl_{k}")
    fm = f2.slider("Amplification factor - medium", 0.0, 2.0, float(a_cfg.factor_medium), 0.05, key=f"wi_fm_{k}")
    fh = f3.slider("Amplification factor - high", 0.0, 2.0, float(a_cfg.factor_high), 0.05, key=f"wi_fh_{k}")
    g_def = float(np.clip(suggested, 0.0, 20.0)) if suggested is not None else 1.0
    g1, g2 = st.columns(2)
    g_sil = g1.slider("Reach growth per hour if silent (%)", -10.0, 20.0, round(g_def, 2), 0.1, key=f"wi_gs_{k}",
                      help="Default is the compound hourly growth observed in the reach estimates, if available." if suggested is not None
                      else "No observed reach growth available; default is a neutral assumption.")
    g_res = g2.slider("Reach growth per hour after a response (%)", -10.0, 20.0, round(g_def, 2), 0.1, key=f"wi_gr_{k}",
                      help="Lower this to explore a denial that slows spread; keep equal to model pure amplification.")
    if suggested is not None:
        st.caption(f"Observed compound reach growth in the data: {suggested:.2f}% per hour (used as the default).")

    hp = cfg().hardening.reach_point
    sim = simulate(reach0, resp, {"low": min(fl, fm), "medium": fm, "high": max(fh, fm)}, horizon, g_sil, g_res, hardening_reach=hp)
    tl, sm = sim.timeline, sim.summary
    med = sm[sm["scenario"].str.contains("medium")].iloc[0]
    m = st.columns(4)
    m[0].metric("Reach at end - remain silent", f"{sm.iloc[0]['reach at end']:,.0f}")
    m[1].metric("Reach at end - denial (medium)", f"{med['reach at end']:,.0f}")
    m[2].metric("Difference in exposure", f"{med['difference vs silence']:+,.0f}")
    m[3].metric("Immediate additional exposure (medium)", f"{med['additional exposure at t=0']:,.0f}")

    f = go.Figure()
    f.add_scatter(x=tl["hours"], y=tl["Remain silent"], name="A: Remain silent", line=dict(color=NAVY, width=3))
    for col, name in zip(["#9DB1C9", BLUE, "#2F4B73"], ["low", "medium", "high"]):
        f.add_scatter(x=tl["hours"], y=tl[f"B: Public denial ({name})"], name=f"B: Public denial ({name})",
                      line=dict(color=col, width=2, dash="dot" if name != "medium" else "solid"))
    f.add_hline(y=hp, line_dash="dash", line_color=GREY, annotation_text=f"Hardening point {hp:,.0f}")
    f.update_xaxes(title="Hours from now")
    f.update_yaxes(title="Estimated reach (hypothetical)")
    show_fig(style_fig(f, "Estimated reach: remain silent versus public denial (HYPOTHETICAL)", 400), key="wi_line")
    hs, hm = sm.iloc[0]["hours to hardening point"], med["hours to hardening point"]
    fmt = lambda x: "not reached in period" if x is None or pd.isna(x) else ("already above" if x == 0 else f"{x:,.0f} h")
    callout(f"<b>Hardening point ({hp:,.0f} people):</b> remaining silent crosses it: <b>{fmt(hs)}</b>; after a public denial (medium): <b>{fmt(hm)}</b>. "
            f"This sets the cost of silence against the cost of speaking. Hypothetical.", "muted")
    fb = go.Figure(go.Bar(x=sm["scenario"], y=sm["reach at end"], marker_color=[NAVY, "#9DB1C9", BLUE, "#2F4B73"],
                          text=[f"{x:,.0f}" for x in sm["reach at end"]], textposition="outside"))
    fb.update_yaxes(title="Reach at end of period")
    show_fig(style_fig(fb, "Exposure comparison at the end of the observation period (HYPOTHETICAL)"), key="wi_bar")
    show_df(sm.round(1))
    st.caption(f"{sim.label} Differences shown are consequences of the assumptions above, not forecasts. "
               f"Growth is compounded hourly and capped at an assumed audience ceiling of 100 million.")


# ============================================================================= PAGE 7
def page_history():
    header("Assessment History and Reports", "Stored assessments, how recommendations changed over time, and downloadable reports.")
    db = get_db()
    allh = db.get_assessments()
    st.markdown("#### Latest stored assessment per claim")
    if allh.empty:
        st.info("No assessments stored yet. Save one from the Decision page or from the Executive Overview.")
    else:
        latest = allh.sort_values("assessment_id").groupby("claim_id").tail(1)
        show_df(latest[["claim_id", "assessed_at", "velocity", "growth", "acceleration", "reach", "amplification_risk",
                        "response_pressure_score", "recommendation", "escalation"]].round(3))

    cid = need_claim()
    if not cid:
        return
    st.divider()
    st.markdown(f"#### Claim {cid}")
    if not hasattr(db, "get_audit_events"):
        get_db.clear()
        db = get_db()
    activity = db.get_audit_events(cid)
    st.markdown("**Claim activity audit**")
    if activity.empty:
        st.caption("No activity events recorded yet.")
    else:
        show_df(activity[["occurred_at", "event_type", "record_type", "record_id",
                         "field_names", "detail"]])
    h = db.get_assessments(cid)
    dh = db.get_decision_history(cid)
    if h.empty:
        st.info("No stored assessments for this claim yet.")
        if st.button("Save current assessment now"):
            db.save_assessment(get_bundle(cid))
            st.rerun()
    else:
        show_df(h[["assessment_id", "assessed_at", "observations_used", "velocity", "growth", "acceleration", "reach",
                   "amplification_risk", "response_pressure_score", "recommendation", "lifecycle_stage"]].round(3))
        if len(h) >= 1:
            hp = h.dropna(subset=["response_pressure_score"])
            if len(hp):
                f = go.Figure()
                f.add_scatter(x=hp["assessed_at"], y=hp["response_pressure_score"], mode="lines+markers+text",
                              text=hp["recommendation"].map(lambda x: x.split(" /")[0].replace("RECOMMEND ", "")), textposition="top center",
                              line=dict(color=NAVY), name="RPS")
                f.update_yaxes(title="RPS", range=[0, 1.05])
                show_fig(style_fig(f, "Response Pressure Score across stored assessments"), key="h_rps")
        st.markdown("**Recommendation changes (decision history)**")
        show_df(dh[["decision_timestamp", "previous_recommendation", "new_recommendation", "reason_for_change"]])
        if len(h) >= 2:
            st.markdown("**Compare two assessments**")
            ids = list(h["assessment_id"])
            x1, x2 = st.columns(2)
            a1 = x1.selectbox("Earlier assessment", ids, index=0)
            a2 = x2.selectbox("Later assessment", ids, index=len(ids) - 1)
            r1, r2 = h[h["assessment_id"] == a1].iloc[0], h[h["assessment_id"] == a2].iloc[0]
            rows = []
            for col in ["velocity", "growth", "acceleration", "reach", "response_pressure_score"]:
                v1, v2 = r1[col], r2[col]
                rows.append({"Measure": col, "Earlier": v1, "Later": v2,
                             "Change": (v2 - v1) if pd.notna(v1) and pd.notna(v2) else None})
            for col in ["amplification_risk", "trend_label", "lifecycle_stage", "recommendation"]:
                rows.append({"Measure": col, "Earlier": r1[col], "Later": r2[col], "Change": "changed" if r1[col] != r2[col] else "unchanged"})
            show_df(pd.DataFrame(rows).astype({"Earlier": str, "Later": str, "Change": str}))
        with st.expander("View stored justification text for an assessment"):
            pick = st.selectbox("Assessment", list(h["assessment_id"]), index=len(h) - 1, key="just_pick")
            st.text(h[h["assessment_id"] == pick].iloc[0]["justification"])

    st.divider()
    st.markdown("#### PDF report")
    b = get_bundle(cid)
    review_due = db.get_claim(cid)["review_due_at"]
    try:
        pdf = build_pdf_report(b, h if not h.empty else None, review_due)
        st.download_button("Download PDF assessment report", pdf, f"assessment_{cid}.pdf", "application/pdf")
    except Exception as ex:  # report must never crash the page
        st.error(f"PDF generation failed: {ex}")
    with st.expander("Database integrity check"):
        rep = db.integrity_report()
        st.json(rep)
        st.caption(f"Database file: {DB_PATH}")


# ============================================================================= MAIN
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    db = get_db()
    st.sidebar.markdown(f"### {APP_TITLE}")
    st.sidebar.caption(APP_SUBTITLE)
    page = st.sidebar.radio("Navigation", PAGES, key="page")
    claims = db.get_claims()
    if len(claims):
        ids = list(claims["claim_id"])
        labels = {r["claim_id"]: claim_label(r) for _, r in claims.iterrows()}
        cur = st.session_state.get("claim_id")
        idx = ids.index(cur) if cur in ids else 0
        st.session_state["claim_id"] = st.sidebar.selectbox("Active claim", ids, index=idx, format_func=lambda i: labels[i])
        selected = db.get_claim(st.session_state["claim_id"])
        if selected["review_due_at"]:
            due = pd.to_datetime(selected["review_due_at"])
            st.sidebar.caption(f"Next review: {due:%Y-%m-%d %H:%M}")
            if due < pd.Timestamp.now():
                st.sidebar.warning("Review is overdue.")
        else:
            st.sidebar.caption("Next review: not scheduled")
    else:
        st.session_state.pop("claim_id", None)
        st.sidebar.info("No claims stored.")
    st.sidebar.divider()
    st.sidebar.caption(DISCLAIMER)

    show_flash()
    {PAGES[0]: page_overview, PAGES[1]: page_input, PAGES[2]: page_velocity, PAGES[3]: page_amplification,
     PAGES[4]: page_decision, PAGES[5]: page_whatif, PAGES[6]: page_history}[page]()


main()
