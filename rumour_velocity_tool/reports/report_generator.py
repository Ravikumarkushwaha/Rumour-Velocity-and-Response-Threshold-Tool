"""PDF assessment report (ReportLab) and CSV export helpers."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from config import APP_TITLE, DISCLAIMER, REC_SILENT

NAVY = colors.HexColor("#14284B")
MUTED = colors.HexColor("#5B7A9D")
GREY = colors.HexColor("#6B7280")
LIGHT = colors.HexColor("#EEF2F7")


def _safe(text) -> str:
    """Escape for ReportLab Paragraph and keep to Latin-1 (built-in fonts)."""
    s = "" if text is None else str(text)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return escape(s).replace("\n", "<br/>")


def _f(x, d=2, suffix="") -> str:
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:,.{d}f}{suffix}"


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=19, leading=23, alignment=0, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=ss["Normal"], textColor=GREY, fontSize=9, leading=12),
        "h": ParagraphStyle("h", parent=ss["Heading2"], textColor=NAVY, fontSize=12.5, spaceBefore=10, spaceAfter=4),
        "b": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.2, leading=12.5),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=7.8, leading=10, textColor=GREY),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontSize=8.8, leading=11),
        "cellb": ParagraphStyle("cb", parent=ss["Normal"], fontSize=8.8, leading=11, textColor=NAVY, fontName="Helvetica-Bold"),
        "rec": ParagraphStyle("r", parent=ss["Normal"], fontSize=13, leading=16, textColor=colors.white, fontName="Helvetica-Bold"),
    }


def _table(rows, col_widths, header=True):
    st = _styles()
    data = [[Paragraph(_safe(c), st["cellb"] if (header and i == 0) or j == 0 and not header else st["cell"])
             for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D3E0")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), LIGHT))
    else:
        style.append(("BACKGROUND", (0, 0), (0, -1), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def _line_chart(xs, ys, title, ylabel, width=250, height=120):
    d = Drawing(width, height + 18)
    d.add(String(0, height + 6, title, fontSize=8.5, fillColor=NAVY, fontName="Helvetica-Bold"))
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if pd.notna(x) and pd.notna(y)]
    if len(pts) < 2:
        d.add(String(10, height / 2, "Not enough points to chart", fontSize=8, fillColor=GREY))
        return d
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 34, 18, width - 44, height - 18
    lp.data = [pts]
    lp.lines[0].strokeColor = NAVY
    lp.lines[0].strokeWidth = 1.4
    lp.lines[0].symbol = None
    lp.xValueAxis.labels.fontSize = 6.5
    lp.yValueAxis.labels.fontSize = 6.5
    lp.xValueAxis.visibleGrid = False
    lp.yValueAxis.visibleGrid = True
    lp.yValueAxis.gridStrokeColor = colors.HexColor("#DDE4EE")
    lp.yValueAxis.valueMin = min(0.0, min(p[1] for p in pts))
    d.add(lp)
    d.add(String(34, 4, "hours since first observation", fontSize=6.5, fillColor=GREY))
    return d


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 10 * mm, "Synthetic demonstration data | Advisory only | Not official Army policy")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_report(bundle, history: Optional[pd.DataFrame] = None,
                     review_due_at: Optional[str] = None) -> bytes:
    """Return the assessment report for an AssessmentBundle as PDF bytes."""
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=17 * mm, title=f"Assessment {bundle.ctx.claim_id}", author="Rumour Velocity Tool")
    W = A4[0] - 36 * mm
    ctx, tr, rc, amp, dec, lc = bundle.ctx, bundle.trend, bundle.reach, bundle.amp, bundle.decision, bundle.lifecycle
    obs = bundle.obs
    story = []

    story += [Paragraph(_safe(APP_TITLE), st["title"]),
              Paragraph("Assessment Report", ParagraphStyle("x", parent=st["sub"], fontSize=11, textColor=MUTED)),
              Spacer(1, 4)]
    period = "n/a" if len(obs) == 0 else f"{obs['timestamp'].iloc[0]:%Y-%m-%d %H:%M} to {obs['timestamp'].iloc[-1]:%Y-%m-%d %H:%M} ({tr.window_hours:.1f} h, {tr.n_observations} observations)"
    story.append(_table([
        ["Claim ID", ctx.claim_id], ["Claim category", ctx.category], ["Claim description", ctx.description],
        ["Verification status", ctx.verification_status], ["Public impact (analyst-rated)", ctx.public_impact],
        ["Assessment date", bundle.assessed_at], ["Observation period", period],
        ["Next review scheduled", review_due_at or "Not scheduled"]], [48 * mm, W - 48 * mm], header=False))
    story.append(Spacer(1, 8))
    ref_count = int(obs["source_reference"].notna().sum()) if "source_reference" in obs else 0
    assessed_count = int((obs["source_reliability"] != "Unassessed").sum()) if "source_reliability" in obs else 0
    story.append(Paragraph("Evidence coverage (not a probability of truth)", st["h"]))
    story.append(Paragraph(
        _safe(f"{len(obs)} valid observation(s); {ref_count} with public source references; "
              f"{assessed_count} with analyst-assessed source reliability. Reach confidence: {rc.confidence}. "
              "The tool does not verify references or source credibility."),
        st["b"]))
    if ref_count:
        evidence_rows = [["Observed at", "Public source reference / note", "Reliability (analyst-rated)"]]
        referenced = obs[obs["source_reference"].notna()].tail(10)
        for _, observation in referenced.iterrows():
            evidence_rows.append([
                str(observation["timestamp"]),
                str(observation["source_reference"]),
                str(observation.get("source_reliability", "Unassessed")),
            ])
        story.append(_table(evidence_rows, [36 * mm, W - 73 * mm, 37 * mm]))
        if ref_count > len(referenced):
            story.append(Paragraph(f"Showing the latest {len(referenced)} of {ref_count} referenced observations.", st["small"]))
    story.append(Spacer(1, 5))

    rec_tbl = Table([[Paragraph(_safe("RECOMMENDATION: " + dec.recommendation), st["rec"])]], colWidths=[W])
    rec_tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                 ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story += [rec_tbl, Spacer(1, 3), Paragraph(_safe(dec.headline), st["b"]),
              Paragraph("<b>Human review status:</b> " + _safe(dec.human_review_status), st["b"])]
    if dec.escalation:
        story.append(Paragraph("<b>Mandatory escalation reasons:</b> " + _safe(" ".join(dec.escalation_reasons)), st["b"]))
    story.append(Paragraph("<b>Note on response type:</b> " + _safe(dec.response_note), st["b"]))

    # --- velocity -----------------------------------------------------------------
    story.append(Paragraph("1. Velocity, growth and acceleration", st["h"]))
    story.append(_table([
        ["Metric", "Value", "Meaning"],
        ["Cumulative mentions (observed)", _f(tr.total_mentions, 0), "Total so far - not a rate of spread"],
        ["Current velocity V(t)", _f(tr.current_velocity, 2, " /h"), "Additional mentions per hour"],
        ["Previous velocity V(t-1)", _f(tr.previous_velocity, 2, " /h"), "Previous interval"],
        ["Percentage growth G(t)", _f(tr.growth_pct, 1, " %"), "Latest interval vs previous count"],
        ["Acceleration A(t)", _f(tr.acceleration, 2, " /h^2"), "Change of velocity per hour"],
        ["Moving average velocity", _f(tr.moving_avg, 2, " /h"), "Smooths temporary fluctuation"],
        ["Trend classification", tr.label, "Configurable velocity thresholds"],
        ["Lifecycle stage", lc.stage, lc.rationale]], [55 * mm, 32 * mm, W - 87 * mm]))
    story.append(Spacer(1, 5))
    s = bundle.series
    if len(s) >= 2:
        t0 = s["timestamp"].iloc[0]
        hrs = [(t - t0).total_seconds() / 3600 for t in s["timestamp"]]
        ch = Table([[_line_chart(hrs, s["mentions"], "Cumulative mentions", "mentions", W / 2 - 4, 115),
                     _line_chart(hrs, s["velocity"], "Velocity (mentions/hour)", "mentions/h", W / 2 - 4, 115)]],
                   colWidths=[W / 2, W / 2])
        story.append(ch)
    story.append(Paragraph(_safe(lc.disclaimer), st["small"]))

    # --- reach ----------------------------------------------------------------------
    story.append(Paragraph("2. Public reach assessment", st["h"]))
    reach_txt = "INSUFFICIENT REACH DATA" if not rc.available else f"{rc.current_reach:,.0f} people ({rc.basis} estimate)"
    story.append(_table([
        ["Item", "Value"], ["Estimated current reach", reach_txt],
        ["Reach growth over observation window", _f(rc.reach_growth_pct_total, 1, " %")],
        ["Reach growth, latest interval", _f(rc.last_interval_growth_pct, 1, " %")],
        ["Independent sources", f"{rc.independent_sources if rc.independent_sources is not None else 'n/a'} ({rc.sources_basis})"],
        ["Engagement (not unique audience)", _f(rc.total_engagement, 0)], ["Data confidence", rc.confidence]],
        [70 * mm, W - 70 * mm]))
    for n in rc.notes:
        story.append(Paragraph("- " + _safe(n), st["small"]))

    # --- amplification ----------------------------------------------------------------
    story.append(Paragraph("3. Amplification risk (hypothetical scenarios)", st["h"]))
    if amp.assessable:
        rows = [["Scenario", "Factor", "Potential additional exposure", "Projected reach", "Ratio", "Risk"]]
        for k in ("low", "medium", "high"):
            x = amp.scenarios[k]
            rows.append([k.title(), f"{x['factor']:.2f}", f"{x['additional']:,.0f}", f"{x['projected']:,.0f}",
                         f"{x['ratio']:.2f}", x["risk"]])
        story.append(_table(rows, [22 * mm, 16 * mm, 50 * mm, 36 * mm, 18 * mm, W - 142 * mm]))
        story.append(Paragraph(f"<b>Classified risk:</b> {_safe(amp.risk)} ({_safe(amp.classification_scenario)} scenario).", st["b"]))
    else:
        story.append(Paragraph("Amplification risk could not be assessed because reach is unavailable.", st["b"]))
    for a in amp.assumptions:
        story.append(Paragraph("- Assumption: " + _safe(a), st["small"]))
    for l in amp.limitations:
        story.append(Paragraph("- Limitation: " + _safe(l), st["small"]))

    # --- balance of risks -------------------------------------------------------------
    hd = bundle.hardening
    story.append(Paragraph("3b. Cost of silence (hardening risk)", st["h"]))
    story.append(_table([
        ["Item", "Value"],
        ["Hardening risk", hd.level + ("" if hd.index is None else f" (index {hd.index:.2f})")],
        ["Time in circulation", f"{hd.hours_in_circulation:.1f} h"],
        ["Hours to hardening reach point (extrapolated)", "n/a" if hd.eta_reach_hours is None else ("already reached" if hd.eta_reach_hours == 0 else f"{hd.eta_reach_hours:,.0f} h")],
        ["Cost of denial (amplification risk)", amp.risk]], [78 * mm, W - 78 * mm]))
    for n_ in hd.notes:
        story.append(Paragraph("- " + _safe(n_), st["small"]))

    # --- threshold ------------------------------------------------------------------------
    story.append(Paragraph("4. Response threshold", st["h"]))
    rps = dec.rps
    if rps is not None and rps.score is not None:
        rows = [["Component", "Raw value", "Normalised (0-1)", "Weight", "Effective weight", "Contribution"]]
        for k, c in rps.components.items():
            rows.append([k.title(), _f(c["raw"], 2), _f(c["normalised"], 3), f"{c['weight']:.2f}",
                         f"{c['effective_weight']:.2f}", _f(c["contribution"], 3)])
        rows.append(["Response Pressure Score", "", "", "", "", _f(rps.score, 3)])
        story.append(_table(rows, [38 * mm, 24 * mm, 30 * mm, 18 * mm, 28 * mm, W - 138 * mm]))
        t = dec.thresholds
        story.append(Paragraph(f"<b>Thresholds applied:</b> silence ceiling {t.silence_max:.2f} (base {t.base_silence_max:.2f}); "
                               f"response threshold {t.response_min:.2f} (base {t.base_response_min:.2f}). "
                               f"<b>Position:</b> {_safe(dec.band)}.", st["b"]))
        for r, ds, dr in t.adjustments:
            story.append(Paragraph(f"- {_safe(r)} (silence {ds:+.2f}, response {dr:+.2f})", st["small"]))
        if rps.note:
            story.append(Paragraph("- " + _safe(rps.note), st["small"]))
    else:
        story.append(Paragraph("Response Pressure Score could not be computed (insufficient data).", st["b"]))
    story.append(Paragraph("Thresholds are experimental, configurable assumptions and are not official Army policy.", st["small"]))

    # --- reasoning --------------------------------------------------------------------------
    story.append(Paragraph("5. Detailed reasoning", st["h"]))
    for r in dec.rationale:
        story.append(Paragraph("- " + _safe(r), st["b"]))
    if dec.blockers:
        story.append(Paragraph("<b>Conditions preventing a silence recommendation:</b> " + _safe(" ".join(dec.blockers)), st["b"]))

    if dec.recommendation == REC_SILENT and dec.silence_justification is not None:
        sj = dec.silence_justification
        story.append(Paragraph("6. Silence justification", st["h"]))
        story.append(Paragraph(_safe(sj.summary), st["b"]))
        story.append(Spacer(1, 3))
        for i, (title, body) in enumerate(sj.sections, 1):
            story.append(KeepTogether([Paragraph(f"<b>{i}. {_safe(title)}</b>", st["b"]), Paragraph(_safe(body), st["b"]), Spacer(1, 3)]))
        story.append(Paragraph("<i>" + _safe(sj.principle) + "</i>", st["b"]))

    n = 7 if dec.recommendation == REC_SILENT else 6
    story.append(Paragraph(f"{n}. Reassessment conditions", st["h"]))
    for c in dec.reassessment_conditions:
        story.append(Paragraph("- " + _safe(c), st["b"]))
    story.append(Paragraph(f"{n + 1}. Limitations", st["h"]))
    for l in ["Scenario outputs are hypothetical estimates, not predictions of real public behaviour.",
              "Reach values are analyst estimates; the tool never infers or manufactures audience figures.",
              "Lifecycle stage describes circulation behaviour only and does not establish truth.",
              "The tool has no access to platform algorithms, private messages, internal Army data or classified information.",
              "It cannot and does not publish any statement; all responses require authorised human decision.",
              DISCLAIMER]:
        story.append(Paragraph("- " + _safe(l), st["b"]))

    if history is not None and len(history):
        story.append(Paragraph(f"{n + 2}. Assessment history", st["h"]))
        rows = [["Assessed at", "Recommendation", "RPS", "Velocity"]]
        for _, h in history.tail(12).iterrows():
            rows.append([str(h["assessed_at"]), h["recommendation"], _f(h["response_pressure_score"], 3), _f(h["velocity"], 2)])
        story.append(_table(rows, [38 * mm, W - 38 * mm - 40 * mm, 20 * mm, 20 * mm]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ----------------------------------------------------------------------- CSV
def assessment_summary_df(bundle) -> pd.DataFrame:
    d, t, r, a = bundle.decision, bundle.trend, bundle.reach, bundle.amp
    row = {"claim_id": bundle.ctx.claim_id, "category": bundle.ctx.category,
           "verification_status": bundle.ctx.verification_status, "public_impact": bundle.ctx.public_impact,
           "assessed_at": bundle.assessed_at, "observations": t.n_observations, "total_mentions": t.total_mentions,
           "velocity_per_hour": t.current_velocity, "previous_velocity": t.previous_velocity,
           "growth_pct": t.growth_pct, "acceleration": t.acceleration, "moving_avg_velocity": t.moving_avg,
           "trend": t.label, "lifecycle_stage": bundle.lifecycle.stage, "hardening_level": bundle.hardening.level, "hardening_index": bundle.hardening.index, "reach": r.current_reach,
           "reach_confidence": r.confidence, "amplification_risk": a.risk, "amplification_ratio": a.ratio,
           "rps": d.rps_score, "band": d.band, "recommendation": d.recommendation, "escalation": d.escalation,
           "human_review_status": d.human_review_status}
    return pd.DataFrame([row])


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
