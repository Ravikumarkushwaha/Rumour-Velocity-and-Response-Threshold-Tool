"""
Strategic Decision Engine.

Outputs one of four statuses:
  REMAIN SILENT | MONITOR | RECOMMEND RESPONSE REVIEW |
  INSUFFICIENT DATA / HUMAN VERIFICATION REQUIRED

SAFEGUARDS (evaluated first, independent of the numerical score)
  * analyst-flagged public-safety issue
  * analyst-flagged serious credible allegation
  * verified misinformation at or above the configured impact level
  -> always escalated to authorised human review.
High amplification risk can raise the response bar but can NEVER suppress a
safeguard escalation.  The system NEVER publishes any statement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import (CONFLICTING, REC_INSUFFICIENT, REC_MONITOR, REC_REVIEW, REC_SILENT,
                    TREND_HIGH, TREND_RAPID, UNVERIFIED, VERIFIED_FALSE, VERIFIED_TRUE, PARTLY_TRUE,
                    AppConfig)
from modules.amplification_engine import AmplificationResult
from modules.reach_engine import ReachAssessment
from modules.silence_justification import SilenceJustification, generate_silence_justification
from modules.threshold_engine import (EffectiveThresholds, RPSInputs, RPSResult, band, compute_rps,
                                      effective_thresholds)
from modules.velocity_engine import TrendMetrics

IMPACT_ORDER = {"Low": 0, "Medium": 1, "High": 2}

HUMAN_REVIEW_TEXT = {
    REC_REVIEW: "REFERRED FOR AUTHORISED HUMAN REVIEW. This tool does not and cannot publish any statement.",
    "ESCALATED": "ESCALATED - IMMEDIATE AUTHORISED HUMAN REVIEW REQUIRED regardless of the numerical score.",
    REC_INSUFFICIENT: "HUMAN VERIFICATION REQUIRED. Observations are inadequate for a confident recommendation.",
    REC_MONITOR: "Not required at present. Analyst continues monitoring and reassesses on new observations.",
    REC_SILENT: "Not required at present. Silence is an active decision: monitoring continues and a human may override at any time.",
}


@dataclass
class ClaimContext:
    claim_id: str
    description: str
    category: str
    verification_status: str = UNVERIFIED
    public_impact: str = "Low"
    public_safety_flag: bool = False
    credible_allegation_flag: bool = False


@dataclass
class DecisionResult:
    recommendation: str
    escalation: bool
    escalation_reasons: list
    rps: Optional[RPSResult]
    thresholds: Optional[EffectiveThresholds]
    band: str
    rationale: list
    blockers: list
    reassessment_conditions: list
    monitoring_items: list
    human_review_status: str
    response_note: str
    silence_justification: Optional[SilenceJustification]
    headline: str = ""

    @property
    def rps_score(self) -> Optional[float]:
        return None if self.rps is None else self.rps.score


def escalation_reasons(ctx: ClaimContext, cfg: AppConfig) -> list:
    reasons = []
    if ctx.public_safety_flag:
        reasons.append("Analyst has flagged an urgent public-safety issue.")
    if ctx.credible_allegation_flag:
        reasons.append("Analyst has flagged a serious, credible allegation or potential harm.")
    need = IMPACT_ORDER.get(cfg.threshold.escalate_verified_false_impact, 2)
    if ctx.verification_status == VERIFIED_FALSE and IMPACT_ORDER.get(ctx.public_impact, 0) >= need:
        reasons.append(f"Verified misinformation with {ctx.public_impact} public impact "
                       f"(safeguard level: {cfg.threshold.escalate_verified_false_impact}).")
    return reasons


def response_type_note(ctx: ClaimContext) -> str:
    if ctx.verification_status == VERIFIED_TRUE:
        return "The claim is verified TRUE: a denial would be inappropriate. Any communication must be factually accurate and decided by authorised humans."
    if ctx.verification_status == PARTLY_TRUE:
        return "The claim is PARTLY true: a blanket denial would be inaccurate. Any clarification must separate accurate from inaccurate elements."
    if ctx.verification_status in (UNVERIFIED, CONFLICTING):
        return "Evidence is not settled: no denial should be drafted until the claim is verified."
    return "The claim is verified false: any correction should be factual, proportionate and approved by authorised humans."


def build_monitoring_items(trend: TrendMetrics, reach: ReachAssessment) -> list:
    items = [
        "Cumulative mentions, velocity, percentage growth and acceleration at every new observation.",
        "Estimated reach and number of independent sources (reach is never inferred; update it when evidence exists).",
        "Whether the claim is picked up by new source categories, in particular credible or mainstream outlets.",
        "Variants, reposts or related claims, so they are not counted as separate independent evidence.",
        "Any change to verification status, public impact or safety indicators.",
        "Hardening indicators (reach, time in circulation, independent sources): is the claim starting to be treated as established fact?",
    ]
    if not reach.available:
        items.insert(2, "PRIORITY: obtain a defensible reach estimate; without it amplification risk cannot be assessed.")
    return items


def build_reassessment_conditions(rec: str, trend: TrendMetrics, reach: ReachAssessment, amp: AmplificationResult,
                                  thr: Optional[EffectiveThresholds], rps: Optional[RPSResult],
                                  cfg: AppConfig, median_dt: Optional[float]) -> list:
    v = cfg.velocity
    nxt = median_dt if median_dt and median_dt > 0 else 6.0
    conds = [f"Reassess at the next observation, and no later than {max(nxt, 1.0):.0f} hour(s) from the last one."]
    if rec == REC_INSUFFICIENT:
        conds.insert(0, f"At least {cfg.data.min_observations} valid observations spanning {cfg.data.min_span_hours:g}+ hours are needed.")
        return conds
    if rec == REC_REVIEW:
        conds.append("De-escalate only after authorised human review and a fall in velocity, reach growth and impact.")
        return conds
    if thr is not None:
        score = "n/a" if rps is None or rps.score is None else f"{rps.score:.2f}"
        conds.append(f"Response Pressure Score reaches {thr.response_min:.2f} (currently {score}) -> refer for response review.")
        if rec == REC_SILENT:
            conds.append(f"Response Pressure Score reaches the silence ceiling {thr.silence_max:.2f} -> move to MONITOR.")
    if trend.current_velocity is not None:
        conds.append(f"Velocity reaches {v.high_min:g} mentions/hour (High) or acceleration reaches +{v.accel_rapid_min:g}/h² "
                     f"(currently {trend.current_velocity:.2f}/h).")
    if reach.available:
        conds.append(f"Estimated reach grows by {cfg.threshold.reach_growth_monitor_pct:g}% or more in one interval, "
                     f"or exceeds {cfg.threshold.reach_cap:,.0f} people.")
    else:
        conds.append("A reach estimate becomes available (it may change the amplification and hardening assessments).")
    conds.append(f"Hardening index reaches {cfg.hardening.high_min:.2f} (High): reach near {cfg.hardening.reach_point:,.0f} people, "
                 f"persisting for {cfg.hardening.persistence_hours:g}+ hours with {cfg.hardening.sources_point}+ independent sources.")
    conds.append("Public impact is re-rated to High, a public-safety or credible-allegation flag is raised, or the claim is verified false with high impact (mandatory escalation).")
    conds.append("Verification status changes (e.g. the claim is found to be true or partly true).")
    return conds


def decide(ctx: ClaimContext, trend: TrendMetrics, reach: ReachAssessment, amp: AmplificationResult,
           cfg: AppConfig, data_sufficient: bool = True, insufficiency_reason: Optional[str] = None,
           hardening=None) -> DecisionResult:
    esc = escalation_reasons(ctx, cfg)
    amp_risk = amp.risk if amp.assessable else "Unknown"
    rps = compute_rps(RPSInputs(trend.current_velocity, trend.growth_pct, trend.acceleration,
                                reach.current_reach), cfg.threshold) if trend.current_velocity is not None else None
    thr = effective_thresholds(cfg.threshold, amp_risk, ctx.public_impact, ctx.verification_status)
    score = None if rps is None else rps.score
    bnd = band(score, thr)
    rationale: list = []
    blockers: list = []
    note = response_type_note(ctx)
    monitoring = build_monitoring_items(trend, reach)

    def finish(rec: str, headline: str, escalated: bool = False) -> DecisionResult:
        conds = build_reassessment_conditions(rec, trend, reach, amp, thr, rps, cfg, trend.median_dt_hours)
        status = HUMAN_REVIEW_TEXT["ESCALATED"] if escalated else HUMAN_REVIEW_TEXT[rec]
        sj = None
        if rec == REC_SILENT:
            sj = generate_silence_justification(ctx, trend, reach, amp, rps, thr, conds, monitoring, cfg, hardening)
        return DecisionResult(rec, escalated, esc, rps, thr, bnd, rationale, blockers, conds, monitoring,
                              status, note, sj, headline)

    # 1. Safeguards override everything, including insufficient data
    if esc:
        rationale.append("SAFEGUARD: " + " ".join(esc))
        rationale.append("Mandatory human escalation applies regardless of the numerical score or amplification risk.")
        if score is not None:
            rationale.append(f"For reference, RPS is {score:.2f} ({bnd.lower()}); it does not affect the escalation.")
        return finish(REC_REVIEW, "Safeguard triggered: escalate to authorised human review.", True)

    # 2. Insufficient data
    if not data_sufficient or score is None:
        rationale.append(insufficiency_reason or "Observations are inadequate.")
        rationale.append("The tool does not force a confident recommendation from inadequate evidence.")
        return finish(REC_INSUFFICIENT, "Insufficient data: human verification and more observations required.")

    reach_rising = (reach.last_interval_growth_pct is not None
                    and reach.last_interval_growth_pct >= cfg.threshold.reach_growth_monitor_pct)
    uncertain = ctx.verification_status in (UNVERIFIED, CONFLICTING)
    rationale.append(f"Response Pressure Score {score:.2f} vs silence ceiling {thr.silence_max:.2f} and response threshold "
                     f"{thr.response_min:.2f}: {bnd.lower()}.")
    for reason, ds, dr in thr.adjustments:
        rationale.append(f"Threshold adjustment - {reason} (silence {ds:+.2f}, response {dr:+.2f}).")
    if rps.note:
        rationale.append(rps.note)
    if hardening is not None and hardening.assessable:
        rationale.append(f"Cost of silence: hardening index {hardening.index:.2f} ({hardening.level}); the claim has been in "
                         f"circulation {hardening.hours_in_circulation:.0f} h. Cost of a denial: amplification risk {amp_risk}.")

    # 3. Response-review zone
    if score >= thr.response_min:
        if uncertain and amp_risk == "High" and ctx.public_impact == "Low":
            blockers.append("Evidence is uncertain, public impact is low and amplification risk is high.")
            rationale.append("Pressure is at response level, but a response is not yet justified: the evidence is uncertain, "
                             "impact is low and a denial carries high amplification risk. Hold, verify and monitor closely; "
                             "refer for human review if impact or evidence changes.")
            return finish(REC_MONITOR, "Pressure at response level, but evidence and impact do not yet justify a response.")
        rationale.append("Circulation pressure is at or above the response threshold; remaining silent could allow the claim to "
                         "be accepted as fact. Referral for authorised human review is recommended.")
        if amp_risk == "High":
            rationale.append("Amplification risk is HIGH. It is a factor for the human reviewers (e.g. wording, channel, timing); "
                             "it does not by itself suppress a necessary response.")
        return finish(REC_REVIEW, "Circulation is substantial: refer for authorised response review.")

    # 4. Silence zone (with blockers)
    if score < thr.silence_max:
        if not reach.available:
            blockers.append("Reach data unavailable: amplification risk cannot be assessed, so silence cannot be called evidence-based.")
        if reach_rising:
            blockers.append(f"Estimated reach grew {reach.last_interval_growth_pct:.0f}% in the latest interval, even though other measures are low.")
        if trend.label in (TREND_HIGH, TREND_RAPID):
            blockers.append(f"Circulation is classified '{trend.label}'.")
        if (uncertain and trend.current_velocity is not None and trend.current_velocity >= cfg.velocity.low_max
                and trend.acceleration is not None and trend.acceleration > 0):
            blockers.append("Circulation is at least moderate and still rising while the evidence is uncertain.")
        if (hardening is not None and cfg.hardening.high_blocks_silence and hardening.assessable
                and hardening.level == "High"):
            blockers.append(f"Hardening risk is High (index {hardening.index:.2f}): the claim is widely seen and persistent, "
                            f"so continued silence may let it be accepted as fact.")
        if ctx.public_impact == "High":
            blockers.append("Public impact is rated High.")
        if ctx.verification_status == VERIFIED_TRUE:
            blockers.append("The claim is verified true; this is a communications-planning matter for humans, not a rumour to leave unanswered by default.")
        if not blockers:
            rationale.append("Pressure is below the silence ceiling, circulation is limited, no urgent public impact is recorded and reach is known.")
            if amp_risk in ("High", "Moderate"):
                rationale.append(f"Amplification risk is {amp_risk}: a public denial could introduce the claim to additional audiences.")
            rationale.append("Continued monitoring is part of this decision; reassessment triggers are defined.")
            return finish(REC_SILENT, "Public intervention not recommended at this stage; monitoring continues.")
        rationale.append("Score is in the silence zone, but silence is not yet evidence-based: " + " ".join(blockers))
        return finish(REC_MONITOR, "Monitor: score is low but specific conditions prevent a silence recommendation.")

    # 5. Monitor zone
    rationale.append("Pressure lies between the silence ceiling and the response threshold: the claim is approaching a threshold "
                     "and evidence/observations are insufficient to justify a public response yet.")
    if hardening is not None and hardening.assessable and hardening.level == "High":
        rationale.append("Hardening risk is High: the claim is widely seen and persistent. The tool does not treat falling velocity "
                         "as proof that the claim is no longer believed; human review of acceptance should be considered.")
    if trend.label == "Declining":
        rationale.append("Circulation is declining; a new response should not be recommended automatically. Current conditions, not historical peak, drive the decision.")
    if reach_rising:
        rationale.append(f"Reach is rising ({reach.last_interval_growth_pct:.0f}% in the latest interval) independently of mention velocity; reach and potential impact are assessed separately.")
    return finish(REC_MONITOR, "Monitor: circulation is approaching a threshold; more observations required.")
