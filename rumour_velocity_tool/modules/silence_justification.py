"""
Silence Justification Engine - the centrepiece of the project.

Whenever the recommendation is REMAIN SILENT this module builds an
evidence-based explanation from the ACTUAL calculated metrics. Wording changes
with the numbers (amplification level, acceleration direction, distance to the
threshold, reach availability); nothing is a fixed template for every claim.

Guiding principle:
  Silence is not the absence of action. It is a deliberate, evidence-based
  strategic communication decision supported by continuous monitoring and
  clearly defined intervention thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

PRINCIPLE = ("Silence is not the absence of action. It is a deliberate, evidence-based strategic communication "
             "decision supported by continuous monitoring and clearly defined intervention thresholds.")


@dataclass
class SilenceJustification:
    summary: str
    sections: list = field(default_factory=list)   # [(title, text)]
    principle: str = PRINCIPLE

    def as_text(self) -> str:
        parts = [self.summary, ""]
        for i, (t, body) in enumerate(self.sections, 1):
            parts.append(f"{i}. {t}\n{body}\n")
        parts.append(self.principle)
        return "\n".join(parts)


def _f(x, d=2, unit="") -> str:
    return "undefined" if x is None else f"{x:,.{d}f}{unit}"


def _accel_phrase(a, vcfg) -> str:
    if a is None:
        return "acceleration cannot be computed yet"
    if a >= vcfg.accel_rapid_min:
        return f"the rate of spread is rising quickly ({a:+.2f} mentions/hour²)"
    if a > 0.25:
        return f"the rate of spread is rising slightly ({a:+.2f} mentions/hour²)"
    if a <= vcfg.accel_decline_max:
        return f"the rate of spread is falling ({a:+.2f} mentions/hour²)"
    if a < -0.25:
        return f"the rate of spread is easing slightly ({a:+.2f} mentions/hour²)"
    return f"the rate of spread is essentially steady ({a:+.2f} mentions/hour²)"


def generate_silence_justification(ctx, trend, reach, amp, rps, thr, conditions: list, monitoring: list,
                                   cfg, hardening=None) -> SilenceJustification:
    v, vcfg = cfg.velocity, cfg.velocity
    score = rps.score if rps is not None else None
    headroom = None if score is None else thr.response_min - score
    ceiling_gap = None if score is None else thr.silence_max - score

    # ---- 1-4 circulation measures ------------------------------------------
    s1 = (f"{trend.total_mentions:,.0f} cumulative mentions were recorded over {trend.window_hours:.1f} hours "
          f"across {trend.n_observations} observations. Circulation is classified as '{trend.label}'. "
          f"Total mentions describe how much has been said so far, not how fast it is spreading.")
    s2 = (f"The current velocity is {_f(trend.current_velocity)} mentions/hour (previous interval: "
          f"{_f(trend.previous_velocity)}; moving average: {_f(trend.moving_avg)}). The configured Low-velocity ceiling is "
          f"{v.low_max:g} and the High-velocity floor is {v.high_min:g} mentions/hour.")
    s3 = (f"Growth over the latest interval is {_f(trend.growth_pct, 1, '%')}. " +
          ("A small percentage on a modest base does not indicate rapid spread." if (trend.growth_pct or 0) < 25
           else "This is a notable relative rise and is why the claim stays under close monitoring."))
    s4 = f"Acceleration is {_f(trend.acceleration)} mentions/hour²: {_accel_phrase(trend.acceleration, vcfg)}."

    # ---- 5 reach --------------------------------------------------------------
    if reach.available:
        s5 = (f"Estimated public reach is {reach.current_reach:,.0f} people ({reach.basis} figure from {reach.reach_points} "
              f"reading(s); data confidence: {reach.confidence}). Independent sources: "
              f"{reach.independent_sources if reach.independent_sources is not None else 'not recorded'} "
              f"({reach.sources_basis}). Engagement counts are not treated as audience.")
        if reach.last_interval_growth_pct is not None:
            s5 += f" Reach changed by {reach.last_interval_growth_pct:+.1f}% in the latest interval."
    else:
        s5 = "INSUFFICIENT REACH DATA. No reach estimate is available and none has been inferred."

    # ---- 6 amplification ------------------------------------------------------
    if amp.assessable:
        lo, me, hi = amp.scenarios["low"], amp.scenarios["medium"], amp.scenarios["high"]
        s6 = (f"Amplification risk is {amp.risk} under the '{amp.classification_scenario}' scenario. An assumed response audience "
              f"of {amp.response_exposure:,.0f} people would add roughly {lo['additional']:,.0f} (low), {me['additional']:,.0f} "
              f"(medium) or {hi['additional']:,.0f} (high) new exposures against a current reach of {amp.current_reach:,.0f}, "
              f"i.e. amplification ratios of {lo['ratio']:.2f}, {me['ratio']:.2f} and {hi['ratio']:.2f}. These are hypothetical "
              f"scenario estimates, not predictions.")
    else:
        s6 = "Amplification risk could not be assessed because reach is unavailable."

    # ---- 7 thresholds -----------------------------------------------------------
    adj_txt = "; ".join(r for r, _, _ in thr.adjustments) if thr.adjustments else "no context adjustments applied"
    s7 = (f"The Response Pressure Score is {_f(score)} on a 0-1 scale. It is below the silence ceiling of {thr.silence_max:.2f} "
          f"by {_f(ceiling_gap)} and {_f(headroom)} below the response-review threshold of {thr.response_min:.2f}. "
          f"Threshold context: {adj_txt}. Thresholds are experimental, not official policy.")

    # ---- 8 why not a denial -------------------------------------------------------
    reasons = []
    if amp.assessable and amp.risk == "High":
        reasons.append(f"a denial could expose the claim to several times its current audience (ratio {amp.ratio:.1f}x), "
                       f"turning a limited rumour into a much larger one")
    elif amp.assessable and amp.risk == "Moderate":
        reasons.append(f"a denial could noticeably widen the audience (ratio {amp.ratio:.2f}x) without a matching benefit")
    else:
        reasons.append("although amplification risk is limited, nothing in the circulation data justifies a public statement")
    reasons.append(f"circulation is below the response threshold ({_f(score)} vs {thr.response_min:.2f})")
    reasons.append(f"no urgent public-impact indicator is recorded (impact rated {ctx.public_impact}; no safety or credible-allegation flag)")
    if trend.label in ("Low velocity", "Declining"):
        reasons.append(f"the claim is not spreading quickly ('{trend.label}')")
    cost = ""
    if hardening is not None and hardening.assessable:
        cost = (f" The cost of staying silent is currently limited: the hardening index is {hardening.index:.2f} ({hardening.level}); "
                f"reach is {hardening.reach_ratio:.1%} of the {cfg.hardening.reach_point:,.0f}-person hardening point and the claim has been "
                f"in circulation {hardening.hours_in_circulation:.0f} h.")
        if hardening.eta_reach_hours:
            cost += (f" If reach keeps growing at the observed rate, the hardening point would be approached in about "
                     f"{hardening.eta_reach_hours:.0f} h (a straight-line extrapolation, not a prediction), which is why the claim stays under watch.")
    s8 = ("A public denial is not currently recommended because " + "; ".join(reasons[:-1]) + "; and " + reasons[-1] +
          ". An official denial is itself a public act that can signal that a claim merits an official answer. " +
          "Declining to respond now is a decision that can be reversed at any time; a published denial cannot be recalled." + cost)

    # ---- 9-10 ----------------------------------------------------------------------
    s9 = "\n".join("- " + m for m in monitoring)
    s10 = "\n".join("- " + c for c in conditions)

    # ---- summary ------------------------------------------------------------------
    amp_clause = (f"The estimated amplification risk is {amp.risk.lower()}: a public denial could introduce the claim to additional audiences "
                  f"(about {amp.ratio:.1f}x the current reach under the {amp.classification_scenario} scenario)."
                  if amp.assessable and amp.risk in ("High", "Moderate")
                  else "Amplification risk is low, but the circulation does not justify a statement.")
    summary = (f"Public intervention is not recommended at this stage. The observed circulation is {trend.label.lower()} "
               f"({trend.total_mentions:,.0f} cumulative mentions), and the current velocity of {_f(trend.current_velocity)} mentions/hour "
               f"is below the configured response threshold (Response Pressure Score {_f(score)} against a threshold of {thr.response_min:.2f}). "
               f"{amp_clause} Continued monitoring is recommended, with reassessment if the rate of circulation, public reach "
               f"or potential impact increases.")

    return SilenceJustification(summary, [
        ("Current circulation", s1), ("Current velocity", s2), ("Percentage growth", s3), ("Acceleration", s4),
        ("Estimated public reach / data availability", s5), ("Amplification risk", s6),
        ("Comparison with response thresholds", s7), ("Why a public denial is not currently recommended", s8),
        ("What continues to be monitored", s9), ("Conditions that would change the recommendation", s10)])
