"""
Central configuration for the Rumour Velocity and Response Threshold Tool.

Every numerical assumption used anywhere in the application lives in this file
so that it can be inspected, documented and changed.  All thresholds are
EXPERIMENTAL, ILLUSTRATIVE values chosen for demonstration with synthetic
data.  They are NOT official Army policy.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "rumour_tool.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"
SYNTHETIC_CSV = BASE_DIR / "data" / "synthetic_rumours.csv"

APP_TITLE = "Rumour Velocity and Response Threshold Tool"
APP_SUBTITLE = "Decision-support prototype  |  Synthetic demonstration data only"
DISCLAIMER = (
    "Independent academic and technical prototype built on open-source concepts and "
    "synthetic data. Thresholds are experimental and are not official Army policy. "
    "The tool never publishes anything; every recommendation is advisory and "
    "requires authorised human review."
)

# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #
CATEGORIES = [
    "Recruitment notification",
    "Incident",
    "Unit-related claim",
    "Family-related claim",
    "Other",
]

VERIFIED_TRUE = "Verified true"
VERIFIED_FALSE = "Verified false"
PARTLY_TRUE = "Partly true"
UNVERIFIED = "Unverified"
CONFLICTING = "Conflicting evidence"
VERIFICATION_STATUSES = [VERIFIED_TRUE, VERIFIED_FALSE, PARTLY_TRUE, UNVERIFIED, CONFLICTING]

SOURCE_TYPES = [
    "News media",
    "Public social media post",
    "Public forum",
    "Blog / website",
    "Video platform",
    "Other",
]

IMPACT_LEVELS = ["Low", "Medium", "High"]
REACH_BASES = ["estimated", "observed"]
SOURCE_RELIABILITY_LEVELS = ["Unassessed", "Low", "Moderate", "High"]

# Recommendations ----------------------------------------------------------- #
REC_SILENT = "REMAIN SILENT"
REC_MONITOR = "MONITOR"
REC_REVIEW = "RECOMMEND RESPONSE REVIEW"
REC_INSUFFICIENT = "INSUFFICIENT DATA / HUMAN VERIFICATION REQUIRED"
ALL_RECOMMENDATIONS = [REC_SILENT, REC_MONITOR, REC_REVIEW, REC_INSUFFICIENT]

# Trend labels -------------------------------------------------------------- #
TREND_LOW = "Low velocity"
TREND_MODERATE = "Moderate velocity"
TREND_HIGH = "High velocity"
TREND_RAPID = "Rapidly accelerating"
TREND_DECLINING = "Declining"
TREND_NA = "Insufficient data"

# Lifecycle stages ---------------------------------------------------------- #
STAGES = ["Emergence", "Growth", "Acceleration", "Stabilisation", "Decline"]


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    min_observations: int = 4          # fewer valid observations -> INSUFFICIENT DATA
    min_span_hours: float = 3.0        # observation window must span at least this long
    aggregation_hours: float = 0.0     # 0 = use observations as supplied
    similarity_threshold: float = 0.80 # TF-IDF cosine similarity for duplicate-claim warning
    irregular_cv: float = 0.5          # coefficient of variation of intervals flagged as irregular


@dataclass
class VelocityConfig:
    moving_avg_window: int = 3         # observations in the rolling average
    low_max: float = 5.0               # mentions/hour: below = Low velocity
    high_min: float = 20.0             # mentions/hour: at or above = High velocity
    accel_rapid_min: float = 2.0       # mentions/hour^2 for "Rapidly accelerating"
    accel_decline_max: float = -1.0    # mentions/hour^2 at or below = declining


@dataclass
class ReachConfig:
    min_reach_points: int = 2          # reach readings needed for reach growth
    min_independent_sources: int = 3   # sources needed for higher confidence


@dataclass
class AmplificationConfig:
    response_exposure: float = 50_000.0  # assumed audience an official statement could reach
    factor_low: float = 0.20             # share of that audience newly exposed to the claim
    factor_medium: float = 0.50
    factor_high: float = 1.00
    classification_scenario: str = "medium"  # scenario used to classify risk
    ratio_moderate_min: float = 0.25     # ratio >= this -> Moderate
    ratio_high_min: float = 1.00         # ratio >= this -> High


@dataclass
class ThresholdConfig:
    # RPS weights (must sum to 1)
    w_velocity: float = 0.35
    w_growth: float = 0.15
    w_accel: float = 0.20
    w_reach: float = 0.30
    # Normalisation caps: value at/above cap maps to 1.0
    velocity_cap: float = 40.0         # mentions/hour
    growth_cap: float = 100.0          # percent per interval
    accel_cap: float = 5.0             # mentions/hour^2
    reach_cap: float = 100_000.0       # people (estimated)
    # Base thresholds on the 0-1 RPS scale
    silence_max: float = 0.30          # RPS below this = silence zone
    response_min: float = 0.60         # RPS at/above this = response-review zone
    min_gap: float = 0.05              # minimum gap between the two thresholds
    # Context adjustments (added to thresholds). Positive = harder to trigger.
    amp_adjust_response: dict = field(default_factory=lambda: {"Low": 0.00, "Moderate": 0.03, "High": 0.07})
    impact_adjust_silence: dict = field(default_factory=lambda: {"Low": 0.00, "Medium": -0.05, "High": -0.10})
    impact_adjust_response: dict = field(default_factory=lambda: {"Low": 0.00, "Medium": -0.05, "High": -0.10})
    evidence_adjust_response: dict = field(default_factory=lambda: {
        VERIFIED_FALSE: -0.05, VERIFIED_TRUE: 0.0, PARTLY_TRUE: 0.0, UNVERIFIED: 0.0, CONFLICTING: 0.0})
    # Silence is blocked if reach is rising at least this much per interval (percent)
    reach_growth_monitor_pct: float = 25.0
    # Safeguard: Verified-false claims at or above this impact are escalated to a human
    escalate_verified_false_impact: str = "High"

    @property
    def weights(self) -> dict:
        return {"velocity": self.w_velocity, "growth": self.w_growth,
                "acceleration": self.w_accel, "reach": self.w_reach}


@dataclass
class LifecycleConfig:
    emergence_total_max: int = 30      # cumulative mentions below this = Emergence
    decline_peak_ratio: float = 0.50   # velocity <= ratio x peak velocity = Decline
    stable_band: float = 0.20          # |V - MA| / MA within this = flat
    stable_accel_abs: float = 0.50     # |A| below this = flat


@dataclass
class HardeningConfig:
    """The 'hardening point': when a claim left alone starts to be accepted as fact.
    Index = reach ratio x persistence/source ratio (all analyst-configurable, illustrative)."""
    reach_point: float = 50_000.0      # people reached at which the claim is treated as widely seen
    persistence_hours: float = 24.0    # hours in circulation at which persistence saturates
    sources_point: int = 5             # independent sources at which corroboration saturates
    elevated_min: float = 0.30         # index >= this -> Elevated
    high_min: float = 0.60             # index >= this -> High
    high_blocks_silence: bool = True   # High hardening risk prevents a silence recommendation
    eta_cap_hours: float = 24 * 30     # ETAs beyond this are reported as "not within 30 days"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    reach: ReachConfig = field(default_factory=ReachConfig)
    amplification: AmplificationConfig = field(default_factory=AmplificationConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    hardening: HardeningConfig = field(default_factory=HardeningConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "AppConfig":
        return copy.deepcopy(self)


def default_config() -> AppConfig:
    return AppConfig()


# Threshold presets used for "comparison of different threshold settings".
PRESETS = {
    "Conservative (high bar for response)": {
        "weights": (0.40, 0.10, 0.20, 0.30), "silence_max": 0.35, "response_min": 0.70},
    "Balanced (default)": {
        "weights": (0.35, 0.15, 0.20, 0.30), "silence_max": 0.30, "response_min": 0.60},
    "Responsive (early review)": {
        "weights": (0.25, 0.15, 0.30, 0.30), "silence_max": 0.20, "response_min": 0.45},
}
