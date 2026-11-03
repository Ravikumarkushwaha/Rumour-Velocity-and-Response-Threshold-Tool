import math

import numpy as np
import pandas as pd
import pytest

from config import (DataConfig, TREND_DECLINING, TREND_HIGH, TREND_LOW, TREND_MODERATE, TREND_NA,
                    TREND_RAPID, VelocityConfig)
from modules.data_processing import aggregate_to_interval, assess_sufficiency, clean_observations
from modules.velocity_engine import (acceleration, classify_trend, compute_series, moving_average,
                                     percentage_growth, summarise, velocity)
from tests.helpers import make_obs


# ------------------------------------------------------------ scalar formulas
def test_velocity_basic():
    assert velocity(150, 100, 5) == pytest.approx(10.0)

def test_velocity_zero_new_mentions():
    assert velocity(100, 100, 2) == 0.0

def test_velocity_zero_or_negative_dt_is_undefined():
    assert velocity(10, 5, 0) is None
    assert velocity(10, 5, -1) is None

def test_velocity_missing_values():
    assert velocity(None, 5, 1) is None
    assert velocity(5, None, 1) is None
    assert velocity(float("nan"), 5, 1) is None
    assert velocity(5, 4, float("nan")) is None

def test_growth_basic():
    assert percentage_growth(150, 100) == pytest.approx(50.0)

def test_growth_zero_denominator_is_safe():
    assert percentage_growth(10, 0) is None

def test_growth_missing_and_zero_both():
    assert percentage_growth(None, 5) is None
    assert percentage_growth(0, 0) is None

def test_growth_can_be_zero():
    assert percentage_growth(50, 50) == 0.0

def test_acceleration_basic():
    assert acceleration(12, 6, 3) == pytest.approx(2.0)

def test_acceleration_negative_is_deceleration():
    assert acceleration(4, 10, 2) == pytest.approx(-3.0)

def test_acceleration_undefined_cases():
    assert acceleration(None, 3, 1) is None
    assert acceleration(5, 3, 0) is None


# ------------------------------------------------------------ series
def test_series_regular_intervals():
    s = compute_series(make_obs([10, 20, 40, 70], step_hours=1))
    assert s["velocity"].tolist()[1:] == [10.0, 20.0, 30.0]
    assert s["acceleration"].tolist()[2:] == [10.0, 10.0]
    assert math.isnan(s["velocity"].iloc[0]) and math.isnan(s["acceleration"].iloc[1])

def test_series_irregular_intervals_use_real_elapsed_hours():
    df = make_obs([0, 10, 40])
    df["timestamp"] = pd.to_datetime(["2026-11-01 00:00", "2026-11-01 02:00", "2026-11-01 03:00"])
    s = compute_series(df)
    assert s["dt_hours"].tolist()[1:] == [2.0, 1.0]
    assert s["velocity"].tolist()[1:] == [5.0, 30.0]          # 10/2h and 30/1h
    assert s["acceleration"].iloc[2] == pytest.approx(25.0)   # (30-5)/1h

def test_series_zero_previous_mentions_growth_nan_not_error():
    s = compute_series(make_obs([0, 5, 10]))
    assert math.isnan(s["growth_pct"].iloc[1])
    assert s["growth_pct"].iloc[2] == pytest.approx(100.0)

def test_series_empty_and_single():
    assert len(compute_series(pd.DataFrame())) == 0
    s = compute_series(make_obs([5]))
    assert len(s) == 1 and math.isnan(s["velocity"].iloc[0])

def test_moving_average_smooths():
    ma = moving_average([np.nan, 10, 30, 20, 40], window=3)
    assert ma.iloc[1] == pytest.approx(10.0)
    assert ma.iloc[2] == pytest.approx(20.0)
    assert ma.iloc[4] == pytest.approx(30.0)

def test_total_mentions_is_not_velocity():
    big_slow = summarise(compute_series(make_obs([10_000, 10_010, 10_020, 10_030])))
    small_fast = summarise(compute_series(make_obs([10, 60, 160, 360])))
    assert big_slow.total_mentions > small_fast.total_mentions
    assert big_slow.current_velocity < small_fast.current_velocity


# ------------------------------------------------------------ classification
V = VelocityConfig()

def test_classification_levels():
    assert classify_trend(1.0, 1.0, 0.0, 1.0, V) == TREND_LOW
    assert classify_trend(10.0, 10.0, 0.0, 10.0, V) == TREND_MODERATE
    assert classify_trend(30.0, 30.0, 0.5, 30.0, V) == TREND_HIGH

def test_classification_rapid_and_declining():
    assert classify_trend(30.0, 20.0, 5.0, 25.0, V) == TREND_RAPID
    assert classify_trend(8.0, 16.0, -4.0, 14.0, V) == TREND_DECLINING

def test_classification_low_velocity_never_rapid():
    assert classify_trend(1.0, 0.1, 9.0, 0.5, V) == TREND_LOW

def test_classification_thresholds_configurable():
    custom = VelocityConfig(low_max=1.0, high_min=3.0)
    assert classify_trend(2.0, 2.0, 0.0, 2.0, custom) == TREND_MODERATE
    assert classify_trend(4.0, 4.0, 0.0, 4.0, custom) == TREND_HIGH

def test_classification_undefined_velocity():
    assert classify_trend(None, None, None, None, V) == TREND_NA

def test_summarise_current_and_previous():
    t = summarise(compute_series(make_obs([10, 20, 40, 70])))
    assert (t.current_velocity, t.previous_velocity, t.acceleration) == (30.0, 20.0, 10.0)
    assert t.growth_pct == pytest.approx(75.0)


# ------------------------------------------------------------ data processing / insufficiency
def test_insufficient_observations_flagged():
    clean, rep = clean_observations(make_obs([1, 2]))
    assert not rep.sufficient and "required" in rep.insufficient_reason
    assert any("INSUFFICIENT" in w for w in rep.warnings)

def test_insufficient_time_span_flagged():
    df = make_obs([1, 2, 3, 4], step_hours=0.25)
    ok, reason = assess_sufficiency(clean_observations(df)[0], DataConfig(min_observations=4, min_span_hours=3))
    assert not ok and "window" in reason

def test_sufficient_when_enough_data():
    _, rep = clean_observations(make_obs([1, 2, 4, 8, 16]))
    assert rep.sufficient

def test_empty_input():
    clean, rep = clean_observations(pd.DataFrame())
    assert len(clean) == 0 and not rep.sufficient

def test_missing_required_columns():
    clean, rep = clean_observations(pd.DataFrame({"foo": [1]}))
    assert rep.errors and len(clean) == 0

def test_missing_mention_count_row_dropped_not_inferred():
    df = make_obs([1, 2, 3, 4, 5]); df.loc[2, "mention_count"] = np.nan
    clean, rep = clean_observations(df)
    assert len(clean) == 4 and rep.dropped["missing_mentions"] == 1

def test_bad_timestamp_dropped():
    df = make_obs([1, 2, 3, 4, 5]); df["timestamp"] = df["timestamp"].astype(object); df.loc[1, "timestamp"] = "not a date"
    clean, rep = clean_observations(df)
    assert len(clean) == 4 and rep.dropped["bad_timestamp"] == 1

def test_duplicate_observations_not_counted_twice():
    df = make_obs([1, 2, 3, 4, 5]); df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    clean, rep = clean_observations(df)
    assert len(clean) == 5 and rep.dropped["exact_duplicates"] == 1

def test_same_timestamp_conflicting_counts_collapsed():
    df = make_obs([1, 2, 3, 4, 5]); extra = df.iloc[[2]].copy(); extra["engagement_count"] = 5
    clean, rep = clean_observations(pd.concat([df, extra], ignore_index=True))
    assert len(clean) == 5 and rep.dropped["duplicate_timestamp"] == 1

def test_cumulative_decrease_dropped():
    clean, rep = clean_observations(make_obs([5, 10, 7, 20, 30]))
    assert 7 not in clean["mention_count"].tolist() and rep.dropped["non_cumulative"] == 1

def test_negative_mentions_dropped():
    clean, rep = clean_observations(make_obs([5, -3, 7, 20, 30]))
    assert len(clean) == 4 and rep.dropped["negative_mentions"] == 1

def test_reach_is_never_inferred():
    clean, rep = clean_observations(make_obs([1, 2, 3, 4, 5], reach=[None, 100, None, None, 200]))
    assert clean["estimated_reach"].isna().sum() == 3

def test_irregular_intervals_detected_and_reported():
    df = make_obs([1, 2, 3, 4, 5, 6])
    df["timestamp"] = pd.to_datetime(["2026-11-01 00:00", "2026-11-01 01:00", "2026-11-01 08:00", "2026-11-01 09:00",
                                      "2026-11-01 20:00", "2026-11-01 21:00"])
    _, rep = clean_observations(df)
    assert rep.irregular_intervals

def test_aggregation_keeps_last_value_per_interval():
    df = make_obs([1, 2, 3, 4, 5, 6], step_hours=1)
    agg = aggregate_to_interval(df, 2.0)
    assert agg["mention_count"].tolist() == [2, 4, 6]

def test_first_observed_filter():
    df = make_obs([1, 2, 3, 4, 5])
    clean, rep = clean_observations(df, first_observed="2026-11-01 02:00")
    assert len(clean) == 3 and rep.dropped["before_first_observed"] == 2
