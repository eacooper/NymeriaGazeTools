"""
Tests for nymeria_gaze_tools.preprocessing

All tests use synthetic DataFrames — no real dataset required.
"""

import numpy as np
import pandas as pd
import pytest

from nymeria_gaze_tools.preprocessing import (
    normalize_timestamps,
    convert_radians_to_degrees,
    compute_binocular_gaze,
    compute_confidence_widths,
    compute_velocity,
    remove_invalid_samples,
    compute_sampling_rate,
    preprocess,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_raw_df(n: int = 10) -> pd.DataFrame:
    """Minimal synthetic raw gaze DataFrame matching Nymeria column names."""
    t0 = 1_000_000_000  # arbitrary start timestamp in microseconds
    timestamps = t0 + np.arange(n) * 100_000  # 10 Hz → 100 ms intervals

    yaw = np.linspace(-0.2, 0.2, n)   # radians
    pitch = np.linspace(-0.3, 0.0, n)  # radians
    margin = 0.05  # confidence interval half-width in radians

    return pd.DataFrame({
        "tracking_timestamp_us":    timestamps,
        "left_yaw_rads_cpf":        yaw,
        "right_yaw_rads_cpf":       yaw + 0.01,
        "pitch_rads_cpf":           pitch,
        "left_yaw_low_rads_cpf":    yaw - margin,
        "left_yaw_high_rads_cpf":   yaw + margin,
        "right_yaw_low_rads_cpf":   yaw + 0.01 - margin,
        "right_yaw_high_rads_cpf":  yaw + 0.01 + margin,
        "pitch_low_rads_cpf":       pitch - margin,
        "pitch_high_rads_cpf":      pitch + margin,
    })


# ---------------------------------------------------------------------------
# normalize_timestamps
# ---------------------------------------------------------------------------

def test_normalize_timestamps_starts_at_zero():
    df = make_raw_df()
    out = normalize_timestamps(df)
    assert out["elapsed_time_s"].iloc[0] == 0.0


def test_normalize_timestamps_units_are_seconds():
    df = make_raw_df(n=11)
    out = normalize_timestamps(df)
    # 10 Hz → 0.1 s intervals; 10 intervals → last value = 1.0 s
    assert abs(out["elapsed_time_s"].iloc[-1] - 1.0) < 1e-6


def test_normalize_timestamps_does_not_modify_original():
    df = make_raw_df()
    _ = normalize_timestamps(df)
    assert "elapsed_time_s" not in df.columns


# ---------------------------------------------------------------------------
# convert_radians_to_degrees
# ---------------------------------------------------------------------------

def test_convert_radians_to_degrees_output_columns_exist():
    df = make_raw_df()
    out = convert_radians_to_degrees(df)
    for col in ["left_yaw_deg", "right_yaw_deg", "pitch_deg",
                "left_yaw_low_deg", "left_yaw_high_deg",
                "right_yaw_low_deg", "right_yaw_high_deg",
                "pitch_low_deg", "pitch_high_deg"]:
        assert col in out.columns, f"Missing column: {col}"


def test_convert_radians_to_degrees_known_value():
    df = make_raw_df(n=1)
    df["left_yaw_rads_cpf"] = np.pi / 2   # 90 degrees
    df["right_yaw_rads_cpf"] = 0.0
    df["pitch_rads_cpf"] = np.pi           # 180 degrees
    # set confidence columns to 0 to avoid NaN
    for col in ["left_yaw_low_rads_cpf", "left_yaw_high_rads_cpf",
                "right_yaw_low_rads_cpf", "right_yaw_high_rads_cpf",
                "pitch_low_rads_cpf", "pitch_high_rads_cpf"]:
        df[col] = 0.0

    out = convert_radians_to_degrees(df)
    assert abs(out["left_yaw_deg"].iloc[0] - 90.0) < 1e-10
    assert abs(out["pitch_deg"].iloc[0] - 180.0) < 1e-10


# ---------------------------------------------------------------------------
# compute_binocular_gaze
# ---------------------------------------------------------------------------

def test_compute_binocular_gaze_is_average():
    df = make_raw_df()
    df = convert_radians_to_degrees(df)
    out = compute_binocular_gaze(df)
    expected = (df["left_yaw_deg"] + df["right_yaw_deg"]) / 2.0
    pd.testing.assert_series_equal(out["avg_yaw_deg"], expected, check_names=False)


def test_compute_binocular_gaze_pitch_unchanged():
    df = make_raw_df()
    df = convert_radians_to_degrees(df)
    before = df["pitch_deg"].copy()
    out = compute_binocular_gaze(df)
    pd.testing.assert_series_equal(out["pitch_deg"], before)


# ---------------------------------------------------------------------------
# compute_confidence_widths
# ---------------------------------------------------------------------------

def test_compute_confidence_widths_columns_exist():
    df = make_raw_df()
    df = convert_radians_to_degrees(df)
    out = compute_confidence_widths(df)
    assert "yaw_confidence_width_deg" in out.columns
    assert "pitch_confidence_width_deg" in out.columns


def test_compute_confidence_widths_are_non_negative():
    df = make_raw_df()
    df = convert_radians_to_degrees(df)
    out = compute_confidence_widths(df)
    assert (out["yaw_confidence_width_deg"] >= 0).all()
    assert (out["pitch_confidence_width_deg"] >= 0).all()


# ---------------------------------------------------------------------------
# remove_invalid_samples
# ---------------------------------------------------------------------------

def test_remove_invalid_samples_drops_nan_rows():
    df = make_raw_df()
    df = convert_radians_to_degrees(df)
    df.loc[2, "left_yaw_deg"] = np.nan
    df.loc[5, "pitch_deg"] = np.nan
    out = remove_invalid_samples(df)
    assert len(out) == len(df) - 2
    assert not out.isnull().any().any()


def test_remove_invalid_samples_keeps_clean_rows():
    df = make_raw_df()
    df = convert_radians_to_degrees(df)
    out = remove_invalid_samples(df)
    assert len(out) == len(df)


# ---------------------------------------------------------------------------
# compute_velocity
# ---------------------------------------------------------------------------

def test_compute_velocity_output_columns_exist():
    df = make_raw_df()
    df = normalize_timestamps(df)
    df = convert_radians_to_degrees(df)
    df = compute_binocular_gaze(df)
    out = compute_velocity(df)
    for col in ["yaw_velocity_deg_s", "pitch_velocity_deg_s", "angular_velocity_deg_s"]:
        assert col in out.columns, f"Missing column: {col}"


def test_compute_velocity_angular_is_magnitude():
    df = make_raw_df()
    df = normalize_timestamps(df)
    df = convert_radians_to_degrees(df)
    df = compute_binocular_gaze(df)
    out = compute_velocity(df)
    expected = np.sqrt(out["yaw_velocity_deg_s"] ** 2 + out["pitch_velocity_deg_s"] ** 2)
    pd.testing.assert_series_equal(
        out["angular_velocity_deg_s"], expected, check_names=False
    )


def test_compute_velocity_fallback_without_timestamps():
    """compute_velocity should not crash when elapsed_time_s is absent."""
    df = make_raw_df()
    df = convert_radians_to_degrees(df)
    df = compute_binocular_gaze(df)
    assert "elapsed_time_s" not in df.columns
    out = compute_velocity(df)
    assert "angular_velocity_deg_s" in out.columns


# ---------------------------------------------------------------------------
# compute_sampling_rate
# ---------------------------------------------------------------------------

def test_compute_sampling_rate_10hz():
    df = make_raw_df(n=11)  # 10 intervals at 100 ms each
    rate = compute_sampling_rate(df)
    assert abs(rate - 10.0) < 0.01


def test_compute_sampling_rate_single_row_returns_nan():
    df = make_raw_df(n=1)
    rate = compute_sampling_rate(df)
    assert np.isnan(rate)


# ---------------------------------------------------------------------------
# preprocess (end-to-end)
# ---------------------------------------------------------------------------

def test_preprocess_returns_expected_columns():
    df = make_raw_df()
    out = preprocess(df)
    expected_cols = [
        "elapsed_time_s",
        "avg_yaw_deg", "pitch_deg",
        "yaw_confidence_width_deg", "pitch_confidence_width_deg",
        "yaw_velocity_deg_s", "pitch_velocity_deg_s", "angular_velocity_deg_s",
    ]
    for col in expected_cols:
        assert col in out.columns, f"Missing column after preprocess(): {col}"


def test_preprocess_no_nan_in_output():
    df = make_raw_df()
    out = preprocess(df)
    assert not out[["avg_yaw_deg", "pitch_deg", "elapsed_time_s"]].isnull().any().any()


def test_preprocess_does_not_modify_input():
    df = make_raw_df()
    original_columns = set(df.columns)
    _ = preprocess(df)
    assert set(df.columns) == original_columns
