"""
Tests for nymeria_gaze_tools.metrics

All tests use synthetic DataFrames — no real dataset required.
"""

import numpy as np
import pandas as pd
import pytest

from nymeria_gaze_tools.metrics import (
    fixation_metrics,
    saccade_metrics,
    session_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_fixations(n: int = 5) -> pd.DataFrame:
    """Synthetic fixation table."""
    return pd.DataFrame({
        "start_s":     np.arange(n, dtype=float),
        "end_s":       np.arange(n, dtype=float) + 0.3,
        "duration_ms": np.full(n, 300.0),
        "centroid_yaw_deg":   np.zeros(n),
        "centroid_pitch_deg": np.zeros(n),
    })


def make_saccades(n_saccades: int = 4, n_artifacts: int = 1) -> pd.DataFrame:
    """Synthetic saccade table with both saccades and artifacts."""
    saccade_rows = pd.DataFrame({
        "event_type":    ["saccade"] * n_saccades,
        "amplitude_deg": np.linspace(2.0, 8.0, n_saccades),
        "duration_ms":   np.full(n_saccades, 50.0),
    })
    artifact_rows = pd.DataFrame({
        "event_type":    ["artifact"] * n_artifacts,
        "amplitude_deg": np.full(n_artifacts, np.nan),
        "duration_ms":   np.full(n_artifacts, 250.0),
    })
    return pd.concat([saccade_rows, artifact_rows], ignore_index=True)


def make_preprocessed_df(n: int = 50, include_depth: bool = True) -> pd.DataFrame:
    """Minimal preprocessed gaze DataFrame (output of preprocess())."""
    t = np.linspace(0, 5.0, n)
    df = pd.DataFrame({
        "elapsed_time_s":          t,
        "avg_yaw_deg":             np.sin(t),
        "pitch_deg":               np.cos(t) * -10,
        "yaw_velocity_deg_s":      np.cos(t),
        "pitch_velocity_deg_s":    np.sin(t),
        "angular_velocity_deg_s":  np.ones(n),
        "yaw_confidence_width_deg":   np.full(n, 2.0),
        "pitch_confidence_width_deg": np.full(n, 2.0),
    })
    if include_depth:
        df["depth_m"] = np.linspace(1.0, 3.0, n)
    return df


# ---------------------------------------------------------------------------
# fixation_metrics
# ---------------------------------------------------------------------------

def test_fixation_metrics_keys():
    fixations = make_fixations()
    result = fixation_metrics(fixations, recording_duration_s=30.0)
    expected_keys = {
        "n_fixations", "fixation_rate_per_min",
        "mean_duration_ms", "median_duration_ms",
        "sd_duration_ms", "iqr_duration_ms",
        "pct_time_in_fixation",
    }
    assert expected_keys == set(result.keys())


def test_fixation_metrics_counts():
    fixations = make_fixations(n=5)
    result = fixation_metrics(fixations, recording_duration_s=60.0)
    assert result["n_fixations"] == 5
    assert abs(result["fixation_rate_per_min"] - 5.0) < 0.01


def test_fixation_metrics_empty_returns_nan():
    empty = pd.DataFrame(columns=["duration_ms"])
    result = fixation_metrics(empty, recording_duration_s=30.0)
    assert result["n_fixations"] is np.nan or np.isnan(result["n_fixations"])
    assert np.isnan(result["mean_duration_ms"])


# ---------------------------------------------------------------------------
# saccade_metrics
# ---------------------------------------------------------------------------

def test_saccade_metrics_keys():
    saccades = make_saccades()
    result = saccade_metrics(saccades)
    expected_keys = {
        "n_saccades", "n_artifacts",
        "mean_amplitude_deg", "median_amplitude_deg", "sd_amplitude_deg",
        "mean_duration_ms", "median_duration_ms",
    }
    assert expected_keys == set(result.keys())


def test_saccade_metrics_counts():
    saccades = make_saccades(n_saccades=4, n_artifacts=2)
    result = saccade_metrics(saccades)
    assert result["n_saccades"] == 4
    assert result["n_artifacts"] == 2


def test_saccade_metrics_empty_returns_nan():
    empty = pd.DataFrame(columns=["event_type", "amplitude_deg", "duration_ms"])
    result = saccade_metrics(empty)
    assert np.isnan(result["n_saccades"])
    assert np.isnan(result["mean_amplitude_deg"])


# ---------------------------------------------------------------------------
# session_summary
# ---------------------------------------------------------------------------

def test_session_summary_returns_single_row():
    df = make_preprocessed_df()
    fixations = make_fixations()
    saccades = make_saccades()
    result = session_summary(df, fixations=fixations, saccades=saccades,
                             sampling_rate_hz=10.0)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_session_summary_has_required_columns():
    df = make_preprocessed_df()
    fixations = make_fixations()
    saccades = make_saccades()
    result = session_summary(df, fixations=fixations, saccades=saccades,
                             sampling_rate_hz=10.0)
    for col in ["recording_duration_s", "sampling_rate_hz",
                "n_fixations", "n_saccades"]:
        assert col in result.columns, f"Missing column: {col}"


def test_session_summary_metadata_kwargs():
    """Extra kwargs should appear as columns in the summary row."""
    df = make_preprocessed_df()
    fixations = make_fixations()
    saccades = make_saccades()
    result = session_summary(df, fixations=fixations, saccades=saccades,
                             sampling_rate_hz=10.0,
                             participant="alice", script="S7-Cooking")
    assert result["participant"].iloc[0] == "alice"
    assert result["script"].iloc[0] == "S7-Cooking"


def test_session_summary_recording_duration():
    df = make_preprocessed_df(n=51)  # elapsed_time_s goes 0 → 5.0
    fixations = make_fixations()
    saccades = make_saccades()
    result = session_summary(df, fixations=fixations, saccades=saccades,
                             sampling_rate_hz=10.0)
    assert abs(result["recording_duration_s"].iloc[0] - 5.0) < 0.1


def test_session_summary_no_depth_returns_nan():
    """depth_m is optional — mean_depth_m/var_depth_m should be NaN when absent."""
    df = make_preprocessed_df(include_depth=False)
    fixations = make_fixations()
    saccades = make_saccades()
    result = session_summary(df, fixations=fixations, saccades=saccades,
                             sampling_rate_hz=10.0)
    assert np.isnan(result["mean_depth_m"].iloc[0])
    assert np.isnan(result["var_depth_m"].iloc[0])
