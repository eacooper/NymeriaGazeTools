"""
preprocessing.py — Signal processing and feature engineering for Nymeria eye gaze data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline on raw eye gaze data.

    Steps:
    1. Normalize timestamps to elapsed time
    2. Convert all radian fields to degrees
    3. Remove invalid samples (NaN/null values)
    4. Compute binocular gaze averages
    5. Compute confidence widths
    6. Compute velocity

    Parameters
    ----------
    df : pd.DataFrame
        Raw eye gaze DataFrame from load_session()
    include_cartesian : bool
        If True, compute Cartesian gaze coordinates (requires depth_m)

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with computed features
    """
    out = df.copy()
    out = normalize_timestamps(out)
    out = convert_radians_to_degrees(out)
    out = remove_invalid_samples(out)
    out = compute_binocular_gaze(out)
    out = compute_confidence_widths(out)
    out = compute_velocity(out)
    return out


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps to elapsed time in seconds from start."""
    out = df.copy()
    t0 = out["tracking_timestamp_us"].iloc[0]
    out["elapsed_time_s"] = (out["tracking_timestamp_us"] - t0) / 1e6
    return out


def compute_sampling_rate(df: pd.DataFrame) -> float:
    """Calculate actual sampling rate from timestamps using mean interval."""
    if len(df) < 2:
        return np.nan

    time_diffs_us = df["tracking_timestamp_us"].diff().dropna()
    mean_interval_s = (time_diffs_us / 1e6).mean()
    return 1.0 / mean_interval_s


def convert_radians_to_degrees(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all radian fields to degrees."""
    out = df.copy()

    # Main gaze angles
    out["left_yaw_deg"] = out["left_yaw_rads_cpf"] * (180.0 / np.pi)
    out["right_yaw_deg"] = out["right_yaw_rads_cpf"] * (180.0 / np.pi)
    out["pitch_deg"] = out["pitch_rads_cpf"] * (180.0 / np.pi)

    # Confidence bounds
    out["left_yaw_low_deg"] = out["left_yaw_low_rads_cpf"] * (180.0 / np.pi)
    out["left_yaw_high_deg"] = out["left_yaw_high_rads_cpf"] * (180.0 / np.pi)
    out["right_yaw_low_deg"] = out["right_yaw_low_rads_cpf"] * (180.0 / np.pi)
    out["right_yaw_high_deg"] = out["right_yaw_high_rads_cpf"] * (180.0 / np.pi)
    out["pitch_low_deg"] = out["pitch_low_rads_cpf"] * (180.0 / np.pi)
    out["pitch_high_deg"] = out["pitch_high_rads_cpf"] * (180.0 / np.pi)

    return out


def remove_invalid_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN or null values in gaze data."""
    out = df.copy()
    return out.dropna().reset_index(drop=True)


def compute_binocular_gaze(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binocular average gaze angles and vergence."""
    out = df.copy()
    out["avg_yaw_deg"] = (out["left_yaw_deg"] + out["right_yaw_deg"]) / 2.0
    out["vergence_deg"] = out["right_yaw_deg"] - out["left_yaw_deg"]
    return out


def compute_confidence_widths(df: pd.DataFrame) -> pd.DataFrame:
    """Compute confidence interval widths for gaze estimates."""
    out = df.copy()
    out["yaw_confidence_width_deg"] = out["left_yaw_high_deg"] - out["left_yaw_low_deg"]
    out["pitch_confidence_width_deg"] = out["pitch_high_deg"] - out["pitch_low_deg"]
    return out


def compute_velocity(df: pd.DataFrame, dt_s: float = 0.1) -> pd.DataFrame:
    """Compute angular velocity using np.gradient.

    Uses elapsed_time_s if available, otherwise falls back to fixed dt_s.
    """
    out = df.copy()

    if "elapsed_time_s" in out.columns:
        time_axis = out["elapsed_time_s"].to_numpy(dtype=float)
    else:
        time_axis = np.arange(len(out), dtype=float) * dt_s

    yaw_deg = out["avg_yaw_deg"].to_numpy(dtype=float)
    pitch_deg = out["pitch_deg"].to_numpy(dtype=float)

    out["yaw_velocity_deg_s"] = np.gradient(yaw_deg, time_axis)
    out["pitch_velocity_deg_s"] = np.gradient(pitch_deg, time_axis)
    out["angular_velocity_deg_s"] = np.sqrt(
        out["yaw_velocity_deg_s"] ** 2 + out["pitch_velocity_deg_s"] ** 2
    )
    return out
