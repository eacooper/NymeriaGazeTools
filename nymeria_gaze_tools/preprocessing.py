"""
preprocessing.py — Signal processing and feature engineering for Nymeria eye gaze data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(
    df: pd.DataFrame,
    trim_start_min: float = 0.0,
    trim_end_min: float = 0.0,
    max_yaw_confidence_width_deg: float | None = None,
    max_pitch_confidence_width_deg: float | None = None,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline on raw eye gaze data.

    Steps:
    1. Trim start/end (optional, e.g. to drop calibration period)
    2. Normalize timestamps to elapsed time
    3. Convert all radian fields to degrees
    4. Remove invalid samples (NaN/null values)
    5. Compute binocular gaze averages
    6. Compute confidence widths
    7. Filter low-confidence samples (optional)
    8. Compute velocity

    Parameters
    ----------
    df : pd.DataFrame
        Raw eye gaze DataFrame from load_session()
    trim_start_min : float
        Minutes to drop from the start (e.g. calibration). Default 0.
    trim_end_min : float
        Minutes to drop from the end. Default 0.
    max_yaw_confidence_width_deg : float, optional
        Drop samples where yaw confidence interval width exceeds this (degrees).
    max_pitch_confidence_width_deg : float, optional
        Drop samples where pitch confidence interval width exceeds this (degrees).
    """
    out = df.copy()
    if trim_start_min > 0 or trim_end_min > 0:
        out = trim_recording(out, trim_start_min=trim_start_min, trim_end_min=trim_end_min)
    out = normalize_timestamps(out)
    out = convert_radians_to_degrees(out)
    out = remove_invalid_samples(out)
    out = compute_binocular_gaze(out)
    out = compute_confidence_widths(out)
    if max_yaw_confidence_width_deg is not None or max_pitch_confidence_width_deg is not None:
        out = filter_low_confidence(
            out,
            max_yaw_width_deg=max_yaw_confidence_width_deg,
            max_pitch_width_deg=max_pitch_confidence_width_deg,
        )
    out = compute_velocity(out)
    return out


def trim_recording(
    df: pd.DataFrame,
    trim_start_min: float = 0.0,
    trim_end_min: float = 0.0,
) -> pd.DataFrame:
    """Trim minutes from the start and/or end of a recording.

    Works on raw tracking_timestamp_us, so call this before normalize_timestamps()
    so that elapsed_time_s starts from 0 after the trim.

    Parameters
    ----------
    df : pd.DataFrame
        Raw eye gaze DataFrame from load_session()
    trim_start_min : float
        Minutes to remove from the start (e.g. to skip calibration).
    trim_end_min : float
        Minutes to remove from the end.
    """
    if trim_start_min < 0 or trim_end_min < 0:
        raise ValueError("trim_start_min and trim_end_min must be non-negative.")

    t_us = df["tracking_timestamp_us"]
    duration_min = (t_us.iloc[-1] - t_us.iloc[0]) / 1e6 / 60.0

    if trim_start_min + trim_end_min >= duration_min:
        raise ValueError(
            f"Total trim ({trim_start_min + trim_end_min:.2f} min) exceeds "
            f"recording duration ({duration_min:.2f} min)."
        )

    cutoff_start = t_us.iloc[0] + trim_start_min * 60 * 1e6
    cutoff_end = t_us.iloc[-1] - trim_end_min * 60 * 1e6

    return df.loc[(t_us >= cutoff_start) & (t_us <= cutoff_end)].reset_index(drop=True)


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


def filter_low_confidence(
    df: pd.DataFrame,
    max_yaw_width_deg: float | None = None,
    max_pitch_width_deg: float | None = None,
) -> pd.DataFrame:
    """Drop samples where the confidence interval is too wide (model is uncertain).

    Call after compute_confidence_widths() and before compute_velocity() — velocity
    at the edges of any resulting gaps will be unreliable for 1-2 samples.

    Parameters
    ----------
    max_yaw_width_deg : float, optional
        Drop rows where yaw confidence width exceeds this value.
    max_pitch_width_deg : float, optional
        Drop rows where pitch confidence width exceeds this value.
    """
    mask = pd.Series(True, index=df.index)
    if max_yaw_width_deg is not None:
        mask &= df["yaw_confidence_width_deg"] <= max_yaw_width_deg
    if max_pitch_width_deg is not None:
        mask &= df["pitch_confidence_width_deg"] <= max_pitch_width_deg
    return df.loc[mask].reset_index(drop=True)


def add_head_compensation(
    gaze_df: pd.DataFrame,
    traj_df: pd.DataFrame,
) -> pd.DataFrame:
    """Subtract head rotation from gaze velocity to isolate eye-only movement.

    Interpolates head angular velocity from the closed-loop trajectory
    (typically ~1032 Hz) onto the gaze timestamps (typically 10 Hz) using
    nearest-neighbour lookup, then subtracts the head rate component-wise
    from the gaze velocity.

    Call after preprocess() — requires yaw_velocity_deg_s and
    pitch_velocity_deg_s to already be present in gaze_df.

    Parameters
    ----------
    gaze_df : pd.DataFrame
        Output of preprocess(). Must contain:
        tracking_timestamp_us, yaw_velocity_deg_s, pitch_velocity_deg_s.
    traj_df : pd.DataFrame
        Output of load_trajectory(). Must contain:
        tracking_timestamp_us, angular_velocity_y_device (yaw, rad/s),
        angular_velocity_x_device (pitch, rad/s).

    Returns
    -------
    pd.DataFrame
        gaze_df with five new columns added:
        - head_yaw_rate_deg_s                : interpolated head yaw rate (deg/s)
        - head_pitch_rate_deg_s              : interpolated head pitch rate (deg/s)
        - eye_yaw_vel                        : head-compensated yaw velocity (deg/s)
        - eye_pitch_vel                      : head-compensated pitch velocity (deg/s)
        - angular_velocity_compensated_deg_s : eye-only speed magnitude (deg/s)
    """
    gaze_ts = gaze_df["tracking_timestamp_us"].to_numpy()
    traj_ts = traj_df["tracking_timestamp_us"].to_numpy()
    traj_wy = traj_df["angular_velocity_y_device"].to_numpy()  # yaw,   rad/s
    traj_wx = traj_df["angular_velocity_x_device"].to_numpy()  # pitch, rad/s

    # Nearest-neighbour interpolation
    idx      = np.searchsorted(traj_ts, gaze_ts)
    idx      = np.clip(idx, 0, len(traj_ts) - 1)
    idx_left = np.clip(idx - 1, 0, len(traj_ts) - 1)
    use_left = np.abs(gaze_ts - traj_ts[idx_left]) < np.abs(gaze_ts - traj_ts[idx])
    idx_nn   = np.where(use_left, idx_left, idx)

    RAD2DEG = 180.0 / np.pi
    out = gaze_df.copy()
    out["head_yaw_rate_deg_s"]   = traj_wy[idx_nn] * RAD2DEG
    out["head_pitch_rate_deg_s"] = traj_wx[idx_nn] * RAD2DEG

    out["eye_yaw_vel"]   = out["yaw_velocity_deg_s"]  - out["head_yaw_rate_deg_s"]
    out["eye_pitch_vel"] = out["pitch_velocity_deg_s"] - out["head_pitch_rate_deg_s"]
    out["angular_velocity_compensated_deg_s"] = np.sqrt(
        out["eye_yaw_vel"] ** 2 + out["eye_pitch_vel"] ** 2
    )
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
