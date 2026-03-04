"""
events.py — Saccade and fixation detection algorithms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nymeria_gaze_tools import (
    DEFAULT_VELOCITY_THRESHOLD_DEG_S,
    DEFAULT_DISPERSION_THRESHOLD_DEG,
    DEFAULT_MIN_FIXATION_MS,
    DEFAULT_MIN_SACCADE_MS,
)


def detect_saccades(
    df: pd.DataFrame,
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD_DEG_S,
    min_duration_ms: float = DEFAULT_MIN_SACCADE_MS,
) -> pd.DataFrame:
    """Detect saccades using I-VT algorithm (velocity threshold)."""
    _require_columns(df, ["angular_velocity_deg_s", "elapsed_time_s"])

    out = df.copy()
    vel = out["angular_velocity_deg_s"].to_numpy(dtype=float)
    time_s = out["elapsed_time_s"].to_numpy(dtype=float)

    above = (~np.isnan(vel)) & (vel > velocity_threshold)
    is_saccade = np.zeros(len(out), dtype=bool)
    saccade_id = np.full(len(out), np.nan)

    runs = _get_runs(above)
    sid = 1
    for start, end in runs:
        duration_ms = (time_s[end - 1] - time_s[start]) * 1000.0
        if duration_ms >= min_duration_ms:
            is_saccade[start:end] = True
            saccade_id[start:end] = sid
            sid += 1

    out["is_saccade"] = is_saccade
    out["saccade_id"] = saccade_id
    return out


def get_saccade_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize detected saccades as one row per event."""
    _require_columns(df, ["is_saccade", "saccade_id", "elapsed_time_s",
                           "avg_yaw_deg", "pitch_deg", "angular_velocity_deg_s"])

    saccade_df = df[df["is_saccade"]].copy()
    if saccade_df.empty:
        return pd.DataFrame(columns=[
            "saccade_id", "onset_s", "offset_s", "duration_ms",
            "amplitude_deg", "peak_velocity_deg_s",
        ])

    rows = []
    for sid, group in saccade_df.groupby("saccade_id"):
        onset_s = group["elapsed_time_s"].iloc[0]
        offset_s = group["elapsed_time_s"].iloc[-1]
        duration_ms = (offset_s - onset_s) * 1000.0

        dyaw = group["avg_yaw_deg"].iloc[-1] - group["avg_yaw_deg"].iloc[0]
        dpitch = group["pitch_deg"].iloc[-1] - group["pitch_deg"].iloc[0]
        amplitude_deg = float(np.sqrt(dyaw**2 + dpitch**2))

        peak_vel = float(group["angular_velocity_deg_s"].max())

        rows.append({
            "saccade_id": int(sid),
            "onset_s": onset_s,
            "offset_s": offset_s,
            "duration_ms": duration_ms,
            "amplitude_deg": amplitude_deg,
            "peak_velocity_deg_s": peak_vel,
        })

    return pd.DataFrame(rows)


def detect_fixations(
    df: pd.DataFrame,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD_DEG,
    min_duration_ms: float = DEFAULT_MIN_FIXATION_MS,
    window_ms: float = 100.0,
) -> pd.DataFrame:
    """Detect fixations using I-DT algorithm (dispersion threshold)."""
    _require_columns(df, ["avg_yaw_deg", "pitch_deg", "elapsed_time_s"])

    out = df.copy()
    yaw = out["avg_yaw_deg"].to_numpy(dtype=float)
    pitch = out["pitch_deg"].to_numpy(dtype=float)
    time_s = out["elapsed_time_s"].to_numpy(dtype=float)
    n = len(out)

    is_fixation = np.zeros(n, dtype=bool)
    fixation_id = np.full(n, np.nan)

    fid = 1
    i = 0
    window_s = window_ms / 1000.0

    while i < n:
        j = i + 1
        while j < n and (time_s[j] - time_s[i]) < window_s:
            j += 1

        if j >= n:
            break

        if _dispersion(yaw[i:j], pitch[i:j]) > dispersion_threshold:
            i += 1
            continue

        while j < n and _dispersion(yaw[i:j + 1], pitch[i:j + 1]) <= dispersion_threshold:
            j += 1

        duration_ms = (time_s[j - 1] - time_s[i]) * 1000.0
        if duration_ms >= min_duration_ms:
            is_fixation[i:j] = True
            fixation_id[i:j] = fid
            fid += 1
            i = j
        else:
            i += 1

    out["is_fixation"] = is_fixation
    out["fixation_id"] = fixation_id
    return out


def get_fixation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize detected fixations as one row per event."""
    _require_columns(df, ["is_fixation", "fixation_id", "elapsed_time_s",
                           "avg_yaw_deg", "pitch_deg"])

    fix_df = df[df["is_fixation"]].copy()
    if fix_df.empty:
        return pd.DataFrame(columns=[
            "fixation_id", "onset_s", "offset_s", "duration_ms",
            "centroid_yaw_deg", "centroid_pitch_deg", "dispersion_deg",
        ])

    rows = []
    for fid, group in fix_df.groupby("fixation_id"):
        onset_s = group["elapsed_time_s"].iloc[0]
        offset_s = group["elapsed_time_s"].iloc[-1]
        duration_ms = (offset_s - onset_s) * 1000.0

        yaw_vals = group["avg_yaw_deg"].to_numpy(dtype=float)
        pitch_vals = group["pitch_deg"].to_numpy(dtype=float)

        rows.append({
            "fixation_id": int(fid),
            "onset_s": onset_s,
            "offset_s": offset_s,
            "duration_ms": duration_ms,
            "centroid_yaw_deg": float(np.nanmean(yaw_vals)),
            "centroid_pitch_deg": float(np.nanmean(pitch_vals)),
            "dispersion_deg": float(_dispersion(yaw_vals, pitch_vals)),
        })

    return pd.DataFrame(rows)


def _dispersion(yaw: np.ndarray, pitch: np.ndarray) -> float:
    yaw_range = float(np.nanmax(yaw) - np.nanmin(yaw))
    pitch_range = float(np.nanmax(pitch) - np.nanmin(pitch))
    return yaw_range + pitch_range


def _get_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Run preprocess(df) first."
        )
