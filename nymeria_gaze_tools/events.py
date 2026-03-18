"""
events.py — Fixation and saccade detection for preprocessed Nymeria eye gaze data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nymeria_gaze_tools import (
    DEFAULT_DISPERSION_THRESHOLD_DEG,
    DEFAULT_MIN_FIXATION_MS,
)

_SAMPLE_RATE_HZ: float = 10.0


def _dispersion(yaw: np.ndarray, pitch: np.ndarray) -> float:
    """Manhattan dispersion: yaw range + pitch range."""
    return float((yaw.max() - yaw.min()) + (pitch.max() - pitch.min()))


def detect_fixations_idt(
    df: pd.DataFrame,
    dispersion_threshold_deg: float = DEFAULT_DISPERSION_THRESHOLD_DEG,
    min_fixation_ms: float = DEFAULT_MIN_FIXATION_MS,
    sample_rate_hz: float = _SAMPLE_RATE_HZ,
) -> list[dict]:
    """Detect fixations using I-DT (sliding window dispersion threshold).

    Slides a minimum-duration window across the data. If the gaze points in
    the window are tightly clustered (dispersion < threshold), it's a fixation.
    Otherwise, drop the first point and slide forward.

    Parameters
    ----------
    df : pd.DataFrame
        Output of preprocess(). Requires: elapsed_time_s, avg_yaw_deg, pitch_deg.
    dispersion_threshold_deg : float
        Max dispersion (yaw range + pitch range) to qualify as a fixation.
    min_fixation_ms : float
        Minimum fixation duration — sets the window size in samples.
    sample_rate_hz : float
        Sampling rate of the data. Used to convert min_fixation_ms to samples.
    """
    times = df["elapsed_time_s"].to_numpy(dtype=float)
    yaw   = df["avg_yaw_deg"].to_numpy(dtype=float)
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    n     = len(df)

    # Minimum window size in samples
    win = max(2, int(round(min_fixation_ms / 1000.0 * sample_rate_hz)))

    fixations: list[dict] = []
    i = 0

    while i <= n - win:
        window_yaw   = yaw[i : i + win]
        window_pitch = pitch[i : i + win]

        if _dispersion(window_yaw, window_pitch) <= dispersion_threshold_deg:
            # Seed qualifies — expand forward until dispersion breaks
            last_good = i + win - 1
            j = i + win

            while j < n:
                if _dispersion(yaw[i : j + 1], pitch[i : j + 1]) <= dispersion_threshold_deg:
                    last_good = j
                    j += 1
                else:
                    break  # eye moved — stop at last good point

            end_idx = last_good + 1  # exclusive
            fixations.append({
                "start_time_s":  float(times[i]),
                "end_time_s":    float(times[last_good]),
                "duration_ms":   float((times[last_good] - times[i]) * 1000.0),
                "avg_yaw_deg":   float(yaw[i:end_idx].mean()),
                "avg_pitch_deg": float(pitch[i:end_idx].mean()),
                "n_samples":     int(end_idx - i),
            })
            i = end_idx  # jump past the full expanded fixation
        else:
            i += 1  # slide forward

    return fixations


_FIXATION_COLUMNS = [
    "start_time_s", "end_time_s", "duration_ms",
    "avg_yaw_deg", "avg_pitch_deg", "n_samples",
]


def get_fixation_table(
    df: pd.DataFrame,
    dispersion_threshold_deg: float = DEFAULT_DISPERSION_THRESHOLD_DEG,
    min_fixation_ms: float = DEFAULT_MIN_FIXATION_MS,
    sample_rate_hz: float = _SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """Run detect_fixations_idt and return results as a tidy DataFrame."""
    fixations = detect_fixations_idt(
        df,
        dispersion_threshold_deg=dispersion_threshold_deg,
        min_fixation_ms=min_fixation_ms,
        sample_rate_hz=sample_rate_hz,
    )

    if not fixations:
        return pd.DataFrame(columns=_FIXATION_COLUMNS)

    return pd.DataFrame(fixations)


# ---------------------------------------------------------------------------
# Saccade detection (inter-fixation)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_SACCADE_MS: float = 200.0

_SACCADE_COLUMNS = [
    "start_time_s", "end_time_s", "duration_ms",
    "from_yaw_deg", "from_pitch_deg", "to_yaw_deg", "to_pitch_deg",
    "amplitude_deg", "peak_velocity_deg_s", "event_type",
]


def detect_saccades(
    fixations: list[dict],
    df: pd.DataFrame = None,
    max_saccade_ms: float = _DEFAULT_MAX_SACCADE_MS,
) -> list[dict]:
    """Derive saccades from gaps between consecutive fixations.

    At 10Hz, velocity-based detection is too noisy — so saccades are defined
    as the movement between two fixation centroids. Gaps longer than
    max_saccade_ms are likely blinks or dropouts and are labeled 'artifact'.
    Pass df to extract peak_velocity_deg_s from the raw velocity signal.
    """
    saccades = []

    for curr, nxt in zip(fixations, fixations[1:]):
        start  = curr["end_time_s"]
        end    = nxt["start_time_s"]
        dur_ms = (end - start) * 1000.0

        d_yaw   = nxt["avg_yaw_deg"]   - curr["avg_yaw_deg"]
        d_pitch = nxt["avg_pitch_deg"] - curr["avg_pitch_deg"]

        if df is not None:
            window = df[(df["elapsed_time_s"] >= start) & (df["elapsed_time_s"] <= end)]
            peak_vel = float(window["angular_velocity_deg_s"].max()) if not window.empty else float("nan")
        else:
            peak_vel = float("nan")

        saccades.append({
            "start_time_s":       float(start),
            "end_time_s":         float(end),
            "duration_ms":        float(dur_ms),
            "from_yaw_deg":       float(curr["avg_yaw_deg"]),
            "from_pitch_deg":     float(curr["avg_pitch_deg"]),
            "to_yaw_deg":         float(nxt["avg_yaw_deg"]),
            "to_pitch_deg":       float(nxt["avg_pitch_deg"]),
            "amplitude_deg":      float(np.sqrt(d_yaw**2 + d_pitch**2)),
            "peak_velocity_deg_s": peak_vel,
            "event_type":         "saccade" if dur_ms <= max_saccade_ms else "artifact",
        })

    return saccades


def get_saccade_table(
    fixations: list[dict],
    df: pd.DataFrame = None,
    max_saccade_ms: float = _DEFAULT_MAX_SACCADE_MS,
) -> pd.DataFrame:
    """Run detect_saccades and return results as a tidy DataFrame."""
    saccades = detect_saccades(fixations, df=df, max_saccade_ms=max_saccade_ms)

    if not saccades:
        return pd.DataFrame(columns=_SACCADE_COLUMNS)

    return pd.DataFrame(saccades)
