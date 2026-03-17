"""
metrics.py — Quantitative summaries for fixations, saccades, and data quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nymeria_gaze_tools.events import get_fixation_table, get_saccade_table
from nymeria_gaze_tools.preprocessing import compute_sampling_rate


def fixation_metrics(
    fixations: pd.DataFrame,
    recording_duration_s: float,
) -> dict:
    """Fixation count, rate, duration stats, and dwell % for one session."""
    if fixations.empty:
        return {k: np.nan for k in [
            "n_fixations", "fixation_rate_per_min",
            "mean_duration_ms", "median_duration_ms", "sd_duration_ms", "iqr_duration_ms",
            "pct_time_in_fixation",
        ]}

    dur = fixations["duration_ms"]
    total_fixation_ms = dur.sum()
    recording_min     = recording_duration_s / 60.0

    return {
        "n_fixations":            int(len(fixations)),
        "fixation_rate_per_min":  round(len(fixations) / recording_min, 2),
        "mean_duration_ms":       round(dur.mean(), 2),
        "median_duration_ms":     round(dur.median(), 2),
        "sd_duration_ms":         round(dur.std(), 2),
        "iqr_duration_ms":        round(float(np.percentile(dur, 75) - np.percentile(dur, 25)), 2),
        "pct_time_in_fixation":   round(total_fixation_ms / (recording_duration_s * 1000) * 100, 2),
    }


def saccade_metrics(saccades: pd.DataFrame) -> dict:
    """Saccade count, amplitude, and duration stats. Artifacts excluded."""
    sac = saccades[saccades["event_type"] == "saccade"] if not saccades.empty else saccades

    if sac.empty:
        return {k: np.nan for k in [
            "n_saccades", "n_artifacts",
            "mean_amplitude_deg", "median_amplitude_deg", "sd_amplitude_deg",
            "mean_duration_ms", "median_duration_ms",
        ]}

    n_artifacts = int((saccades["event_type"] == "artifact").sum()) if not saccades.empty else 0

    return {
        "n_saccades":           int(len(sac)),
        "n_artifacts":          n_artifacts,
        "mean_amplitude_deg":   round(sac["amplitude_deg"].mean(), 3),
        "median_amplitude_deg": round(sac["amplitude_deg"].median(), 3),
        "sd_amplitude_deg":     round(sac["amplitude_deg"].std(), 3),
        "mean_duration_ms":     round(sac["duration_ms"].mean(), 2),
        "median_duration_ms":   round(sac["duration_ms"].median(), 2),
    }


def session_summary(
    df: pd.DataFrame,
    fixations: pd.DataFrame = None,
    saccades: pd.DataFrame = None,
    **metadata,
) -> pd.DataFrame:
    """Single-row summary combining fixation and saccade metrics for one session.

    Computes fixation/saccade tables internally if not provided.
    Pass metadata as kwargs (e.g. participant='Alice', activity='cooking') —
    these become the first columns, making pd.concat across sessions easy.
    """
    if fixations is None:
        fixations = get_fixation_table(df)

    if saccades is None:
        saccades = get_saccade_table(fixations.to_dict("records"))

    recording_duration_s = float(df["elapsed_time_s"].iloc[-1])

    row = {
        **metadata,
        "recording_duration_s":  round(recording_duration_s, 2),
        "sampling_rate_hz":      round(compute_sampling_rate(df), 2),
        "mean_vergence_deg":     round(df["vergence_deg"].mean(), 3),
        **fixation_metrics(fixations, recording_duration_s),
        **saccade_metrics(saccades),
    }

    return pd.DataFrame([row])
