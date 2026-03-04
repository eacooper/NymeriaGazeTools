"""
metrics.py — Scalar summary metrics for eye gaze sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def session_summary(df: pd.DataFrame) -> dict:
    """Compute overall session statistics."""
    _require_columns(df, ["elapsed_time_s", "avg_yaw_deg", "pitch_deg",
                           "vergence_deg", "depth_m", "angular_velocity_deg_s"])

    n = len(df)
    duration_s = float(df["elapsed_time_s"].iloc[-1] - df["elapsed_time_s"].iloc[0])
    sampling_hz = (n / duration_s) if duration_s > 0 else float("nan")
    missing_pct = float(df["avg_yaw_deg"].isna().mean() * 100)

    def _stats7(series: pd.Series) -> dict:
        s = series.dropna()
        if len(s) == 0:
            return {k: float("nan") for k in ["mean", "std", "min", "max", "median", "skew", "kurtosis"]}
        return {
            "mean":     float(s.mean()),
            "std":      float(s.std(ddof=1)),
            "min":      float(s.min()),
            "max":      float(s.max()),
            "median":   float(s.median()),
            "skew":     float(scipy_stats.skew(s, nan_policy="omit")),
            "kurtosis": float(scipy_stats.kurtosis(s, nan_policy="omit")),
        }

    def _stats5(series: pd.Series) -> dict:
        s = series.dropna()
        if len(s) == 0:
            return {k: float("nan") for k in ["mean", "std", "min", "max", "median"]}
        return {
            "mean":   float(s.mean()),
            "std":    float(s.std(ddof=1)),
            "min":    float(s.min()),
            "max":    float(s.max()),
            "median": float(s.median()),
        }

    vel = df["angular_velocity_deg_s"].dropna()
    angular_velocity_stats = {
        "mean":   float(vel.mean())   if len(vel) else float("nan"),
        "std":    float(vel.std(ddof=1)) if len(vel) else float("nan"),
        "median": float(vel.median()) if len(vel) else float("nan"),
        "p95":    float(np.nanpercentile(vel, 95)) if len(vel) else float("nan"),
    }

    return {
        "duration_s":       duration_s,
        "n_samples":        n,
        "sampling_rate_hz": sampling_hz,
        "missing_gaze_pct": missing_pct,
        "yaw":              _stats7(df["avg_yaw_deg"]),
        "pitch":            _stats7(df["pitch_deg"]),
        "depth":            _stats5(df["depth_m"]),
        "vergence":         {
            "mean":   float(df["vergence_deg"].mean()),
            "std":    float(df["vergence_deg"].std(ddof=1)),
            "median": float(df["vergence_deg"].median()),
        },
        "angular_velocity": angular_velocity_stats,
    }


def saccade_metrics(df: pd.DataFrame) -> dict:
    """Compute saccade statistics (requires detect_saccades to be run first)."""
    from nymeria_gaze_tools.events import get_saccade_table

    _require_columns(df, ["is_saccade", "elapsed_time_s"])

    duration_s = float(df["elapsed_time_s"].iloc[-1] - df["elapsed_time_s"].iloc[0])
    tbl = get_saccade_table(df)

    if tbl.empty:
        nan3 = {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        return {
            "count": 0,
            "rate_per_min": 0.0,
            "amplitude_deg": nan3,
            "peak_velocity_deg_s": nan3,
            "main_sequence_r2": float("nan"),
        }

    count = len(tbl)
    rate = (count / duration_s * 60.0) if duration_s > 0 else float("nan")

    amp = tbl["amplitude_deg"]
    vel = tbl["peak_velocity_deg_s"]

    def _s3(s: pd.Series) -> dict:
        return {"mean": float(s.mean()), "std": float(s.std(ddof=1)), "median": float(s.median())}

    r2 = float("nan")
    valid = tbl[(amp > 0) & (vel > 0)]
    if len(valid) >= 3:
        log_amp = np.log(valid["amplitude_deg"])
        log_vel = np.log(valid["peak_velocity_deg_s"])
        slope, intercept, r_value, *_ = scipy_stats.linregress(log_amp, log_vel)
        r2 = float(r_value ** 2)

    return {
        "count":               count,
        "rate_per_min":        rate,
        "amplitude_deg":       _s3(amp),
        "peak_velocity_deg_s": _s3(vel),
        "main_sequence_r2":    r2,
    }


def fixation_metrics(df: pd.DataFrame) -> dict:
    """Compute fixation statistics (requires detect_fixations to be run first)."""
    from nymeria_gaze_tools.events import get_fixation_table

    _require_columns(df, ["is_fixation", "elapsed_time_s"])

    duration_s = float(df["elapsed_time_s"].iloc[-1] - df["elapsed_time_s"].iloc[0])
    tbl = get_fixation_table(df)

    coverage_pct = float(df["is_fixation"].mean() * 100)

    if tbl.empty:
        nan3 = {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        return {
            "count": 0,
            "rate_per_min": 0.0,
            "duration_ms": nan3,
            "coverage_pct": coverage_pct,
            "spatial_dispersion_deg": float("nan"),
        }

    count = len(tbl)
    rate = (count / duration_s * 60.0) if duration_s > 0 else float("nan")
    dur = tbl["duration_ms"]

    return {
        "count":       count,
        "rate_per_min": rate,
        "duration_ms": {
            "mean":   float(dur.mean()),
            "std":    float(dur.std(ddof=1)),
            "median": float(dur.median()),
        },
        "coverage_pct":           coverage_pct,
        "spatial_dispersion_deg": float(tbl["dispersion_deg"].mean()),
    }


def quality_report(df: pd.DataFrame) -> dict:
    """Assess data quality for a session."""
    _require_columns(df, ["avg_yaw_deg", "elapsed_time_s", "depth_m"])

    missing_rows = int(df["avg_yaw_deg"].isna().sum())
    missing_pct  = float(missing_rows / len(df) * 100)

    dt_us = df["elapsed_time_s"].diff().dropna() * 1000.0
    median_dt = float(dt_us.median())
    temporal_gap_count = int((dt_us > 2 * median_dt).sum())
    sampling_jitter_ms = float(dt_us.std(ddof=1))

    depth_cap_pct = float((df["depth_m"] == 4.0).mean() * 100)

    result = {
        "missing_rows":           missing_rows,
        "missing_pct":            missing_pct,
        "temporal_gap_count":     temporal_gap_count,
        "sampling_jitter_ms_std": sampling_jitter_ms,
        "depth_at_cap_pct":       depth_cap_pct,
    }

    if "yaw_confidence_width_deg" in df.columns:
        result["confidence_width_yaw_mean_deg"] = float(
            df["yaw_confidence_width_deg"].mean()
        )
    else:
        result["confidence_width_yaw_mean_deg"] = float("nan")

    if "pitch_confidence_width_deg" in df.columns:
        result["confidence_width_pitch_mean_deg"] = float(
            df["pitch_confidence_width_deg"].mean()
        )
    else:
        result["confidence_width_pitch_mean_deg"] = float("nan")

    return result


def print_gaze_statistics(df: pd.DataFrame) -> None:
    """Print formatted table of gaze statistics."""
    from scipy.stats import skew, kurtosis

    cols = ["left_yaw_deg", "right_yaw_deg", "avg_yaw_deg", "pitch_deg", "depth_m", "vergence_deg"]
    labels = ["Left Yaw", "Right Yaw", "Avg Yaw", "Pitch", "Depth", "Vergence"]

    hdr = f"{'Metric':<14}{'Min':>9}{'Max':>9}{'Mean':>9}{'Std':>9}{'Median':>9}{'Skew':>8}{'Kurt':>8}"
    print(hdr)
    print("─" * len(hdr))

    for col, label in zip(cols, labels):
        v = df[col].dropna()
        u = " m" if col == "depth_m" else " °"
        print(f"{label:<14}"
              f"{v.min():>7.2f}{u}"
              f"{v.max():>7.2f}{u}"
              f"{v.mean():>7.2f}{u}"
              f"{v.std():>7.2f}{u}"
              f"{v.median():>7.2f}{u}"
              f"{skew(v):>8.2f}"
              f"{kurtosis(v):>8.2f}")


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            "Ensure preprocessing.preprocess(df) has been run."
        )
