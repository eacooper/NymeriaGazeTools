"""
analysis.py — End-to-end analysis pipelines for eye gaze sessions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nymeria_gaze_tools import (
    DEFAULT_VELOCITY_THRESHOLD_DEG_S,
    DEFAULT_DISPERSION_THRESHOLD_DEG,
)
from nymeria_gaze_tools.io import load_session, filter_sessions
from nymeria_gaze_tools.preprocessing import preprocess
from nymeria_gaze_tools.events import detect_saccades, detect_fixations
from nymeria_gaze_tools.metrics import session_summary, saccade_metrics, fixation_metrics, quality_report


def _run_full_pipeline(
    df: pd.DataFrame,
    velocity_threshold: float,
    dispersion_threshold: float,
    include_cartesian: bool,
) -> dict:
    df_proc = preprocess(df, include_cartesian=include_cartesian)
    df_proc = detect_saccades(df_proc, velocity_threshold=velocity_threshold)
    df_proc = detect_fixations(df_proc, dispersion_threshold=dispersion_threshold)
    return {
        "df":        df_proc,
        "summary":   session_summary(df_proc),
        "saccades":  saccade_metrics(df_proc),
        "fixations": fixation_metrics(df_proc),
        "quality":   quality_report(df_proc),
    }


def _load_or_pass(uid_or_df, data_root, source: str) -> pd.DataFrame:
    if isinstance(uid_or_df, pd.DataFrame):
        return uid_or_df
    return load_session(uid_or_df, data_root=data_root, source=source)


def _flatten_summary(result: dict) -> dict[str, float]:
    """Flatten nested result dict to {metric_name: scalar_value}."""
    flat = {}
    for k in ["duration_s", "n_samples", "sampling_rate_hz", "missing_gaze_pct"]:
        flat[k] = result["summary"].get(k, float("nan"))
    for group, stats in result["summary"].items():
        if isinstance(stats, dict):
            for stat, val in stats.items():
                flat[f"{group}_{stat}"] = val
    for k in ["count", "rate_per_min", "main_sequence_r2"]:
        flat[f"saccade_{k}"] = result["saccades"].get(k, float("nan"))
    for sub, stats in result["saccades"].items():
        if isinstance(stats, dict):
            for stat, val in stats.items():
                flat[f"saccade_{sub}_{stat}"] = val
    for k in ["count", "rate_per_min", "coverage_pct", "spatial_dispersion_deg"]:
        flat[f"fixation_{k}"] = result["fixations"].get(k, float("nan"))
    for sub, stats in result["fixations"].items():
        if isinstance(stats, dict):
            for stat, val in stats.items():
                flat[f"fixation_{sub}_{stat}"] = val
    return flat


_RADAR_METRICS = [
    "saccade_count",
    "saccade_amplitude_deg_mean",
    "saccade_peak_velocity_deg_s_mean",
    "fixation_count",
    "fixation_duration_ms_mean",
    "fixation_coverage_pct",
    "yaw_std",
    "pitch_std",
    "angular_velocity_mean",
]


def _normalize_radar(flat_a: dict, flat_b: dict) -> dict:
    """Min-max normalize two flat metric dicts."""
    result = {}
    for k in _RADAR_METRICS:
        va = flat_a.get(k, float("nan"))
        vb = flat_b.get(k, float("nan"))
        lo = min(va, vb)
        hi = max(va, vb)
        rng = hi - lo if (hi - lo) != 0 else 1.0
        result[k] = {"a": (va - lo) / rng, "b": (vb - lo) / rng}
    return result


def analyze_session(
    sequence_uid_or_df: str | pd.DataFrame,
    data_root: str | Path = None,
    source: str = "local",
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD_DEG_S,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD_DEG,
    include_cartesian: bool = False,
) -> dict:
    """Run full analysis pipeline on a single session.

    Returns dict with keys: df, summary, saccades, fixations, quality.
    """
    raw = _load_or_pass(sequence_uid_or_df, data_root, source)
    return _run_full_pipeline(raw, velocity_threshold, dispersion_threshold, include_cartesian)


def compare_two_sessions(
    uid_a: str | pd.DataFrame,
    uid_b: str | pd.DataFrame,
    data_root: str | Path = None,
    source: str = "local",
    labels: tuple[str, str] = ("Session A", "Session B"),
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD_DEG_S,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD_DEG,
) -> dict:
    """Analyze two sessions and compute difference profile.

    Returns dict with keys: session_a, session_b, labels, delta, radar_metrics.
    """
    raw_a = _load_or_pass(uid_a, data_root, source)
    raw_b = _load_or_pass(uid_b, data_root, source)

    result_a = _run_full_pipeline(raw_a, velocity_threshold, dispersion_threshold, False)
    result_b = _run_full_pipeline(raw_b, velocity_threshold, dispersion_threshold, False)

    flat_a = _flatten_summary(result_a)
    flat_b = _flatten_summary(result_b)

    delta = {k: flat_a.get(k, float("nan")) - flat_b.get(k, float("nan"))
             for k in flat_a}

    return {
        "session_a":     result_a,
        "session_b":     result_b,
        "labels":        labels,
        "delta":         delta,
        "radar_metrics": _normalize_radar(flat_a, flat_b),
    }


def compare_participant_sessions(
    catalog: pd.DataFrame,
    participant: str,
    data_root: str | Path = None,
    source: str = "local",
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD_DEG_S,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD_DEG,
) -> dict:
    """Analyze all sessions for one participant.

    Returns dict with keys: participant, session_count, per_session, activity_metrics, radar_metrics.
    """
    subset = filter_sessions(catalog, participant=participant, has_gaze_data=True)
    uids = subset["sequence_uid"].tolist()

    per_session: dict[str, dict] = {}
    for uid in uids:
        raw = load_session(uid, data_root=data_root, source=source)
        per_session[uid] = _run_full_pipeline(
            raw, velocity_threshold, dispersion_threshold, False
        )

    rows = []
    for uid, res in per_session.items():
        meta_row = subset[subset["sequence_uid"] == uid].iloc[0]
        flat = _flatten_summary(res)
        flat["uid"] = uid
        flat["script"] = meta_row["script"]
        rows.append(flat)

    activity_metrics = pd.DataFrame(rows).set_index("uid") if rows else pd.DataFrame()
    radar_metrics = _normalize_group_radar(
        {uid: _flatten_summary(res) for uid, res in per_session.items()}
    )

    return {
        "participant":      participant,
        "session_count":    len(uids),
        "per_session":      per_session,
        "activity_metrics": activity_metrics,
        "radar_metrics":    radar_metrics,
    }


def analyze_group(
    catalog: pd.DataFrame,
    group_by: str,
    group_value: str,
    data_root: str | Path = None,
    source: str = "local",
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD_DEG_S,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD_DEG,
) -> dict:
    """Analyze all sessions matching a metadata filter.

    Returns dict with keys: group_by, group_value, session_count, session_metrics, aggregate.
    """
    if group_by not in catalog.columns:
        raise ValueError(
            f"Column '{group_by}' not found in catalog. "
            f"Available: {list(catalog.columns)}"
        )

    subset = catalog[
        (catalog[group_by] == group_value) & (catalog["has_gaze_data"] == True)
    ].reset_index(drop=True)

    rows = []
    for uid in subset["sequence_uid"].tolist():
        raw = load_session(uid, data_root=data_root, source=source)
        res = _run_full_pipeline(raw, velocity_threshold, dispersion_threshold, False)
        flat = _flatten_summary(res)
        flat["uid"] = uid
        rows.append(flat)

    session_metrics = pd.DataFrame(rows).set_index("uid") if rows else pd.DataFrame()

    aggregate = {}
    for col in session_metrics.select_dtypes(include=np.number).columns:
        s = session_metrics[col].dropna()
        aggregate[col] = {
            "mean":   float(s.mean()),
            "std":    float(s.std(ddof=1)),
            "median": float(s.median()),
            "n":      int(len(s)),
        }

    return {
        "group_by":        group_by,
        "group_value":     group_value,
        "session_count":   len(subset),
        "session_metrics": session_metrics,
        "aggregate":       aggregate,
    }


def compare_groups(
    catalog: pd.DataFrame,
    group_by: str,
    data_root: str | Path = None,
    source: str = "local",
    groups: list[str] = None,
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD_DEG_S,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD_DEG,
) -> dict:
    """Compare sessions across all values of a metadata column.

    Returns dict with keys: group_by, groups, group_metrics, session_metrics, metric_distributions.
    """
    if group_by not in catalog.columns:
        raise ValueError(
            f"Column '{group_by}' not found in catalog. "
            f"Available: {list(catalog.columns)}"
        )

    all_values = catalog[group_by].dropna().unique().tolist()
    if groups is not None:
        all_values = [g for g in groups if g in all_values]

    all_session_rows = []
    group_agg_rows   = []
    metric_distributions: dict[str, dict[str, np.ndarray]] = {}

    for gval in all_values:
        grp_result = analyze_group(
            catalog, group_by, gval,
            data_root=data_root, source=source,
            velocity_threshold=velocity_threshold,
            dispersion_threshold=dispersion_threshold,
        )
        sm = grp_result["session_metrics"].copy()
        if not sm.empty:
            sm[group_by] = gval
            all_session_rows.append(sm)

        agg_row = {"group": gval}
        for metric, stats in grp_result["aggregate"].items():
            agg_row[f"{metric}_mean"]   = stats["mean"]
            agg_row[f"{metric}_median"] = stats["median"]
            agg_row[f"{metric}_std"]    = stats["std"]
            agg_row[f"{metric}_n"]      = stats["n"]
            metric_distributions.setdefault(metric, {})[gval] = (
                sm[metric].dropna().to_numpy() if metric in sm.columns else np.array([])
            )
        group_agg_rows.append(agg_row)

    session_metrics = (
        pd.concat(all_session_rows) if all_session_rows else pd.DataFrame()
    )
    group_metrics = (
        pd.DataFrame(group_agg_rows).set_index("group")
        if group_agg_rows else pd.DataFrame()
    )

    return {
        "group_by":             group_by,
        "groups":               all_values,
        "group_metrics":        group_metrics,
        "session_metrics":      session_metrics,
        "metric_distributions": metric_distributions,
    }


def _normalize_group_radar(flat_map: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Min-max normalize radar metrics across all sessions."""
    result = {}
    for uid in flat_map:
        result[uid] = {}

    for metric in _RADAR_METRICS:
        vals = {uid: flat_map[uid].get(metric, float("nan")) for uid in flat_map}
        valid = [v for v in vals.values() if not np.isnan(v)]
        lo = min(valid) if valid else 0.0
        hi = max(valid) if valid else 1.0
        rng = hi - lo if (hi - lo) != 0 else 1.0
        for uid, val in vals.items():
            result[uid][metric] = float("nan") if np.isnan(val) else (val - lo) / rng

    return result
