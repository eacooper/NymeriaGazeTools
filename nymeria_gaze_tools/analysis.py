"""
analysis.py — Session-level analysis workflows for Nymeria eye gaze data.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd
import plotly.graph_objects as go

from nymeria_gaze_tools.events import get_fixation_table, get_saccade_table
from nymeria_gaze_tools.io import load_session
from nymeria_gaze_tools.metrics import session_summary
from nymeria_gaze_tools.preprocessing import preprocess, compute_sampling_rate
from nymeria_gaze_tools.viz import plot_gaze_timeseries


class SessionResult(NamedTuple):
    fixations: pd.DataFrame
    saccades:  pd.DataFrame
    summary:   pd.DataFrame
    fig:       go.Figure
    df:        pd.DataFrame


class GroupResult(NamedTuple):
    summaries: pd.DataFrame
    dfs:       list[pd.DataFrame]


def analyze_session(
    raw_df: pd.DataFrame,
    meta: dict = None,
    show: bool = True,
    **preprocess_kwargs,
) -> SessionResult:
    """Preprocess, detect fixations/saccades, compute summary, and plot gaze timeseries.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output of load_session(). Sampling rate is computed here before NaN removal.
    meta : dict, optional
        Session metadata — passed to plot title and summary columns.
    show : bool
        Call fig.show() automatically. Set False to handle display yourself.
    **preprocess_kwargs
        Forwarded to preprocess() (e.g. trim_start_min, max_yaw_confidence_width_deg).
    """
    meta              = meta or {}
    sampling_rate_hz  = compute_sampling_rate(raw_df)
    df                = preprocess(raw_df, **preprocess_kwargs)
    fixations         = get_fixation_table(df, sample_rate_hz=sampling_rate_hz)
    saccades          = get_saccade_table(fixations.to_dict("records"), df=df)
    summary           = session_summary(df, fixations=fixations, saccades=saccades,
                                        sampling_rate_hz=sampling_rate_hz, **meta)
    fig               = plot_gaze_timeseries(df, meta=meta)

    if show:
        fig.show()

    return SessionResult(fixations=fixations, saccades=saccades, summary=summary, fig=fig, df=df)


def analyze_sessions(
    sessions_df: pd.DataFrame,
    data_root: str | Path = None,
    show: bool = False,
    **preprocess_kwargs,
) -> GroupResult:
    """Batch-analyze a filtered set of sessions. Prints progress to stdout."""
    n = len(sessions_df)
    summaries, dfs = [], []

    for i, (_, row) in enumerate(sessions_df.iterrows(), start=1):
        uid = row["sequence_uid"]
        print(f"[{i}/{n}] {uid}")
        try:
            raw    = load_session(uid, data_root=data_root)
            result = analyze_session(raw, meta=row.to_dict(), show=show, **preprocess_kwargs)
            summaries.append(result.summary)
            dfs.append(result.df)
        except Exception as e:
            print(f"  Skipping {uid}: {e}")

    return GroupResult(
        summaries=pd.concat(summaries, ignore_index=True),
        dfs=dfs,
    )
