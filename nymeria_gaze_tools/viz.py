"""
viz.py — Interactive visualizations for Nymeria eye gaze data.

All functions return a Plotly figure — call .show() or display in a notebook.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def _confidence_band(
    fig: go.Figure,
    row: int,
    t: pd.Series,
    lo: pd.Series,
    hi: pd.Series,
    main: pd.Series,
    name: str,
    color_rgb: str,
    color_hex: str,
) -> None:
    """Add a line + shaded confidence band to a subplot row."""
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines",
                             line=dict(width=0), showlegend=False), row=row, col=1)
    fig.add_trace(go.Scatter(x=t, y=lo, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor=f"rgba({color_rgb},0.12)", showlegend=False),
                  row=row, col=1)
    fig.add_trace(go.Scatter(x=t, y=main, name=name,
                             line=dict(width=1.3, color=color_hex), opacity=0.85),
                  row=row, col=1)


def plot_gaze_timeseries(
    df: pd.DataFrame,
    meta: dict = None,
    title: str = None,
    height: int = 750,
) -> go.Figure:
    """3-panel interactive plot: Yaw (L/R/avg + CI bands), Pitch (+ CI), Depth."""
    meta = meta or {}
    t    = df["elapsed_time_s"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Horizontal Gaze — Yaw", "Vertical Gaze — Pitch", "Gaze Depth"),
    )

    _confidence_band(fig, 1, t,
                     df["left_yaw_low_deg"],  df["left_yaw_high_deg"],  df["left_yaw_deg"],
                     "Left Eye", "65,105,225", "royalblue")
    _confidence_band(fig, 1, t,
                     df["right_yaw_low_deg"], df["right_yaw_high_deg"], df["right_yaw_deg"],
                     "Right Eye", "255,140,0", "darkorange")
    fig.add_trace(go.Scatter(x=t, y=df["avg_yaw_deg"], name="Avg (binocular)",
                             line=dict(width=2, color="black")), row=1, col=1)

    _confidence_band(fig, 2, t,
                     df["pitch_low_deg"], df["pitch_high_deg"], df["pitch_deg"],
                     "Pitch", "46,139,87", "seagreen")

    fig.add_trace(go.Scatter(x=t, y=df["depth_m"], name="Depth",
                             line=dict(width=1, color="mediumpurple")), row=3, col=1)

    _title = title or f"Gaze Over Time — {meta.get('script', '')} | {meta.get('location', '')}"

    fig.update_layout(
        height=height,
        title_text=_title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis3=dict(title="Time (s)", rangeslider=dict(visible=True, thickness=0.05)),
    )
    fig.update_yaxes(title_text="Yaw (°)",   row=1, col=1)
    fig.update_yaxes(title_text="Pitch (°)", row=2, col=1)
    fig.update_yaxes(title_text="Depth (m)", row=3, col=1)

    return fig
