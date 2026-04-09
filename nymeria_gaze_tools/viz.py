"""
viz.py — Interactive visualizations for Nymeria eye gaze data.

All functions return a Plotly figure — call .show() or display in a notebook.
"""

from __future__ import annotations

import math

import numpy as np
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
    fixations: pd.DataFrame = None,
    meta: dict = None,
    title: str = None,
    height: int = 750,
) -> go.Figure:
    """Interactive gaze time series: Yaw, Pitch, Depth (+ optional fixation shading).

    Pass fixations to shade fixation windows on Yaw and Pitch panels.
    """
    meta = meta or {}
    t    = df["elapsed_time_s"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Horizontal Gaze — Yaw", "Vertical Gaze — Pitch", "Gaze Depth"),
    )

    # fixation shading — build all shapes in one batch for performance
    if fixations is not None and not fixations.empty:
        yref_map = {1: "y domain", 2: "y2 domain"}
        shapes = [
            dict(type="rect",
                 x0=r["start_time_s"], x1=r["end_time_s"],
                 y0=0, y1=1,
                 xref="x", yref=yref_map[row],
                 fillcolor="rgba(0,180,0,0.15)", line_width=0)
            for _, r in fixations.iterrows()
            for row in (1, 2)
        ]
        fig.update_layout(shapes=shapes)

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


def plot_gaze_scatter(
    df: pd.DataFrame,
    x: str = "avg_yaw_deg",
    y: str = "pitch_deg",
    color: str = "elapsed_time_s",
    colormap: str = "viridis",
    meta: dict = None,
    title: str = None,
) -> go.Figure:
    """2D gaze scatter plot colored by a third variable.

    x, y, color accept any column name in df. Default: yaw vs pitch colored by time.
    colormap accepts any Plotly colorscale name (e.g. 'viridis', 'plasma', 'turbo').
    """
    meta   = meta or {}
    _title = title or f"{x}  vs  {y}  —  colored by {color}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y],
        mode="markers",
        marker=dict(size=3, color=df[color], colorscale=colormap, opacity=0.6,
                    colorbar=dict(title=color)),
        hovertemplate=f"{x}: %{{x:.2f}}<br>{y}: %{{y:.2f}}<br>{color}: %{{marker.color:.2f}}<extra></extra>",
    ))

    # crosshairs at origin for spatial reference
    fig.add_hline(y=0, line=dict(color="gray", width=0.7, dash="dash"))
    fig.add_vline(x=0, line=dict(color="gray", width=0.7, dash="dash"))

    fig.update_layout(
        title_text=_title,
        xaxis_title=x,
        yaxis_title=y,
        width=650, height=550,
        autosize=False,
    )

    return fig


def plot_gaze_heatmap(
    df: pd.DataFrame,
    x: str = "avg_yaw_deg",
    y: str = "pitch_deg",
    bins: int = 60,
    colormap: str = "YlOrRd",
    title: str = None,
) -> go.Figure:
    """2D density heatmap (count per bin) for any two gaze columns.

    Empty bins are transparent. colormap accepts any Plotly colorscale name.
    """
    _title = title or f"Joint distribution:  {x}  ×  {y}"

    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=df[x], y=df[y],
        nbinsx=bins, nbinsy=bins,
        colorscale=colormap,
        zmin=1,
        colorbar=dict(title="Count"),
    ))

    fig.add_hline(y=0, line=dict(color="white", width=0.8, dash="dash"))
    fig.add_vline(x=0, line=dict(color="white", width=0.8, dash="dash"))

    fig.update_layout(
        title_text=_title,
        xaxis_title=x,
        yaxis_title=y,
        width=550, height=550,
        autosize=False,
    )

    return fig


def plot_velocity_trace(
    df: pd.DataFrame,
    fixations: pd.DataFrame = None,
    meta: dict = None,
    title: str = None,
    height: int = 400,
) -> go.Figure:
    """Angular velocity over time with optional fixation shading."""
    meta   = meta or {}
    t      = df["elapsed_time_s"]
    _title = title or f"Angular Velocity — {meta.get('script', '')} | {meta.get('location', '')}".strip(" — |")

    fig = go.Figure()

    if fixations is not None and not fixations.empty:
        shapes = [
            dict(type="rect",
                 x0=r["start_time_s"], x1=r["end_time_s"],
                 y0=0, y1=1, yref="y domain",
                 fillcolor="rgba(0,180,0,0.15)", line_width=0)
            for _, r in fixations.iterrows()
        ]
        fig.update_layout(shapes=shapes)

    fig.add_trace(go.Scatter(
        x=t, y=df["angular_velocity_deg_s"],
        mode="lines",
        line=dict(width=0.8, color="crimson"),
        name="Angular velocity",
    ))

    fig.update_layout(
        title_text=_title,
        height=height,
        xaxis=dict(title="Time (s)", rangeslider=dict(visible=True, thickness=0.05)),
        yaxis_title="Speed (°/s)",
        showlegend=False,
    )

    return fig


def plot_main_sequence(
    saccades: pd.DataFrame,
    title: str = None,
) -> go.Figure:
    """Saccade amplitude vs peak velocity (main sequence).

    Saccades in blue, artifacts in gray.
    Requires peak_velocity_deg_s — run get_saccade_table(fixations, df=df).
    """
    sac = saccades[saccades["event_type"] == "saccade"]
    art = saccades[saccades["event_type"] == "artifact"]

    fig = go.Figure()

    if not art.empty:
        fig.add_trace(go.Scatter(
            x=art["amplitude_deg"], y=art["peak_velocity_deg_s"],
            mode="markers", name="Artifact",
            marker=dict(size=4, color="lightgray", opacity=0.5),
        ))

    if not sac.empty:
        fig.add_trace(go.Scatter(
            x=sac["amplitude_deg"], y=sac["peak_velocity_deg_s"],
            mode="markers", name="Saccade",
            marker=dict(size=5, color="royalblue", opacity=0.7),
        ))


    fig.update_layout(
        title_text=title or "Main Sequence — Saccade Amplitude vs Peak Velocity",
        xaxis_title="Amplitude (°)",
        yaxis_title="Peak Velocity (°/s)",
        width=580, height=500,
        autosize=False,
    )

    return fig


def plot_population_density(
    dfs: list[pd.DataFrame],
    x: str = "avg_yaw_deg",
    y: str = "pitch_deg",
    bins: int = 80,
    colormap: str = "viridis",
    title: str = None,
) -> go.Figure:
    """Population-level 2D density map averaged across participants.

    Each df is normalized by its sample count before averaging so all
    participants contribute equally regardless of session length.
    """
    _title = title or f"Population density:  {x}  ×  {y}  (n={len(dfs)})"

    # consistent grid across all participants
    x_all    = pd.concat([df[x].dropna() for df in dfs])
    y_all    = pd.concat([df[y].dropna() for df in dfs])
    x_edges  = np.linspace(x_all.min(), x_all.max(), bins + 1)
    y_edges  = np.linspace(y_all.min(), y_all.max(), bins + 1)

    maps = []
    for df in dfs:
        h, _, _ = np.histogram2d(df[x].dropna(), df[y].dropna(), bins=[x_edges, y_edges])
        total = h.sum()
        if total > 0:
            maps.append(h / total)

    z         = np.mean(maps, axis=0).T          # (ybins, xbins) for go.Heatmap
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x_centers, y=y_centers, z=z,
        colorscale=colormap,
        colorbar=dict(title="Density"),
    ))

    fig.add_hline(y=0, line=dict(color="white", width=0.8, dash="dash"))
    fig.add_vline(x=0, line=dict(color="white", width=0.8, dash="dash"))

    fig.update_layout(
        title_text=_title,
        xaxis_title=x,
        yaxis_title=y,
        width=550, height=550,
        autosize=False,
    )

    return fig


def plot_population_density_grid(
    groups: dict[str, list[pd.DataFrame]],
    x: str = "avg_yaw_deg",
    y: str = "pitch_deg",
    bins: int = 80,
    colormap: str = "viridis",
    title: str = None,
    panel_size: int = 420,
    n_cols: int = None,
) -> go.Figure:
    """Side-by-side population density heatmaps for multiple groups.

    All panels share the same axis ranges and color scale for fair comparison.

    Parameters
    ----------
    groups : dict mapping label -> list of preprocessed DataFrames
        e.g. {"Male": [df1, df2], "Female": [df3, df4]}
    panel_size : int
        Width and height of each individual panel in pixels.
    n_cols : int, optional
        Number of columns. Defaults to all panels in one row.
        e.g. n_cols=2 wraps 6 groups into a 3x2 grid.
    """
    labels = [lbl for lbl, dfs in groups.items() if dfs]
    n      = len(labels)

    if n == 0:
        raise ValueError("No groups with data provided.")

    n_cols = n_cols or n
    n_rows = math.ceil(n / n_cols)

    # shared grid so all panels use identical axes
    all_dfs   = [df for dfs in groups.values() for df in dfs]
    x_all     = pd.concat([df[x].dropna() for df in all_dfs])
    y_all     = pd.concat([df[y].dropna() for df in all_dfs])
    x_edges   = np.linspace(x_all.min(), x_all.max(), bins + 1)
    y_edges   = np.linspace(y_all.min(), y_all.max(), bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # compute per-group density; track shared color range
    z_maps = []
    for lbl in labels:
        maps = []
        for df in groups[lbl]:
            h, _, _ = np.histogram2d(df[x].dropna(), df[y].dropna(),
                                     bins=[x_edges, y_edges])
            total = h.sum()
            if total > 0:
                maps.append(h / total)
        z_maps.append(np.mean(maps, axis=0).T if maps else np.zeros((bins, bins)))

    zmin = min(z.min() for z in z_maps)
    zmax = max(z.max() for z in z_maps)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{lbl}  (n={len(groups[lbl])})" for lbl in labels],
        horizontal_spacing=0.06,
        vertical_spacing=0.1,
    )

    for idx, (lbl, z) in enumerate(zip(labels, z_maps)):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        show_colorbar = idx == n - 1
        fig.add_trace(
            go.Heatmap(
                x=x_centers, y=y_centers, z=z,
                colorscale=colormap,
                zmin=zmin, zmax=zmax,
                colorbar=dict(title="Density") if show_colorbar else None,
                showscale=show_colorbar,
            ),
            row=row, col=col,
        )
        fig.add_hline(y=0, line=dict(color="white", width=0.8, dash="dash"),
                      row=row, col=col)
        fig.add_vline(x=0, line=dict(color="white", width=0.8, dash="dash"),
                      row=row, col=col)

    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y, col=1)

    fig.update_layout(
        title_text=title or f"Population density by group:  {x}  ×  {y}",
        width=panel_size * n_cols + 80,
        height=panel_size * n_rows + 80,
        autosize=False,
    )

    return fig
