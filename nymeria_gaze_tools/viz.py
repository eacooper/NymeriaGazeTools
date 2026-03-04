"""
viz.py — Visualization functions for eye gaze data using Plotly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Low-level plotting functions (maximally reusable)

def plot_scatter_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    size: str = None,
    colorscale: str = "Viridis",
    marker_size: int = 3,
    opacity: float = 0.6,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    show_crosshairs: bool = True,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Generic 2D scatter plot."""
    fig = go.Figure()

    marker_dict = dict(
        size=marker_size if size is None else df[size],
        opacity=opacity,
    )

    if color is not None:
        marker_dict['color'] = df[color]
        marker_dict['colorscale'] = colorscale
        marker_dict['colorbar'] = dict(title=color)

    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y],
        mode='markers',
        marker=marker_dict,
        showlegend=False,
    ))

    if show_crosshairs:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or y,
        title=title,
        width=width,
        height=height,
    )

    return fig


def plot_timeseries(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str] | str,
    labels: list[str] = None,
    colors: list[str] = None,
    show_confidence: bool = False,
    confidence_cols: dict = None,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """Generic time series plot with optional confidence bands."""
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    fig = go.Figure()

    for i, col in enumerate(y_cols):
        color = colors[i] if colors and i < len(colors) else None
        label = labels[i] if labels and i < len(labels) else col

        # Add confidence band if requested
        if show_confidence and confidence_cols and col in confidence_cols:
            low_col, high_col = confidence_cols[col]

            # Upper bound
            fig.add_trace(go.Scatter(
                x=df[x],
                y=df[high_col],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
            ))

            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=df[x],
                y=df[low_col],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba(100,100,100,0.2)' if not color else None,
                showlegend=False,
            ))

        # Main trace
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[col],
            mode='lines',
            name=label,
            line=dict(color=color) if color else {},
        ))

    fig.update_layout(
        xaxis_title=xlabel or x,
        yaxis_title=ylabel,
        title=title,
        width=width,
        height=height,
        hovermode='x unified',
    )

    return fig


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 80,
    show_kde: bool = True,
    show_mean: bool = True,
    show_median: bool = True,
    color: str = 'blue',
    xlabel: str = None,
    title: str = None,
    width: int = 600,
    height: int = 400,
) -> go.Figure:
    """Distribution plot with optional KDE, mean, and median lines."""
    data = df[column].dropna().values

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        histnorm='probability density' if show_kde else '',
        marker_color=color,
        opacity=0.6,
        name='Histogram',
    ))

    # KDE
    if show_kde:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 500)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode='lines',
            name='KDE',
            line=dict(color=color, width=2.5),
        ))

    # Mean line
    if show_mean:
        mean_val = data.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean = {mean_val:.2f}",
            annotation_position="top",
        )

    # Median line
    if show_median:
        median_val = np.median(data)
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Median = {median_val:.2f}",
            annotation_position="bottom",
        )

    fig.update_layout(
        xaxis_title=xlabel or column,
        yaxis_title='Density' if show_kde else 'Count',
        title=title or f'{column} Distribution',
        width=width,
        height=height,
    )

    return fig


def plot_heatmap_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    bins: int = 60,
    colorscale: str = 'YlOrRd',
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    show_crosshairs: bool = True,
    width: int = 600,
    height: int = 600,
) -> go.Figure:
    """2D histogram heatmap."""
    fig = go.Figure()

    fig.add_trace(go.Histogram2d(
        x=df[x],
        y=df[y],
        nbinsx=bins,
        nbinsy=bins,
        colorscale=colorscale,
        colorbar=dict(title='Count'),
    ))

    if show_crosshairs:
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.6)
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.6)

    fig.update_layout(
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or y,
        title=title or f'{x} vs {y}',
        width=width,
        height=height,
    )

    return fig


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    fit_type: str = 'power',
    colorscale: str = 'Viridis',
    marker_size: int = 20,
    opacity: float = 0.7,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Scatter plot with power-law or linear trend line."""
    fig = go.Figure()

    x_data = df[x].values
    y_data = df[y].values

    marker_dict = dict(size=marker_size, opacity=opacity)

    if color is not None:
        marker_dict['color'] = df[color]
        marker_dict['colorscale'] = colorscale
        marker_dict['colorbar'] = dict(title=color)

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=marker_dict,
        showlegend=False,
    ))

    # Fit trend line
    valid = (x_data > 0.3) & ~np.isnan(x_data) & ~np.isnan(y_data)
    if valid.sum() > 5:
        if fit_type == 'power':
            coeffs = np.polyfit(np.log(x_data[valid]), np.log(y_data[valid]), 1)
            x_fit = np.linspace(x_data[valid].min(), x_data[valid].max(), 200)
            y_fit = np.exp(coeffs[1]) * x_fit ** coeffs[0]

            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name=f'Power law: y = {np.exp(coeffs[1]):.1f} · x^{coeffs[0]:.2f}',
                line=dict(color='red', width=2),
            ))
        elif fit_type == 'linear':
            coeffs = np.polyfit(x_data[valid], y_data[valid], 1)
            x_fit = np.linspace(x_data[valid].min(), x_data[valid].max(), 200)
            y_fit = coeffs[0] * x_fit + coeffs[1]

            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                name=f'Linear: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}',
                line=dict(color='red', width=2),
            ))

    fig.update_layout(
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or y,
        title=title,
        width=width,
        height=height,
        showlegend=True,
    )

    return fig


# High-level convenience wrappers for gaze-specific plots

def plot_gaze_scatter_temporal(
    df: pd.DataFrame,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Gaze scatter plot colored by time."""
    return plot_scatter_2d(
        df,
        x='avg_yaw_deg',
        y='pitch_deg',
        color='elapsed_time_s',
        colorscale='Viridis',
        xlabel='Yaw (°)',
        ylabel='Pitch (°)',
        title='Gaze Map (Colored by Time)',
        width=width,
        height=height,
    )


def plot_gaze_scatter_depth(
    df: pd.DataFrame,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Gaze scatter plot colored by depth."""
    return plot_scatter_2d(
        df,
        x='avg_yaw_deg',
        y='pitch_deg',
        color='depth_m',
        colorscale='Plasma',
        xlabel='Yaw (°)',
        ylabel='Pitch (°)',
        title='Gaze Map (Colored by Depth)',
        width=width,
        height=height,
    )


def plot_gaze_overview(
    df: pd.DataFrame,
    width: int = 1400,
    height: int = 600,
) -> go.Figure:
    """Dual gaze scatter plots: time and depth side-by-side."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Colored by Time', 'Colored by Depth'),
    )

    # Time-colored scatter
    fig.add_trace(
        go.Scatter(
            x=df['avg_yaw_deg'],
            y=df['pitch_deg'],
            mode='markers',
            marker=dict(
                size=3,
                color=df['elapsed_time_s'],
                colorscale='Viridis',
                colorbar=dict(title='Time (s)', x=0.45),
                opacity=0.6,
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Depth-colored scatter
    fig.add_trace(
        go.Scatter(
            x=df['avg_yaw_deg'],
            y=df['pitch_deg'],
            mode='markers',
            marker=dict(
                size=3,
                color=df['depth_m'],
                colorscale='Plasma',
                colorbar=dict(title='Depth (m)', x=1.02),
                opacity=0.6,
            ),
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Add crosshairs to both
    for col in [1, 2]:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=col)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=col)

    fig.update_xaxes(title_text='Yaw (°)', row=1, col=1)
    fig.update_xaxes(title_text='Yaw (°)', row=1, col=2)
    fig.update_yaxes(title_text='Pitch (°)', row=1, col=1)
    fig.update_yaxes(title_text='Pitch (°)', row=1, col=2)

    fig.update_layout(
        title='2D Gaze Map',
        width=width,
        height=height,
    )

    return fig


def plot_gaze_timeseries(
    df: pd.DataFrame,
    show_confidence: bool = True,
    width: int = 1000,
    height: int = 750,
) -> go.Figure:
    """Time series of yaw, pitch, and depth with optional confidence bands."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Horizontal Gaze — Yaw', 'Vertical Gaze — Pitch', 'Gaze Depth'),
    )

    t = df['elapsed_time_s']

    # Yaw
    if show_confidence:
        for eye, color in [('left', 'royalblue'), ('right', 'darkorange')]:
            # Confidence band
            fig.add_trace(go.Scatter(
                x=t, y=df[f'{eye}_yaw_high_deg'],
                mode='lines', line=dict(width=0),
                showlegend=False,
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=t, y=df[f'{eye}_yaw_low_deg'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=f'rgba(100,100,100,0.15)',
                showlegend=False,
            ), row=1, col=1)

            # Main line
            fig.add_trace(go.Scatter(
                x=t, y=df[f'{eye}_yaw_deg'],
                mode='lines', name=f'{eye.capitalize()} Eye',
                line=dict(color=color, width=1.3),
            ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=df['avg_yaw_deg'],
        mode='lines', name='Avg (binocular)',
        line=dict(color='black', width=2),
    ), row=1, col=1)

    # Pitch
    if show_confidence:
        fig.add_trace(go.Scatter(
            x=t, y=df['pitch_high_deg'],
            mode='lines', line=dict(width=0),
            showlegend=False,
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=t, y=df['pitch_low_deg'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(100,100,100,0.15)',
            showlegend=False,
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=df['pitch_deg'],
        mode='lines', name='Pitch',
        line=dict(color='seagreen', width=1.3),
    ), row=2, col=1)

    # Depth
    fig.add_trace(go.Scatter(
        x=t, y=df['depth_m'],
        mode='lines', name='Depth',
        line=dict(color='mediumpurple', width=1),
    ), row=3, col=1)

    fig.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig.update_yaxes(title_text='Yaw (°)', row=1, col=1)
    fig.update_yaxes(title_text='Pitch (°)', row=2, col=1)
    fig.update_yaxes(title_text='Depth (m)', row=3, col=1)

    fig.update_layout(
        title='Gaze Over Time',
        width=width,
        height=height,
        hovermode='x unified',
    )

    return fig


def plot_distribution_summary(
    df: pd.DataFrame,
    columns: list[str] = None,
    colors: list[str] = None,
    width: int = 1600,
    height: int = 500,
) -> go.Figure:
    """Three distribution plots side-by-side."""
    if columns is None:
        columns = ['avg_yaw_deg', 'pitch_deg', 'depth_m']

    if colors is None:
        colors = ['royalblue', 'seagreen', 'purple']

    labels = {
        'avg_yaw_deg': 'Average Yaw (°)',
        'pitch_deg': 'Pitch (°)',
        'depth_m': 'Depth (m)',
    }

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[labels.get(col, col) for col in columns],
    )

    for i, (col, color) in enumerate(zip(columns, colors), 1):
        data = df[col].dropna().values

        # Histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=80,
            histnorm='probability density',
            marker_color=color,
            opacity=0.6,
            showlegend=False,
        ), row=1, col=i)

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 500)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode='lines',
            line=dict(color=color, width=2.5),
            showlegend=False,
        ), row=1, col=i)

        # Mean and median
        fig.add_vline(x=data.mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean={data.mean():.2f}",
                     annotation_position="top right", row=1, col=i)
        fig.add_vline(x=np.median(data), line_dash="dot", line_color="orange",
                     annotation_text=f"Median={np.median(data):.2f}",
                     annotation_position="bottom right", row=1, col=i)

        fig.update_xaxes(title_text=labels.get(col, col), row=1, col=i)
        fig.update_yaxes(title_text='Density', row=1, col=i)

    fig.update_layout(
        title='Gaze Distribution Summary',
        width=width,
        height=height,
    )

    return fig


def plot_joint_distribution_summary(
    df: pd.DataFrame,
    width: int = 1600,
    height: int = 500,
) -> go.Figure:
    """Three joint distribution heatmaps side-by-side."""
    pairs = [
        ('avg_yaw_deg', 'pitch_deg', 'Yaw (°)', 'Pitch (°)', 'YlOrRd'),
        ('avg_yaw_deg', 'depth_m', 'Yaw (°)', 'Depth (m)', 'YlGnBu'),
        ('pitch_deg', 'depth_m', 'Pitch (°)', 'Depth (m)', 'PuRd'),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{xl} × {yl}' for _, _, xl, yl, _ in pairs],
    )

    for i, (xc, yc, xl, yl, cmap) in enumerate(pairs, 1):
        fig.add_trace(go.Histogram2d(
            x=df[xc],
            y=df[yc],
            nbinsx=60,
            nbinsy=60,
            colorscale=cmap,
            colorbar=dict(title='Count', x=0.3 * i - 0.05) if i == 3 else None,
            showscale=(i == 3),
        ), row=1, col=i)

        # Crosshairs
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.6, row=1, col=i)
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.6, row=1, col=i)

        fig.update_xaxes(title_text=xl, row=1, col=i)
        fig.update_yaxes(title_text=yl, row=1, col=i)

    fig.update_layout(
        title='Joint Distributions',
        width=width,
        height=height,
    )

    return fig


def plot_velocity_trace(
    df: pd.DataFrame,
    threshold: float = None,
    width: int = 1000,
    height: int = 400,
) -> go.Figure:
    """Velocity time series with optional threshold line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['elapsed_time_s'],
        y=df['angular_velocity_deg_s'],
        mode='lines',
        name='Angular velocity',
        line=dict(color='crimson', width=0.8),
    ))

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f'Saccade threshold ({threshold}°/s)',
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Speed (°/s)',
        title='Eye Velocity Over Time',
        width=width,
        height=height,
        hovermode='x',
    )

    return fig


def plot_velocity_histogram(
    df: pd.DataFrame,
    threshold: float = None,
    bins: int = 120,
    width: int = 800,
    height: int = 400,
) -> go.Figure:
    """Velocity distribution with optional threshold line."""
    vel_data = df['angular_velocity_deg_s'].dropna().values

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=vel_data,
        nbinsx=bins,
        marker_color='crimson',
        opacity=0.7,
        showlegend=False,
    ))

    # Mean
    fig.add_vline(
        x=vel_data.mean(),
        line_dash="dash",
        line_color="black",
        annotation_text=f'Mean = {vel_data.mean():.1f} °/s',
        annotation_position="top right",
    )

    # Median
    fig.add_vline(
        x=np.median(vel_data),
        line_dash="dot",
        line_color="orange",
        annotation_text=f'Median = {np.median(vel_data):.1f} °/s',
        annotation_position="top left",
    )

    # Threshold
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="solid",
            line_color="red",
            line_width=2,
            annotation_text=f'Threshold = {threshold} °/s',
            annotation_position="bottom right",
        )

    fig.update_layout(
        xaxis_title='Angular Speed (°/s)',
        yaxis_title='Count',
        title='Eye Velocity Distribution',
        width=width,
        height=height,
    )

    fig.update_xaxes(range=[0, None])

    return fig


def plot_main_sequence(
    saccade_table: pd.DataFrame,
    fit_type: str = 'power',
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Main sequence plot: saccade amplitude vs peak velocity."""
    return plot_scatter_with_trend(
        saccade_table,
        x='amplitude_deg',
        y='peak_velocity_deg_s',
        color='duration_ms',
        fit_type=fit_type,
        colorscale='Viridis',
        marker_size=20,
        xlabel='Saccade Amplitude (°)',
        ylabel='Peak Velocity (°/s)',
        title='Main Sequence',
        width=width,
        height=height,
    )


def plot_fixation_spatial(
    fixation_table: pd.DataFrame,
    color_by: str = 'duration_ms',
    size_by: str = 'duration_ms',
    colorscale: str = 'Hot',
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Fixation locations sized and colored by duration or dispersion."""
    size_data = fixation_table[size_by]
    sizes = (size_data / size_data.max() * 40 + 5).values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fixation_table['centroid_yaw_deg'],
        y=fixation_table['centroid_pitch_deg'],
        mode='markers',
        marker=dict(
            size=sizes,
            color=fixation_table[color_by],
            colorscale=colorscale,
            colorbar=dict(title=color_by),
            opacity=0.75,
            line=dict(color='black', width=0.3),
        ),
        showlegend=False,
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        xaxis_title='Yaw (°)',
        yaxis_title='Pitch (°)',
        title='Fixation Locations (size & color = duration)',
        width=width,
        height=height,
    )

    return fig


def plot_fixation_summary(
    fixation_table: pd.DataFrame,
    width: int = 1400,
    height: int = 600,
) -> go.Figure:
    """Fixation duration histogram and spatial map side-by-side."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Duration Distribution', 'Spatial Locations'),
    )

    # Duration histogram
    dur_data = fixation_table['duration_ms'].values

    fig.add_trace(go.Histogram(
        x=dur_data,
        nbinsx=40,
        marker_color='teal',
        opacity=0.7,
        showlegend=False,
    ), row=1, col=1)

    fig.add_vline(
        x=dur_data.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f'Mean = {dur_data.mean():.0f} ms',
        row=1, col=1,
    )

    fig.add_vline(
        x=np.median(dur_data),
        line_dash="dot",
        line_color="orange",
        annotation_text=f'Median = {np.median(dur_data):.0f} ms',
        row=1, col=1,
    )

    # Spatial map
    sizes = (dur_data / dur_data.max() * 40 + 5)

    fig.add_trace(go.Scatter(
        x=fixation_table['centroid_yaw_deg'],
        y=fixation_table['centroid_pitch_deg'],
        mode='markers',
        marker=dict(
            size=sizes,
            color=dur_data,
            colorscale='Hot',
            colorbar=dict(title='Duration (ms)', x=1.02),
            opacity=0.75,
            line=dict(color='black', width=0.3),
        ),
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

    fig.update_xaxes(title_text='Fixation Duration (ms)', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_xaxes(title_text='Yaw (°)', row=1, col=2)
    fig.update_yaxes(title_text='Pitch (°)', row=1, col=2)

    fig.update_layout(
        title='Fixation Summary',
        width=width,
        height=height,
    )

    return fig
