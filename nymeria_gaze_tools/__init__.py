"""
nymeria_gaze_tools
==================
Python toolkit for analyzing eye gaze data from the Nymeria dataset.
"""

__version__ = "0.1.0"

# I-DT fixation detection defaults
DEFAULT_DISPERSION_THRESHOLD_DEG: float = 1.0
DEFAULT_MIN_FIXATION_MS: float = 200.0

# ---------------------------------------------------------------------------
# Re-export public API so callers can do: import nymeria_gaze_tools as ngt
# ---------------------------------------------------------------------------

from nymeria_gaze_tools.io import (
    load_metadata,
    load_session,
    load_sessions,
    # load_trajectory,
    filter_sessions,
    list_participants,
    list_activities,
    list_locations,
)

from nymeria_gaze_tools.preprocessing import (
    preprocess,
    # add_head_compensation,
    compute_sampling_rate,
    normalize_timestamps,
    convert_radians_to_degrees,
    remove_invalid_samples,
    compute_binocular_gaze,
    compute_confidence_widths,
    compute_velocity,
)

from nymeria_gaze_tools.events import (
    detect_fixations_idt,
    get_fixation_table,
    detect_saccades,
    get_saccade_table,
)

from nymeria_gaze_tools.metrics import (
    gaze_signal_metrics,
    fixation_metrics,
    saccade_metrics,
    session_summary,
)

from nymeria_gaze_tools.viz import (
    plot_gaze_timeseries,
    plot_gaze_scatter,
    plot_gaze_heatmap,
    plot_velocity_trace,
    plot_main_sequence,
    plot_population_density,
    plot_population_density_grid,
    plot_gaze_position_boxplots,
)

from nymeria_gaze_tools.analysis import (
    analyze_session,
    analyze_sessions,
    SessionResult,
    GroupResult,
)
