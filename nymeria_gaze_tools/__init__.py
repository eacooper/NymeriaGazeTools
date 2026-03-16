"""
nymeria_gaze_tools
==================
Python toolkit for analyzing eye gaze data from the Nymeria dataset.
"""

__version__ = "0.1.0"

# I-VT saccade detection default
DEFAULT_VELOCITY_THRESHOLD_DEG_S: float = 30.0

# I-DT fixation detection defaults
DEFAULT_DISPERSION_THRESHOLD_DEG: float = 1.0
DEFAULT_MIN_FIXATION_MS: float = 200.0

# Minimum saccade duration
DEFAULT_MIN_SACCADE_MS: float = 20.0

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

# from nymeria_gaze_tools.metrics import (
#     session_summary,
#     saccade_metrics,
#     fixation_metrics,
#     quality_report,
#     print_gaze_statistics,
# )

# from nymeria_gaze_tools.analysis import (
#     analyze_session,
#     compare_two_sessions,
#     compare_participant_sessions,
#     analyze_group,
#     compare_groups,
# )

# from nymeria_gaze_tools.stats import (
#     descriptive_stats,
#     run_anova,
#     run_ttest,
#     correlation_matrix,
#     compare_groups_statistically,
# )

# from nymeria_gaze_tools.viz import (
#     # Low-level plotting functions
#     plot_scatter_2d,
#     plot_timeseries,
#     plot_distribution,
#     plot_heatmap_2d,
#     plot_scatter_with_trend,
#     # High-level gaze-specific functions
#     plot_gaze_scatter_temporal,
#     plot_gaze_scatter_depth,
#     plot_gaze_overview,
#     plot_gaze_timeseries,
#     plot_distribution_summary,
#     plot_joint_distribution_summary,
#     plot_velocity_trace,
#     plot_velocity_histogram,
#     plot_main_sequence,
#     plot_fixation_spatial,
#     plot_fixation_summary,
# )
