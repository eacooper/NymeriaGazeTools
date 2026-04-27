# Tests

## Running the tests

```bash
# from the repo root
pytest tests/              # run all tests
pytest tests/ -v           # verbose (shows each test name)
pytest tests/test_io.py    # one file only
pytest tests/ -k "filter"  # tests whose name contains "filter"
```

No real Nymeria data needed — all tests use synthetic DataFrames.

## What each file covers

### test_io.py
Functions in `nymeria_gaze_tools/io.py`:
- `_read_metadata_csv` — age-label remapping (`19-25` → `18-24`), `has_gaze_data` cast to bool
- `filter_sessions` — filtering by script, participant, gender, `has_gaze_data`; combined filters; unknown column raises `ValueError`
- `list_participants / list_activities / list_locations` — returns sorted unique values

### test_preprocessing.py
Functions in `nymeria_gaze_tools/preprocessing.py`, tested individually and end-to-end:
- `normalize_timestamps` — starts at 0, units are seconds, does not mutate input
- `convert_radians_to_degrees` — correct output columns, known-value accuracy
- `compute_binocular_gaze` — yaw is the left/right average, pitch is unchanged
- `compute_confidence_widths` — output columns exist, values are non-negative
- `remove_invalid_samples` — drops NaN rows, passes clean data through
- `compute_velocity` — output columns exist, angular = √(yaw²+pitch²), works without timestamps
- `compute_sampling_rate` — correct Hz from timestamps, NaN for single-row input
- `preprocess` (end-to-end) — expected output columns, no NaN in core columns, input not mutated

### test_metrics.py
Functions in `nymeria_gaze_tools/metrics.py`:
- `fixation_metrics` — correct keys, correct count/rate, all-NaN on empty input
- `saccade_metrics` — correct keys, correct saccade/artifact counts, all-NaN on empty input
- `session_summary` — returns a single-row DataFrame, has required columns, kwargs become columns, recording duration is correct, depth columns are NaN when `depth_m` is absent
