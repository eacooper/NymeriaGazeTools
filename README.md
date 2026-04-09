# NymeriaGazeTools

NymeriaGazeTools is a Python toolkit for analyzing eye gaze data from [Project Nymeria](https://www.projectaria.com/datasets/nymeria/), Meta's large-scale dataset of egocentric recordings captured during everyday activities.

It handles the full analysis pipeline — loading and filtering sessions, preprocessing raw signals, detecting fixations and saccades, computing summary metrics, and generating interactive visualizations.

## Installation

```bash
pip install nymeria_gaze_tools
```

Requires Python 3.10 or higher.

To load data directly from HuggingFace, set your token as an environment variable:

```bash
export HF_TOKEN=your_token
```

## Getting your data

The recommended way is to download directly from HuggingFace. This gives you the processed dataset ready to use:

```bash
python downloadScripts/download_from_hf.py --output /path/to/data
```

If you have the official `nymeria_download_urls.json` from Meta's Nymeria page and want to work from the raw recordings, use:

```bash
python downloadScripts/download_eyegaze.py
```

See the [wiki](wiki.md) for full details on both scripts.

## Quick start

```python
import nymeria_gaze_tools as ngt

# Load the session catalog and pick a session
catalog = ngt.load_metadata(data_root="/path/to/data")
sessions = ngt.filter_sessions(catalog, script="S7-Cooking", has_gaze_data=True)

# Load and analyze a single session
raw = ngt.load_session(sessions["sequence_uid"].iloc[0], data_root="/path/to/data")
result = ngt.analyze_session(raw)

# Result contains preprocessed data, fixations, saccades, summary, and a plot
print(result.summary)
```

## Examples

The `EDA/` folder contains Jupyter notebooks demonstrating how to use the toolkit:

- **individual_gaze_analysis.ipynb** — steps through the full pipeline manually for a single session: load, preprocess, detect fixations and saccades, then plot gaze time series, scatter, heatmaps, velocity trace, and the saccade main sequence
- **quick_analysis.ipynb** — uses the high-level `analyze_session()` and `analyze_sessions()` API to run the full pipeline in one call, covering both single-session and small-batch analysis, with a population density plot at the end
- **population_gaze_heatmaps.ipynb** — generates population-level gaze density heatmaps across all 20 activities, broken down by age group and by gender, using `plot_population_density_grid()`
