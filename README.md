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

- **individual_gaze_analysis.ipynb** — load a single session, run the full pipeline, and explore gaze patterns through time series, heatmaps, and the saccade main sequence

More notebooks covering population-level and cross-activity analyses are in progress.
