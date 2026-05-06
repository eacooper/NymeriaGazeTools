# NymeriaGazeTools

NymeriaGazeTools is a Python toolkit for analyzing eye gaze data from [Project Nymeria](https://www.projectaria.com/datasets/nymeria/), Meta's large-scale dataset of egocentric recordings captured during everyday activities.

It handles the full analysis pipeline — loading and filtering sessions, preprocessing raw signals, detecting fixations and saccades, computing summary metrics, and generating interactive visualizations.

## Installation

Requires Python 3.10 or higher.

**1. Clone the repo**

```bash
git clone https://github.com/eacooper/NymeriaGazeTools.git
cd NymeriaGazeTools
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
```

**3. Install the package**

```bash
pip install -e .
```

**4. Install Jupyter** (needed to run the example notebooks)

```bash
pip install jupyter ipywidgets
```

## Getting your data

The data is not included in the repo. The recommended way is to download from [HuggingFace](https://huggingface.co/datasets/nymeriagazedata/eye-gaze-data):

```bash
export HF_TOKEN=your_token
python downloadScripts/download_from_hf.py --output /path/to/data --folder processed
```

If you have the official `nymeria_download_urls.json` from Meta's Nymeria page and want to work from the raw recordings, use:

```bash
python downloadScripts/download_eyegaze.py
```

See the [wiki](https://github.com/eacooper/NymeriaGazeTools/wiki) for full details on both scripts.

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

The `examples/` folder contains Jupyter notebooks demonstrating how to use the toolkit:

- **individual_gaze_analysis.ipynb** — steps through the full pipeline manually for a single session: load, preprocess, detect fixations and saccades, then plot gaze time series, scatter, heatmaps, velocity trace, and the saccade main sequence *(experimental)*
- **quick_analysis.ipynb** — uses the high-level `analyze_session()` and `analyze_sessions()` API to run the full pipeline in one call, covering both single-session and small-batch analysis, with a population density plot at the end
- **population_gaze_heatmaps.ipynb** — generates population-level gaze density heatmaps across all 20 activities, broken down by age group and by gender, using `plot_population_density_grid()`
