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

**3. Install the package and Jupyter dependencies**

```bash
pip install -e .
pip install jupyter ipywidgets
```

**4. Register the environment as a Jupyter kernel**

```bash
python -m ipykernel install --user --name=nymeria --display-name "Python (nymeria)"
```

This registers the virtual environment as a named kernel so any Jupyter installation on your machine (Anaconda, JupyterLab, system) can use it.

**5. Launch Jupyter and select the kernel**

```bash
jupyter notebook
```

When a notebook opens, go to **Kernel > Change Kernel > Python (nymeria)**. You only need to do this once per notebook — Jupyter remembers the kernel choice.

> **Troubleshooting:** If you see `ModuleNotFoundError` for `plotly` or `nymeria_gaze_tools`, the notebook is running on the wrong kernel. Check the kernel name in the top-right corner of the notebook and switch it to **Python (nymeria)**.

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

The `examples/` folder contains Jupyter notebooks demonstrating how to use the toolkit. You can view each notebook with interactive visualizations rendered online — no installation needed:

- **preprocessing_example.ipynb** ([view interactive notebook](https://eacooper.github.io/NymeriaGazeTools/examples/preprocessing_example.html)) — walks through preprocessing one step at a time for a single session, trimming the first 2 minutes and last 1 minute before normalizing timestamps, converting units, cleaning samples, computing binocular gaze, confidence widths, and velocity
- **individual_gaze_analysis.ipynb** ([view interactive notebook](https://eacooper.github.io/NymeriaGazeTools/examples/individual_gaze_analysis.html)) — steps through the full pipeline manually for a single session: load, preprocess, detect fixations and saccades, then plot gaze time series, scatter, heatmaps, velocity trace, and the saccade main sequence *(experimental)*
- **quick_analysis.ipynb** ([view interactive notebook](https://eacooper.github.io/NymeriaGazeTools/examples/quick_analysis.html)) — uses the high-level `analyze_session()` and `analyze_sessions()` API to run the full pipeline in one call, covering both single-session and small-batch analysis, with a population density plot at the end
- **population_gaze_heatmaps.ipynb** ([view interactive notebook](https://eacooper.github.io/NymeriaGazeTools/examples/population_gaze_heatmaps.html)) — generates population-level gaze density heatmaps across all 20 activities, broken down by age group and by gender, using `plot_population_density_grid()`
- **population_gaze_boxplots.ipynb** ([view interactive notebook](https://eacooper.github.io/NymeriaGazeTools/examples/population_gaze_boxplots.html)) — plots the distribution of raw yaw and pitch gaze positions across the population for each activity, grouped by age group and gender, using `plot_gaze_position_boxplots()`
- **cooking_analysis.ipynb** ([view interactive notebook](https://eacooper.github.io/NymeriaGazeTools/examples/cooking_analysis.html)) — deep-dives into S7-Cooking, the largest activity in the dataset (154 sessions). Computes gaze signal metrics for every session, then breaks down mean and variance of yaw, pitch, and depth by age group using `plot_gaze_position_boxplots()`


## Projects built with this toolkit

**Decoding Gaze** ([site](https://devi-sivakumar-ds.github.io/Decoding-Gaze/viz/) · [repo](https://github.com/devi-sivakumar-ds/Decoding-Gaze)) — an interactive D3.js visualization built for the Information Visualization course. It uses the HuggingFace dataset and `nymeria_gaze_tools` to explore how gaze position on an AR glasses lens varies by activity, with an interactive AR overlay design tool.

