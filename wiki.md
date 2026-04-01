# NymeriaGazeTools Wiki

Welcome to the NymeriaGazeTools wiki!
---

## 1. What is NymeriaGazeTools?

NymeriaGazeTools is a Python toolkit for analyzing eye gaze data from Meta's Nymeria dataset — a large-scale collection of recordings that capture where people look as they go through everyday activities like cooking, walking, or having a conversation.

The dataset contains over 1,100 recording sessions. Each session tracks a person's eye movements continuously, giving you a detailed log of where their gaze went, for how long, and how fast it moved. That's a substantial amount of data, and turning raw gaze signals into meaningful insights takes several steps.

This toolkit handles that process. It takes care of cleaning the raw data, identifying meaningful gaze events, and computing summary statistics — so you can focus on asking research questions rather than wrestling with data wrangling.

---

## 2. Key Concepts

Before diving in, here are a few terms that come up throughout this toolkit.

**Fixation** — A brief period where your eyes remain stable on a single point in space. This is when visual processing actually happens; the moments in between are largely skipped over by the brain.

**Saccade** — The fast, ballistic movement your eyes make when shifting from one fixation to the next. These typically last under 100 milliseconds and happen constantly, even when you feel like you're looking steadily at something.

**Yaw and Pitch** — The horizontal and vertical angles describing gaze direction. Yaw captures left-right movement; pitch captures up-down.

**Binocular Gaze** — A single gaze estimate derived by combining signals from both eyes. It tends to be more stable and reliable than relying on either eye individually.

**Vergence** — The degree to which your two eyes angle toward each other. When focusing on something nearby, the eyes converge inward; for distant objects, they align more in parallel.

**Confidence** — An uncertainty estimate attached to each gaze sample by the model. Samples with low confidence — often caused by blinks or partial occlusion — are typically filtered out before analysis.

**Velocity** — The rate at which gaze angle changes over time. High angular velocity indicates a saccade in progress; values near zero suggest the eye is holding a fixation.

---

## 3. About the Dataset

The Nymeria dataset contains **1,100 sessions** recorded across **236 participants**. Most participants completed 5 sessions (129 out of 236).

**Participants**

The dataset is roughly gender-balanced: 120 male and 116 female participants. Ethnicity coverage skews toward Caucasian participants (90), with East Asian (35), South Asian (34), African American (30), and Southeast Asian (30) each representing a meaningful share. Hispanic (10) and Other/Mixed (7) participants make up the remainder. Age ranges from 18 to 50, with the largest groups in the 25–30 (58 participants) and 18–24 (49 participants) brackets.

*[image: Who's in the Study — ethnicity, age group, gender breakdown]*

**Activities**

Sessions span 20 scripted activities:

| Script | Activity |
|---|---|
| S1 | Relax at home |
| S2 | Where is X |
| S3 | Welcome to my place |
| S4 | Body stretch |
| S5 | Workout |
| S6 | Dance |
| S7 | Cooking |
| S8 | Having a meal |
| S9 | Making a mess |
| S10 | Housekeeping |
| S11 | Laundry |
| S12 | Game night |
| S13 | Charades |
| S14 | By my desk |
| S15 | Do as I command |
| S16 | Simon says |
| S17 | In the office |
| S18 | Hike |
| S19 | Fresh air |
| S20 | Party |

Cooking (S7) has the most coverage with 154 sessions, followed by Simon Says (S16, 126) and Where is X (S2, 118). Some activities have limited representation — By My Desk (S14, 17 sessions) and Body Stretch (S4, 19 sessions) are the smallest.

*[image: Who's Doing What — recording volume by ethnicity, gender, age group, gaze type]*

**Gaze calibration**

969 sessions use personalized eye gaze (calibrated to the individual), while 131 use general gaze. This toolkit loads personalized gaze data if available, otherwise falls back to general.

**Sampling rate**

Most sessions are sampled at 10 Hz (943 out of 1,100). Twenty-six sessions from early recording dates were captured at 30 Hz. These come from 8 participants: angela_harrell, barbara_wheeler, christina_jones, heather_becker, james_johnson, jason_smith, shelby_arroyo, and virginia_rivera. The toolkit handles both rates correctly — `compute_sampling_rate()` infers the rate per session from the timestamps.

---

## 4. Getting Started

**Installation**

```bash
pip install nymeria_gaze_tools
```

**Getting your data**

The easiest way is to download the processed dataset from HuggingFace using the provided script:

```bash
python downloadScripts/download_from_hf.py --output /path/to/data
```

Alternatively, if you want the toolkit to fetch data on demand without downloading everything upfront, pass `source="huggingface"` to `load_metadata()` or `load_session()`:

```python
import nymeria_gaze_tools as ngt

catalog = ngt.load_metadata(source="huggingface")
raw = ngt.load_session("sequence_uid_here", source="huggingface")
```

This downloads and caches individual files as needed. You will need an HF token set in your environment:

```bash
export HF_TOKEN=your_token
```

**Your first analysis**

```python
import nymeria_gaze_tools as ngt

# Load the catalog and filter to sessions you care about
catalog = ngt.load_metadata(data_root="/path/to/data")
sessions = ngt.filter_sessions(catalog, script="S7-Cooking", has_gaze_data=True)

# Load a session and run the full pipeline
raw = ngt.load_session(sessions["sequence_uid"].iloc[0], data_root="/path/to/data")
result = ngt.analyze_session(raw)

# result.df         — preprocessed gaze data
# result.fixations  — fixation table
# result.saccades   — saccade table
# result.summary    — one-row metrics summary
# result.fig        — interactive Plotly figure
print(result.summary)
```

---

## 5. Your Data

The Nymeria dataset is organized around **sessions** — each one is a single recording of a participant going through an activity. There are over 1,100 of them in total.

There are two types of files you'll work with:

- **metadata.csv** — A catalog of all sessions. Each row is one session and includes details like the participant, activity, location, and demographic information.
- **Eye gaze CSVs** — One file per session, containing the raw gaze measurements sampled at roughly 10 times per second.

**Loading your data**

Load the metadata catalog with `load_metadata()`, which returns a DataFrame with one row per session. From there, use `filter_sessions()` to narrow down by participant, activity, location, age group, gender, or ethnicity. Once you have your filtered list, pass the `sequence_uid` values to `load_session()` or `load_sessions()` to load the corresponding gaze data.

To quickly check what is available in your catalog, use `list_participants()`, `list_activities()`, and `list_locations()`.

---

## 6. Preprocessing

Raw gaze data straight from the dataset isn't quite ready for analysis. The preprocessing step cleans it up and prepares it in a consistent format. Run the full pipeline at once with `preprocess()`, or apply individual steps if you need more control. Here's what happens, in order:

1. **Trim the recording** *(optional, default: no trimming)* — The start of a session often includes a calibration period you don't want in your analysis. You can trim a set number of minutes from the start or end.

2. **Normalize timestamps** — The raw timestamps are in microseconds from some arbitrary clock. `normalize_timestamps()` converts them to elapsed time in seconds, starting from zero.

3. **Convert units** — Gaze angles in the raw data are stored in radians. `convert_radians_to_degrees()` converts them to degrees, which are much easier to interpret.

4. **Compute sampling rate** — The actual recording rate is inferred from the timestamp intervals before any rows are removed, so it reflects the true signal rate. Use `compute_sampling_rate()` on the raw DataFrame before preprocessing.

5. **Remove invalid samples** — `remove_invalid_samples()` drops any rows with missing or null values.

6. **Compute binocular gaze** — `compute_binocular_gaze()` averages the left and right eye signals into a single estimate of where the person is looking. Vergence is also computed at this step.

7. **Compute confidence widths** — `compute_confidence_widths()` calculates how wide the model's confidence intervals are for each sample, which tells you how certain the model was.

8. **Filter low-confidence samples** *(optional, no default threshold)* — If a confidence interval is too wide — say, because the person blinked — that sample is dropped. You set the threshold.

9. **Compute velocity** — `compute_velocity()` derives angular velocity from the gaze angle over time. If timestamps aren't available, it assumes a 10 Hz sampling rate as a fallback.

---

## 7. Detecting Fixations & Saccades

Once your data is preprocessed, the next step is identifying the two core gaze events: fixations and saccades.

**Fixations**

The toolkit uses an algorithm called I-DT (Identification by Dispersion Threshold). The idea is straightforward: it slides a time window across the data and checks whether the gaze points inside it are tightly clustered. If they are, that window is labeled a fixation. It then keeps expanding the window forward until the gaze starts moving again.

Two parameters control this:
- **Dispersion threshold** *(default: 1°)* — How spread out the gaze can be and still count as a fixation. A smaller value means stricter fixations.
- **Minimum fixation duration** *(default: 200 ms)* — The shortest event that qualifies as a fixation. Anything briefer is ignored.

Use `get_fixation_table()` to get the result as a DataFrame, where each row is one fixation with its start time, end time, duration, average position, and sample count. For direct access to the list of fixation dicts, use `detect_fixations_idt()`.

**Saccades**

Saccades are derived from the gaps between consecutive fixations. At 10 Hz, the signal is too coarse to detect saccades directly from velocity alone — so instead, each gap between two fixations is treated as a saccade. The toolkit computes its amplitude (how far the eye moved) and peak velocity from the raw signal.

Gaps longer than **200 ms** are flagged as artifacts rather than saccades — these are likely blinks or signal dropouts.

Use `get_saccade_table()` to get the result as a DataFrame, where each row is one inter-fixation event labeled either `saccade` or `artifact`. For direct access to the list of dicts, use `detect_saccades()`.

---

## 8. Metrics

After detecting fixations and saccades, the toolkit can compute a concise set of numbers that summarize gaze behavior for a session. These are designed to be stacked across many sessions for comparison.

**Fixation metrics**

`fixation_metrics()` returns the following:

| Metric | What it means |
|---|---|
| `n_fixations` | Total number of fixations detected |
| `fixation_rate_per_min` | How many fixations occurred per minute |
| `mean_duration_ms` | Average fixation length in milliseconds |
| `median_duration_ms` | Median fixation length — less sensitive to outliers |
| `sd_duration_ms` | Standard deviation of fixation durations |
| `iqr_duration_ms` | Interquartile range of durations — spread of the middle 50% |
| `pct_time_in_fixation` | Percentage of the recording spent in a fixation |

**Saccade metrics**

`saccade_metrics()` returns the following:

| Metric | What it means |
|---|---|
| `n_saccades` | Total number of saccades detected |
| `n_artifacts` | Number of inter-fixation gaps flagged as artifacts (likely blinks or dropouts) |
| `mean_amplitude_deg` | Average distance the eye traveled per saccade, in degrees |
| `median_amplitude_deg` | Median saccade amplitude |
| `sd_amplitude_deg` | Standard deviation of saccade amplitudes |
| `mean_duration_ms` | Average saccade duration |
| `median_duration_ms` | Median saccade duration |

**Session summary**

`session_summary()` combines all of the above — plus recording duration, sampling rate, and mean vergence — into a single-row DataFrame for a session. This makes it straightforward to concatenate results across hundreds of sessions and run group-level analyses. For a complete single-session pipeline that returns all of these together, use `analyze_session()`, or `analyze_sessions()` for batch processing.

---

## 9. Visualizations

All plots are interactive — you can zoom, pan, and hover over data points. They're built with Plotly and work well in Jupyter notebooks.

---

**Gaze Time Series**

Shows yaw, pitch, and gaze depth over the full duration of a session, each in its own panel. The left and right eye signals are plotted separately alongside the binocular average, with shaded confidence bands. If you pass in a fixation table, fixation windows are shaded in green. Use `plot_gaze_timeseries(df, fixations=fixations)`.

---

**Gaze Scatter**

Plots any two gaze-related columns against each other, with a third variable mapped to color. By default it shows yaw vs pitch colored by time, which reveals where in the visual field the person was looking and in what order. Useful for exploring spatial gaze patterns. Use `plot_gaze_scatter(df)`.

---

**Gaze Heatmap**

A 2D density map showing where gaze was concentrated across the session. The color intensity reflects how many samples fell in each region. Good for identifying hotspots — areas the person looked at most frequently. Use `plot_gaze_heatmap(df)`.

---

**Velocity Trace**

Shows angular velocity over time. Spikes correspond to saccades; flat, low regions are fixations. If you pass in a fixation table, those periods are shaded in green. Helpful for visually validating your fixation detection results. Use `plot_velocity_trace(df, fixations=fixations)`.

---

**Main Sequence**

*TO DO*

---

**Population Density**

Similar to the heatmap, but built from multiple sessions at once. Each session is normalized by its sample count before averaging, so longer recordings don't dominate the result. Useful for understanding group-level gaze patterns across participants or activities. Use `plot_population_density(dfs)` for a single group, or `plot_population_density_grid(groups)` to compare multiple groups side by side.

---

## 10. Download Scripts

Two scripts are provided in `downloadScripts/` for getting the data onto your machine.

**download_from_hf.py**

Downloads the dataset from HuggingFace. This is the recommended path for most users.

```bash
python downloadScripts/download_from_hf.py --output /path/to/data
```

By default it downloads both `processed/` and `raw/`. If you only want the processed dataset to use with this toolkit:

```bash
python downloadScripts/download_from_hf.py --output /path/to/data --folder processed
```

If you want to explore the raw recordings yourself:

```bash
python downloadScripts/download_from_hf.py --output /path/to/data --folder raw
```

The script is resume-safe — it skips files that already exist. You can also do a dry run to preview what would be downloaded:

```bash
python downloadScripts/download_from_hf.py --output /path/to/data --dry-run
```

Requires `HF_TOKEN` set in your environment.

**download_eyegaze.py**

Downloads the dataset directly from Meta's Nymeria page using the official download URLs. Use this if you need the raw recordings or do not have HuggingFace access.

Before running, download the `nymeria_download_urls.json` file from the [Nymeria dataset page](https://www.projectaria.com/datasets/nymeria/) and place it at `data/nymeria_download_urls.json`. Then run:

```bash
python downloadScripts/download_eyegaze.py
```

This downloads the raw recordings, extracts the gaze CSVs, and builds the processed directory structure the toolkit expects.
