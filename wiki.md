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

**Confidence** — An uncertainty estimate attached to each gaze sample by the model. Samples with low confidence — often caused by blinks or partial occlusion — are typically filtered out before analysis.

**Velocity** — The rate at which gaze angle changes over time. High angular velocity indicates a saccade in progress; values near zero suggest the eye is holding a fixation.

---

## 3. About the Dataset

The Nymeria dataset contains **1,100 sessions** recorded across **236 participants**. Most participants completed 5 sessions (129 out of 236).

**Participants**

The dataset is roughly gender-balanced: 120 male and 116 female participants. Ethnicity coverage skews toward Caucasian participants (90), with East Asian (35), South Asian (34), African American (30), and Southeast Asian (30) each representing a meaningful share. Hispanic (10) and Other/Mixed (7) participants make up the remainder. Age ranges from 18 to 50, with the largest groups in the 25–30 (58 participants) and 18–24 (49 participants) brackets.

<img width="400" height="400" alt="Participant_Ethnicity_Distribution" src="https://github.com/user-attachments/assets/23004e07-a4b9-4bef-a5b1-14ef27ad23a4" />


<img width="400" height="400" alt="Participant_AgeGroup_Distribution" src="https://github.com/user-attachments/assets/f3b7bc20-2ca4-4ef9-a91a-2e243fcf21e4" />


<img width="400" height="400" alt="Participant_Gender_Distribution" src="https://github.com/user-attachments/assets/b42d2429-341c-44bc-8c8d-70c1be065e12" />

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

<img width="732" height="484" alt="RecordingVolume_ethnicityxscript" src="https://github.com/user-attachments/assets/d40cff90-feb3-45dd-acd3-48ffc1a02249" />
<img width="672" height="464" alt="RecordingVolume_agexscript" src="https://github.com/user-attachments/assets/c920f9c9-d6d6-4fd2-9001-1c198e1175ba" />
<img width="432" height="492" alt="RecordingVolume_genderxscript" src="https://github.com/user-attachments/assets/39de7b23-b686-447b-baab-604a8fe96ce7" />
<img width="478" height="512" alt="RecordingVolume_gazeTypexscript" src="https://github.com/user-attachments/assets/b664c3e8-11a3-46ab-a3da-b9d5579cb52b" />
<img width="1076" height="747" alt="RecordingsPerParticipant_Histogram" src="https://github.com/user-attachments/assets/68247c07-27e6-498e-9582-71373fb151a3" />


**Gaze calibration**

969 sessions use personalized eye gaze (calibrated to the individual), while 131 use general gaze. This toolkit loads personalized gaze data if available, otherwise falls back to general.

**Sampling rate**

Most sessions are sampled at 10 Hz (943 out of 1,100). Twenty-six sessions from early recording dates were captured at 30 Hz. These come from 8 participants: angela_harrell, barbara_wheeler, christina_jones, heather_becker, james_johnson, jason_smith, shelby_arroyo, and virginia_rivera. The toolkit handles both rates correctly — `compute_sampling_rate()` infers the rate per session from the timestamps.

---

## 4. Getting the Data

Two scripts are provided in `downloadScripts/` for getting the data onto your machine.

**download_from_hf.py**

Downloads the processed dataset from HuggingFace. This is the recommended path for most users.

The dataset is hosted at [huggingface.co/datasets/nymeria-gaze](https://huggingface.co/datasets/nymeria-gaze). You will need a HuggingFace account and an access token set in your environment:

```bash
export HF_TOKEN=your_token
```

Then run:

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

**download_eyegaze.py**

Downloads the dataset directly from Meta's Nymeria page using the official download URLs. Use this if you need the raw recordings or do not have HuggingFace access.

Before running, download the `nymeria_download_urls.json` file from the [Nymeria dataset page](https://www.projectaria.com/datasets/nymeria/) and place it at `data/nymeria_download_urls.json`. Then run:

```bash
python downloadScripts/download_eyegaze.py
```

This downloads the raw recordings, extracts the gaze CSVs, and builds the processed directory structure the toolkit expects.

---

## 5. Getting Started

**Installation**

```bash
pip install nymeria_gaze_tools
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

## 6. Your Data

The Nymeria dataset is organized around **sessions** — each one is a single recording of a participant going through an activity. There are over 1,100 of them in total.

There are two types of files you'll work with:

- **metadata.csv** — A catalog of all sessions. Each row is one session and includes details like the participant, activity, location, and demographic information.
- **Eye gaze CSVs** — One file per session, containing the raw gaze measurements sampled at roughly 10 times per second.

**Loading your data**

Load the metadata catalog with `load_metadata()`, which returns a DataFrame with one row per session. From there, use `filter_sessions()` to narrow down by participant, activity, location, age group, gender, or ethnicity. Once you have your filtered list, pass the `sequence_uid` values to `load_session()` or `load_sessions()` to load the corresponding gaze data.

Both `load_metadata()` and `load_session()` accept a `source` argument. The default is `"local"`, which reads from your local `data/processed/` directory. If you haven't downloaded the data, you can load directly from HuggingFace by passing `source="huggingface"` — this requires `HF_TOKEN` to be set in your environment (see Section 4).

To quickly check what is available in your catalog, use `list_participants()`, `list_activities()`, and `list_locations()`.

**Known data issues**

Some sessions in the raw metadata carry the age group label `"19-25"`, which is a mislabeling — the correct label for that bracket is `"18-24"`. `load_metadata()` corrects this automatically at load time, so the returned catalog always uses `"18-24"`. No manual fix is needed in your code.

---

## 7. Preprocessing

Raw gaze data straight from the dataset isn't quite ready for analysis. The preprocessing step cleans it up and prepares it in a consistent format. Run the full pipeline at once with `preprocess()`, or apply individual steps if you need more control. Here's what happens, in order:

1. **Trim the recording** *(optional, default: no trimming)* — The start of a session often includes a calibration period you don't want in your analysis. You can trim a set number of minutes from the start or end.

2. **Normalize timestamps** — The raw timestamps are in microseconds from some arbitrary clock. `normalize_timestamps()` converts them to elapsed time in seconds, starting from zero.

3. **Convert units** — Gaze angles in the raw data are stored in radians. `convert_radians_to_degrees()` converts them to degrees, which are much easier to interpret.

4. **Compute sampling rate** — The actual recording rate is inferred from the timestamp intervals before any rows are removed, so it reflects the true signal rate. Use `compute_sampling_rate()` on the raw DataFrame before preprocessing.

5. **Remove invalid samples** — `remove_invalid_samples()` drops any rows with missing or null values.

6. **Compute binocular gaze** — `compute_binocular_gaze()` averages the left and right eye yaw signals into `avg_yaw_deg`. Note that the dataset provides separate left and right yaw measurements, but only a single pitch signal — so there is no left/right averaging for pitch, just `pitch_deg` directly.

7. **Compute confidence widths** — `compute_confidence_widths()` calculates how wide the model's confidence intervals are for each sample, which tells you how certain the model was.

8. **Filter low-confidence samples** *(optional, no default threshold)* — If a confidence interval is too wide — say, because the person blinked — that sample is dropped. Thresholds are set independently for yaw and pitch via `max_yaw_confidence_width_deg` and `max_pitch_confidence_width_deg`.

9. **Compute velocity** — `compute_velocity()` produces three columns: `yaw_velocity_deg_s` and `pitch_velocity_deg_s` for the component rates, and `angular_velocity_deg_s` for the combined speed magnitude. If timestamps aren't available, it assumes a 10 Hz sampling rate as a fallback.

---

## 8. Detecting Fixations & Saccades *(Experimental)*

> **Note:** Fixation and saccade detection in this toolkit is experimental. The 10 Hz sampling rate of most Nymeria sessions is lower than what standard eye-tracking algorithms are designed for, which limits detection accuracy. Treat these results as approximate and interpret them with care.

Once your data is preprocessed, you can identify fixations and saccades using `get_fixation_table()` and `get_saccade_table()`. For a complete single-session pipeline, use `analyze_session()`.

Fixations are detected using an I-DT (Identification by Dispersion Threshold) algorithm. Saccades are derived from the gaps between consecutive fixations rather than velocity peaks, since the 10 Hz signal is too coarse for direct velocity-based detection. Gaps longer than 200 ms are flagged as artifacts (likely blinks or dropouts) rather than saccades.

---

## 9. Metrics

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

`session_summary()` combines all of the above — plus recording duration and sampling rate — into a single-row DataFrame for a session. This makes it straightforward to concatenate results across hundreds of sessions and run group-level analyses. For a complete single-session pipeline that returns all of these together, use `analyze_session()`, or `analyze_sessions()` for batch processing. The batch function returns a `GroupResult` with two fields: `.summaries` — a single concatenated DataFrame with one row per session, ready for group-level analysis — and `.dfs` — a list of preprocessed DataFrames, one per session, which you can pass directly into the population density plots.

---

## 10. Visualizations

All plots are interactive — you can zoom, pan, and hover over data points. They're built with Plotly and work well in Jupyter notebooks.

---

**Gaze Time Series**

Shows yaw, pitch, and gaze depth over the full duration of a session, each in its own panel. The left and right eye signals are plotted separately alongside the binocular average, with shaded confidence bands. If you pass in a fixation table, fixation windows are shaded in green. Use `plot_gaze_timeseries(df, fixations=fixations)`.
<img width="1124" height="750" alt="gazetimeseries" src="https://github.com/user-attachments/assets/86b2d241-414e-4705-ac32-45c790cdc824" />

---

**Gaze Scatter**

Plots any two gaze-related columns against each other, with a third variable mapped to color. By default it shows yaw vs pitch colored by time, which reveals where in the visual field the person was looking and in what order. Useful for exploring spatial gaze patterns. Use `plot_gaze_scatter(df)`.
<img width="650" height="550" alt="gazescatter" src="https://github.com/user-attachments/assets/b197a55a-4ab8-4cae-960c-67c98aa31dfe" />

---

**Gaze Heatmap**

A 2D density map showing where gaze was concentrated across the session. The color intensity reflects how many samples fell in each region. Good 
<img width="550" height="550" alt="gazeheatmap" src="https://github.com/user-attachments/assets/6011e552-8bb3-4dcf-bd06-8c9cadd9c550" />
for identifying hotspots — areas the person looked at most frequently. Use `plot_gaze_heatmap(df)`.

---

**Velocity Trace**

Shows angular velocity over time. Spikes correspond to saccades; flat, low regions are fixations. If you pass in a fixation table, those periods are shaded in green. Helpful for visually validating your fixation detection results. Use `plot_velocity_trace(df, fixations=fixations)`.

<img width="1124" height="400" alt="gazevelocity" src="https://github.com/user-attachments/assets/9312762b-098f-40f0-8fe4-21302e56f721" />

---

**Main Sequence** *(Experimental)*

Plots saccade amplitude against peak velocity. Saccades appear in blue, artifacts in gray. Because saccade detection is experimental at 10 Hz, treat this as a rough qualitative check rather than precise measurement. Use `plot_main_sequence(saccades)`.

<img width="580" height="500" alt="mainsequence" src="https://github.com/user-attachments/assets/0325bea5-5bf3-4885-aef7-42d9712215f8" />

---

**Population Density**

Similar to the heatmap, but built from multiple sessions at once. Each session is normalized by its sample count before averaging, so longer recordings don't dominate the result. Useful for understanding group-level gaze patterns across participants or activities. Use `plot_population_density(dfs)` for a single group, or `plot_population_density_grid(groups)` to compare multiple groups side by side.
<img width="920" height="500" alt="population heatmaps" src="https://github.com/user-attachments/assets/0b2383f9-a0e1-4e5b-b5f3-36542da3edb8" />

---

**Gaze Position Boxplots**

The 1D counterpart to the population density map. Pools all gaze samples across sessions in each group and draws one box per group, making it easy to compare the spread and central tendency of yaw or pitch across demographics. Pass a `groups` dict mapping labels to lists of preprocessed DataFrames, and set `column` to `"avg_yaw_deg"` or `"pitch_deg"`. Use `plot_gaze_position_boxplots(groups, column="avg_yaw_deg")`.
<img width="1400" height="500" alt="boxwhisker" src="https://github.com/user-attachments/assets/cc7fb676-fb78-4d20-8828-3819f2d036f5" />

