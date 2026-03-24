# NymeriaGazeTools Wiki

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

## 3. Getting Started

*TO DO*

---

## 4. Your Data

The Nymeria dataset is organized around **sessions** — each one is a single recording of a participant going through an activity. There are over 1,100 of them in total.

There are two types of files you'll work with:

- **metadata.csv** — A catalog of all sessions. Each row is one session and includes details like the participant, activity, location, and demographic information.
- **Eye gaze CSVs** — One file per session, containing the raw gaze measurements sampled at roughly 10 times per second.

**Loading your data**

You start by loading the metadata catalog, which gives you an overview of all available sessions. From there, you can filter down to the sessions you actually care about — by participant, activity, location, age group, gender, or ethnicity. Once you have your filtered list, you load the corresponding gaze data for those sessions.

The toolkit also lets you quickly check what participants, activities, and locations are available in your catalog, which is handy when you're first getting oriented.

---

## 5. Preprocessing

Raw gaze data straight from the dataset isn't quite ready for analysis. The preprocessing step cleans it up and prepares it in a consistent format. Here's what happens, in order:

1. **Trim the recording** *(optional, default: no trimming)* — The start of a session often includes a calibration period you don't want in your analysis. You can trim a set number of minutes from the start or end.

2. **Normalize timestamps** — The raw timestamps are in microseconds from some arbitrary clock. This converts them to elapsed time in seconds, starting from zero.

3. **Convert units** — Gaze angles in the raw data are stored in radians. These get converted to degrees, which are much easier to interpret.

4. **Compute sampling rate** — The actual recording rate is inferred from the timestamp intervals before any rows are removed, so it reflects the true signal rate.

5. **Remove invalid samples** — Any rows with missing or null values are dropped.

6. **Compute binocular gaze** — The left and right eye signals are averaged into a single estimate of where the person is looking. Vergence is also computed at this step.

7. **Compute confidence widths** — Each sample comes with a confidence interval from the model. This step calculates how wide those intervals are, which tells you how certain the model was.

8. **Filter low-confidence samples** *(optional, no default threshold)* — If a confidence interval is too wide — say, because the person blinked — that sample is dropped. You set the threshold.

9. **Compute velocity** — Angular velocity is derived from the gaze angle over time. If timestamps aren't available, it assumes a 10 Hz sampling rate as a fallback.

You can run all of these steps at once, or apply them individually if you need more control.

---

## 6. Detecting Fixations & Saccades

Once your data is preprocessed, the next step is identifying the two core gaze events: fixations and saccades.

**Fixations**

The toolkit uses an algorithm called I-DT (Identification by Dispersion Threshold). The idea is straightforward: it slides a time window across the data and checks whether the gaze points inside it are tightly clustered. If they are, that window is labeled a fixation. It then keeps expanding the window forward until the gaze starts moving again.

Two parameters control this:
- **Dispersion threshold** *(default: 1°)* — How spread out the gaze can be and still count as a fixation. A smaller value means stricter fixations.
- **Minimum fixation duration** *(default: 200 ms)* — The shortest event that qualifies as a fixation. Anything briefer is ignored.

The result is a table where each row is one fixation, with its start time, end time, duration, average position, and sample count.

**Saccades**

Saccades are derived from the gaps between consecutive fixations. At 10 Hz, the signal is too coarse to detect saccades directly from velocity alone — so instead, each gap between two fixations is treated as a saccade. The toolkit computes its amplitude (how far the eye moved) and peak velocity from the raw signal.

Gaps longer than **200 ms** are flagged as artifacts rather than saccades — these are likely blinks or signal dropouts.

The result is a table where each row is one inter-fixation event, labeled either `saccade` or `artifact`.

---

## 7. Metrics

After detecting fixations and saccades, the toolkit can compute a concise set of numbers that summarize gaze behavior for a session. These are designed to be stacked across many sessions for comparison.

**Fixation metrics**

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

You can combine all of the above — plus recording duration, sampling rate, and mean vergence — into a single-row summary for a session. This makes it straightforward to concatenate results across hundreds of sessions and run group-level analyses.

---

## 8. Visualizations

All plots are interactive — you can zoom, pan, and hover over data points. They're built with Plotly and work well in Jupyter notebooks.

---

**Gaze Time Series**

Shows yaw, pitch, and gaze depth over the full duration of a session, each in its own panel. The left and right eye signals are plotted separately alongside the binocular average, with shaded confidence bands. If you pass in a fixation table, fixation windows are shaded in green. 

---

**Gaze Scatter**

Plots any two gaze-related columns against each other, with a third variable mapped to color. By default it shows yaw vs pitch colored by time, which reveals where in the visual field the person was looking and in what order. Useful for exploring spatial gaze patterns.

---

**Gaze Heatmap**

A 2D density map showing where gaze was concentrated across the session. The color intensity reflects how many samples fell in each region. Good for identifying hotspots — areas the person looked at most frequently.

---

**Velocity Trace**

Shows angular velocity over time. Spikes correspond to saccades; flat, low regions are fixations. If you pass in a fixation table, those periods are shaded in green. Helpful for visually validating your fixation detection results.

---

**Main Sequence**

*TO DO*

---

**Population Density**

Similar to the heatmap, but built from multiple sessions at once. Each session is normalized by its sample count before averaging, so longer recordings don't dominate the result. Useful for understanding group-level gaze patterns across participants or activities.

---
