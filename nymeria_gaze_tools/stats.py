"""
stats.py — Statistical tests and descriptive statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def descriptive_stats(
    session_metrics: pd.DataFrame,
    metrics: list[str] = None,
) -> pd.DataFrame:
    """Compute descriptive statistics for session metrics."""
    if metrics is None:
        cols = session_metrics.select_dtypes(include=np.number).columns.tolist()
    else:
        cols = [c for c in metrics if c in session_metrics.columns]

    rows = []
    for col in cols:
        s = session_metrics[col].dropna()
        rows.append({
            "metric":   col,
            "mean":     float(s.mean())                            if len(s) else float("nan"),
            "std":      float(s.std(ddof=1))                      if len(s) else float("nan"),
            "median":   float(s.median())                         if len(s) else float("nan"),
            "skew":     float(scipy_stats.skew(s, nan_policy="omit"))     if len(s) else float("nan"),
            "kurtosis": float(scipy_stats.kurtosis(s, nan_policy="omit")) if len(s) else float("nan"),
            "n":        int(len(s)),
        })

    return pd.DataFrame(rows).set_index("metric")


def run_anova(
    session_metrics: pd.DataFrame,
    metric: str,
    group_col: str,
) -> dict:
    """Run one-way ANOVA comparing a metric across groups."""
    if metric not in session_metrics.columns:
        raise ValueError(f"Metric '{metric}' not found in session_metrics.")
    if group_col not in session_metrics.columns:
        raise ValueError(f"Group column '{group_col}' not found in session_metrics.")

    groups = [
        grp[metric].dropna().to_numpy()
        for _, grp in session_metrics.groupby(group_col)
    ]
    group_labels = sorted(session_metrics[group_col].dropna().unique().tolist())

    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups.")

    f_stat, p_val = scipy_stats.f_oneway(*groups)
    f_stat = float(f_stat)
    p_val  = float(p_val)

    grand_mean = float(session_metrics[metric].dropna().mean())
    ss_total  = sum(((x - grand_mean) ** 2).sum() for x in groups)
    group_means = [float(g.mean()) for g in groups]
    group_ns    = [len(g) for g in groups]
    ss_between  = sum(n * (m - grand_mean) ** 2 for n, m in zip(group_ns, group_means))
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else float("nan")

    result = {
        "F":           f_stat,
        "p_value":     p_val,
        "eta_squared": eta_sq,
        "significant": p_val < 0.05,
    }

    if p_val < 0.05 and len(groups) >= 3:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            data_col = session_metrics[[metric, group_col]].dropna()
            tukey = pairwise_tukeyhsd(
                endog=data_col[metric],
                groups=data_col[group_col],
                alpha=0.05,
            )
            result["post_hoc"] = pd.DataFrame(
                data=tukey._results_table.data[1:],
                columns=tukey._results_table.data[0],
            )
        except ImportError:
            result["post_hoc"] = None
    else:
        result["post_hoc"] = None

    return result


def run_ttest(
    group_a: np.ndarray | pd.Series,
    group_b: np.ndarray | pd.Series,
    paired: bool = False,
    alternative: str = "two-sided",
) -> dict:
    """Run independent or paired t-test between two groups."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if paired:
        result = scipy_stats.ttest_rel(a, b, alternative=alternative)
        diff = a - b
        n = len(diff)
        se = float(np.std(diff, ddof=1) / np.sqrt(n))
        ci_half = scipy_stats.t.ppf(0.975, df=n - 1) * se
        mean_diff = float(diff.mean())
        ci = (mean_diff - ci_half, mean_diff + ci_half)
        pooled_std = float(np.std(diff, ddof=1))
        cohen_d = float(mean_diff / pooled_std) if pooled_std != 0 else float("nan")
    else:
        result = scipy_stats.ttest_ind(a, b, alternative=alternative, equal_var=False)
        mean_diff = float(a.mean() - b.mean())
        pooled_std = float(np.sqrt(
            (np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2
        ))
        cohen_d = float(mean_diff / pooled_std) if pooled_std != 0 else float("nan")
        se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
        df_welch = (
            (np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)) ** 2
            / (
                (np.var(a, ddof=1) / len(a)) ** 2 / (len(a) - 1)
                + (np.var(b, ddof=1) / len(b)) ** 2 / (len(b) - 1)
            )
        )
        ci_half = scipy_stats.t.ppf(0.975, df=df_welch) * se
        ci = (mean_diff - ci_half, mean_diff + ci_half)

    return {
        "t":                      float(result.statistic),
        "p_value":                float(result.pvalue),
        "cohen_d":                cohen_d,
        "confidence_interval_95": (float(ci[0]), float(ci[1])),
    }


def correlation_matrix(
    session_metrics: pd.DataFrame,
    metrics: list[str] = None,
    method: str = "both",
) -> dict:
    """Compute Pearson and/or Spearman correlation matrices."""
    if metrics is None:
        cols = session_metrics.select_dtypes(include=np.number).columns.tolist()
    else:
        cols = [c for c in metrics if c in session_metrics.columns]

    data = session_metrics[cols].dropna()
    result = {}

    if method in ("pearson", "both"):
        result["pearson"] = data.corr(method="pearson")
        n = len(data)
        r = result["pearson"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
        p_vals = 2 * scipy_stats.t.sf(np.abs(t_stat), df=n - 2)
        result["p_values"] = pd.DataFrame(p_vals, index=cols, columns=cols)

    if method in ("spearman", "both"):
        result["spearman"] = data.corr(method="spearman")

    return result


def compare_groups_statistically(
    groups_result: dict,
    metric: str,
    test: str = "auto",
) -> dict:
    """Pick the right statistical test for a comparison (auto-selects based on group count)."""
    sm = groups_result["session_metrics"]
    group_col = groups_result["group_by"]
    groups = groups_result["groups"]

    n_groups = len(groups)
    if test == "auto":
        test = "anova" if n_groups > 2 else "ttest"

    if test == "anova":
        return run_anova(sm, metric=metric, group_col=group_col)

    elif test == "ttest":
        if n_groups != 2:
            raise ValueError(
                f"t-test requires exactly 2 groups; found {n_groups}. "
                "Use test='anova' or restrict groups."
            )
        dist = groups_result["metric_distributions"].get(metric, {})
        a = dist.get(groups[0], np.array([]))
        b = dist.get(groups[1], np.array([]))
        return run_ttest(a, b)

    else:
        raise ValueError(f"Unknown test '{test}'. Use 'auto', 'anova', or 'ttest'.")
