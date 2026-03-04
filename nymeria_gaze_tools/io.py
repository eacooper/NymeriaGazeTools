"""
io.py — Data loading and filtering for eye gaze sessions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


HF_REPO_ID   = "nymeriagazedata/eye-gaze-data"
HF_REPO_TYPE = "dataset"
HF_CACHE_DIR = Path.home() / ".cache" / "nymeria_gaze_tools"
HF_TOKEN     = os.environ.get("HF_TOKEN")


def _resolve_data_root(data_root: str | Path | None) -> Path:
    if data_root is None:
        return Path.cwd() / "data" / "processed"
    return Path(data_root)


def load_metadata(
    data_root: str | Path = None,
    source: str = "local",
    repo_id: str = HF_REPO_ID,
    token: str = None,
) -> pd.DataFrame:
    """Load metadata.csv and return as DataFrame."""
    if source == "local":
        root = _resolve_data_root(data_root)
        meta_path = root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found at: {meta_path}\n"
                "Set data_root or ensure data/processed/metadata.csv exists."
            )
        df = pd.read_csv(meta_path)
        if "has_gaze_data" in df.columns:
            df["has_gaze_data"] = df["has_gaze_data"].map(
                lambda v: str(v).strip().lower() == "true"
            )
        return df

    elif source == "huggingface":
        raise NotImplementedError("HuggingFace loading not yet implemented.")
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'local' or 'huggingface'.")


def load_session(
    sequence_uid: str,
    data_root: str | Path = None,
    source: str = "local",
    repo_id: str = HF_REPO_ID,
    token: str = None,
) -> pd.DataFrame:
    """Load a single eye gaze CSV by sequence_uid."""
    if source == "local":
        root = _resolve_data_root(data_root)
        uid = sequence_uid.removesuffix(".csv")
        csv_path = root / "eye_gaze" / f"{uid}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Eye gaze CSV not found: {csv_path}\n"
                "Check sequence_uid and data_root."
            )
        return pd.read_csv(csv_path)

    elif source == "huggingface":
        raise NotImplementedError("HuggingFace loading not yet implemented.")
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'local' or 'huggingface'.")


def load_sessions(
    sequence_uids: list[str],
    data_root: str | Path = None,
    source: str = "local",
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Batch-load multiple sessions."""
    return {
        uid: load_session(uid, data_root=data_root, source=source, **kwargs)
        for uid in sequence_uids
    }


def filter_sessions(
    catalog: pd.DataFrame,
    participant: str = None,
    script: str = None,
    participant_gender: str = None,
    participant_age_group: str = None,
    participant_ethnicity: str = None,
    location: str = None,
    session_id: str = None,
    has_gaze_data: bool = None,
    **extra_filters,
) -> pd.DataFrame:
    """Return filtered metadata catalog (all filters are AND-combined)."""
    mask = pd.Series(True, index=catalog.index)

    if participant is not None:
        mask &= catalog["fake_name"] == participant
    if script is not None:
        mask &= catalog["script"] == script
    if participant_gender is not None:
        mask &= catalog["participant_gender"] == participant_gender
    if participant_age_group is not None:
        mask &= catalog["participant_age_group"] == participant_age_group
    if participant_ethnicity is not None:
        mask &= catalog["participant_ethnicity"] == participant_ethnicity
    if location is not None:
        mask &= catalog["location"] == location
    if session_id is not None:
        mask &= catalog["session_id"] == session_id
    if has_gaze_data is not None:
        mask &= catalog["has_gaze_data"] == has_gaze_data

    for col, value in extra_filters.items():
        if col not in catalog.columns:
            raise ValueError(
                f"Column '{col}' not found in catalog. "
                f"Available: {list(catalog.columns)}"
            )
        mask &= catalog[col] == value

    return catalog[mask].reset_index(drop=True)


def list_participants(catalog: pd.DataFrame) -> list[str]:
    """Return sorted unique participant fake names."""
    return sorted(catalog["fake_name"].dropna().unique().tolist())


def list_activities(catalog: pd.DataFrame) -> list[str]:
    """Return sorted unique activity/script names."""
    return sorted(catalog["script"].dropna().unique().tolist())


def list_locations(catalog: pd.DataFrame) -> list[str]:
    """Return sorted unique recording locations."""
    return sorted(catalog["location"].dropna().unique().tolist())
