"""
Tests for nymeria_gaze_tools.io

All tests use synthetic DataFrames and temporary CSV files — no real dataset required.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nymeria_gaze_tools.io import (
    filter_sessions,
    list_participants,
    list_activities,
    list_locations,
    _read_metadata_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_catalog() -> pd.DataFrame:
    """Minimal synthetic metadata catalog."""
    return pd.DataFrame({
        "sequence_uid":           ["uid_001", "uid_002", "uid_003", "uid_004"],
        "fake_name":              ["alice", "bob", "alice", "carol"],
        "script":                 ["S7-Cooking", "S7-Cooking", "S1-Relax_at_home", "S1-Relax_at_home"],
        "participant_gender":     ["Female", "Male", "Female", "Female"],
        "participant_age_group":  ["25-30", "18-24", "25-30", "31-40"],
        "participant_ethnicity":  ["Caucasian", "East Asian", "Caucasian", "South Asian"],
        "location":               ["Loc_01", "Loc_02", "Loc_01", "Loc_03"],
        "has_gaze_data":          [True, True, False, True],
    })


# ---------------------------------------------------------------------------
# _read_metadata_csv — age label remapping
# ---------------------------------------------------------------------------

def test_age_label_remap():
    """load_metadata must correct the '19-25' mislabeling to '18-24'."""
    df = pd.DataFrame({
        "participant_age_group": ["19-25", "18-24", "25-30"],
        "has_gaze_data": ["True", "False", "True"],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = Path(f.name)

    result = _read_metadata_csv(tmp_path)
    assert "19-25" not in result["participant_age_group"].values
    assert (result["participant_age_group"] == "18-24").sum() == 2


def test_has_gaze_data_cast_to_bool():
    """has_gaze_data should be boolean after loading."""
    df = pd.DataFrame({
        "has_gaze_data": ["True", "False", "true", "false"],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = Path(f.name)

    result = _read_metadata_csv(tmp_path)
    assert result["has_gaze_data"].dtype == bool
    assert result["has_gaze_data"].tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# filter_sessions
# ---------------------------------------------------------------------------

def test_filter_by_script():
    catalog = make_catalog()
    result = filter_sessions(catalog, script="S7-Cooking")
    assert len(result) == 2
    assert (result["script"] == "S7-Cooking").all()


def test_filter_by_participant():
    catalog = make_catalog()
    result = filter_sessions(catalog, participant="alice")
    assert len(result) == 2
    assert (result["fake_name"] == "alice").all()


def test_filter_by_gender():
    catalog = make_catalog()
    result = filter_sessions(catalog, participant_gender="Female")
    assert len(result) == 3
    assert (result["participant_gender"] == "Female").all()


def test_filter_combined():
    catalog = make_catalog()
    result = filter_sessions(catalog, script="S7-Cooking", participant_gender="Female")
    assert len(result) == 1
    assert result["fake_name"].iloc[0] == "alice"


def test_filter_has_gaze_data():
    catalog = make_catalog()
    result = filter_sessions(catalog, has_gaze_data=True)
    assert len(result) == 3
    assert result["has_gaze_data"].all()


def test_filter_no_match_returns_empty():
    catalog = make_catalog()
    result = filter_sessions(catalog, script="S99-Nonexistent")
    assert len(result) == 0


def test_filter_unknown_column_raises():
    catalog = make_catalog()
    with pytest.raises(ValueError, match="not found in catalog"):
        filter_sessions(catalog, nonexistent_column="value")


# ---------------------------------------------------------------------------
# list_participants / list_activities / list_locations
# ---------------------------------------------------------------------------

def test_list_participants_sorted_unique():
    catalog = make_catalog()
    result = list_participants(catalog)
    assert result == sorted({"alice", "bob", "carol"})


def test_list_activities_sorted_unique():
    catalog = make_catalog()
    result = list_activities(catalog)
    assert result == sorted({"S7-Cooking", "S1-Relax_at_home"})


def test_list_locations_sorted_unique():
    catalog = make_catalog()
    result = list_locations(catalog)
    assert result == sorted({"Loc_01", "Loc_02", "Loc_03"})
