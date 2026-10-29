"""
Tests for src/data_loader.py

Two kinds of test:
1. Logic tests: use a tiny hand-made table (3 rows) that follows the UCI format.
   These rows are TEST FIXTURES that check the validation code itself; they are
   never used for analysis or modelling.
2. Real-data tests: check the actual downloaded hour.csv. They are skipped
   automatically if the file has not been placed in data/raw/ yet.
Run with:  pytest -v
"""

import pandas as pd
import pytest

from config import project_config as cfg
from src.data_loader import (
    DataValidationError,
    load_hour_data,
    summarise_data,
    validate_hour_data,
)


def make_fixture() -> pd.DataFrame:
    """Three syntactically valid rows in the UCI hour.csv layout (test fixture only)."""
    return pd.DataFrame({
        "instant": [1, 2, 3],
        "dteday": ["2011-01-01", "2011-01-01", "2011-01-01"],
        "season": [1, 1, 1], "yr": [0, 0, 0], "mnth": [1, 1, 1], "hr": [0, 1, 2],
        "holiday": [0, 0, 0], "weekday": [6, 6, 6], "workingday": [0, 0, 0],
        "weathersit": [1, 1, 1], "temp": [0.24, 0.22, 0.22], "atemp": [0.29, 0.27, 0.27],
        "hum": [0.81, 0.80, 0.80], "windspeed": [0.0, 0.0, 0.0],
        "casual": [3, 8, 5], "registered": [13, 32, 27], "cnt": [16, 40, 32],
    })


def test_valid_fixture_passes():
    assert validate_hour_data(make_fixture()) == []


def test_missing_column_detected():
    df = make_fixture().drop(columns=["hr"])
    assert any("Missing columns" in p for p in validate_hour_data(df))


def test_out_of_range_hour_detected():
    df = make_fixture()
    df.loc[0, "hr"] = 24
    assert any(p.startswith("hr:") for p in validate_hour_data(df))


def test_cnt_inconsistency_detected():
    df = make_fixture()
    df.loc[1, "cnt"] = 999
    assert any("cnt != casual + registered" in p for p in validate_hour_data(df))


def test_strict_load_raises(tmp_path):
    df = make_fixture()
    df.loc[0, "season"] = 7
    bad_file = tmp_path / "hour.csv"
    df.to_csv(bad_file, index=False)
    with pytest.raises(DataValidationError):
        load_hour_data(bad_file)


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the official dataset"):
        load_hour_data(tmp_path / "does_not_exist.csv")


def test_load_sorts_and_adds_datetime(tmp_path):
    df = make_fixture().iloc[::-1]          # deliberately reversed order
    f = tmp_path / "hour.csv"
    df.to_csv(f, index=False)
    loaded = load_hour_data(f)
    assert loaded["datetime"].is_monotonic_increasing
    assert loaded["hr"].tolist() == [0, 1, 2]


# ---------------- Real-data tests (need data/raw/hour.csv) ----------------
needs_real_data = pytest.mark.skipif(
    not cfg.HOUR_CSV.exists(), reason="data/raw/hour.csv not downloaded yet"
)


@needs_real_data
def test_real_file_matches_uci_schema():
    df = load_hour_data()  # strict=True: raises if anything is wrong
    assert len(df) == cfg.EXPECTED_HOUR_ROWS
    assert set(cfg.HOUR_COLUMNS).issubset(df.columns)


@needs_real_data
def test_real_file_date_range_and_no_duplicate_timestamps():
    s = summarise_data(load_hour_data())
    assert s["date_min"] == "2011-01-01"
    assert s["date_max"] == "2012-12-31"
    assert s["duplicate_timestamps"] == 0
