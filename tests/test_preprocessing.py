"""
Tests for feature engineering and preprocessing: leakage, chronological split, encoding.
Real-data tests are skipped if data/raw/hour.csv is missing.
"""
import pandas as pd
import pytest

from config import project_config as cfg
from src.feature_engineering import ALL_FEATURES, TREE_FEATURES, add_features
from src.preprocessing import build_preprocessor, chronological_split


def test_no_leakage_columns_in_features():
    # cnt, casual and registered must never be model inputs
    for col in cfg.LEAKAGE_COLUMNS + ["instant", "dteday", "atemp"]:
        assert col not in ALL_FEATURES
        assert col not in TREE_FEATURES


def test_weather_4_merged_into_3_and_interaction_created():
    df = pd.DataFrame({"weathersit": [1, 3, 4], "workingday": [1, 0, 1], "hr": [8, 17, 23]})
    out = add_features(df)
    assert out["weathersit"].tolist() == [1, 3, 3]
    assert out["hr_workingday"].tolist() == ["W08", "N17", "W23"]
    assert df["weathersit"].tolist() == [1, 3, 4]   # original not modified


def test_unknown_preprocessor_kind_rejected():
    with pytest.raises(ValueError):
        build_preprocessor("something_else")


needs_real_data = pytest.mark.skipif(not cfg.HOUR_CSV.exists(), reason="hour.csv not present")


@needs_real_data
def test_chronological_split_on_real_data():
    from src.data_loader import load_hour_data
    df = add_features(load_hour_data())
    train, val, test = chronological_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["datetime"].max() < val["datetime"].min()
    assert val["datetime"].max() < test["datetime"].min()
    assert train["datetime"].max() <= pd.Timestamp("2012-06-30 23:00")
    assert test["datetime"].min() >= pd.Timestamp("2012-10-01")
