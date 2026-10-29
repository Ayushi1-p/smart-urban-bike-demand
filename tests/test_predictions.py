"""
Tests for src/predict.py: input validation, unit conversion, and real predictions
from the saved models (skipped if models have not been trained yet).
"""
import pytest

from config import project_config as cfg
from src import predict as P

VALID = {"hr": 17, "season": 3, "mnth": 7, "weekday": 3, "holiday": 0, "yr": 1,
         "weathersit": 1, "temp_c": 25.0, "humidity_pct": 55, "windspeed": 13.0}


def test_valid_input_has_no_problems():
    assert P.validate_inputs(VALID) == []


@pytest.mark.parametrize("field,value", [
    ("hr", 24), ("hr", -1), ("season", 5), ("mnth", 0), ("weekday", 7),
    ("temp_c", 45.0), ("humidity_pct", 120), ("windspeed", -2), ("weathersit", 0),
])
def test_out_of_range_rejected(field, value):
    assert P.validate_inputs({**VALID, field: value})


def test_wrong_types_rejected():
    assert P.validate_inputs({**VALID, "hr": "five"})
    assert P.validate_inputs({**VALID, "hr": 3.5})
    assert P.validate_inputs({k: v for k, v in VALID.items() if k != "temp_c"})


def test_invalid_input_raises():
    with pytest.raises(P.InvalidInputError):
        P.to_model_row({**VALID, "hr": 30})


def test_workingday_rule():
    assert P.derive_workingday(3, 0) == 1   # Wednesday
    assert P.derive_workingday(0, 0) == 0   # Sunday
    assert P.derive_workingday(6, 0) == 0   # Saturday
    assert P.derive_workingday(3, 1) == 0   # Wednesday holiday


def test_unit_conversion_matches_uci_scale():
    row = P.to_model_row({**VALID, "temp_c": 39.0, "humidity_pct": 100, "windspeed": 67.0, "weathersit": 4})
    assert row["temp"].iloc[0] == pytest.approx(1.0)
    assert row["hum"].iloc[0] == pytest.approx(1.0)
    assert row["windspeed"].iloc[0] == pytest.approx(1.0)
    assert row["weathersit"].iloc[0] == 3          # 4 merged into 3
    assert row["hr_workingday"].iloc[0] == "W17"
    for col in cfg.LEAKAGE_COLUMNS:
        assert col not in row.columns


def test_level_from_count():
    t = {"low_upper": 64.0, "medium_upper": 198.0}
    assert P.level_from_count(10, t) == "Low"
    assert P.level_from_count(64, t) == "Medium"
    assert P.level_from_count(198, t) == "High"


needs_models = pytest.mark.skipif(not P.models_available(), reason="models not trained yet")


@needs_models
def test_saved_models_predict():
    count = P.predict_count(VALID)
    assert count >= 0
    label, proba = P.predict_level(VALID)
    assert label in ["Low", "Medium", "High"]
    assert sum(proba.values()) == pytest.approx(1.0, abs=1e-6)


@needs_models
def test_night_lower_than_evening_peak():
    # Sanity check: 3 am on a working day should be predicted lower than 5 pm.
    assert P.predict_count({**VALID, "hr": 3}) < P.predict_count({**VALID, "hr": 17})
