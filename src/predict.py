"""
predict.py
The bridge between the dashboard (app.py) and the trained models.

The app never trains anything. It calls these functions, which:
    1. validate the user's inputs,
    2. convert real units (°C, %, km/h) into the 0-1 scale the UCI data uses,
    3. build a one-row DataFrame with exactly the training features,
    4. load the saved pipelines (preprocessing + model) and predict.
"""
import json
from functools import lru_cache

import joblib
import pandas as pd

from config import project_config as cfg
from src.feature_engineering import ALL_FEATURES, make_input_row

REG_MODEL = cfg.MODELS_DIR / "regression_model.joblib"
REG_META = cfg.MODELS_DIR / "regression_model_meta.json"
CLF_MODEL = cfg.MODELS_DIR / "classification_model.joblib"
CLF_META = cfg.MODELS_DIR / "classification_model_meta.json"


class InvalidInputError(ValueError):
    """Raised when user inputs are outside the ranges the model was trained on."""


# Allowed ranges in REAL units (from the UCI documentation)
INPUT_RANGES = {
    "hr": (0, 23), "season": (1, 4), "mnth": (1, 12), "weekday": (0, 6),
    "holiday": (0, 1), "yr": (0, 1), "weathersit": (1, 4),
    "temp_c": (cfg.TEMP_MIN_C, cfg.TEMP_MAX_C),   # -8 .. 39 °C
    "humidity_pct": (0, cfg.HUM_MAX),              # 0 .. 100 %
    "windspeed": (0, cfg.WINDSPEED_MAX),           # 0 .. 67 (UCI unit)
}
INTEGER_FIELDS = ["hr", "season", "mnth", "weekday", "holiday", "yr", "weathersit"]


def models_available() -> bool:
    return all(p.exists() for p in [REG_MODEL, REG_META, CLF_MODEL, CLF_META])


@lru_cache(maxsize=1)
def load_models():
    """Load both pipelines and their metadata once (cached)."""
    if not models_available():
        raise FileNotFoundError("Trained models not found. Run:  python -m src.regression  and  "
                                "python -m src.classification")
    with open(REG_META) as f:
        reg_meta = json.load(f)
    with open(CLF_META) as f:
        clf_meta = json.load(f)
    return joblib.load(REG_MODEL), reg_meta, joblib.load(CLF_MODEL), clf_meta


def derive_workingday(weekday: int, holiday: int) -> int:
    """UCI rule: working day = Monday-Friday and not a holiday (weekday 0 = Sunday, 6 = Saturday)."""
    return int(1 <= weekday <= 5 and holiday == 0)


def validate_inputs(inputs: dict) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems = []
    for field, (low, high) in INPUT_RANGES.items():
        if field not in inputs or inputs[field] is None:
            problems.append(f"{field} is missing")
            continue
        value = inputs[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{field} must be a number")
            continue
        if field in INTEGER_FIELDS and float(value) != int(value):
            problems.append(f"{field} must be a whole number")
        if not low <= value <= high:
            problems.append(f"{field} = {value} is outside the allowed range {low} to {high}")
    return problems


def to_model_row(inputs: dict) -> pd.DataFrame:
    """Validate, convert units to the UCI 0-1 scale and build the feature row."""
    problems = validate_inputs(inputs)
    if problems:
        raise InvalidInputError("; ".join(problems))
    weekday, holiday = int(inputs["weekday"]), int(inputs["holiday"])
    row = make_input_row(
        hr=int(inputs["hr"]), season=int(inputs["season"]), mnth=int(inputs["mnth"]),
        weekday=weekday, workingday=derive_workingday(weekday, holiday), holiday=holiday,
        yr=int(inputs["yr"]), weathersit=int(inputs["weathersit"]),
        temp=(inputs["temp_c"] - cfg.TEMP_MIN_C) / (cfg.TEMP_MAX_C - cfg.TEMP_MIN_C),
        hum=inputs["humidity_pct"] / cfg.HUM_MAX,
        windspeed=inputs["windspeed"] / cfg.WINDSPEED_MAX,
    )
    return row[ALL_FEATURES]


def predict_count(inputs: dict) -> float:
    """Regression: expected number of bikes rented in that hour (never negative)."""
    reg_model, _, _, _ = load_models()
    return max(0.0, float(reg_model.predict(to_model_row(inputs))[0]))


def predict_level(inputs: dict) -> tuple[str, dict]:
    """Classification: Low / Medium / High plus the model's probability for each class."""
    _, _, clf_model, clf_meta = load_models()
    row = to_model_row(inputs)
    label = str(clf_model.predict(row)[0])
    proba = dict(zip([str(c) for c in clf_model.classes_], clf_model.predict_proba(row)[0]))
    ordered = {c: float(proba[c]) for c in clf_meta["classes"]}
    return label, ordered


def level_from_count(count: float, thresholds: dict) -> str:
    """Which band a regression prediction falls in, using the training thresholds."""
    if count < thresholds["low_upper"]:
        return "Low"
    if count < thresholds["medium_upper"]:
        return "Medium"
    return "High"
