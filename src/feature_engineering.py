"""
feature_engineering.py
Turns the EDA decisions (docs/eda_findings.md) into code.

Every change here depends only on values in the SAME row (no averages over the whole
dataset), so it cannot leak information from the test period into training.
"""
import pandas as pd

# ---- Feature lists used by every model ------------------------------------------
NUMERIC_FEATURES = ["temp", "hum", "windspeed"]                     # continuous, get scaled
CATEGORICAL_FEATURES = ["season", "mnth", "hr", "weekday", "weathersit", "hr_workingday"]
BINARY_FEATURES = ["yr", "holiday", "workingday"]                   # already 0/1
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Tree models split on raw numbers, so they do not need one-hot columns or the
# combined hr_workingday column (a tree can combine hr and workingday by itself).
TREE_FEATURES = NUMERIC_FEATURES + ["season", "mnth", "hr", "weekday", "weathersit"] + BINARY_FEATURES

# Feature used by Simple Linear Regression (one input only): temp has the strongest
# linear correlation with cnt among the allowed inputs (+0.405, EDA section 4).
SIMPLE_FEATURE = "temp"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with the engineered features added.

    1. weathersit 4 -> 3   only 3 hours have code 4 (heavy rain/snow), too few to learn
                           from, so they are grouped with 3 as "bad weather".
    2. hr_workingday       the daily shape differs: two commute peaks on working days,
                           one midday hump on weekends (EDA Figure 02). A linear model
                           cannot learn that from hr and workingday separately, so we
                           combine them, e.g. "W17" = 5 pm on a working day,
                           "N17" = 5 pm on a weekend/holiday.
    3. atemp is not used   it is 0.988 correlated with temp (same information twice).
    """
    df = df.copy()
    df["weathersit"] = df["weathersit"].clip(upper=3)
    prefix = df["workingday"].map({1: "W", 0: "N"})
    df["hr_workingday"] = prefix + df["hr"].astype(str).str.zfill(2)
    return df


def make_input_row(hr, season, mnth, weekday, workingday, holiday, yr,
                   weathersit, temp, hum, windspeed) -> pd.DataFrame:
    """Build a one-row DataFrame in the same format as training data (used by the app)."""
    row = pd.DataFrame([{
        "season": season, "yr": yr, "mnth": mnth, "hr": hr, "holiday": holiday,
        "weekday": weekday, "workingday": workingday, "weathersit": weathersit,
        "temp": temp, "hum": hum, "windspeed": windspeed,
    }])
    return add_features(row)
