"""
preprocessing.py
1. Chronological train / validation / test split.
2. scikit-learn preprocessing (ColumnTransformer) for linear, polynomial and tree models.

Why a ColumnTransformer inside a Pipeline?
    The scaler and one-hot encoder are FITTED (learn means, std devs, categories) only
    when pipeline.fit() is called on the training data. During cross-validation the
    whole pipeline is re-fitted inside every fold, so validation rows never influence
    the scaling. This prevents data leakage.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from config import project_config as cfg
from src.feature_engineering import (BINARY_FEATURES, CATEGORICAL_FEATURES,
                                     NUMERIC_FEATURES, TREE_FEATURES)


def chronological_split(df: pd.DataFrame):
    """
    Split by date (never shuffled):
        train      : up to TRAIN_END       (2012-06-30)
        validation : up to VALIDATION_END  (2012-09-30)
        test       : everything after      (Oct-Dec 2012)
    """
    train_end = pd.Timestamp(cfg.TRAIN_END) + pd.Timedelta(hours=23)
    val_end = pd.Timestamp(cfg.VALIDATION_END) + pd.Timedelta(hours=23)
    train = df[df["datetime"] <= train_end]
    val = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)]
    test = df[df["datetime"] > val_end]
    # Safety checks: no overlap, nothing lost, time order respected
    assert len(train) + len(val) + len(test) == len(df)
    assert train["datetime"].max() < val["datetime"].min() < test["datetime"].min()
    return train.copy(), val.copy(), test.copy()


def split_summary(train, val, test) -> pd.DataFrame:
    """Small table with the size and dates of each split (goes into the report)."""
    rows = []
    total = len(train) + len(val) + len(test)
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        rows.append({"split": name, "rows": len(part), "percent": round(100 * len(part) / total, 1),
                     "start": part["datetime"].min(), "end": part["datetime"].max()})
    return pd.DataFrame(rows)


def build_preprocessor(kind: str = "linear", poly_degree: int = 2) -> ColumnTransformer:
    """
    kind = "simple"  : only temp, scaled                     (Simple Linear Regression)
    kind = "linear"  : scaled numerics + one-hot categories  (Linear, Ridge, Logistic, KNN, SVM, NB)
    kind = "poly"    : like "linear" but numerics expanded with PolynomialFeatures
                       (temp, hum, windspeed, temp^2, temp*hum, ...)
    kind = "tree"    : raw values, no scaling, no one-hot    (Decision Tree, Random Forest)
    """
    one_hot = OneHotEncoder(handle_unknown="ignore")

    if kind == "simple":
        return ColumnTransformer([("num", StandardScaler(), ["temp"])])
    if kind == "linear":
        return ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", one_hot, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ])
    if kind == "poly":
        numeric = Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ])
        return ColumnTransformer([
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", one_hot, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ])
    if kind == "tree":
        return ColumnTransformer([("all", "passthrough", TREE_FEATURES)])
    raise ValueError(f"Unknown preprocessor kind: {kind}")
