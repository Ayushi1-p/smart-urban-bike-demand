"""
evaluation.py
Metric functions shared by the regression and classification scripts.

Regression metrics (all in "bikes per hour" except R^2):
    MAE  = average of |actual - predicted|          -> "on average we are off by X bikes"
    MSE  = average of (actual - predicted)^2        -> punishes big mistakes heavily
    RMSE = square root of MSE                       -> same units as cnt, still punishes big errors
    R^2  = 1 - (model error / error of always predicting the mean)
           1.0 = perfect, 0 = no better than the mean, negative = worse than the mean
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def clip_counts(pred):
    """Rental counts cannot be negative; linear models can predict < 0, so clip at 0."""
    return np.clip(pred, 0, None)


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "MSE": round(mse, 1),
        "RMSE": round(float(np.sqrt(mse)), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }
