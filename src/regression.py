"""
regression.py  -  Task A: predict cnt (bikes rented in one hour).   Syllabus Unit III

Workflow (run:  python -m src.regression)
    1. Load data, add features, split by time (train / validation / test).
    2. For each of the 7 required models, and for two target versions
       (raw cnt and log(cnt + 1)), tune hyperparameters with TimeSeriesSplit
       cross-validation on the TRAINING period only.
    3. Score every tuned model on the VALIDATION period and choose the best (lowest RMSE).
    4. Refit each tuned model on train + validation and score it ONCE on the TEST period.
    5. Save tables (outputs/metrics), figures (outputs/figures) and the chosen model (models/).
"""
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from config import project_config as cfg
from src.data_loader import load_hour_data
from src.evaluation import clip_counts, regression_metrics
from src.feature_engineering import ALL_FEATURES, add_features
from src.preprocessing import build_preprocessor, chronological_split, split_summary

TARGET = cfg.TARGET

# ---------------------------------------------------------------------------------
# The 7 required models. "grid" = hyperparameter values tried by cross-validation.
#   Polynomial: degree 2 or 3 on temp/hum/windspeed (not on one-hot columns, which
#               would create thousands of meaningless columns).
#   Ridge / Lasso / Elastic Net use the degree-2 polynomial features, so regularisation
#   has something to control. alpha = strength of the penalty on large coefficients.
# ---------------------------------------------------------------------------------
REGRESSION_MODELS = [
    {"name": "Dummy (mean)", "prep": "linear", "model": DummyRegressor(strategy="mean"), "grid": {}},
    {"name": "Simple Linear (temp)", "prep": "simple", "model": LinearRegression(), "grid": {}},
    {"name": "Multiple Linear", "prep": "linear", "model": LinearRegression(), "grid": {}},
    {"name": "Polynomial", "prep": "poly", "model": LinearRegression(),
     "grid": {"prep__num__poly__degree": [2, 3]}},
    {"name": "Ridge", "prep": "poly", "model": Ridge(),
     "grid": {"model__alpha": [0.1, 1, 10, 100]}},
    {"name": "Lasso", "prep": "poly", "model": Lasso(max_iter=20000),
     "grid": {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}},
    {"name": "Elastic Net", "prep": "poly", "model": ElasticNet(max_iter=20000),
     "grid": {"model__alpha": [0.001, 0.01, 0.1, 1], "model__l1_ratio": [0.2, 0.5, 0.8]}},
]

CV = TimeSeriesSplit(n_splits=5)  # each fold trains on earlier hours, validates on later ones


def make_estimator(spec, log_target):
    """Pipeline = preprocessing + model. Optionally train on log(cnt + 1)."""
    pipe = Pipeline([("prep", build_preprocessor(spec["prep"])), ("model", spec["model"])])
    grid = dict(spec["grid"])
    if log_target:
        # Model learns log1p(cnt); predictions are converted back with expm1 automatically.
        pipe = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        grid = {f"regressor__{k}": v for k, v in grid.items()}
    return pipe, grid


def tune(spec, log_target, X, y):
    """GridSearchCV with time-series folds on the training period. Returns the best fitted model."""
    est, grid = make_estimator(spec, log_target)
    search = GridSearchCV(est, grid or {}, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X, y)
    params = {k.replace("regressor__", "").replace("model__", "").replace("prep__num__poly__", "poly_"): v
              for k, v in search.best_params_.items()}
    return search.best_estimator_, params, -search.best_score_


def predict(model, X):
    return clip_counts(model.predict(X))


# ---------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------
def save(fig, name):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / f"{name}.png", dpi=120)
    plt.close(fig)
    print(f"  saved figure: {name}.png")


def plot_validation_comparison(results):
    r = results.sort_values("val_RMSE", ascending=False)
    labels = r["model"] + " [" + r["target"] + "]"
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(r))
    ax.barh(y - 0.2, r["val_RMSE"], height=0.4, label="RMSE", color="indianred")
    ax.barh(y + 0.2, r["val_MAE"], height=0.4, label="MAE", color="steelblue")
    ax.set_yticks(y, labels)
    ax.set(title="Regression models on the validation period (Jul-Sep 2012) - lower is better",
           xlabel="Error (bikes per hour)")
    ax.legend()
    save(fig, "09_regression_validation_comparison")


def plot_degree_curve(X_tr, y_tr, X_val, y_val):
    """Underfitting vs overfitting: training and validation error for polynomial degrees 1-4."""
    rows = []
    for d in [1, 2, 3, 4]:
        pipe = Pipeline([("prep", build_preprocessor("poly", poly_degree=d)), ("model", LinearRegression())])
        pipe.fit(X_tr, y_tr)
        rows.append({"degree": d,
                     "train_RMSE": regression_metrics(y_tr, predict(pipe, X_tr))["RMSE"],
                     "val_RMSE": regression_metrics(y_val, predict(pipe, X_val))["RMSE"]})
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["degree"], curve["train_RMSE"], marker="o", label="Training RMSE")
    ax.plot(curve["degree"], curve["val_RMSE"], marker="o", label="Validation RMSE")
    ax.set(title="Polynomial degree vs error (numeric features)", xlabel="Polynomial degree",
           ylabel="RMSE (bikes per hour)", xticks=[1, 2, 3, 4])
    ax.legend()
    save(fig, "10_polynomial_degree_curve")
    return curve


def plot_test_diagnostics(name, y_test, pred, test_df):
    resid = y_test - pred

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, pred, alpha=0.3, s=10)
    lim = max(y_test.max(), pred.max())
    ax.plot([0, lim], [0, lim], color="red", linestyle="--", label="Perfect prediction")
    ax.set(title=f"Actual vs predicted - {name} (test, Oct-Dec 2012)",
           xlabel="Actual bikes per hour", ylabel="Predicted bikes per hour")
    ax.legend()
    save(fig, "11_regression_actual_vs_predicted")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(pred, resid, alpha=0.3, s=10)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set(title="Residuals vs predicted", xlabel="Predicted bikes per hour",
                ylabel="Residual (actual - predicted)")
    axes[1].hist(resid, bins=50, color="steelblue")
    axes[1].set(title="Distribution of residuals", xlabel="Residual (bikes)", ylabel="Number of hours")
    mae_by_hr = pd.Series(np.abs(resid.values), index=test_df["hr"].values).groupby(level=0).mean()
    axes[2].bar(mae_by_hr.index, mae_by_hr.values, color="indianred")
    axes[2].set(title="Mean absolute error by hour of day", xlabel="Hour of day",
                ylabel="MAE (bikes)", xticks=range(0, 24, 2))
    save(fig, "12_regression_residuals")

    # Two weeks of the test period: does the model follow the daily rhythm?
    window = test_df["datetime"] < test_df["datetime"].min() + pd.Timedelta(days=14)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(test_df.loc[window, "datetime"], y_test[window], label="Actual", linewidth=1.2)
    ax.plot(test_df.loc[window, "datetime"], pred[window], label="Predicted", linewidth=1.2)
    ax.set(title=f"First two weeks of the test period - {name}", xlabel="Date", ylabel="Bikes per hour")
    ax.legend()
    save(fig, "13_regression_test_two_weeks")
    return mae_by_hr


def plot_coefficients(model, name):
    """For linear-type models: the 15 largest coefficients (what pushes demand up or down)."""
    pipe = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    est = pipe.named_steps["model"]
    if not hasattr(est, "coef_"):
        return None
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(est.coef_, index=names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color=["indianred" if v < 0 else "steelblue" for v in top.values])
    ax.set(title=f"Largest coefficients - {name}", xlabel="Coefficient (effect on the model's target scale)")
    save(fig, "14_regression_top_coefficients")
    return coefs, int((coefs == 0).sum())


# ---------------------------------------------------------------------------------
def run_regression():
    df = add_features(load_hour_data())
    train, val, test = chronological_split(df)
    splits = split_summary(train, val, test)
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    splits.to_csv(cfg.METRICS_DIR / "split_summary.csv", index=False)
    print("Chronological split:\n", splits.to_string(index=False), "\n")

    X_tr, y_tr = train[ALL_FEATURES], train[TARGET]
    X_val, y_val = val[ALL_FEATURES], val[TARGET]
    X_trval = pd.concat([X_tr, X_val])
    y_trval = pd.concat([y_tr, y_val])
    X_te, y_te = test[ALL_FEATURES], test[TARGET]

    # ---- Steps 2 + 3: tune on train (time-series CV), score on validation ----
    rows, tuned = [], {}
    for spec in REGRESSION_MODELS:
        for log_target in ([False] if spec["name"].startswith("Dummy") else [False, True]):
            label = "log(cnt+1)" if log_target else "cnt"
            model, params, cv_rmse = tune(spec, log_target, X_tr, y_tr)
            tr = regression_metrics(y_tr, predict(model, X_tr))
            va = regression_metrics(y_val, predict(model, X_val))
            rows.append({"model": spec["name"], "target": label, "best_params": json.dumps(params),
                         "cv_RMSE": round(cv_rmse, 2), "train_RMSE": tr["RMSE"], "train_R2": tr["R2"],
                         **{f"val_{k}": v for k, v in va.items()}})
            tuned[(spec["name"], label)] = (spec, log_target, model)
            print(f"{spec['name']:22s} [{label:10s}] val RMSE={va['RMSE']:7.2f}  MAE={va['MAE']:7.2f}  R2={va['R2']:.4f}  {params}")

    results = pd.DataFrame(rows)
    results.to_csv(cfg.METRICS_DIR / "regression_validation_results.csv", index=False)

    # Select the best model on VALIDATION RMSE (MAE breaks ties). The test set is not used here.
    best_row = results.sort_values(["val_RMSE", "val_MAE"]).iloc[0]
    best_key = (best_row["model"], best_row["target"])
    print(f"\nSelected on validation: {best_key[0]} [{best_key[1]}]  RMSE={best_row['val_RMSE']}")

    # ---- Step 4: refit each tuned configuration on train+validation, test ONCE ----
    test_rows, final_models = [], {}
    for (name, label), (spec, log_target, model) in tuned.items():
        final = sklearn.base.clone(model).fit(X_trval, y_trval)   # same hyperparameters, more data
        te = regression_metrics(y_te, predict(final, X_te))
        test_rows.append({"model": name, "target": label, "selected_on_validation": (name, label) == best_key,
                          **{f"test_{k}": v for k, v in te.items()}})
        final_models[(name, label)] = final
    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(cfg.METRICS_DIR / "regression_test_results.csv", index=False)

    # ---- Step 5: figures + saved model ----
    print("\nFigures:")
    plot_validation_comparison(results)
    curve = plot_degree_curve(X_tr, y_tr, X_val, y_val)
    best_model = final_models[best_key]
    best_pred = predict(best_model, X_te)
    mae_by_hr = plot_test_diagnostics(f"{best_key[0]} [{best_key[1]}]", y_te.reset_index(drop=True),
                                      pd.Series(best_pred), test.reset_index(drop=True))
    plot_coefficients(best_model, f"{best_key[0]} [{best_key[1]}]")

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, cfg.MODELS_DIR / "regression_model.joblib")
    best_test = test_results[test_results["selected_on_validation"]].iloc[0]
    meta = {
        "task": "regression", "target": TARGET,
        "selected_model": best_key[0], "target_transform": best_key[1],
        "best_params": json.loads(best_row["best_params"]),
        "selection_rule": "lowest validation RMSE (Jul-Sep 2012); tuned with 5-fold TimeSeriesSplit on train",
        "validation_metrics": {k: best_row[f"val_{k}"] for k in ["MAE", "MSE", "RMSE", "R2"]},
        "test_metrics": {k: best_test[f"test_{k}"] for k in ["MAE", "MSE", "RMSE", "R2"]},
        "final_fit": "train + validation (2011-01-01 to 2012-09-30)",
        "features": ALL_FEATURES, "random_seed": cfg.RANDOM_SEED,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(cfg.MODELS_DIR / "regression_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ---- Printed summary (send this output for interpretation) ----
    pd.set_option("display.width", 200)
    print("\n=== VALIDATION RESULTS (sorted by RMSE) ===")
    print(results.sort_values("val_RMSE")[["model", "target", "train_RMSE", "cv_RMSE", "val_MAE",
                                           "val_RMSE", "val_R2", "best_params"]].to_string(index=False))
    print("\n=== TEST RESULTS (Oct-Dec 2012, models refit on train+validation) ===")
    print(test_results.sort_values("test_RMSE").to_string(index=False))
    print("\n=== POLYNOMIAL DEGREE CURVE ===")
    print(curve.to_string(index=False))
    print("\n=== TEST MAE BY HOUR (selected model) ===")
    print(mae_by_hr.round(1).to_string())
    print("\n=== FEATURE SELECTION BY L1 PENALTY (coefficients set exactly to 0) ===")
    for (name, label), m in final_models.items():
        if name in ("Lasso", "Elastic Net"):
            pipe = m.regressor_ if isinstance(m, TransformedTargetRegressor) else m
            coef = pipe.named_steps["model"].coef_
            print(f"  {name:12s} [{label:10s}] {int((coef == 0).sum())} of {len(coef)} coefficients are 0")
    print(f"\nSaved model: models/regression_model.joblib  ->  {best_key[0]} [{best_key[1]}]")


if __name__ == "__main__":
    run_regression()
