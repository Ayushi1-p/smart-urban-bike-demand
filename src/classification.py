"""
classification.py  -  Task B: classify an hour as Low / Medium / High demand.   Syllabus Unit IV

Workflow (run:  python -m src.classification)
    1. Same features and chronological split as regression.
    2. Demand thresholds = 33.3% and 66.7% quantiles (tertiles) of cnt in the TRAINING
       period only. The same two numbers label validation, test and app inputs.
    3. Tune the 7 required classifiers with TimeSeriesSplit CV on train (score = macro F1).
    4. Score on validation, choose the best by macro F1.
    5. Refit on train + validation, score ONCE on test. Save tables, figures and the model.

Why macro F1?  Accuracy can look good if a model only predicts the most common class.
Macro F1 averages the F1 of Low, Medium and High equally, so every class counts.
"""
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import project_config as cfg
from src.data_loader import load_hour_data
from src.feature_engineering import ALL_FEATURES, TREE_FEATURES, add_features
from src.preprocessing import build_preprocessor, chronological_split

CLASSES = ["Low", "Medium", "High"]
SEED = cfg.RANDOM_SEED
CV = TimeSeriesSplit(n_splits=5)

# ---------------------------------------------------------------------------------
# Scaling: KNN (distances), SVM (distances/kernels), Logistic Regression (gradient
# based) need scaled inputs -> "linear" preprocessor. Trees split on one feature at a
# time, so scaling changes nothing -> "tree" preprocessor (raw values).
# ---------------------------------------------------------------------------------
CLASSIFIERS = [
    {"name": "Dummy (most frequent)", "prep": "linear", "model": DummyClassifier(strategy="most_frequent"),
     "grid": {}},
    {"name": "Logistic Regression", "prep": "linear", "model": LogisticRegression(max_iter=3000),
     "grid": {"model__C": [0.1, 1, 10]}},
    {"name": "KNN", "prep": "linear", "model": KNeighborsClassifier(),
     "grid": {"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]}},
    {"name": "Decision Tree", "prep": "tree", "model": DecisionTreeClassifier(random_state=SEED),
     "grid": {"model__max_depth": [5, 10, 15, None], "model__min_samples_leaf": [1, 5, 20]}},
    {"name": "Random Forest", "prep": "tree",
     "model": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
     "grid": {"model__max_depth": [10, 20, None], "model__min_samples_leaf": [1, 5]}},
    # Tuned without probabilities (faster); the final fits wrap the tuned SVC in
    # CalibratedClassifierCV so ROC-AUC can be computed (see fit_final).
    {"name": "SVM (RBF kernel)", "prep": "linear", "model": SVC(kernel="rbf", random_state=SEED),
     "grid": {"model__C": [0.1, 1, 10]}, "proba_refit": True},
    {"name": "Naive Bayes (Gaussian)", "prep": "linear_dense", "model": GaussianNB(),
     "grid": {"model__var_smoothing": [1e-9, 1e-6, 1e-3]}},
]


# ---------------------------------------------------------------------------------
def compute_thresholds(train_cnt: pd.Series) -> dict:
    """Tertile cut points from TRAINING data only."""
    low, high = train_cnt.quantile([1 / 3, 2 / 3])
    return {"low_upper": float(low), "medium_upper": float(high)}


def label_demand(cnt, thresholds) -> pd.Series:
    """Low: cnt < low_upper | Medium: low_upper <= cnt < medium_upper | High: cnt >= medium_upper"""
    cnt = pd.Series(cnt)
    return pd.Series(np.select([cnt < thresholds["low_upper"], cnt < thresholds["medium_upper"]],
                               ["Low", "Medium"], default="High"), index=cnt.index)


def classification_metrics(y_true, y_pred, proba=None, classes=None) -> dict:
    m = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }
    # One-vs-rest ROC-AUC, macro average (each class vs the other two, then averaged).
    if proba is not None and pd.Series(y_true).nunique() == 3:
        y_bin = label_binarize(y_true, classes=classes)  # one column per class: 1 = this class
        m["roc_auc_ovr_macro"] = round(roc_auc_score(y_bin, proba, average="macro"), 4)
    else:
        m["roc_auc_ovr_macro"] = np.nan
    return m


def make_pipeline(spec):
    return Pipeline([("prep", build_preprocessor(spec["prep"])), ("model", clone(spec["model"]))])


def proba_in_class_order(model, X):
    """predict_proba columns follow model.classes_ (alphabetical); reorder to Low, Medium, High."""
    proba = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def fit_final(spec, params, X, y):
    """Fit a fresh pipeline with the tuned hyperparameters (and probabilities for SVM)."""
    pipe = make_pipeline(spec)
    pipe.set_params(**params)
    if spec.get("proba_refit"):
        # SVC does not output probabilities by itself. CalibratedClassifierCV fits the
        # tuned SVC and learns (with internal 5-fold CV) how to turn its scores into
        # probabilities. This is the method recommended by scikit-learn 1.9+.
        svc = pipe.named_steps["model"]
        pipe.set_params(model=CalibratedClassifierCV(svc, ensemble=False))
    return pipe.fit(X, y)


# ---------------------------------------------------------------------------------
def save(fig, name):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / f"{name}.png", dpi=120)
    plt.close(fig)
    print(f"  saved figure: {name}.png")


def plot_class_distribution(dist):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dist.T.plot(kind="bar", ax=ax, color=["#6baed6", "#fd8d3c", "#d62728"])
    ax.set(title="Demand classes in each split (thresholds from training data)",
           xlabel="Split", ylabel="Share of hours (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Class")
    save(fig, "15_class_distribution")


def plot_metric_comparison(results):
    r = results.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro",
                                    "roc_auc_ovr_macro"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    r.plot(kind="bar", ax=ax)
    ax.set(title="Classifiers on the validation period (Jul-Sep 2012) - higher is better",
           xlabel="", ylabel="Score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "16_classification_validation_comparison")


def plot_confusion_matrices(final_models, X_te, y_te, best_name):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (name, model) in zip(axes.flat, final_models.items()):
        cm = confusion_matrix(y_te, model.predict(X_te), labels=CLASSES)
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name + ("  (selected)" if name == best_name else ""), fontsize=10)
    axes.flat[-1].axis("off")
    fig.suptitle("Confusion matrices on the test period (Oct-Dec 2012). Rows = actual, columns = predicted")
    save(fig, "17_confusion_matrices_test")


def plot_roc(model, X_te, y_te, name):
    proba = proba_in_class_order(model, X_te)
    y_bin = label_binarize(y_te, classes=CLASSES)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, c in enumerate(CLASSES):
        RocCurveDisplay.from_predictions(y_bin[:, i], proba[:, i], name=f"{c} vs rest", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title(f"One-vs-rest ROC curves - {name} (test)")
    ax.legend(loc="lower right")
    save(fig, "18_roc_curves_test")


def plot_feature_importance(rf_model):
    names = rf_model.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    imp = pd.Series(rf_model.named_steps["model"].feature_importances_, index=names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp.index, imp.values, color="steelblue")
    ax.set(title="Random Forest feature importance (mean decrease in impurity)", xlabel="Importance")
    save(fig, "19_random_forest_feature_importance")
    return imp.sort_values(ascending=False)


# ---------------------------------------------------------------------------------
def run_classification():
    df = add_features(load_hour_data())
    train, val, test = chronological_split(df)

    thresholds = compute_thresholds(train[cfg.TARGET])
    for part in (train, val, test):
        part["demand_level"] = label_demand(part[cfg.TARGET], thresholds)

    dist = pd.DataFrame({name: part["demand_level"].value_counts(normalize=True).reindex(CLASSES).mul(100).round(1)
                         for name, part in [("train", train), ("validation", val), ("test", test)]})
    counts = pd.DataFrame({name: part["demand_level"].value_counts().reindex(CLASSES)
                           for name, part in [("train", train), ("validation", val), ("test", test)]})
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(cfg.METRICS_DIR / "class_distribution.csv")

    X_tr, y_tr = train[ALL_FEATURES], train["demand_level"]
    X_val, y_val = val[ALL_FEATURES], val["demand_level"]
    X_trval, y_trval = pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val])
    X_te, y_te = test[ALL_FEATURES], test["demand_level"]

    # ---- tune on train, score on validation ----
    rows, tuned = [], {}
    for spec in CLASSIFIERS:
        search = GridSearchCV(make_pipeline(spec), spec["grid"], cv=CV, scoring="f1_macro", n_jobs=-1)
        search.fit(X_tr, y_tr)
        params = search.best_params_
        model = fit_final(spec, params, X_tr, y_tr)
        pred = model.predict(X_val)
        m = classification_metrics(y_val, pred, proba_in_class_order(model, X_val), CLASSES)
        train_f1 = round(f1_score(y_tr, model.predict(X_tr), average="macro"), 4)
        clean = {k.replace("model__", ""): v for k, v in params.items()}
        rows.append({"model": spec["name"], "best_params": json.dumps(clean), "cv_f1_macro": round(search.best_score_, 4),
                     "train_f1_macro": train_f1, **m})
        tuned[spec["name"]] = (spec, params)
        print(f"{spec['name']:24s} val F1={m['f1_macro']:.4f}  acc={m['accuracy']:.4f}  "
              f"AUC={m['roc_auc_ovr_macro']}  {clean}")

    results = pd.DataFrame(rows)
    results.to_csv(cfg.METRICS_DIR / "classification_validation_results.csv", index=False)
    best_name = results.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]["model"]
    print(f"\nSelected on validation (macro F1): {best_name}")

    # ---- refit on train + validation, test ONCE ----
    test_rows, final_models = [], {}
    for name, (spec, params) in tuned.items():
        model = fit_final(spec, params, X_trval, y_trval)
        pred = model.predict(X_te)
        m = classification_metrics(y_te, pred, proba_in_class_order(model, X_te), CLASSES)
        test_rows.append({"model": name, "selected_on_validation": name == best_name, **m})
        final_models[name] = model
    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(cfg.METRICS_DIR / "classification_test_results.csv", index=False)

    best_model = final_models[best_name]
    report = classification_report(y_te, best_model.predict(X_te), labels=CLASSES, digits=4)
    (cfg.METRICS_DIR / "classification_report_test.txt").write_text(report)

    print("\nFigures:")
    plot_class_distribution(dist)
    plot_metric_comparison(results)
    plot_confusion_matrices(final_models, X_te, y_te, best_name)
    plot_roc(best_model, X_te, y_te, best_name)
    importance = plot_feature_importance(final_models["Random Forest"])

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, cfg.MODELS_DIR / "classification_model.joblib")
    best_test = test_results[test_results["selected_on_validation"]].iloc[0].to_dict()
    best_val = results[results["model"] == best_name].iloc[0].to_dict()
    meta = {
        "task": "classification", "classes": CLASSES, "thresholds": thresholds,
        "threshold_rule": "tertiles (33.3% / 66.7% quantiles) of cnt in the training period 2011-01-01 to 2012-06-30",
        "labeling": "Low: cnt < low_upper; Medium: low_upper <= cnt < medium_upper; High: cnt >= medium_upper",
        "selected_model": best_name, "best_params": json.loads(best_val["best_params"]),
        "selection_rule": "highest validation macro F1 (Jul-Sep 2012); tuned with 5-fold TimeSeriesSplit on train",
        "validation_metrics": {k: best_val[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                          "roc_auc_ovr_macro"]},
        "test_metrics": {k: best_test[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                     "roc_auc_ovr_macro"]},
        "final_fit": "train + validation (2011-01-01 to 2012-09-30)",
        "features": TREE_FEATURES if tuned[best_name][0]["prep"] == "tree" else ALL_FEATURES,
        "random_seed": SEED,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(cfg.MODELS_DIR / "classification_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    pd.set_option("display.width", 220)
    print("\n=== THRESHOLDS (from training data) ===")
    print(f"Low: cnt < {thresholds['low_upper']:.1f} | Medium: {thresholds['low_upper']:.1f} <= cnt < "
          f"{thresholds['medium_upper']:.1f} | High: cnt >= {thresholds['medium_upper']:.1f}")
    print("\n=== CLASS DISTRIBUTION (number of hours, % in brackets) ===")
    print((counts.astype(str) + " (" + dist.astype(str) + "%)").to_string())
    print("\n=== VALIDATION RESULTS (sorted by macro F1) ===")
    print(results.sort_values("f1_macro", ascending=False).to_string(index=False))
    print("\n=== TEST RESULTS (Oct-Dec 2012, refit on train+validation) ===")
    print(test_results.sort_values("f1_macro", ascending=False).to_string(index=False))
    print(f"\n=== CLASSIFICATION REPORT - {best_name} (test) ===")
    print(report)
    print("=== CONFUSION MATRIX - selected model (test; rows = actual Low/Medium/High) ===")
    print(confusion_matrix(y_te, best_model.predict(X_te), labels=CLASSES))
    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
    print(importance.round(4).to_string())
    print(f"\nSaved model: models/classification_model.joblib  ->  {best_name}")


if __name__ == "__main__":
    run_classification()
