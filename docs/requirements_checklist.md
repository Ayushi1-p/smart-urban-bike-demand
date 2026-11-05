# Requirements Checklist

Where each CSE570 topic and each report section is demonstrated in the project.

## Unit I – Python fundamentals and data preparation
| Topic | Evidence |
|---|---|
| Python essentials, modules, functions | `src/*.py`, `config/project_config.py` |
| NumPy arrays / operations | notebook 01 §3; `np.log1p`, `np.clip`, `np.select` in `src/` |
| pandas DataFrames, loading CSV | `src/data_loader.py`, notebook 01 |
| Data types, statistical summary | notebook 01 §2; `data_loader` output |
| Missing values | `summarise_data`, notebook 01 §4 (none found; strategy documented) |
| Duplicates | `summarise_data` (0 rows, 0 timestamps) |
| Categorical encoding | `OneHotEncoder` in `src/preprocessing.py`; notebook 01 §5 |
| Feature scaling | `StandardScaler` in pipelines; notebook 01 §7 |
| Train/test split | `chronological_split` (train/validation/test), notebook 01 §6 |
| Data dictionary | `docs/data_dictionary.md`, dashboard Dataset page |

## Unit II – EDA and feature engineering
| Topic | Evidence |
|---|---|
| Univariate / multivariate analysis | figures 01, 02–07 (`src/eda.py`), notebook 02 |
| Correlation, heatmap | figure 06 |
| Outlier detection | figure 08, IQR numbers in `docs/eda_findings.md` |
| Feature selection | atemp dropped; Lasso zero coefficients |
| Feature transformation | log(cnt+1) target; polynomial features |
| Imbalance handling | weather 4 → 3; class distribution check, macro F1 (figure 15) |
| Pipelines, ColumnTransformer | `src/preprocessing.py` |
| Time/season/weather analysis | figures 02–05, 07 |
| Leakage identification | `LEAKAGE_COLUMNS`, `tests/test_preprocessing.py` |

## Unit III – Regression
| Topic | Evidence |
|---|---|
| Dummy, simple, multiple, polynomial, Ridge, Lasso, Elastic Net | `src/regression.py` |
| Hyperparameter tuning, cross-validation | GridSearchCV + TimeSeriesSplit(5) |
| Over/underfitting | figure 10; train vs CV vs validation RMSE columns |
| Residual analysis | figures 11–13 |
| MAE, MSE, RMSE, R² + comparison table | `outputs/metrics/regression_*_results.csv`, figure 09 |

## Unit IV – Classification
| Topic | Evidence |
|---|---|
| Dummy, Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes | `src/classification.py` |
| Target creation, class distribution | thresholds in `classification_model_meta.json`; `class_distribution.csv` |
| Scaling where needed | "linear" vs "tree" preprocessors; KNN experiment in notebook 04 |
| Confusion matrix, accuracy, precision, recall, F1, ROC-AUC (OvR macro) | figures 16–18, `classification_*_results.csv`, `classification_report_test.txt` |
| Tuning, CV, comparison | GridSearchCV + TimeSeriesSplit(5), macro F1 |

## Application and quality
| Requirement | Evidence |
|---|---|
| Streamlit dashboard (overview, data, EDA, comparison, 2 forms, limitations) | `app.py` |
| Backend modules, saved models | `src/predict.py`, `models/*.joblib` + `*_meta.json` |
| Input validation | `validate_inputs`, `tests/test_predictions.py` |
| Automated tests | 30 tests (`python -m pytest -v`) |
| Reproducibility | `requirements.txt`, seed 42, versions in meta JSON, README commands |

## Report sections (`report/report.md`)
| Section | Source of content |
|---|---|
| 1 Title page | placeholders to fill |
| 2 Abstract | real results |
| 3 Introduction | `docs/methodology.md` |
| 4 Dataset | `data/README.md`, `docs/data_dictionary.md` |
| 5 Preprocessing | `src/preprocessing.py`, `src/feature_engineering.py` |
| 6 EDA | figures 01–08, `docs/eda_findings.md` |
| 7 Methodology | `docs/figures/methodology_flowchart.png` |
| 8 Model development | `src/regression.py`, `src/classification.py` |
| 9 Results | `outputs/metrics/*.csv`, figures 09–19, `docs/*_results.md` |
| 10 Conclusion & limitations | results docs, dashboard limitations page |
| 11 References | dataset, paper, library papers |
| 12 GitHub & LinkedIn evidence | to fill with real links and dated screenshots |

## Still to do (by the student)
- [ ] Run notebooks on the real data and insert the KNN scaling numbers into report §9.4
- [ ] Fill the title page and section 12; insert dated screenshots
- [ ] Export the report to PDF
- [ ] Make the repository public; publish the LinkedIn post
- [ ] Clean-environment check: fresh clone → install → train → `pytest` → `streamlit run app.py`
