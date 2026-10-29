# 🚲 Smart Urban Bike Demand Prediction System

An end-to-end machine-learning project (M.Tech Data Science & Analytics, **CSE570**) that
estimates **hourly bike-sharing demand** from weather, seasonal and calendar conditions,
using the real UCI Bike Sharing dataset. It includes an interactive Streamlit dashboard.

> **Decision-support prototype trained on historical data** (Capital Bikeshare,
> Washington D.C., 2011–2012). It is **not** connected to live stations or live weather.

![Dashboard overview](docs/screenshots/overview.png)

## Problem statement
Bike-sharing demand changes sharply by hour, season, weather and type of day. If an
operator under-estimates demand, stations run empty. If it over-estimates, bikes and
redistribution staff sit idle. This project builds two connected models:

| Task | Type | Target |
|---|---|---|
| A | Regression | `cnt`: bikes rented in one hour |
| B | Multiclass classification | `demand_level`: **Low** (< 64) / **Medium** (64–197) / **High** (≥ 198) bikes per hour. Thresholds = tertiles of the **training period only** |

## Dataset
- **Source:** UCI Machine Learning Repository, *Bike Sharing* (ID 275):
  https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset · DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894) · licence **CC BY 4.0**
- **Verified facts (hour.csv):** 17,379 rows (the UCI page states 17,389), 17 columns,
  0 missing values, 0 duplicate rows, 2011-01-01 to 2012-12-31. 165 hours have no record
  (for example the Hurricane Sandy shutdown, 29–30 Oct 2012).
- **Inputs:** season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, hum,
  windspeed, plus the engineered `hr_workingday`.
- **Excluded:** `casual` and `registered` (they add up to `cnt` → target leakage),
  `atemp` (0.988 correlated with `temp`), `instant`, `dteday`.

Details: [`data/README.md`](data/README.md) · [`docs/data_dictionary.md`](docs/data_dictionary.md) · [`docs/eda_findings.md`](docs/eda_findings.md)

## Methodology
```
hour.csv ─► validate schema ─► EDA ─► feature engineering ─► chronological split
                                                               │
            ┌──────────────────────────────────────────────────┤
            ▼                                                   ▼
  Regression (7 models × raw/log target)            Classification (7 models)
  Pipeline: ColumnTransformer + model               thresholds from TRAIN only
  tune: GridSearchCV + TimeSeriesSplit(5)           tune: GridSearchCV + TimeSeriesSplit(5)
  select: lowest validation RMSE                    select: highest validation macro F1
            └──────────────► refit on train+validation ─► test ONCE ◄───┘
                                         │
                              saved pipelines (joblib)
                                         │
                         Streamlit dashboard (app.py) via src/predict.py
```

| Split | Period | Rows |
|---|---|---|
| Train | 2011-01-01 → 2012-06-30 | 13,003 (74.8%) |
| Validation | 2012-07-01 → 2012-09-30 | 2,208 (12.7%) |
| Test | 2012-10-01 → 2012-12-31 | 2,168 (12.5%) |

**Leakage prevention:** leakage columns are excluded centrally and checked by tests; all
scaling and encoding is inside scikit-learn pipelines, fitted on training folds only;
class thresholds come from the training period; the test set was used once.

## Results (test period Oct–Dec 2012, actual outputs)

### Regression: selected model **Lasso on log(cnt+1)** (alpha = 0.001)
| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **Lasso [log] (selected on validation)** | **42.49** | **67.72** | **0.887** |
| Ridge [log] | 40.72 | 66.45 | 0.891 |
| Multiple Linear [log] | 43.04 | 70.39 | 0.878 |
| Multiple Linear [raw] | 68.63 | 94.82 | 0.779 |
| Simple Linear (temp) | 145.09 | 203.89 | −0.023 |
| Dummy (mean) | 156.87 | 204.43 | −0.028 |

MAE is 73% lower than the baseline. The log transform gave the largest single
improvement. Full tables: [`docs/regression_results.md`](docs/regression_results.md).

### Classification: selected model **Logistic Regression** (C = 10)
| Model | Accuracy | Macro F1 | ROC-AUC (OvR macro) |
|---|---|---|---|
| **Logistic Regression (selected on validation)** | **0.839** | **0.816** | **0.946** |
| SVM (RBF) | 0.837 | 0.822 | 0.944 |
| Random Forest | 0.833 | 0.816 | 0.948 |
| Decision Tree | 0.807 | 0.780 | 0.914 |
| KNN | 0.750 | 0.750 | 0.919 |
| Naive Bayes | 0.774 | 0.722 | 0.874 |
| Dummy | 0.459 | 0.210 | 0.500 |

Medium is the hardest class (F1 0.67). Full analysis:
[`docs/classification_results.md`](docs/classification_results.md).

## Dashboard
| Prediction (regression) | Prediction (classification) |
|---|---|
| ![](docs/screenshots/predict_regression.png) | ![](docs/screenshots/predict_classification.png) |

Pages: Overview · Dataset · interactive EDA · Model comparison · Predict demand (two
forms) · Source & limitations.

## Technologies
Python 3.13 · NumPy · pandas · scikit-learn · Matplotlib · Seaborn · Streamlit · Altair ·
Joblib · pytest · Git/GitHub

## Project structure
```
app.py                  Streamlit dashboard (frontend)
config/project_config.py  paths, seed, split dates, leakage columns
src/data_loader.py      load + validate hour.csv
src/eda.py              EDA figures and numbers
src/feature_engineering.py  engineered features, feature lists
src/preprocessing.py    chronological split, ColumnTransformer pipelines
src/regression.py       Task A: 7 regression models
src/classification.py   Task B: 7 classifiers
src/evaluation.py       metric functions
src/predict.py          input validation + prediction (used by the app)
tests/                  pytest tests (30)
outputs/figures, outputs/metrics   generated figures and result tables
models/                 saved pipelines (re-created by training)
docs/                   data dictionary, findings, results, development log
```

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# download the UCI zip and put hour.csv, day.csv, Readme.txt into data/raw/

python -m src.data_loader         # validate the dataset
python -m src.eda                 # EDA figures (outputs/figures 01-08)
python -m src.regression          # train + evaluate regression, saves models/regression_model.joblib
python -m src.classification      # train + evaluate classification, saves models/classification_model.joblib
python -m pytest -v               # 30 tests
streamlit run app.py              # dashboard at http://localhost:8501
```
Trained `.joblib` files are not committed. The two training commands recreate them
(random seed 42).

## Limitations
- Historical data from a single city (Washington D.C., 2011–2012). Not directly
  applicable to other cities, including Indian cities, or to the system today.
- Observed weather is used as input. Real use would need forecasts, which add error.
- The model only knows 2011 and 2012 demand levels. Class thresholds are fixed from the
  training period.
- The test period is one season and includes the Hurricane Sandy shutdown.
- Point predictions only; no confidence intervals.

## Future work
- Retrain on official Indian city data if it becomes available (monsoon and festival
  features).
- Live integration: operator GBFS station feed + weather-forecast API + scheduled retraining.
- Tree-based / boosted regressors, prediction intervals, permutation importance,
  station-level forecasting.

## Citation and licence
- Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
- Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence, 2*(2–3), 113–127. https://doi.org/10.1007/s13748-013-0040-3

Dataset licence: CC BY 4.0.
