# Methodology and Problem Statement

_Living document. Sections are filled in as each stage is actually implemented.
Status markers: ✅ implemented & verified · 🟡 designed, not yet implemented._

---

## 1. Problem statement (CSE570 Q1, Part 1) ✅

### 1.1 Context
Public bike-sharing systems let people rent a bicycle at one station and return it at
another. Demand changes sharply by hour of day, season, weather and working-day status.
If an operator under-estimates demand, stations run empty and riders are turned away;
if it over-estimates, bikes sit idle and redistribution staff and vehicles are wasted.
A reliable estimate of hourly demand under given conditions supports decisions such as
how many bikes to have in service and when to schedule redistribution or maintenance.

### 1.2 Machine-learning tasks and target variables

| Task | Type | Target | Definition |
|---|---|---|---|
| A | Regression | `cnt` | Total number of bikes rented in one hour (casual + registered users) |
| B | Multiclass classification | `demand_level` | `Low` / `Medium` / `High`, derived from `cnt` using thresholds computed **from the training period only** |

Task B exists because operators often act on bands rather than exact numbers
("is this a high-demand hour?"), and because it lets the project cover the
classification half of the syllabus with the same real data.

### 1.3 Inputs
Calendar and time (`season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`)
and weather (`weathersit`, `temp`, `atemp`, `hum`, `windspeed`), plus engineered
features to be documented in §4.
`casual`, `registered` and `cnt` are **never** inputs (target leakage; see §5).

### 1.4 Measurable objectives / success criteria
Results are reported as measured on a held-out, later time period, whatever they turn out to be.

1. **Beat a naive baseline.** The selected regression model must achieve lower MAE and
   RMSE than `DummyRegressor` (predicts the training mean) on the test period; the
   selected classifier must achieve higher macro-F1 than `DummyClassifier`.
2. **Compare required algorithms fairly.** All 7 regression and 7 classification
   approaches in the CSE570 syllabus are trained on the same chronological split and
   compared on the same metrics (MAE, MSE, RMSE, R² / precision, recall, F1, ROC-AUC,
   confusion matrix).
3. **No leakage.** Automated tests confirm that no target-derived column reaches any
   model and that every preprocessing step is fitted on training data only.
4. **Usable prototype.** A Streamlit dashboard returns a regression and a
   classification prediction for user-entered conditions, using the saved models.

### 1.5 Intended users and impact
- Bike-sharing operations planners (fleet sizing, redistribution timing, maintenance windows).
- Students and analysts studying how weather and calendar effects drive mobility demand.

This is a **decision-support prototype** trained on historical data. It does not
measure or claim reductions in cost, traffic or emissions.

### 1.6 Key assumptions and constraints
- **Historical, single-city data.** Capital Bikeshare, Washington D.C., 2011–2012.
  Patterns may differ for other cities and for today's system.
- **Weather is assumed known.** Predictions use the weather of the hour being
  predicted; in real use this would come from a forecast, adding error not measured here.
- **System-wide demand only.** The data has no station-level information.
- **Not live.** No real-time bike or weather feed is connected (see README, "Future work").
- **Normalised weather values.** UCI publishes temperature, humidity and wind speed
  scaled to 0–1; the app converts between real units and this scale using the
  constants documented by UCI.

---

## 2. Evaluation design ✅

Hourly demand has strong time structure and a growth trend between 2011 and 2012.
A random shuffle would let the model "see the future", inflating scores. The primary
evaluation therefore uses a **chronological split**:

| Split | Period | Purpose |
|---|---|---|
| Train | 2011-01-01 → 2012-06-30 | fit models and all preprocessing |
| Validation | 2012-07-01 → 2012-09-30 | choose models and tune hyperparameters |
| Test | 2012-10-01 → 2012-12-31 | final, one-time evaluation |

Hyperparameter search inside the training period uses `TimeSeriesSplit`
(each fold validates on data later than its training data). Row counts: train 13,003, validation 2,208, test 2,168.

Known consequence to discuss in the report: the test quarter is autumn/winter only,
so test scores describe performance in that season, not the whole year.

---

## 3. Data preparation ✅
See `report/report.md` §5 and `src/data_loader.py`, `src/preprocessing.py`.

## 4. Feature engineering ✅
See `docs/eda_findings.md` and `src/feature_engineering.py`.

## 5. Leakage prevention ✅ (rules and automated tests in `tests/test_preprocessing.py`)
- `cnt`, `casual`, `registered` are listed in `LEAKAGE_COLUMNS` and excluded centrally.
- `instant` (row index) and the raw date string are excluded as features.
- Scalers, encoders and selectors live inside scikit-learn `Pipeline`s and are fitted
  on training rows only; cross-validation refits them inside each fold.
- Demand-level thresholds are computed from training `cnt` only and reused unchanged
  for validation, test and app predictions.

## 6. Models ✅
`src/regression.py`, `src/classification.py`; see report §8.

## 7. Results ✅
`docs/regression_results.md`, `docs/classification_results.md`.
