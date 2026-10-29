# Regression Results (Task A: predict `cnt`)

Source: actual output of `python -m src.regression`, run on 29-Oct-2026 on the real
hour.csv. Tables: `outputs/metrics/regression_validation_results.csv`,
`outputs/metrics/regression_test_results.csv`. Figures 09–14 in `outputs/figures/`.

## 1. Chronological split
| Split | Rows | % | Period |
|---|---|---|---|
| Train | 13,003 | 74.8 | 2011-01-01 → 2012-06-30 |
| Validation | 2,208 | 12.7 | 2012-07-01 → 2012-09-30 |
| Test | 2,168 | 12.5 | 2012-10-01 → 2012-12-31 |

## 2. Validation results (model selection), sorted by RMSE
Hyperparameters were tuned with 5-fold `TimeSeriesSplit` on the training period only.

| Model | Target | Train RMSE | CV RMSE | Val MAE | Val RMSE | Val R² | Best params |
|---|---|---|---|---|---|---|---|
| **Lasso** | **log(cnt+1)** | 52.92 | 85.61 | **47.55** | **74.44** | **0.8962** | alpha = 0.001 |
| Elastic Net | log(cnt+1) | 52.60 | 85.25 | 47.65 | 75.36 | 0.8936 | alpha = 0.001, l1_ratio = 0.2 |
| Ridge | log(cnt+1) | 52.36 | 85.42 | 50.16 | 80.95 | 0.8772 | alpha = 0.1 |
| Polynomial (deg 2) | log(cnt+1) | 52.37 | 85.43 | 50.20 | 81.03 | 0.8770 | degree = 2 |
| Multiple Linear | log(cnt+1) | 60.92 | 94.55 | 65.89 | 109.47 | 0.7755 | – |
| Polynomial (deg 2) | cnt | 63.35 | 87.11 | 84.09 | 110.12 | 0.7729 | degree = 2 |
| Ridge | cnt | 63.35 | 87.15 | 84.10 | 110.13 | 0.7728 | alpha = 0.1 |
| Lasso | cnt | 63.36 | 86.71 | 84.11 | 110.14 | 0.7728 | alpha = 0.001 |
| Elastic Net | cnt | 63.41 | 87.41 | 84.45 | 110.60 | 0.7708 | alpha = 0.001, l1_ratio = 0.8 |
| Multiple Linear | cnt | 64.59 | 87.34 | 87.05 | 113.79 | 0.7575 | – |
| Simple Linear (temp) | cnt | 146.15 | 157.13 | 174.29 | 227.34 | 0.0319 | – |
| Simple Linear (temp) | log(cnt+1) | 162.55 | 174.17 | 183.84 | 252.84 | −0.1975 | – |
| Dummy (mean) | cnt | 160.78 | 173.97 | 197.83 | 260.65 | −0.2726 | – |

**Selected model:** Lasso regression on log(cnt + 1), with degree-2 polynomial weather
features and one-hot calendar features, alpha = 0.001. It had the lowest validation RMSE.

## 3. Final test result (Oct–Dec 2012, evaluated once)
Each tuned model was refit on train + validation, then scored once on the test period.

| Model | Target | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| **Lasso (selected)** | log(cnt+1) | **42.49** | 4585.6 | **67.72** | **0.8872** |
| Ridge | log(cnt+1) | 40.72 | 4415.7 | 66.45 | 0.8913 |
| Polynomial | log(cnt+1) | 40.72 | 4416.6 | 66.46 | 0.8913 |
| Elastic Net | log(cnt+1) | 41.56 | 4455.6 | 66.75 | 0.8904 |
| Multiple Linear | log(cnt+1) | 43.04 | 4955.2 | 70.39 | 0.8781 |
| Multiple Linear | cnt | 68.63 | 8991.7 | 94.82 | 0.7787 |
| Simple Linear (temp) | cnt | 145.09 | 41570.7 | 203.89 | −0.0230 |
| Dummy (mean) | cnt | 156.87 | 41790.9 | 204.43 | −0.0284 |

- Against the baseline, the selected model reduces MAE from 156.87 to 42.49 (−73%)
  and RMSE from 204.43 to 67.72 (−67%).
- On the test set, Ridge / Polynomial (log) scored about 1.3 bikes lower RMSE than
  Lasso. **We keep Lasso**, because the model was chosen on the validation period
  before looking at test results. Switching now would be tuning on the test set. The
  top four log-target models are within ~1.3 bikes of each other on test, so they are
  practically equivalent.

## 4. Interpretation
1. **Log transform was the biggest improvement.** For the same polynomial model,
   validation RMSE fell from 110.12 to 81.03 and MAE from 84.09 to 50.20. The target is
   right-skewed (skewness 1.28), and effects such as "rush hour" and "rain" are closer to
   multiplicative (a % change) than additive. A log target models that better and cannot
   predict negative counts.
2. **One feature is not enough.** Simple Linear Regression on temperature alone
   (val R² 0.03) is barely better than the mean baseline. Demand is driven mainly by hour
   and day type, which one weather variable cannot capture. This is **underfitting**
   (high error on training data too: train RMSE 146).
3. **Regularisation and feature selection.** With the log target, Lasso and Elastic Net
   beat plain polynomial regression (val RMSE 74.4 vs 81.0). Lasso set **37 of 110**
   coefficients exactly to 0, removing features that did not help.
4. **Tuning hit the edge of the grid.** The best alpha for Lasso and Elastic Net
   (0.001) and for Ridge (0.1) was the smallest value tried. Only light regularisation
   was needed. A wider grid (smaller alpha) is listed as future work. It was not added
   after seeing test results, to keep the test set untouched.
5. **Over/underfitting check.** Train RMSE (52.9) < validation RMSE (74.4) for the
   selected model: a moderate generalisation gap, expected because the validation
   period is a later, busier time than most of the training data. The polynomial degree
   curve (Figure 10, raw target) shows training and validation error both fall slowly
   from degree 1 to 4 (val 113.8 → 107.2). With only 3 numeric features, higher degrees
   add few columns, so clear overfitting does not appear in this range. CV chose degree 2.
6. **Dummy R² is negative on validation (−0.27).** The baseline predicts the training
   mean, which includes the quieter year 2011. Jul–Sep 2012 is much busier, so predicting
   the old mean is worse than predicting the validation period's own mean. That gives
   R² < 0, and it is evidence of the growth trend.
7. **Where the model makes mistakes (Figure 12).** Test MAE by hour: 2.5–19 bikes at
   night (0–6 h), but **105.1 at 08:00** and **86.4 / 88.5 at 17:00 / 18:00**. The largest
   errors are at commute peaks, where demand is highest and varies most from day to day.

## 5. Why R² alone is not enough to choose a model
- R² compares the model to "predict the period's average", so it depends on how much the
  target varies in that period. The selected model's **error fell** from validation to
  test (RMSE 74.4 → 67.7), yet its **R² also fell** (0.896 → 0.887), because the winter
  test period varies less.
- MAE and RMSE are in bikes per hour, which an operator can understand: "on average we
  are off by about 42 bikes".
- RMSE punishes large errors (rush hours), MAE treats all errors equally. Reporting both
  shows whether errors come from a few big misses.

## 6. Limitations
- Weather inputs are the observed values for that hour. In real use they would come from
  a forecast, which adds error not measured here.
- The test period is one season (Oct–Dec), and the Sandy shutdown (29–30 Oct 2012) is
  inside it.
- Linear models with one-hot features cannot capture every interaction. Tree-based
  regressors are outside the Unit III syllabus list and are left as future work.
