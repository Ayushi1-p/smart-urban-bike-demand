# EDA Findings (from `python -m src.eda`, run on 29-Oct-2026 on the real hour.csv)

All numbers below were copied from the actual output of `src/eda.py`.
Figures are in `outputs/figures/`.

## 1. Target distribution (Figure 01)
- Skewness of `cnt` = **1.28**: strongly right-skewed. Most hours have low demand,
  a few hours have very high demand. Mean (189.5) > median (142).
- **Decision:** test a log transform of the target (`log1p(cnt)`) for regression and
  keep it only if it improves validation results.

## 2. Hour of day (Figures 02, 07)
- Busiest hours: **17:00 (461.5)**, **18:00 (425.5)**, **08:00 (359.0)**, 16:00 (312.0), 19:00 (311.5).
- These are morning and evening commute times.
- Average demand on working days (193.2) and weekends/holidays (181.4) is similar,
  but the **shape** of the day is different: two commute peaks on working days, a
  single midday hump on weekends (Figure 02).
- Pearson correlation of `hr` with `cnt` is only 0.394 because the relationship is not
  a straight line (low at night, peaks at 8 and 17–18).
- **Decision:** treat `hr` as a category (one-hot) for linear models, and add an
  hour × working-day interaction feature.

## 3. Season, month and year (Figures 03, 04)
- Average rentals per hour: Winter **111.1**, Spring **208.3**, Summer **236.0**, Fall **198.9**.
- 2011: **143.8**, 2012: **234.7**, about **63% higher** in 2012. The system was growing.
- **Decision:** a random train/test split would mix 2012 hours into training and
  overstate accuracy. We use the chronological split
  (train: Jan 2011–Jun 2012, validation: Jul–Sep 2012, test: Oct–Dec 2012).
- `yr` is kept as a feature. Training already contains 2012 (Jan–Jun), so the model
  has seen yr = 1.

### Season coding check
`Readme.txt` inside the UCI zip lists seasons as 1 = spring … 4 = winter, while the UCI
web page lists 1 = winter … 4 = fall. The data settles it: season 1 contains months
[1, 2, 3, 12], season 2 [3, 4, 5, 6], season 3 [6, 7, 8, 9], season 4 [9, 10, 11, 12]
(checked 29-Oct-2026). These are astronomical seasons (boundaries around the 21st), so
**1 = winter, 2 = spring, 3 = summer, 4 = fall**, as on the web page. Winter also has
the lowest demand (111.1), which agrees.

## 4. Weather (Figure 05)
- Hours per weather situation: 1 = **11,413**, 2 = **4,544**, 3 = **1,419**, 4 = **3**.
- Category 4 (heavy rain/snow) has only 3 hours, too few to learn from.
- **Decision:** merge category 4 into category 3 ("bad weather") for modelling. The
  same mapping is applied in the dashboard.
- Correlations with `cnt`: temp **+0.405**, atemp +0.401, hum **−0.323**,
  weathersit −0.142, windspeed +0.093. Warmer, drier hours have more rentals.

## 5. Correlation and feature selection (Figure 06)
- `registered` (0.972) and `casual` (0.695) are highly correlated with `cnt` because
  `cnt = casual + registered`. This confirms they are **leakage** and must be excluded.
- `temp` vs `atemp` correlation = **0.988**: almost the same information.
- **Decision:** drop `atemp` and keep `temp` (multicollinearity; easier to explain).
- `holiday`, `weekday` and `workingday` have weak linear correlation (−0.031 to 0.030)
  but affect the daily shape (see section 2), so they are kept.

## 6. Outliers (Figure 08)
- IQR method: Q1 = 40, Q3 = 281, IQR = 241, upper limit = 642.5.
- **505 hours (2.91%)** are above the upper limit.
- Figure 08 shows which hours these are. They are genuine busy commute hours, not
  data errors: all values passed the range checks in `data_loader.py`.
- **Decision:** keep them. Removing them would teach the model that rush hours do not
  exist. MAE (less sensitive to outliers) is reported alongside RMSE.

## 7. Missing hours
- 165 of 17,544 hours have no row. They are concentrated on a few dates:
  - 2012-10-29 (23 hours) and 2012-10-30 (13): Hurricane Sandy; Capital Bikeshare
    suspended service on Monday 29 Oct 2012.
  - 2011-01-26 (8), 2011-01-27 (16): the January 2011 snowstorm in Washington D.C.
  - 2011-08-27 (6), 2011-08-28 (7): weekend of Hurricane Irene (to be confirmed).
  - 2011-01-18 (12), 2011-02-22 (6): cause not identified.
- These are periods when the system was closed or not recording, not missing cell
  values (the file has 0 missing values).
- **Decision:** do not invent rows. Note in the report that the test period
  (Oct–Dec 2012) contains the Sandy shutdown.

## 8. Imbalance
- The input feature `weathersit` is imbalanced (handled in section 4).
- The classification target (Low/Medium/High) will be balanced by design: tertile
  thresholds are computed on the training period. Its class distribution in the
  validation and test periods will be checked and reported (Day 3).