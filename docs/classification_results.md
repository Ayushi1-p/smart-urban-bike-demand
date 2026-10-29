# Classification Results (Task B: Low / Medium / High demand)

Source: actual output of `python -m src.classification`, run on 29-Oct-2026 on the real
hour.csv. Tables: `outputs/metrics/classification_validation_results.csv`,
`outputs/metrics/classification_test_results.csv`, `outputs/metrics/class_distribution.csv`,
`outputs/metrics/classification_report_test.txt`. Figures 15–19 in `outputs/figures/`.

## 1. How the classes were defined
Thresholds are the tertiles (33.3% and 66.7% quantiles) of `cnt` in the **training
period only** (2011-01-01 to 2012-06-30):

| Class | Rule |
|---|---|
| Low | cnt < **64** bikes/hour |
| Medium | 64 ≤ cnt < **198** |
| High | cnt ≥ 198 |

The same two numbers are applied to validation, test and dashboard predictions. They are
stored in `models/classification_model_meta.json`.

## 2. Class distribution
| Class | Train | Validation (Jul–Sep 2012) | Test (Oct–Dec 2012) |
|---|---|---|---|
| Low | 4,334 (33.3%) | 498 (22.6%) | 636 (29.3%) |
| Medium | 4,319 (33.2%) | 383 (17.3%) | 537 (24.8%) |
| High | 4,350 (33.5%) | 1,327 (60.1%) | 995 (45.9%) |

- Training is balanced by construction, so no resampling or class weighting was needed
  for training.
- Validation and test are **not** balanced: High is 60% and 46%. Demand grew in 2012
  (EDA: +63%), so more hours exceed the fixed threshold of 198. This is a real
  **distribution shift**, not an error. Recomputing thresholds on the test period would
  leak test information, so we did not do it.
- Because of this, models are selected and compared with **macro F1**, which weighs
  Low, Medium and High equally, not with accuracy.

## 3. Validation results (model selection), sorted by macro F1
Tuned with 5-fold `TimeSeriesSplit` on the training period (scoring = macro F1).

| Model | Best params | CV F1 | Train F1 | Val accuracy | Val precision | Val recall | **Val macro F1** | Val ROC-AUC (OvR macro) |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | C = 10 | 0.7819 | 0.8715 | 0.9207 | 0.8968 | 0.8881 | **0.8922** | 0.9800 |
| Random Forest | max_depth = None, min_samples_leaf = 1 | 0.7279 | 1.0000 | 0.9004 | 0.8612 | 0.8924 | 0.8743 | 0.9767 |
| Decision Tree | max_depth = 10, min_samples_leaf = 20 | 0.7494 | 0.8627 | 0.8981 | 0.8579 | 0.8744 | 0.8656 | 0.9719 |
| Naive Bayes (Gaussian) | var_smoothing = 1e-9 | 0.7741 | 0.6649 | 0.8922 | 0.8434 | 0.8673 | 0.8543 | 0.9655 |
| SVM (RBF) | C = 10 | 0.7354 | 0.9753 | 0.8718 | 0.8370 | 0.8751 | 0.8493 | 0.9563 |
| KNN | k = 15, weights = distance | 0.6883 | 1.0000 | 0.7948 | 0.7757 | 0.8377 | 0.7824 | 0.9462 |
| Dummy (most frequent) | – | 0.1492 | 0.1671 | 0.6010 | 0.2003 | 0.3333 | 0.2503 | 0.5000 |

**Selected model: Logistic Regression (C = 10)**, highest validation macro F1.

## 4. Final test result (Oct–Dec 2012, evaluated once)
| Model | Accuracy | Precision (macro) | Recall (macro) | **Macro F1** | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (selected)** | 0.8386 | 0.8374 | 0.8041 | **0.8160** | 0.9461 |
| SVM (RBF) | 0.8372 | 0.8285 | 0.8183 | 0.8217 | 0.9441 |
| Random Forest | 0.8326 | 0.8197 | 0.8133 | 0.8158 | 0.9477 |
| Decision Tree | 0.8072 | 0.7851 | 0.7786 | 0.7802 | 0.9142 |
| KNN | 0.7500 | 0.7624 | 0.7669 | 0.7498 | 0.9186 |
| Naive Bayes | 0.7735 | 0.7883 | 0.7114 | 0.7217 | 0.8736 |
| Dummy | 0.4589 | 0.1530 | 0.3333 | 0.2097 | 0.5000 |

Per-class results of the selected model (test):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.9621 | 0.8381 | 0.8958 | 636 |
| Medium | 0.7269 | 0.6145 | 0.6660 | 537 |
| High | 0.8233 | 0.9598 | 0.8863 | 995 |

Confusion matrix (rows = actual, columns = predicted, order Low / Medium / High):
```
         Low  Medium  High
Low      533     86     17
Medium    19    330    188
High       2     38    955
```

## 5. Interpretation
1. **All six real models beat the baseline by a wide margin.** Test macro F1 is 0.82
   for the selected model vs 0.21 for the Dummy.
2. **Why accuracy alone misleads.** On validation, the Dummy (always "High") reaches
   **60.1% accuracy** while its macro F1 is only 0.25. It never predicts Low or Medium.
3. **Medium is the hardest class** (F1 0.67). It is the middle band, so its hours sit
   next to both thresholds. Most errors go to the **neighbouring** class: 188 Medium hours
   were predicted as High. Low ↔ High confusions are rare (17 + 2 of 2,168 hours), so
   the model almost never makes a "two-step" mistake.
4. **The model leans towards High on the test period.** High recall is 0.96 but
   precision 0.82. Demand levels changed over time (section 2), so borderline Medium hours
   look like High.
5. **On test, SVM was marginally higher** (0.8217 vs 0.8160). We keep Logistic
   Regression because it was selected on validation. The top three are within 0.006 F1
   of each other on test, so they are practically equivalent.
6. **Validation → test drop** (0.892 → 0.816). Possible reasons: autumn/winter is a
   different season from the summer validation period; the class mix changes; the test
   period contains the Hurricane Sandy shutdown.
7. **Overfitting.** Random Forest and KNN reach train macro F1 = **1.000** (they memorise
   the training hours) but 0.874 / 0.782 on validation. The tuned Decision Tree is
   limited by min_samples_leaf = 20, and its train F1 (0.863) is close to validation.
8. **Why Logistic Regression does well here.** It receives one-hot hour and the combined
   hour × working-day feature, so it can learn a separate weight for "5 pm on a working
   day". The main pattern is easy to separate this way, and regularisation (C = 10)
   keeps it stable.
9. **Why KNN is weakest (apart from Naive Bayes on test).** It measures distance across
   ~110 mostly one-hot columns. In many dimensions every point looks roughly equally far
   away ("curse of dimensionality"), so neighbours are less informative.
10. **Naive Bayes** assumes features are independent given the class. That is clearly
    false here (hr and hr_workingday hold overlapping information), which explains the
    lowest test F1 of the real models. Its train F1 (0.665) is lower than validation
    because the validation mix is 60% High, which is easier for it.
11. **CV F1 is lower than validation F1 for most models.** Early TimeSeriesSplit folds
    train on only a few months of 2011, so the CV estimate is pessimistic. The validation
    period also has many clear High hours.
12. **ROC-AUC (one-vs-rest, macro) = 0.946 on test.** The predicted probabilities rank
    hours well, even where the final hard label (especially Medium) is wrong.

## 6. Feature importance (Random Forest, Figure 19)
hr **0.479**, temp 0.126, hum 0.102, windspeed 0.064, weekday 0.051, mnth 0.048,
yr 0.042, workingday 0.032, season 0.029, weathersit 0.025, holiday 0.004.

Hour of day is by far the most important feature, followed by temperature and humidity.
This agrees with the EDA. Caveat: impurity-based importance tends to favour continuous
features with many distinct values (temp, hum, windspeed), so windspeed may be
over-stated. Permutation importance would be a more reliable check (future work).

## 7. Limitations
- Thresholds are fixed from the training period. In a growing system the share of
  "High" hours rises over time. In operation, thresholds should be reviewed periodically
  (for example every season) using only past data.
- The classes are ordered (Low < Medium < High), but the models treat them as unordered.
  Ordinal methods are possible future work.
