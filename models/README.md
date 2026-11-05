# Models

Created by the training scripts (not committed to Git; recreate them with the commands below):

| File | Created by | Content |
|---|---|---|
| `regression_model.joblib` | `python -m src.regression` | selected pipeline (preprocessing + Lasso on log(cnt+1)) |
| `regression_model_meta.json` | same | parameters, validation/test metrics, features, package versions |
| `classification_model.joblib` | `python -m src.classification` | selected pipeline (preprocessing + Logistic Regression) |
| `classification_model_meta.json` | same | Low/Medium/High thresholds, metrics, versions |

Each `.joblib` file holds the **whole pipeline**, so the dashboard applies exactly the same
transformations as training. Random seed: 42.
