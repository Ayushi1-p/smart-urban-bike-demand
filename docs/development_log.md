# Development Log

One entry per real working session, with actual dates. Commit timestamps on GitHub are
the authoritative record. Add the commit messages from `git log --oneline` where marked.

---

## 21 Oct 2026 – Planning
- Verified the official UCI dataset page (licence CC BY 4.0, DOI, file list) and the
  paper citation (Crossref).
- Drafted the project structure, configuration and data-loader design, and the plan.

## 25 Oct 2026 – Environment setup
- Installed Python 3.13.15, VS Code with the Python extension, and Git.
- Created the project folder and a virtual environment (`.venv`); installed packages
  from `requirements.txt`.
- Created the folder structure.

## 28 Oct 2026 – Data loading and documentation
- Downloaded the official UCI zip; placed hour.csv, day.csv and Readme.txt in `data/raw/`.
- Added `config/project_config.py` and `src/data_loader.py` (schema validation).
- Ran `python -m src.data_loader`: 17,379 rows (UCI page states 17,389), 17 columns,
  0 missing values, 0 duplicates, 165 hours without a record.
- Updated `EXPECTED_HOUR_ROWS` to the verified 17,379; added `docs/data_dictionary.md`.
- Set up Git, created the GitHub repository, first push.
- Commits: [paste from git log]

## 29 Oct 2026 – EDA, models and dashboard
- `src/eda.py`: figures 01–08 and key numbers; `docs/eda_findings.md`. Season coding
  verified from the data (1 = winter).
- `src/feature_engineering.py`, `src/preprocessing.py`, `src/evaluation.py`,
  `src/regression.py`. Issue: pytest could not import `config` → fixed by placing
  `pytest.ini` in the project root. 13 tests passed.
- Regression run: selected Lasso [log(cnt+1)]; test MAE 42.49, RMSE 67.72, R² 0.887.
  `docs/regression_results.md`.
- Discussed whether to switch to Indian data: no official public hourly dataset found;
  kept UCI and documented it as a limitation.
- `src/classification.py`: selected Logistic Regression; test macro F1 0.816.
  `docs/classification_results.md`.
- `app.py`, `src/predict.py`, `tests/test_predictions.py`: dashboard working, 30 tests.
  Fixed chart order/axis and truncated metrics after reviewing screenshots.
- README with verified results.
- Commits: [paste from git log]

## 30 Oct 2026 – Notebooks and documentation
- Notebooks 01–04 (executed on the real data).
- Methodology flowchart, `report/report.md`, `docs/viva_questions.md`,
  `docs/linkedin_post_draft.md`, `docs/requirements_checklist.md`.
- Commits: [paste from git log]

## Next
- Repository made public (date: ____), LinkedIn post published (date: ____).
- Report PDF export, dated screenshots, final checks.
