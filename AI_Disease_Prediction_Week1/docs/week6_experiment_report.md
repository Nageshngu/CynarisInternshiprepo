# Week 6 — Core Model and Experimentation Report

## Method
Used the same 800-row supplied CSV and stratified 560/120/120 train/validation/test split from Week 3 (random state 42). Primary model: Random Forest with balanced class weights. Preprocessing is fitted on the training partition only. Compared three configurations; selected by validation macro F1, with balanced accuracy tie-break. The held-out test partition was used once after model selection.

## Configuration comparison
| Run | Trees | Max depth | Min leaf | Validation accuracy | Validation macro F1 |
|---|---:|---|---:|---:|---:|
| RF_1 | 100 | 5 | 2 | 85.83% | 85.64% |
| RF_2 | 200 | 10 | 2 | 82.50% | 82.40% |
| RF_3 | 300 | None | 1 | 80.83% | 80.66% |

## Best model
**RF_1**, parameters `{'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 2}`. Validation accuracy: **85.83%**, validation macro F1: **85.64%**. Held-out test accuracy: **84.17%**, test macro F1: **84.13%**.

## Experiment tracking
The training script supports MLflow local tracking, recording hyperparameters and train/validation metrics for each run. **MLflow was not installed in the build environment, so these completed experiments are recorded in `reports/week6_experiments.json` rather than an MLflow tracking store.** To fulfill the MLflow requirement, install `mlflow` and rerun `python scripts/train_week6.py`; the script will then write MLflow runs to `mlruns/`. Do not claim MLflow logging has been completed before rerunning.

## Artifacts
- `scripts/train_week6.py` reproducible training + optional MLflow logging
- `reports/week6_experiments.json` full configurations, metrics, classification report and confusion matrix
- `models/week6_best_random_forest.joblib` fitted selected model

## Limitations
Dataset provenance and medical validity have not been established. Scores on a small educational dataset do not demonstrate safe clinical performance. SHAP and the API/UI are separate future integration tasks. No deployment or medical decision use is warranted.
