# Week 3 Baseline Report — Data Pipeline + Baseline Model

## Objective
Implement CSV ingestion, validation, preprocessing, stratified data splitting, baseline model training, and an evaluation benchmark for an educational symptom-based disease classifier.

## Dataset and checks
- Input: `data/raw/healthcare_symptoms.csv`
- 800 records, 17 columns, eight equally represented diagnosis labels (100 per class).
- `Patient_ID` excluded as an identifier; `Diagnosis` is the target.
- CSV validation checks missing labels, duplicate patient identifiers and duplicate feature/label rows.
- Features include symptom indicators, demographic and clinical measurements. This is **not** a symptoms-only model.
- Dataset provenance, licensing, label verification, and clinical representativeness are not established. Do not use for patient care.

## Pipeline
1. Load and validate CSV.
2. Stratified split 70/15/15 with seed 42: 560 training, 120 validation, 120 test.
3. Fit preprocessing on **training only** using `Pipeline`/`ColumnTransformer`: median numeric imputation + standard scaling; most-frequent categorical imputation + one-hot encoding for Gender.
4. Fit `LogisticRegression(max_iter=2000, random_state=42)` on training split.
5. Compare with a majority-class `DummyClassifier` on validation and held-out test sets.
6. Save model as `models/baseline_logistic_regression.joblib` and metrics in `reports/metrics.json`.

## Reproduced results
| Metric | Validation model | Test model | Test dummy |
|---|---:|---:|---:|
| Accuracy | 0.8083 | 0.8000 | 0.1250 |
| Balanced accuracy | 0.8083 | 0.8000 | 0.1250 |
| Macro precision | 0.8115 | 0.8130 | 0.0156 |
| Macro recall | 0.8083 | 0.8000 | 0.1250 |
| Macro F1 | 0.8068 | 0.7989 | 0.0278 |

The test accuracy is 96/120, not a medical safety benchmark. The dummy benchmark achieves 15/120 because classes are balanced. See `reports/metrics.json` for full per-class classification report and confusion matrix. These scores reflect one deterministic split; no external validation, uncertainty intervals, subgroup evaluation, or probability calibration were completed.

## Limitations and next steps
- Some features (e.g., HbA1c and blood pressure) require measurements and cannot be inferred from free-text symptoms.
- No claims of reliable diagnosis, treatment recommendations, or clinical triage are supported by this experiment.
- Verify provenance and labels; add repeated cross-validation, external testing, calibration, red-flag rules reviewed by clinicians, SHAP explanations, FastAPI integration, and Streamlit UI in subsequent milestones.
- Avoid uploading raw health data to public repositories without rights and privacy review.

## Reproduction
```bash
pip install -r requirements.txt
python scripts/train_baseline.py
python -m unittest discover -s tests
```
