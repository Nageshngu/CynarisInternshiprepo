# AI-Powered Disease Prediction and Symptom Checker

**Week 1 + Week 3 + Week 6 milestones completed (educational prototype).** Not a medical device. Predictions must not be used for diagnosis, treatment, or decisions to delay medical care.
Week 1: Research + Architecture Design

Educational prototype only. Not a medical device or clinical diagnosis tool. Never rely on model outputs to determine treatment or delay medical care. Urgent symptoms require professional assessment.

Week 1 deliverables

docs/research_review.md: three technical references and their relevance (technical review; not a claim of clinical validation)

docs/architecture.md: planned FastAPI + Streamlit + SHAP architecture

docs/evaluation_plan.md: data splits, metrics, limitations and safety checks

docs/week1_submission.md: milestone summary to adapt before submission

scripts/inspect_dataset.py: runnable data inspection

src/data_pipeline.py: preprocessing/split skeleton

backend/main.py: initial API skeleton (no diagnostic endpoint)

frontend/app.py: initial Streamlit screen (no predictions)

data/raw/: supplied CSV, JSON and test cases

Dataset

The provided CSV has 800 rows, 17 columns and 8 diagnosis classes with 100 records per class. This does not establish clinical validity. Do not interpret example labels or scripted test expectations as ground truth.

Run Week 1 inspection

python -m pip install -r requirements.txt
python scripts/inspect_dataset.py

Run starter UI and API (optional)

uvicorn backend.main:app --reload
streamlit run frontend/app.py

## Week 3: Data pipeline + baseline model
- CSV ingestion and validation: `src/data_pipeline.py`
- Stratified 70/15/15 split; preprocessing fitted only on training data
- Logistic Regression baseline and majority-class benchmark: `scripts/train_baseline.py`
- Saved model: `models/baseline_logistic_regression.joblib`
- Machine-readable results: `reports/metrics.json`
- Full report: `docs/week3_baseline_report.md`
- Ready-to-adapt milestone summary: `docs/week3_submission.md`
- Basic automated checks: `tests/test_pipeline.py`

**Measured results:** 800 records, 8 labels; validation accuracy 80.83%, held-out test accuracy 80.00%, majority-class test benchmark 12.50%. Dataset-only results; no clinical validation.

## Reproduce Week 3
Requires Python 3.10+.
```bash
python -m pip install -r requirements.txt
python scripts/train_baseline.py
python -m unittest discover -s tests
```

## Week 1 materials retained
- `docs/research_review.md`: three technical references
- `docs/architecture.md`: planned FastAPI + Streamlit + SHAP design
- `docs/evaluation_plan.md`: evaluation plan and safety constraints
- `docs/week1_submission.md`: Week 1 summary
- `scripts/inspect_dataset.py`: data inspection
- `backend/main.py` and `frontend/app.py`: starter placeholders (not integrated prediction apps)
- `data/raw/`: supplied dataset and reference samples

## Limitations
This model uses clinical measurements in addition to symptoms. Data provenance, licensing and medical validity are unverified. The API/UI and SHAP per-prediction explanations are planned work, not Week 3 deliverables. Do not publish patient data without authorization.

## Week 6: Core model + experimentation
- Three Random Forest hyperparameter configurations trained and compared.
- Reproducible script: `python scripts/train_week6.py`
- Full evidence: `reports/week6_experiments.json` and `docs/week6_experiment_report.md`
- Submission summary: `docs/week6_submission.md`
- Selected model: `models/week6_best_random_forest.joblib`
- MLflow support is coded, but MLflow was unavailable during packaging; after installing requirements, rerun the training script to create real MLflow experiment logs.
- View tracking runs with `mlflow ui --backend-store-uri ./mlruns` after rerunning.


## Week 9: Integration + Testing

FastAPI model-serving endpoint: `python -m uvicorn backend.main:app --port 8000`

Streamlit UI (separate terminal): `python -m streamlit run frontend/app.py`

Integration tests: `python -m pytest tests/test_week9_integration.py -v`

50-concurrent-request load test (API running): `python scripts/load_test_week9.py --concurrency 50 --requests 200`

See `docs/week9_integration_guide.md` and `docs/week9_submission.md`. Benchmark measurements must be generated on the user's machine. This project is not clinically validated.
