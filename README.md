# AI-Powered Disease Prediction and Symptom Checker

**Week 1: Research + Architecture Design**

Educational prototype only. **Not a medical device or clinical diagnosis tool.** Never rely on model outputs to determine treatment or delay medical care. Urgent symptoms require professional assessment.

## Week 1 deliverables
- `docs/research_review.md`: three technical references and their relevance (technical review; not a claim of clinical validation)
- `docs/architecture.md`: planned FastAPI + Streamlit + SHAP architecture
- `docs/evaluation_plan.md`: data splits, metrics, limitations and safety checks
- `docs/week1_submission.md`: milestone summary to adapt before submission
- `scripts/inspect_dataset.py`: runnable data inspection
- `src/data_pipeline.py`: preprocessing/split skeleton
- `backend/main.py`: initial API skeleton (no diagnostic endpoint)
- `frontend/app.py`: initial Streamlit screen (no predictions)
- `data/raw/`: supplied CSV, JSON and test cases

## Dataset
The provided CSV has 800 rows, 17 columns and 8 diagnosis classes with 100 records per class. This does not establish clinical validity. Do not interpret example labels or scripted test expectations as ground truth.

## Run Week 1 inspection
```bash
python -m pip install -r requirements.txt
python scripts/inspect_dataset.py
```

## Run starter UI and API (optional)
```bash
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

## Next milestones
1. Review data provenance, licensing, label quality, and representativeness.
2. Add leakage-safe preprocessing and stratified train/validation/test split.
3. Train and compare baseline classifiers; record test results only after evaluation.
4. Add calibrated probabilities, abstention, SHAP, red-flag handling, and clinician-reviewed safety content.
5. Integrate FastAPI inference with Streamlit, with privacy safeguards.

## GitHub
Create a repository named `AI-Disease-Prediction-Symptom-Checker`, then commit this folder. Review whether you have permission to publish the supplied data before pushing `data/raw/` to a public repository.
