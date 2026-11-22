# Planned system architecture

```text
User -> Streamlit UI -> FastAPI input validation -> feature preprocessing
                                            -> trained disease classifier
                                            -> uncertainty / abstention policy
                                            -> SHAP feature attribution
                                            -> safety review / red-flag messaging
     <- educational result and explanation <- JSON response
```

## Components
- `data/raw`: uploaded example data and supplied tests
- `src/data_pipeline.py`: feature selection and leakage-safe train/validation/test splitting
- `src/train.py` (future): model training and serialized pipeline
- `backend/main.py`: API health endpoint; prediction endpoint deferred until validation
- `frontend/app.py`: Streamlit initial UI
- `src/explain.py` (future): per-prediction SHAP analysis

## Design decisions
Exclude `Patient_ID` from features. Use `Diagnosis` as target. Encode `Gender` within the training pipeline; fit transformations only on training data. Use stratified splits when class sizes permit. Report limitations of a small, potentially synthetic dataset. Avoid collecting personally identifying patient data. Do not present class scores as clinical probabilities without calibration and validation.
