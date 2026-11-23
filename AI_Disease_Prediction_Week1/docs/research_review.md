# Week 1: Technical reference review

These are technical resources relevant to implementation, not clinical evidence that symptom-based disease prediction is safe.

## 1. Scikit-learn — Random Forest and ensemble methods
Source: https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees

**Method:** Ensemble of decision trees; suitable baseline for mixed tabular clinical-style features after appropriate encoding.
**Project relevance:** Compare Random Forest to simpler Logistic Regression and a boosting model.
**Limitation:** Strong benchmark performance on a supplied dataset does not establish generalization to patients.

## 2. Lundberg & Lee (2017) — SHAP
Paper: https://arxiv.org/abs/1705.07874
Code: https://github.com/shap/shap

**Method:** SHAP attributes feature contributions to an individual model output relative to a background distribution.
**Project relevance:** Show which symptoms or measurements influenced a prediction.
**Limitation:** SHAP explains model behavior, not disease causation or medical correctness.

## 3. FastAPI — API framework
Repository: https://github.com/fastapi/fastapi
Documentation: https://fastapi.tiangolo.com/

**Method:** Typed Python web API with request validation and automatic schema documentation.
**Project relevance:** Separate model inference from Streamlit presentation.
**Limitation:** Framework choice does not address medical safety, calibration, security or dataset bias.

## Research gaps to address
Locate and critically review peer-reviewed clinical validation studies, investigate dataset provenance and licensing, and obtain domain expert review before any patient-facing use.
