
# Week 1 – Research and Architecture Design

**Project:** HealthBridge Hospitals – AI-Powered Disease Prediction and Symptom Checker

**Internship:** Cynaris Internship

**Checkpoint:** 1 – Research and Architecture Design

## 1. Objective

Design an educational AI-powered disease prediction and symptom checker using machine learning, FastAPI, Streamlit and planned SHAP explainability.

## 2. Research References

### Reference 1 – Random Forests

- Author: Leo Breiman
- Year: 2001
- Journal: Machine Learning, 45, 5–32
- DOI: https://doi.org/10.1023/A:1010933404324

Key findings:
- Ensemble learning combines multiple decision trees.
- Random Forest can capture nonlinear feature relationships.
- Random Forest provides a candidate model for comparison with Logistic Regression.

### Reference 2 – Scikit-learn: Machine Learning in Python

- Authors: Fabian Pedregosa et al.
- Year: 2011
- Journal: Journal of Machine Learning Research
- Link: https://jmlr.org/papers/v12/pedregosa11a.html

Key findings:
- Provides machine learning algorithms and preprocessing tools.
- Supports reproducible training and evaluation.
- Provides pipelines for consistent feature transformation.

### Reference 3 – A Unified Approach to Interpreting Model Predictions

- Authors: Scott M. Lundberg and Su-In Lee
- Year: 2017
- Conference: NeurIPS
- Link: https://arxiv.org/abs/1705.07874

Key findings:
- Introduces SHAP-based model explainability.
- Supports individual prediction interpretation.
- Provides a framework for understanding feature contributions.

## 3. Proposed System Architecture

Training workflow:

Healthcare Symptom Dataset
→ Data Ingestion
→ Data Validation and Preprocessing
→ Training, Validation and Testing Split
→ Logistic Regression Baseline
→ Random Forest Experiments
→ Model Evaluation
→ Saved Model

Application workflow:

Streamlit User Interface
→ FastAPI Backend
→ Trained Model
→ Prediction Results
→ Planned SHAP Explanations and Safety Warnings

## 4. Dataset Description

Previously reported dataset characteristics:

- Total records: 800
- Total columns: 17
- Diagnosis categories: 8
- Records per category: 100
- Target variable: Diagnosis
- Dataset: data/raw/healthcare_symptoms.csv

The dataset contains demographic information, symptoms and health-related measurements.

## 5. Data Pipeline Skeleton

The data pipeline is located in src/data_pipeline.py.

Core functions:

- load_dataset(): Loads the healthcare symptom dataset.
- split_data(): Separates data into training, validation and testing subsets.
- make_preprocessor(): Defines feature preprocessing transformations.

Planned preprocessing steps:

1. Load and inspect the dataset.
2. Validate required columns.
3. Inspect missing values and duplicate records.
4. Remove patient identifiers from training features.
5. Separate features and target labels.
6. Prepare numerical and categorical features.
7. Perform stratified dataset splitting.
8. Fit preprocessing on training data only.

## 6. Evaluation Strategy

Dataset splitting:

| Subset | Percentage | Records |
|---|---:|---:|
| Training | 70% | 560 |
| Validation | 15% | 120 |
| Testing | 15% | 120 |

Planned random seed: 42

Evaluation metrics:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Confusion matrix
- Class-specific performance

## 7. Repository Implementation Evidence

| File | Purpose |
|---|---|
| README.md | Project overview and setup |
| docs/research_review.md | Research references |
| docs/architecture.md | System architecture |
| docs/evaluation_plan.md | Evaluation methodology |
| docs/week1_submission.md | Week 1 milestone report |
| src/data_pipeline.py | Data pipeline skeleton |
| scripts/inspect_dataset.py | Dataset inspection |
| backend/main.py | FastAPI backend |
| frontend/app.py | Streamlit frontend |
| requirements.txt | Project dependencies |

## 8. Environment Setup

Run from the project directory with the virtual environment activated.

```powershell
python -m pip install -r requirements.txt
python scripts/inspect_dataset.py
```

## 9. Limitations

- Limited dataset size.
- Restricted diagnosis categories.
- Potential demographic bias.
- Lack of independent clinical validation.
- SHAP explanations do not establish clinical reliability.

The application is strictly an educational prototype and must not be used for medical diagnosis or treatment decisions.

## 10. GitHub Repository

https://github.com/Nageshngu/CynarisInternshiprepo/tree/healthbridge-final/AI_Disease_Prediction_Week1

## 11. Conclusion

Week 1 establishes the research foundation, proposed system architecture, evaluation strategy and data pipeline skeleton for the HealthBridge Hospitals project.

The research references support subsequent preprocessing, baseline model development, Random Forest experimentation and model explainability.
