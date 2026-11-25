# Week 3 — Data Pipeline and Baseline Model Report

**Project:** HealthBridge Hospitals — AI-Powered Disease Prediction and Symptom Checker  
**Milestone:** Checkpoint 2 | Cynaris Internship  
**Status:** Revised documentation

## 1. Objective
Build a reproducible data ingestion and preprocessing pipeline, train a multiclass baseline classifier, and establish an evaluation benchmark for later model experiments. This is an educational prototype, not a medical device.

## 2. Dataset
The previously reported dataset (`data/raw/healthcare_symptoms.csv`) contains 800 records, 17 columns and eight diagnosis classes with 100 records per class. Features include demographics, symptoms and health measurements. `Diagnosis` is the target; `Patient_ID` is excluded from training. Dataset provenance and clinical representativeness have not been independently verified.

## 3. Data ingestion and preprocessing
The pipeline is maintained in `src/data_pipeline.py`. The intended workflow is:

1. Load the CSV using pandas and inspect its dimensions, column names and data types.
2. Check required columns, missing values, duplicates, class counts and invalid values. Document any corrections rather than silently discarding records.
3. Separate the diagnosis target from predictors and exclude patient identifiers.
4. Use a stratified 70/15/15 split with random seed 42: 560 training, 120 validation and 120 test records.
5. Prepare numerical features and encode categorical features using the training-fitted preprocessing pipeline. Scaling numerical features and one-hot encoding nominal categories are suitable for Logistic Regression; verify the exact transformers and missing-value policy in the current implementation.
6. Fit preprocessing only on training data and apply the same transformations to validation and test data to prevent leakage.
7. Train the classifier and serialize the fitted model/pipeline using joblib.

The exact missing-value counts, duplicate counts, transformer settings and feature ranges should be recorded from the dataset inspection and current source code. They were not independently checked for this revision.

## 4. Baseline model selection
Logistic Regression was selected because it is computationally efficient, provides a simple multiclass benchmark and offers interpretable coefficients after preprocessing. Its linear decision boundaries provide a useful comparison for later nonlinear models such as Random Forest. It may underfit interactions between symptoms and health measurements.

## 5. Evaluation methodology
Use the validation set for model selection and the held-out test set for final evaluation. Accuracy measures the fraction of correct predictions. Balanced accuracy measures mean class recall. Macro precision, macro recall and macro F1 weight classes equally. A confusion matrix and per-class classification report should be retained to identify frequently confused diagnoses.

| Metric | Previously reported result |
|---|---:|
| Test accuracy | 80.00% |
| Macro F1 | 79.89% |
| Test samples (planned split) | 120 |
| Balanced accuracy | Not verified |
| Macro precision | Not verified |
| Macro recall | Not verified |

These are previously reported baseline figures, not a new training run. Confirm them against the generated evaluation output before submission. Do not substitute later Random Forest metrics for the Week 3 baseline.

## 6. Results analysis
The reported accuracy suggests that the baseline captures useful patterns within the supplied dataset, while approximately one in five test predictions is incorrect. The macro F1 score is close to accuracy, but this does not establish equal performance across diagnoses or demographic groups. Class-specific error analysis is needed to identify which categories are confused. No clinical accuracy or external generalization claim can be made from these results.

## 7. Limitations and improvements
- Small dataset and limited diagnosis categories: obtain larger, representative datasets and independent validation.
- Linear model assumptions: compare Random Forest and other nonlinear classifiers using the same split and metrics.
- Potential overfitting and leakage: use training-only preprocessing and stratified cross-validation for tuning.
- Unverified subgroup performance: evaluate age and gender groups, report sample sizes and uncertainty.
- Unverified probability calibration: evaluate calibration before interpreting model scores as probabilities.
- Incomplete class-level evidence: retain confusion matrices, per-class precision/recall/F1 and error analysis.

## 8. Reproducibility and environment setup
Run the following in Windows PowerShell from the repository root. Python 3.11 is the previously used CI target; record the actual local Python and package versions. Create a virtual environment only if one does not already exist.

```powershell
cd "C:\CynarisInternship\HealthBridge Hospitals"
# If needed: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
cd AI_Disease_Prediction_Week1
python --version
python -m pip install -r requirements.txt
python -m pip freeze
python scripts/inspect_dataset.py
python scripts/train_baseline.py
```

Expected baseline artifact: `models/baseline_logistic_regression.joblib`. Verify the actual generated report and model paths after running the script. Keep the dataset version, split seed, model parameters and dependency versions with the evaluation evidence.

## 9. Deliverables
| Path | Purpose |
|---|---|
| `data/raw/healthcare_symptoms.csv` | Input dataset |
| `src/data_pipeline.py` | Ingestion, splitting and preprocessing |
| `scripts/inspect_dataset.py` | Dataset inspection |
| `scripts/train_baseline.py` | Baseline training and evaluation |
| `models/baseline_logistic_regression.joblib` | Saved baseline model |
| `requirements.txt` | Dependencies |
| `reports/week3_baseline_report.md` | Revised baseline report |

Repository: https://github.com/Nageshngu/CynarisInternshiprepo/tree/healthbridge-final/AI_Disease_Prediction_Week1

## 10. Conclusion
Week 3 establishes the preprocessing pipeline and Logistic Regression benchmark for subsequent experimentation. The reported results provide a starting point for model comparison, not evidence of clinical safety. This system must not be used for diagnosis, treatment decisions or delaying medical care.

**Verification note:** This revision uses previously reported project details. The current source code, dataset and baseline metrics have not been rerun or independently inspected for this report. Confirm implementation-specific details and attach actual evaluation output before resubmission.
