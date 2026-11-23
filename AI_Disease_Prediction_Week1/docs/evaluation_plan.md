# Evaluation plan (not completed)

## Supplied CSV overview
800 records; 17 columns including `Patient_ID` and `Diagnosis`; 8 balanced diagnosis classes. Initial file inspection shows no missing values. No model has been trained or evaluated in Week 1.

## Proposed split
Stratified 70% train / 15% validation / 15% test = 560 / 120 / 120 records. Ensure no patient or duplicated record appears across partitions; investigate whether labels are synthetic or rule-generated.

## Planned metrics
Macro and per-class precision, recall, F1-score, confusion matrix, balanced accuracy, and one-vs-rest ROC-AUC where applicable. Evaluate calibration, false negatives, out-of-distribution inputs and abstention behavior. Report uncertainty intervals if feasible. Do not assert a universal acceptable medical accuracy threshold.

## Safety criteria
A high model score alone is not evidence of clinical readiness. Test red-flag symptom handling independently of the classifier; use clinician-reviewed guidance, avoid treatment prescriptions, and do not deploy for real-world triage without appropriate validation and governance.

## Supplied test case caveat
The provided TXT file contains example expected outputs (including clinical labels and specific confidence values). These are specifications to inspect, not verified diagnoses or observed model outcomes.
