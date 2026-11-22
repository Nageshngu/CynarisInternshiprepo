"""Train a reproducible Week 3 baseline; hold out test data until final evaluation."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, balanced_accuracy_score
from src.data_pipeline import load_dataset, split_data, make_preprocessor

DATA = ROOT / 'data/raw/healthcare_symptoms.csv'

def metrics(y, pred):
    p,r,f,_ = precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    return {'accuracy':round(float(accuracy_score(y,pred)),4), 'balanced_accuracy':round(float(balanced_accuracy_score(y,pred)),4), 'macro_precision':round(float(p),4), 'macro_recall':round(float(r),4), 'macro_f1':round(float(f),4)}

def main():
    df = load_dataset(DATA)
    (Xtr,Xv,Xte),(ytr,yv,yte) = split_data(df)
    baseline = Pipeline([('preprocess',make_preprocessor(Xtr)),('classifier',LogisticRegression(max_iter=2000,random_state=42))])
    dummy = DummyClassifier(strategy='most_frequent')
    baseline.fit(Xtr,ytr)
    dummy.fit(Xtr,ytr)
    results = {'dataset_rows':len(df),'classes':sorted(df['Diagnosis'].unique().tolist()),'split':{'train':len(Xtr),'validation':len(Xv),'test':len(Xte)},'model':'LogisticRegression(max_iter=2000, random_state=42)','benchmark':'DummyClassifier(strategy=most_frequent)','validation':{'model':metrics(yv,baseline.predict(Xv)),'dummy':metrics(yv,dummy.predict(Xv))},'test':{'model':metrics(yte,baseline.predict(Xte)),'dummy':metrics(yte,dummy.predict(Xte))},'classification_report':classification_report(yte,baseline.predict(Xte),output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yte,baseline.predict(Xte),labels=baseline.classes_).tolist(),'confusion_matrix_labels':baseline.classes_.tolist()}
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'reports').mkdir(exist_ok=True)
    joblib.dump(baseline,ROOT/'models/baseline_logistic_regression.joblib')
    (ROOT/'reports/metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:results[k] for k in ['split','validation','test']},indent=2))
    return results

if __name__=='__main__':main()
