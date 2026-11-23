"""Week 6: compare primary Random Forest configurations on validation data.
MLflow logging is enabled when mlflow is installed. Test set is evaluated once.
"""
from pathlib import Path
import json, sys, joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.data_pipeline import load_dataset, split_data, make_preprocessor
try:
    import mlflow
except ImportError:
    mlflow=None

def evaluate(y,pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'macro_precision':float(p),'macro_recall':float(r),'macro_f1':float(f)}

def main():
    df=load_dataset(ROOT/'data/raw/healthcare_symptoms.csv')
    (Xtr,Xv,Xte),(ytr,yv,yte)=split_data(df,random_state=42)
    configs=[{'n_estimators':100,'max_depth':5,'min_samples_leaf':2}, {'n_estimators':200,'max_depth':10,'min_samples_leaf':2}, {'n_estimators':300,'max_depth':None,'min_samples_leaf':1}]
    records=[]; models=[]
    if mlflow:
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("DiseasePrediction-Week6")
    for idx,params in enumerate(configs,1):
        model=Pipeline([('preprocess',make_preprocessor(Xtr)),('classifier',RandomForestClassifier(**params,random_state=42,n_jobs=-1,class_weight='balanced'))])
        model.fit(Xtr,ytr)
        train=evaluate(ytr,model.predict(Xtr)); val=evaluate(yv,model.predict(Xv))
        row={'run':f'RF_{idx}','parameters':params,'train':train,'validation':val}
        records.append(row);models.append(model)
        if mlflow:
            with mlflow.start_run(run_name=row['run']):
                mlflow.log_params({**params,'random_state':42,'class_weight':'balanced'})
                mlflow.log_metrics({'train_'+k:v for k,v in train.items()})
                mlflow.log_metrics({'val_'+k:v for k,v in val.items()})
    # Selection uses validation macro-F1, with balanced accuracy tie-break.
    best_idx=max(range(len(records)),key=lambda i:(records[i]['validation']['macro_f1'],records[i]['validation']['balanced_accuracy']))
    best=records[best_idx]; model=models[best_idx]
    pred=model.predict(Xte)
    result={'dataset_rows':len(df),'split':{'train':len(Xtr),'validation':len(Xv),'test':len(Xte)},'selection_metric':'validation_macro_f1','runs':records,'best_run':best['run'],'best_parameters':best['parameters'],'best_validation':best['validation'],'held_out_test':evaluate(yte,pred),'test_classification_report':classification_report(yte,pred,output_dict=True,zero_division=0),'test_confusion_matrix':confusion_matrix(yte,pred,labels=model.classes_).tolist(),'class_labels':model.classes_.tolist(),'mlflow_logged':bool(mlflow),'tracking_uri': mlflow.get_tracking_uri() if mlflow else None}
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'reports/week6_experiments.json').write_text(json.dumps(result,indent=2))
    joblib.dump(model,ROOT/'models/week6_best_random_forest.joblib')
    print(json.dumps({'best_run':result['best_run'],'best_parameters':result['best_parameters'],'best_validation':result['best_validation'],'held_out_test':result['held_out_test'],'mlflow_logged':result['mlflow_logged']},indent=2))
    return result
if __name__=='__main__':main()
