"""Week 3 ingestion and leakage-safe preprocessing utilities."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = 'Diagnosis'
EXCLUDE = ['Patient_ID', TARGET]
CATEGORICAL = ['Gender']

def load_dataset(path):
    df = pd.read_csv(Path(path))
    if TARGET not in df.columns:
        raise ValueError(f'Missing target column: {TARGET}')
    if df[TARGET].isna().any():
        raise ValueError('Target contains missing labels')
    if df['Patient_ID'].duplicated().any():
        raise ValueError('Duplicate patient identifiers detected; review before splitting')
    if df.drop(columns=['Patient_ID']).duplicated().any():
        raise ValueError('Duplicate feature-label rows detected; review before splitting')
    return df

def split_data(df, random_state=42):
    X = df.drop(columns=EXCLUDE)
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.30, stratify=y, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.50, stratify=y_temp, random_state=random_state)
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

def make_preprocessor(X):
    categorical = [c for c in CATEGORICAL if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical)
    ])
