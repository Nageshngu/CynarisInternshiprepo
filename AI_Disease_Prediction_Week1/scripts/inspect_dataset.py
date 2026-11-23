from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'raw' / 'healthcare_symptoms.csv')
print('Rows, columns:', df.shape)
print('Columns:', df.columns.tolist())
print('Missing values:', int(df.isna().sum().sum()))
print('Duplicate rows:', int(df.duplicated().sum()))
print('Diagnosis counts:')
print(df['Diagnosis'].value_counts().to_string())
