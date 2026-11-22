"""Week 1 preprocessing skeleton; no model training is performed here."""
from sklearn.model_selection import train_test_split

TARGET = 'Diagnosis'
EXCLUDE = ['Patient_ID', TARGET]


def split_data(df, random_state=42):
    """Create stratified 70/15/15 split; inspect duplicates/groups first."""
    X = df.drop(columns=EXCLUDE)
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return (X_train, X_val, X_test), (y_train, y_val, y_test)


def make_preprocessor(X):
    categorical = [c for c in CATEGORICAL if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical)
    ])

