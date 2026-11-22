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
