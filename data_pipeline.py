import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath="creditcard.csv", sample_normal=5000):
    df = pd.read_csv(filepath)
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0].sample(n=sample_normal, random_state=42)
    balanced_df = pd.concat([normal, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    feature_cols = [c for c in balanced_df.columns if c not in ['Time', 'Class']]
    X = balanced_df[feature_cols]
    y = balanced_df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "raw_df": balanced_df,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler
    }