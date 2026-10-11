import pandas as pd
from sklearn.ensemble import IsolationForest

def sanitize_data(X_train, y_train_poisoned, contamination=0.08):
    inspect_matrix = X_train.copy()
    inspect_matrix['_target_flag'] = y_train_poisoned
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_preds = iso.fit_predict(inspect_matrix)
    
    clean_mask = (iso_preds == 1)
    quarantine_mask = (iso_preds == -1)
    
    X_sanitized = X_train[clean_mask]
    y_sanitized = y_train_poisoned[clean_mask]
    
    quarantined_indices = set(X_train[quarantine_mask].index)
    quarantined_df = X_train[quarantine_mask].copy()
    quarantined_df.insert(0, "Assigned_Label", y_train_poisoned[quarantine_mask])
    quarantined_df.insert(1, "Isolation_Status", "QUARANTINED_BY_IFOREST")
    
    return X_sanitized, y_sanitized, quarantined_indices, quarantined_df, int(clean_mask.sum()), int(quarantine_mask.sum())