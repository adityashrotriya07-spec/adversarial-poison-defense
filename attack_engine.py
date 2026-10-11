import numpy as np
import pandas as pd

def inject_label_flipping(X_train, y_train, poison_rate=0.25):
    fraud_indices = y_train[y_train == 1].index
    n_poison = int(len(fraud_indices) * poison_rate)
    poisoned_idx = np.random.choice(fraud_indices, size=n_poison, replace=False)
    
    y_train_poisoned = y_train.copy()
    y_train_poisoned.loc[poisoned_idx] = 0  # Invert malicious fraud (1) to benign (0)
    
    # Build an empirical audit trail table of the exact altered records
    tampered_records = X_train.loc[poisoned_idx].copy()
    tampered_records.insert(0, "Original_Label", y_train.loc[poisoned_idx])
    tampered_records.insert(1, "Injected_Poison_Label", y_train_poisoned.loc[poisoned_idx])
    tampered_records.insert(2, "Attack_Status", "LABEL_CORRUPTED (1 -> 0)")
    
    return y_train_poisoned, set(poisoned_idx), n_poison, tampered_records