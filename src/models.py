"""
src/models.py

Placeholders for modeling pipelines:
- Churn prediction
- LTV estimation
- Fraud classifier

Fill in with domain-specific feature engineering and model persistence.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import joblib

def churn_model_train(X: pd.DataFrame, y: pd.Series, output_path: str = "models/churn_rf.joblib") -> Tuple[object, dict]:
    """
    Train a simple RandomForest for churn prediction.
    Returns (model, metrics).
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    preds = m.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    joblib.dump(m, output_path)
    return m, {"auc": float(auc)}

def ltv_model_train(X: pd.DataFrame, y: pd.Series, output_path: str = "models/ltv_rf.joblib") -> Tuple[object, dict]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    preds = m.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    joblib.dump(m, output_path)
    return m, {"rmse": float(rmse)}
