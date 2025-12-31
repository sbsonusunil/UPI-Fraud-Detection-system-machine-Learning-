# src/features/build_features.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_features(df: pd.DataFrame):
    """
    Build features and preprocessing pipeline for UPI fraud detection.
    """

    df = df.copy()

    # ---------------- Target ----------------
    target_col = "fraud_flag"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # ---------------- Feature Engineering ----------------
    if "amount (INR)" in X.columns:
        X["amount_log"] = np.log1p(X["amount (INR)"])

    # ---------------- Column Types ----------------
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

    # ---------------- Preprocessor ----------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor
