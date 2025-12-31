# pipelines/train_pipeline.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ======================================================
# ADD PROJECT ROOT TO PYTHON PATH (WINDOWS SAFE)
# ======================================================
CURRENT_FILE = os.path.abspath(__file__)
PIPELINES_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(PIPELINES_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ======================================================
# PATHS
# ======================================================
DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "upi_fraud_dataset.csv"
)

PREPROCESSOR_PATH = os.path.join(
    PROJECT_ROOT, "models", "preprocessor.pkl"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "xgb_model.pkl"
)

# ======================================================
# MAIN TRAINING PIPELINE
# ======================================================
def main():
    print("Starting UPI Fraud Training Pipeline")

    # ---------------- Load data ----------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # ---------------- Target ----------------
    TARGET_COL = "fraud_flag"
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column `{TARGET_COL}` not found")

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL]

# --------------------------------------------------
# FEATURE ENGINEERING (must match preprocessor)
# --------------------------------------------------
    X["amount_log"] = np.log1p(X["amount (INR)"])

    X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)
    # --------------------------------------------------
# FIX: Convert day_of_week to numeric
# --------------------------------------------------
    if X["day_of_week"].dtype == "object":
        day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
        "Mon": 0,
        "Tue": 1,
        "Wed": 2,
        "Thu": 3,
        "Fri": 4,
        "Sat": 5,
        "Sun": 6,
    }
    X["day_of_week"] = X["day_of_week"].map(day_map)

    X["day_of_week_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["day_of_week_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

    X["year"] = pd.to_datetime(df["timestamp"]).dt.year
    X["month"] = pd.to_datetime(df["timestamp"]).dt.month
    X["day"] = pd.to_datetime(df["timestamp"]).dt.day
    X["minute"] = pd.to_datetime(df["timestamp"]).dt.minute


    # ---------------- Load existing preprocessor ----------------
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            "preprocessor.pkl not found. "
            "Make sure it exists in models/ folder."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Existing preprocessor loaded")

    # ---------------- Train-test split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ---------------- Apply preprocessing ----------------
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # ---------------- Train model ----------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train_prep, y_train)

    # ---------------- Evaluate ----------------
    preds = model.predict_proba(X_test_prep)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"ROC-AUC Score: {auc:.4f}")

    # ---------------- Save model ----------------
    joblib.dump(model, MODEL_PATH)
    print("Model saved to models/xgb_model.pkl")

    print("Training Pipeline Completed Successfully")


# ======================================================
if __name__ == "__main__":
    main()
