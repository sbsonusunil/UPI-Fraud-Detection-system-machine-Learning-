import os, sys, joblib, pandas as pd, numpy as np

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "upi_fraud_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgb_model.pkl")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")

def engineer_features(df):
    X = df.copy()
    X["amount_log"] = np.log1p(X["amount (INR)"])

    # day_of_week fix
    if X["day_of_week"].dtype == "object":
        day_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6,
                   "Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
        X["day_of_week"] = X["day_of_week"].map(day_map)

    X["hour_sin"] = np.sin(2*np.pi*X["hour_of_day"]/24)
    X["hour_cos"] = np.cos(2*np.pi*X["hour_of_day"]/24)
    X["day_of_week_sin"] = np.sin(2*np.pi*X["day_of_week"]/7)
    X["day_of_week_cos"] = np.cos(2*np.pi*X["day_of_week"]/7)

    ts = pd.to_datetime(df["timestamp"])
    X["year"] = ts.dt.year
    X["month"] = ts.dt.month
    X["day"] = ts.dt.day
    X["minute"] = ts.dt.minute
    return X

def main():
    df = pd.read_csv(DATA_PATH)
    X = engineer_features(df)

    pre = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    Xp = pre.transform(X)
    probs = model.predict_proba(Xp)[:,1]

    out = df.copy()
    out["fraud_probability"] = probs
    out.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "inference_output.csv"), index=False)
    print("Inference completed → data/processed/inference_output.csv")

if __name__ == "__main__":
    main()
