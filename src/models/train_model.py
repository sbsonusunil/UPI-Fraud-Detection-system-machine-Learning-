# src/models/train_model.py

from xgboost import XGBClassifier

def train_xgb_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="auc",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
