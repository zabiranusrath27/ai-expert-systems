from data_prep import get_splits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import xgboost as xgb
import joblib, os

X_train, X_test, y_train, y_test = get_splits()

# RandomForest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
print("RF Report")
rf_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf.predict(X_test)))

os.makedirs("models", exist_ok=True)
joblib.dump((rf, X_train.columns.tolist()), "../models/model_rf.pkl")

# XGBoost
xg = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
xg.fit(X_train, y_train)
print("XGB Report")
xg_pred = xg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, xg_pred))
print(classification_report(y_test, xg.predict(X_test)))
joblib.dump((xg, X_train.columns.tolist()), "../models/model_xgb.pkl")

