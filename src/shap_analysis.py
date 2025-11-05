# # src/shap_analysis.py

# import pandas as pd
# import xgboost as xgb
# import shap
# import json
# from sklearn.model_selection import train_test_split
# import os

# # ===========================
# # 1. READ DATA
# # ===========================
# df = pd.read_csv("data/ai4i2020.csv")  # ✅ use your actual CSV

# feature_cols = [
#     "Air temperature [K]",
#     "Process temperature [K]",
#     "Rotational speed [rpm]",
#     "Torque [Nm]",
#     "Tool wear [min]"
# ]
# target_col = "Machine failure"

# X = df[feature_cols]
# y = df[target_col]

# # ===========================
# # 2. TRAIN MODEL (XGBoost)
# # ===========================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = xgb.XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     eval_metric="logloss",
#     random_state=42
# )
# model.fit(X_train, y_train)

# # ===========================
# # 3. SHAP EXPLAINER
# # ===========================
# explainer = shap.Explainer(model, X_train)
# shap_values = explainer(X_test)

# # ===========================
# # 4. EXPORT JSON
# # ===========================
# probs = model.predict_proba(X_test)[:, 1]  # failure probability
# output = []

# for i in range(len(X_test)):
#     row = {
#         "row_index": int(X_test.index[i]),
#         "features": {col: float(X_test.iloc[i][col]) for col in feature_cols},
#         "predicted_failure_probability": float(probs[i]),
#         "shap_values": {
#             feature_cols[j]: float(shap_values.values[i][j])
#             for j in range(len(feature_cols))
#         },
#         "expected_value": float(shap_values.base_values[i])
#     }
#     output.append(row)

# mean_abs_shap = shap_values.abs.mean(0).values
# global_importance = [
#     {"feature": feature_cols[i], "mean_abs_shap": float(mean_abs_shap[i])}
#     for i in range(len(feature_cols))
# ]

# final_json = {
#     "metadata": {
#         "model": "XGBoost",
#         "target": target_col,
#         "n_test_samples": len(X_test)
#     },
#     "global_feature_importance": global_importance,
#     "rows": output
# }

# os.makedirs("shap_results", exist_ok=True)

# with open("shap_results/shap_results.json", "w") as f:
#     json.dump(final_json, f, indent=4)

# # ===========================
# # 5. EXPORT TO EXCEL
# # ===========================
# df_global = pd.DataFrame(global_importance)
# df_global.to_excel("shap_results/shap_summary.xlsx", index=False)

# print("✅ SHAP analysis generated:")
# print("   → shap_results/shap_results.json")
# print("   → shap_results/shap_summary.xlsx")
# import pandas as pd
# import xgboost as xgb
# import numpy as np
# import shap
# import json
# from sklearn.model_selection import train_test_split

# # 1. READ FILE
# df = pd.read_csv("../data/ai4i2020.csv")

# # ✅ FIX — clean column names
# df.columns = (
#     df.columns
#     .str.replace('[', '', regex=False)
#     .str.replace(']', '', regex=False)
#     .str.replace(' ', '_')
# )

# # 2. SELECT FEATURES AND TARGET
# feature_cols = [
#     "Air_temperature_K",
#     "Process_temperature_K",
#     "Rotational_speed_rpm",
#     "Torque_Nm",
#     "Tool_wear_min"
# ]
# target_col = "Machine_failure"

# X = df[feature_cols]
# y = df[target_col]

# # 3. TRAIN / TEST SPLIT
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 4. TRAIN MODEL (XGBoost)
# model = xgb.XGBClassifier(
#     eval_metric="logloss",
#     random_state=42,
# )
# model.fit(X_train, y_train)

# # 5. SHAP EXPLAINER
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)

# # 6. GLOBAL IMPORTANCE
# mean_abs_shap = np.abs(shap_values).mean(axis=0)
# global_importance = [
#     {"feature": feature_cols[i], "mean_abs_shap": float(mean_abs_shap[i])}
#     for i in range(len(feature_cols))
# ]

# # 7. SAVE RESULT TO JSON
# final_json = {
#     "metadata": {
#         "model": "XGBoost",
#         "target": target_col,
#         "n_test_samples": len(X_test)
#     },
#     "global_feature_importance": global_importance,
# }

# with open("shap_results.json", "w") as f:
#     json.dump(final_json, f, indent=4)

# print("✅ SHAP analysis saved to shap_results.json")


# src/shap_analysis.py

import pandas as pd
import xgboost as xgb
import shap
import json
import numpy as np
from sklearn.model_selection import train_test_split
from openpyxl import Workbook

# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_csv("../data/ai4i2020.csv")
df.columns = df.columns.str.replace(r"[\[\]<>\(\)]", "", regex=True)
# Columns to analyze
feature_cols = [
    "Air temperature K",
    "Process temperature K",
    "Rotational speed rpm",
    "Torque Nm",
    "Tool wear min",
]
target_col = "Machine failure"

X = df[feature_cols]
y = df[target_col]

# ------------------------------
# 2. Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 3. Train XGBoost Model
# ------------------------------
model = xgb.XGBClassifier(
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# ✅ Instead of TreeExplainer, use general SHAP Explainer
explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_test)

# Convert SHAP values into df
shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
shap_df["predicted_prob"] = model.predict_proba(X_test)[:, 1]

# Save Excel
output_file = "shap_results.xlsx"
shap_df.to_excel(output_file, index=False)

print(f"✅ SHAP analysis completed. Excel saved as: {output_file}")

# ------------------------------
# 5. Compute Global Feature Importance
# ------------------------------
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

global_importance = [
    {"feature": feature_cols[i], "mean_abs_shap": float(mean_abs_shap[i])}
    for i in range(len(feature_cols))
]

# ------------------------------
# 6. Store JSON outputs
# ------------------------------
probs = model.predict_proba(X_test)[:, 1]

output = []
for i in range(len(X_test)):
    row = {
        "row_index": int(X_test.index[i]),
        "features": {col: float(X_test.iloc[i][col]) for col in feature_cols},
        "predicted_failure_prob": float(probs[i]),
        "shap_values": {feature_cols[j]: float(shap_values.values[i][j]) for j in range(len(feature_cols))},
        "expected_value": float(shap_values.base_values[i])
    }
    output.append(row)

final_json = {
    "metadata": {
        "model": "XGBoost",
        "target": target_col,
        "n_test_samples": len(X_test)
    },
    "global_feature_importance": global_importance,
    "rows": output
}

with open("../shap_results.json", "w") as f:
    json.dump(final_json, f, indent=4)

print("✅ SHAP JSON saved as shap_results.json")

# ------------------------------
# 7. Export results to Excel
# ------------------------------
excel_path = "../shap_results.xlsx"
wb = Workbook()

# Sheet 1 — Global Feature Importance
ws1 = wb.active
ws1.title = "Global Importance"
ws1.append(["Feature", "Mean |SHAP| Importance"])

for item in global_importance:
    ws1.append([item["feature"], item["mean_abs_shap"]])

# Sheet 2 — Per-row SHAP values
ws2 = wb.create_sheet("Instance Level SHAP")
ws2.append(["Row Index", "Predicted Prob", "Expected Value"] + feature_cols)

for i, row in enumerate(output):
    ws2.append([
        row["row_index"],
        row["predicted_failure_prob"],
        row["expected_value"],
        *row["features"].values()
    ])

wb.save(excel_path)
print(f"✅ SHAP Excel saved as shap_results.xlsx at {excel_path}")

