import joblib
import pandas as pd
import os


def load_model(path="models/model_rf.pkl"):
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
    model_path = os.path.join(base_dir, path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model file not found at: {model_path}")

    model, feature_names = joblib.load(model_path)
    return model, feature_names


def predict(input_data, model_path="models/model_rf.pkl"):

    # 🔥 Fix column name mismatch
    rename_map = {
        "Air temperature [K]": "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Torque [Nm]": "Torque_Nm",
        "Tool wear [min]": "Tool_wear_min",
        "Rotational speed [rpm]": "Rotational_speed_rpm",
        "Type": "Type"
    }

    corrected_input = {rename_map[k]: v for k, v in input_data.items() if k in rename_map}

    model, feature_names = joblib.load(model_path)

    df = pd.DataFrame([corrected_input])
    # df = pd.get_dummies(df)  # encode Type (L/M/H)
    failure_map = {
    "L": {"TWF": 1, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0},
    "M": {"TWF": 0, "HDF": 1, "PWF": 0, "OSF": 0, "RNF": 0},
    "H": {"TWF": 0, "HDF": 0, "PWF": 1, "OSF": 0, "RNF": 0},
    }

    failure_cols = failure_map[input_data["Type"]]
    df = pd.DataFrame([{**corrected_input, **failure_cols}])


    aligned_df = pd.DataFrame(columns=feature_names)
    aligned_df.loc[0] = 0  # initialize zeros

    for col in df.columns:
        if col in aligned_df.columns:
            aligned_df[col] = df[col].astype(float)

    failure_prob = float(model.predict_proba(aligned_df)[0][1])
    prediction = int(model.predict(aligned_df)[0])

    # --- Rule Engine ---
    if failure_prob < 0.30:
        advice = "✅ Machine within expected operating range."
        rule_fired = "No rule triggered"
    elif 0.30 <= failure_prob < 0.70:
        advice = "⚠️ Tool recalibration recommended soon."
        rule_fired = "KL-01 (Temperature + Tool Wear)"
    else:
        advice = "🚨 Inspect load alignment / lubrication / replace tool."
        rule_fired = "KL-03 / KL-04 (High torque + high temp)"

    return {
        "prediction": prediction,
        "failure_probability": failure_prob,
        "advice": advice,
        "rule_fired": rule_fired
    }

# import joblib
# import pandas as pd
# import os


# def load_model(path="models/model_rf.pkl"):
#     base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
#     model_path = os.path.join(base_dir, path)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f" Model file not found at: {model_path}")

#     model, feature_names = joblib.load(model_path)
#     return model, feature_names


# def predict(input_data, model_path="models/model_rf.pkl"):

#     model, feature_names = joblib.load(model_path)

#     # Convert single input to DataFrame
#     df = pd.DataFrame([input_data])

#     # Apply one-hot encoding (same as training)
#     df = pd.get_dummies(df)

#     # Create empty row with ALL feature names used during training
#     aligned_df = pd.DataFrame(columns=feature_names)
#     aligned_df.loc[0] = 0  # initialize all with zero

#     # Copy only matching columns
#     for col in df.columns:
#         if col in aligned_df.columns:
#             aligned_df[col] = df[col].astype(float)

#     # ---- MODEL PREDICTION ----
#     failure_prob = float(model.predict_proba(aligned_df)[0][1])
#     prediction = int(model.predict(aligned_df)[0])

#     # ---- RULE ENGINE ----
#     if failure_prob < 0.30:
#         advice = "✅ Machine within expected operating range."
#         rule_fired = "No rule triggered"
#     elif 0.30 <= failure_prob < 0.70:
#         advice = "⚠️ Tool recalibration recommended soon."
#         rule_fired = "KL-01 (Temperature + Tool Wear)"
#     else:
#         advice = "🚨 Inspect load alignment / lubrication / replace tool."
#         rule_fired = "KL-03 / KL-04 (High torque + high temp)"

#     return {
#         "prediction": prediction,
#         "failure_probability": failure_prob,
#         "advice": advice,
#         "rule_fired": rule_fired
#     }

# # import joblib
# # import pandas as pd
# # import os
# # from src.rules import rule_based_recommendation


# # def load_model(path="models/rf_model.pkl"):
# #     """
# #     Load model and feature names from .pkl stored during training
# #     """
# #     #  Resolve path relative to project root
# #     base_dir = os.path.dirname(os.path.dirname(__file__))  # goes one level up from /src
# #     model_path = os.path.join(base_dir, path)

# #     if not os.path.exists(model_path):
# #         raise FileNotFoundError(f" Model file not found at: {model_path}")

# #     model, feature_names = joblib.load(model_path)
# #     return model, feature_names


# # def predict(input_data, model_path):

# #     # Load trained model and feature list
# #     model, feature_names = joblib.load(model_path)

# #     # Convert input into DataFrame
# #     df = pd.DataFrame([input_data])

# #     # One-hot encode like training
# #     df = pd.get_dummies(df)

# #     # ✅ Add missing columns (model expects the same columns as during training)
# #     missing_cols = set(feature_names) - set(df.columns)
# #     for col in missing_cols:
# #         df[col] = 0

# #     # ✅ Ensure correct ordering of columns
# #     df = df[feature_names]

# #     # Model prediction
# #     failure_prob = model.predict_proba(df)[0][1]
# #     prediction = 1 if failure_prob > 0.5 else 0

# #     # ✅ Expert rule-based reasoning
# #     if failure_prob < 0.30:
# #         advice = "✅ Machine within expected operating range."
# #         rule = "No rule triggered"
# #     elif 0.30 <= failure_prob < 0.70:
# #         advice = "⚠️ Tool recalibration recommended in the next production window."
# #         rule = "KL-01 (tool wear + high temperature)"
# #     else:
# #         advice = "🚨 Inspect load alignment, lubrication, and replace tool immediately."
# #         rule = "KL-03 / KL-04 (high torque, high temperature)"

# #     return {
# #         "prediction": prediction,
# #         "failure_probability": failure_prob,
# #         "advice": advice,
# #         "rule_fired": rule
# #     }
# import joblib
# import pandas as pd
# import os


# def load_model(path="models/model_rf.pkl"):
#     base_dir = os.path.dirname(os.path.dirname(__file__))
#     model_path = os.path.join(base_dir, path)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found at: {model_path}")

#     return joblib.load(model_path)  # returns (model, feature_names)
# def predict(input_data, model_path="models/model_rf.pkl"):
#     model, feature_names = joblib.load(model_path)

#     # Convert input into DataFrame
#     df = pd.DataFrame([input_data])

#     # One-hot encode categorical values (ensures same format as training)
#     df = pd.get_dummies(df)

#     # Create empty frame with training feature names
#     aligned_df = pd.DataFrame(columns=feature_names)
#     aligned_df.loc[0] = 0  # initialize all values as 0

#     # Insert only matching columns
#     for col in df.columns:
#         if col in aligned_df.columns:
#             aligned_df[col] = df[col].astype(float)

#     # Final inference
#     failure_prob = model.predict_proba(aligned_df)[0][1]
#     prediction = 1 if failure_prob > 0.5 else 0

#     # Rule based logic
#     if failure_prob < 0.30:
#         advice = "✅ Machine within expected operating range."
#         rule = "No rule triggered"
#     elif 0.30 <= failure_prob < 0.70:
#         advice = "⚠️ Tool recalibration recommended soon."
#         rule = "KL-01 (Temp + Wear)"
#     else:
#         advice = "🚨 Inspect load alignment / lubrication / replace tool."
#         rule = "KL-03 / KL-04 (High torque + high temp)"

#     return {
#         "prediction": prediction,
#         "failure_probability": float(failure_prob),
#         "advice": advice,
#         "rule": rule,
#     }

# # def predict(input_data, model_path):
# #     model, feature_names = joblib.load(model_path)

# #     # Step 1: rename user keys to match training dataset column names
# #     rename_map = {
# #         "Air temperature [K]": "Air_temperature_K",
# #         "Process temperature [K]": "Process_temperature_K",
# #         "Torque [Nm]": "Torque_Nm",
# #         "Tool wear [min]": "Tool_wear_min",
# #         "Type": "Type",
# #     }

# #     input_data = {rename_map.get(k, k): v for k, v in input_data.items()}

# #     # Step 2: Convert to dataframe
# #     df = pd.DataFrame([input_data])

# #     # Step 3: One-hot encode Type (same as training)
# #     df = pd.get_dummies(df)
# #     df = df.astype(float)     # ensure numeric dtype

# #     # Step 4: Create a full feature row initialized to 0
# #     aligned_df = pd.DataFrame(columns=feature_names)
# #     aligned_df.loc[0] = 0

# #     # Step 5: Overwrite matching model features
# #     for col in df.columns:
# #         if col in aligned_df.columns:
# #             aligned_df.loc[0, col] = df[col].iloc[0]

# #     # ---- DEBUG ----
# #     print("\n--- ALIGNED DF SENT TO MODEL ---")
# #     print(aligned_df.head())
# #     print("--------------------------------\n")

# #     # Step 6: Model prediction
# #     failure_prob = model.predict_proba(aligned_df)[0][1]
# #     prediction = int(failure_prob > 0.50)

# #     # Rule-based expert advice layer
# #     if failure_prob < 0.30:
# #         advice = "✅ Machine within expected operating range."
# #     elif failure_prob < 0.70:
# #         advice = "⚠️ Tool recalibration recommended soon."
# #     else:
# #         advice = "🚨 Immediate inspection required."

# #     return {
# #         "prediction": prediction,
# #         "failure_probability": failure_prob,
# #         "advice": advice,
# #     }

# # def predict(input_data, model_path):

# #     rename_map = {
# #         "Air temperature [K]": "Air_temperature_K",
# #         "Process temperature [K]": "Process_temperature_K",
# #         "Torque [Nm]": "Torque_Nm",
# #         "Tool wear [min]": "Tool_wear_min",
# #         "Type": "Type",
# #     }

# #     input_data = {rename_map.get(k, k): v for k, v in input_data.items()}

# #     model, feature_names = joblib.load(model_path)

# #     df = pd.DataFrame([input_data])

# #     df["Type"] = df["Type"].astype(str)

# #     df = pd.get_dummies(df)

# #     # ✅ Ensure all dummy columns are int (not bool)
# #     df = df.astype(int)

# #     # ✅ Proper feature alignment
# #     aligned_df = pd.DataFrame(columns=feature_names)
# #     aligned_df.loc[0] = 0  # initialize row with zeros
# #     aligned_df.update(df)  # update matching columns
# #     aligned_df = aligned_df.astype(float)

# #     failure_prob = model.predict_proba(aligned_df)[0][1]
# #     prediction = int(failure_prob > 0.50)

# #     if failure_prob < 0.30:
# #         advice = "✅ Machine within expected operating range."
# #         rule = "No rule triggered"
# #     elif failure_prob < 0.70:
# #         advice = "⚠️ Tool recalibration recommended soon."
# #         rule = "XAI Rule KL-01"
# #     else:
# #         advice = "🚨 Immediate inspection required."
# #         rule = "XAI Rule KL-03"

# #     return {
# #         "prediction": prediction,
# #         "failure_probability": failure_prob,
# #         "advice": advice,
# #         "rule_fired": rule,
# #     }

# # def predict(input_data, model_path):
# #     # Load model + trained feature list
# #     model, feature_names = joblib.load(model_path)

# #     # Convert input to DataFrame
# #     df = pd.DataFrame([input_data])

# #     # ✅ One-hot encode
# #     df = pd.get_dummies(df)

# #     # ✅ Fast column alignment
# #     # Create empty row with ALL features, then overlay real values
# #     aligned_df = pd.DataFrame(columns=feature_names)
# #     aligned_df.loc[0] = 0  # initialize with zeros
# #     aligned_df.update(df)  # replace only matching columns

# #     # Prediction
# #     failure_prob = model.predict_proba(aligned_df)[0][1]
# #     prediction = int(failure_prob > 0.50)

# #     # ✅ Decision logic + expert recommendation
# #     if failure_prob < 0.30:
# #         advice = "✅ Machine within expected operating range."
# #         rule = "No rule triggered"
# #     elif failure_prob < 0.70:
# #         advice = "⚠️ Tool recalibration recommended in the next production window."
# #         rule = "KL-01 (tool wear + high temperature)"
# #     else:
# #         advice = "🚨 Inspect alignment / lubrication and replace tool immediately."
# #         rule = "KL-03 / KL-04 (overload / overheating)"

# #     return {
# #         "prediction": prediction,
# #         "failure_probability": failure_prob,
# #         "advice": advice,
# #         "rule_fired": rule,
# #     }


