# test_inference_debug.py
import os
import joblib
import pandas as pd
from pprint import pprint

# Adjust path if needed
MODEL_PATH = "models/model_rf.pkl"   # must be relative to project root

def load_and_print_model_info(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model, feature_names = joblib.load(model_path)
    print("Loaded model:", type(model))
    print("Number of features model expects:", len(feature_names))
    print("First 80 feature names (or fewer):")
    pprint(feature_names[:80])
    return model, feature_names

def build_aligned_df(input_data, feature_names):
    # replicate inference renaming (if using the same mapping)
    rename_map = {
        "Air temperature [K]": "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Torque [Nm]": "Torque_Nm",
        "Tool wear [min]": "Tool_wear_min",
        "Rotational speed [rpm]": "Rotational_speed_rpm",
        "Type": "Type"
    }
    corrected = {rename_map.get(k, k): v for k, v in input_data.items()}
    print("\nCorrected input keys and values:")
    pprint(corrected)

    df = pd.DataFrame([corrected])
    print("\nDataFrame before dummies:")
    print(df)

    df = pd.get_dummies(df)
    print("\nDataFrame AFTER get_dummies (columns):")
    print(df.columns.tolist())
    print(df.dtypes.to_dict())

    # Build aligned_df
    aligned_df = pd.DataFrame(columns=feature_names)
    aligned_df.loc[0] = 0  # initialize zeros

    # Copy matching columns only
    for col in df.columns:
        if col in aligned_df.columns:
            aligned_df.loc[0, col] = df.loc[0, col]
        else:
            print(f"NOTE: input column '{col}' NOT in model features")

    # Ensure numeric dtype
    aligned_df = aligned_df.astype(float)

    print("\nAligned DF (first 60 columns) that will be sent to model (showing 0/values):")
    # show only first 60 columns to keep output readable
    cols_to_show = aligned_df.columns[:60].tolist()
    print(aligned_df[cols_to_show].iloc[0].to_dict())

    return aligned_df

def run_tests(model, feature_names):
    tests = {
        "Low risk": {
            "Air temperature [K]": 295.0,
            "Process temperature [K]": 300.0,
            "Rotational speed [rpm]": 900.0,
            "Torque [Nm]": 10.0,
            "Tool wear [min]": 20.0,
            "Type": "M",
        },
        "Medium risk": {
            "Air temperature [K]": 305.0,
            "Process temperature [K]": 330.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 30.0,
            "Tool wear [min]": 180.0,
            "Type": "L",
        },
        "High risk": {
            "Air temperature [K]": 320.0,
            "Process temperature [K]": 350.0,
            "Rotational speed [rpm]": 2500.0,
            "Torque [Nm]": 70.0,
            "Tool wear [min]": 250.0,
            "Type": "H",
        }
    }

    for name, inp in tests.items():
        print("\n" + "="*60)
        print("TEST:", name)
        aligned = build_aligned_df(inp, feature_names)
        probs = model.predict_proba(aligned)[0]
        pred = model.predict(aligned)[0]
        print(f"PREDICTION: {pred}  PROBS: {probs} (failure_prob={probs[1]:.4f})")
        print("="*60 + "\n")

def main():
    model, feature_names = load_and_print_model_info(MODEL_PATH)
    run_tests(model, feature_names)

if __name__ == "__main__":
    main()
