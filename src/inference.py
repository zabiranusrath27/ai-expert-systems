import joblib
import pandas as pd
import os
from src.rules import rule_based_recommendation


def load_model(path="models/rf_model.pkl"):
    """
    Load model and feature names from .pkl stored during training
    """
    #  Resolve path relative to project root
    base_dir = os.path.dirname(os.path.dirname(__file__))  # goes one level up from /src
    model_path = os.path.join(base_dir, path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model file not found at: {model_path}")

    model, feature_names = joblib.load(model_path)
    return model, feature_names


def predict(input_dict: dict, model_path="models/rf_model.pkl"):
    """
    Predict machine failure using ML model + rule engine.
    Returns prediction, probability and recommendations.
    """
    model, feature_names = load_model(model_path)

    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df)

    df = df.reindex(columns=feature_names, fill_value=0)

    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    advice = rule_based_recommendation(input_dict)

    return {
        "prediction": int(pred),                  
        "failure_probability": round(float(prob), 4),
        "advice": advice
    }

