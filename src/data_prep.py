import pandas as pd
from sklearn.model_selection import train_test_split
import re

def get_splits(path="data/ai4i2020.csv"):
    df = pd.read_csv(path)

    target = "Machine failure"

    # One-hot encode categorical columns
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)

    # ✅ FIX: Clean invalid characters from column names for XGBoost
    X.columns = (
        X.columns
        .str.replace(r"[\[\]<>]", "", regex=True)   # remove [ ] <
        .str.replace(r"\s+", "_", regex=True)       # replace spaces with _
        .str.strip()                                # remove trailing spaces
    )

    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)