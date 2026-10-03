from pathlib import Path
import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {TARGET, ID_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    if df[TARGET].isna().any():
        raise ValueError("Target contains unexpected values")
    df["tenure_group"] = pd.cut(df["tenure"], [-1, 5, 11, 23, 47, float("inf")], labels=["0-5", "6-11", "12-23", "24-47", "48+"])
    return df

def validate_data(df: pd.DataFrame) -> dict:
    return {"rows": len(df), "columns": len(df.columns), "duplicates": int(df.duplicated().sum()), "missing": df.isna().sum().to_dict(), "churn_rate": float(df[TARGET].mean())}
