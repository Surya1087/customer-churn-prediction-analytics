import sqlite3
from pathlib import Path
import pandas as pd

def write_database(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out = out.rename(columns={"customerID":"customer_id", "SeniorCitizen":"senior_citizen", "Partner":"partner", "Dependents":"dependents", "tenure":"tenure_months", "PhoneService":"phone_service", "MultipleLines":"multiple_lines", "InternetService":"internet_service", "OnlineSecurity":"online_security", "OnlineBackup":"online_backup", "DeviceProtection":"device_protection", "TechSupport":"tech_support", "StreamingTV":"streaming_tv", "StreamingMovies":"streaming_movies", "Contract":"contract", "PaperlessBilling":"paperless_billing", "PaymentMethod":"payment_method", "MonthlyCharges":"monthly_charges", "TotalCharges":"total_charges", "Churn":"churn"})
    for c in ["partner", "dependents", "phone_service", "paperless_billing"]:
        if c in out: out[c] = out[c].map({"Yes":1,"No":0}).fillna(out[c])
    keep = [c for c in out.columns if c not in {"tenure_group"}]
    with sqlite3.connect(path) as con:
        out[keep].to_sql("telco_customers", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_churn ON telco_customers(churn)")

def query(path: str | Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(path) as con:
        return pd.read_sql_query(sql, con)
