import joblib
import pandas as pd
import numpy as np

def load_model(path="models/churn_pipeline.joblib"):
    return joblib.load(path)

def predict_risk(model, customer: dict, threshold=.5):
    probabilities = np.asarray(model.predict_proba(pd.DataFrame([customer])))
    probability = float(probabilities[0, 1])
    risk = "High" if probability >= threshold else ("Medium" if probability >= threshold * .6 else "Low")
    return {"probability": probability, "prediction": int(probability >= threshold), "risk": risk}
