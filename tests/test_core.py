import pandas as pd
from churn_analytics.data import load_data, validate_data
from churn_analytics.predict import predict_risk

def test_load_and_validate(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"customerID":["a"],"Churn":["Yes"],"tenure":[2],"TotalCharges":["10"]}).to_csv(p,index=False)
    assert validate_data(load_data(p))["churn_rate"] == 1.0

def test_prediction_shape():
    class M:
        def predict_proba(self, x): return [[.2, .8]]
    result = predict_risk(M(), {"x": 1})
    assert result["prediction"] == 1 and result["risk"] == "High"

def test_artifact_loading(tmp_path):
    import joblib
    from churn_analytics.predict import load_model
    path = tmp_path / "model.joblib"
    joblib.dump({"ok": True}, path)
    assert load_model(path)["ok"] is True
