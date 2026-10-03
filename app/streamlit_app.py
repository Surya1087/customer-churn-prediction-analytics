from pathlib import Path
import json
import sys
from urllib.request import urlretrieve
import pandas as pd
import streamlit as st
import plotly.express as px
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churn_analytics.data import load_data
from churn_analytics.predict import load_model, predict_risk
from churn_analytics.segmentation import segment
from churn_analytics.explainability import feature_importance

ROOT = Path(__file__).parents[1]
st.set_page_config(page_title="Customer Churn Analytics", layout="wide")
st.title("Customer Churn Prediction & Business Analytics")
metrics_path = ROOT / "models/metrics.json"; data_path = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
@st.cache_resource
def ensure_runtime_files():
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        urlretrieve("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv", data_path)
    if not metrics_path.exists() or not (ROOT / "models/final_model.joblib").exists():
        raise FileNotFoundError("Saved model artifacts are missing from the deployment repository.")

try:
    ensure_runtime_files()
except Exception as error:
    st.error("The application could not prepare its dataset and model artifacts.")
    st.caption(str(error))
    st.stop()
metrics = json.loads(metrics_path.read_text()); df = load_data(data_path)
page = st.sidebar.radio("Section", ["Executive Overview", "Exploratory Analytics", "Model Performance", "Prediction", "Segmentation", "Business Insights"])

def table_metrics():
    rows = []
    for name, m in metrics["metrics"].items():
        rows.append({"Model": name.replace("_", " ").title(), **{k: f"{m[k]:.2%}" for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}})
    return pd.DataFrame(rows)

if page == "Executive Overview":
    c1,c2,c3 = st.columns(3); c1.metric("Customers", f"{len(df):,}"); c2.metric("Observed churn", f"{df.Churn.mean():.2%}"); c3.metric("Selected model", metrics["selected_model"].replace("_", " ").title())
    st.info("Observed patterns and model importance are associations, not proof of causation.")
elif page == "Exploratory Analytics":
    st.subheader("Observed churn rates")
    feature = st.selectbox("Group by", ["Contract", "PaymentMethod", "tenure_group", "InternetService", "TechSupport"])
    summary = df.groupby(feature, observed=False).agg(customers=("Churn","size"), churn_rate=("Churn","mean")).reset_index()
    st.plotly_chart(px.bar(summary, x=feature, y="churn_rate", text_auto=".1%", title=f"Observed churn by {feature}"), use_container_width=True); st.dataframe(summary)
elif page == "Model Performance":
    st.subheader("Candidate model comparison")
    st.dataframe(table_metrics(), use_container_width=True)
    chart = pd.DataFrame([{"Model": k, "ROC-AUC": v["roc_auc"], "F1": v["f1"]} for k,v in metrics["metrics"].items()]).melt("Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(chart, x="Model", y="Score", color="Metric", barmode="group", text_auto=".2f"), use_container_width=True)
    st.write(f"**Selection rationale:** {metrics['selection_rule']} Final model: **{metrics['selected_model']}**. CV ROC-AUC and held-out metrics were calculated during training; no model is hardcoded as best.")
    X_explain = df.drop(columns=["Churn", "customerID", "tenure_group"])
    importance = feature_importance(load_model(ROOT / "models/final_model.joblib"), X_explain, df["Churn"]).head(15)
    st.subheader("Global predictive importance")
    st.caption("Permutation importance is used as a compatible, model-agnostic fallback. Importance indicates predictive association, not causation.")
    st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h"), use_container_width=True)
elif page == "Prediction":
    threshold = st.slider("High-risk threshold", .3, .8, .5, .05)
    customer = {"gender": st.selectbox("Gender", ["Male","Female"]), "SeniorCitizen": st.selectbox("Senior citizen", [0,1]), "Partner": st.selectbox("Partner", ["Yes","No"]), "Dependents": st.selectbox("Dependents", ["Yes","No"]), "tenure": st.number_input("Tenure months",0,100,12), "PhoneService":"Yes", "MultipleLines":"No", "InternetService":st.selectbox("Internet service",["DSL","Fiber optic","No"]), "OnlineSecurity":"No", "OnlineBackup":"No", "DeviceProtection":"No", "TechSupport":"No", "StreamingTV":"No", "StreamingMovies":"No", "Contract":st.selectbox("Contract",["Month-to-month","One year","Two year"]), "PaperlessBilling":"Yes", "PaymentMethod":st.selectbox("Payment method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]), "MonthlyCharges":st.number_input("Monthly charges",0.,300.,70.), "TotalCharges":st.number_input("Total charges",0.,10000.,840.)}
    if st.button("Predict"):
        try:
            result = predict_risk(load_model(ROOT / "models/final_model.joblib"), customer, threshold); st.metric("Churn probability", f"{result['probability']:.2%}"); st.success(f"Predicted class: {'Churn' if result['prediction'] else 'Retain'} | Risk: {result['risk']}")
        except Exception: st.error("Prediction failed. Check the model artifact and input schema.")
elif page == "Segmentation":
    labelled, summary, scores = segment(df); st.subheader("K-Means customer segmentation"); st.write(f"Selected k={max(scores, key=scores.get)} using silhouette score."); st.dataframe(summary); st.plotly_chart(px.scatter(labelled, x="tenure", y="MonthlyCharges", color="cluster", hover_data=["Churn"]), use_container_width=True)
else:
    st.subheader("Business insights")
    st.write(f"The dataset contains {len(df):,} customers and an observed churn rate of {df.Churn.mean():.2%}.")
    st.write("Customers in the displayed groups showed higher or lower observed churn rates; these are descriptive associations, not causal claims.")
    st.dataframe(df.groupby("Contract", observed=False)["Churn"].agg(["count","mean"]).rename(columns={"mean":"observed_churn_rate"}))
    st.write("Important predictive signals should be interpreted with the model evaluation and, where available, SHAP/permutation explanations; predictive importance does not establish business causation.")
