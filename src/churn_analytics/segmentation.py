import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

def segment(df, k_values=range(2, 7), seed=42):
    work = df[FEATURES].copy().fillna(df[FEATURES].median())
    scaled = StandardScaler().fit_transform(work)
    scores = {k: silhouette_score(scaled, KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(scaled)) for k in k_values}
    k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(scaled)
    out = df.copy(); out["cluster"] = labels
    summary = out.groupby("cluster").agg(customers=("cluster", "size"), churn_rate=("Churn", "mean"), avg_tenure=("tenure", "mean"), avg_monthly_charges=("MonthlyCharges", "mean")).reset_index()
    return out, summary, scores
