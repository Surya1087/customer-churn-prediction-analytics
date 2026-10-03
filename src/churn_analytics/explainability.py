import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def feature_importance(model, X, y):
    """Return real permutation importance; avoids inventing explanations."""
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    return pd.DataFrame({"feature": X.columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)

def shap_values(model, X):
    try:
        import shap
        transformed = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        pre = transformed.named_steps["preprocessor"]; estimator = transformed.named_steps["model"]
        matrix = pre.transform(X)
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(matrix)
        return values[1] if isinstance(values, list) else values
    except Exception:
        return None
