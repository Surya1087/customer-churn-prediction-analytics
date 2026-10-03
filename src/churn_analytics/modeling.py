from pathlib import Path
import json
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from .preprocessing import make_preprocessor, DROP

def _xgb(seed):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        # XGBoost can be installed but unusable when macOS libomp is absent.
        print(f"XGBoost unavailable; continuing without it: {exc}")
        return None
    return XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1)

def train(df, artifact_dir="models", seed=42):
    X = df.drop(columns=[c for c in DROP if c in df]); y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    candidates = {
        "dummy": (DummyClassifier(strategy="most_frequent"), {}),
        # scikit-learn 1.8+ derives the default L2 penalty from l1_ratio=0;
        # leaving penalty unspecified avoids its deprecation warning.
        "logistic_regression": (LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed, l1_ratio=0), {"model__C": [.1, 1, 10], "model__solver": ["lbfgs"], "model__l1_ratio": [0]}),
        "decision_tree": (DecisionTreeClassifier(class_weight="balanced", random_state=seed), {"model__criterion": ["gini", "entropy"], "model__max_depth": [4, 8, None], "model__min_samples_split": [2, 10], "model__min_samples_leaf": [1, 3]}),
        "random_forest": (RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1), {"model__n_estimators": [150, 300], "model__max_depth": [8, None], "model__min_samples_split": [2, 10], "model__min_samples_leaf": [1, 3], "model__max_features": ["sqrt"]}),
    }
    xgb = _xgb(seed)
    if xgb is not None:
        candidates["xgboost"] = (xgb, {"model__n_estimators": [150, 300], "model__max_depth": [3, 6], "model__learning_rate": [.05, .1], "model__subsample": [.8], "model__colsample_bytree": [.8]})
    results, fitted = {}, {}
    for name, (estimator, grid) in candidates.items():
        pipe = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True) if grid else pipe
        search.fit(X_train, y_train); fitted[name] = search
        p = search.predict_proba(X_test)[:, 1]; pred = (p >= .5).astype(int)
        results[name] = {"accuracy": accuracy_score(y_test,pred), "precision": precision_score(y_test,pred,zero_division=0), "recall": recall_score(y_test,pred), "f1": f1_score(y_test,pred), "roc_auc": roc_auc_score(y_test,p), "confusion_matrix": confusion_matrix(y_test,pred).tolist(), "cv_roc_auc": getattr(search, "best_score_", None), "best_params": getattr(search, "best_params_", {})}
    eligible = [n for n in results if n != "dummy"]
    best = max(eligible, key=lambda n: (results[n]["roc_auc"], results[n]["f1"]))
    out = Path(artifact_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "final_model.joblib"); joblib.dump(fitted[best], out / "churn_pipeline.joblib")
    (out / "metrics.json").write_text(json.dumps({"seed": seed, "selected_model": best, "selection_rule": "Highest held-out ROC-AUC after CV tuning; F1-score breaks ties. Accuracy alone is not used.", "xgboost_available": xgb is not None, "metrics": results}, indent=2, default=str))
    return results
