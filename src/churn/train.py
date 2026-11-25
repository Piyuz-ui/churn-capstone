"""Stage 6 - Model experiments and export.

Trains several models on the same reusable preprocessing pipeline, tracks each
run, picks the best by ROC-AUC on the held-out test set, and exports it.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor, random_oversample, split_xy
from .tracking import Run


def candidate_models():
    rs = config.RANDOM_STATE
    return {
        "logreg_balanced": (LogisticRegression(max_iter=2000, class_weight="balanced"), False),
        "logreg_oversampled": (LogisticRegression(max_iter=2000), True),
        "random_forest": (RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                 class_weight="balanced", random_state=rs, n_jobs=-1), False),
        "gradient_boosting": (GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                         learning_rate=0.05, random_state=rs), True),
    }


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"roc_auc": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred)}


def run_experiments(df: pd.DataFrame):
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE,
                                              stratify=y, random_state=config.RANDOM_STATE)
    results, fitted = [], {}
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    for name, (est, oversample) in candidate_models().items():
        run = Run(name)
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        Xf, yf = random_oversample(X_tr, y_tr) if oversample else (X_tr, y_tr)
        cv_auc = cross_val_score(Pipeline([("prep", build_preprocessor()), ("model", est)]),
                                 X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(Xf, yf)
        metrics = evaluate(pipe, X_te, y_te)
        run.log_params(model=type(est).__name__, oversample=oversample,
                       train_rows=len(Xf), test_rows=len(X_te))
        run.log_metrics(cv_roc_auc=cv_auc, **metrics)
        results.append(run.finish())
        fitted[name] = pipe

    table = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best_name = table.iloc[0]["name"]
    best = fitted[best_name]
    joblib.dump(best, config.MODELS / "churn_model.joblib")
    joblib.dump(best.named_steps["prep"], config.MODELS / "preprocessor.joblib")

    proba = best.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, (proba >= 0.5).astype(int))
    return {"table": table, "best_name": best_name, "best": best,
            "X_test": X_te, "y_test": y_te, "proba": proba, "cm": cm}


def feature_importance(pipe: Pipeline, top: int = 15) -> pd.DataFrame:
    names = pipe.named_steps["prep"].named_steps["transform"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        vals = abs(model.coef_[0])
    return (pd.DataFrame({"feature": names, "importance": vals})
            .sort_values("importance", ascending=False).head(top).reset_index(drop=True))


def predict(df_new: pd.DataFrame, model_path=config.MODELS / "churn_model.joblib") -> pd.Series:
    """Score new customers with the exported model (same preprocessing applied)."""
    model = joblib.load(model_path)
    X = df_new.drop(columns=[c for c in (config.TARGET, config.ID_COL) if c in df_new])
    return pd.Series(model.predict_proba(X)[:, 1], index=df_new.index, name="churn_probability")
