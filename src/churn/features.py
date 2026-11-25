"""Stage 5 - Feature engineering and reusable preprocessing.

`add_features` creates business features. `build_preprocessor` returns a
scikit-learn ColumnTransformer, so the exact same transformations are applied
at training time and at prediction time (no train/serve skew, no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds engineered features. Stateless, so safe inside a Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["num_addons"] = (X[ADDON_COLS] == "Yes").sum(axis=1)
        X["avg_monthly_spend"] = np.where(X["tenure"] > 0, X["TotalCharges"] / X["tenure"].clip(lower=1),
                                          X["MonthlyCharges"])
        X["charge_ratio"] = X["MonthlyCharges"] / (X["avg_monthly_spend"] + 1e-6)
        X["tenure_group"] = pd.cut(X["tenure"], bins=[-1, 12, 24, 48, 72],
                                   labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
        X["is_auto_pay"] = X["PaymentMethod"].str.contains("automatic").astype(int)
        X["no_support_bundle"] = ((X["TechSupport"] == "No") & (X["OnlineSecurity"] == "No")).astype(int)
        X["SeniorCitizen"] = X["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna(X["SeniorCitizen"].astype(str))
        return X


ENGINEERED_NUMERIC = ["num_addons", "avg_monthly_spend", "charge_ratio",
                      "is_auto_pay", "no_support_bundle"]
ENGINEERED_CATEGORICAL = ["tenure_group"]


def build_preprocessor() -> Pipeline:
    numeric = config.NUMERIC_COLS + ENGINEERED_NUMERIC
    binary = config.BINARY_COLS
    categorical = config.CATEGORICAL_COLS + ENGINEERED_CATEGORICAL
    transform = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("bin", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), binary),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("features", FeatureBuilder()), ("transform", transform)])


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[config.TARGET, config.ID_COL])
    y = (df[config.TARGET] == "Yes").astype(int)
    return X, y


def random_oversample(X: pd.DataFrame, y: pd.Series, seed: int = config.RANDOM_STATE):
    """Balance classes by resampling the minority class (training data only)."""
    rng = np.random.default_rng(seed)
    minority = y.value_counts().idxmin()
    idx_min = np.where(y.values == minority)[0]
    idx_maj = np.where(y.values != minority)[0]
    extra = rng.choice(idx_min, len(idx_maj) - len(idx_min), replace=True)
    idx = np.concatenate([idx_maj, idx_min, extra])
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def build_ml_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the preprocessor on the full clean data and return a numeric table (for export)."""
    X, y = split_xy(df)
    pre = build_preprocessor()
    arr = pre.fit_transform(X)
    names = pre.named_steps["transform"].get_feature_names_out()
    out = pd.DataFrame(arr, columns=[n.split("__", 1)[1] for n in names])
    out[config.TARGET] = y.values
    return out
