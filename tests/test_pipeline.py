"""Unit and integration tests for the churn workflow. Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn import config  # noqa: E402
from churn.acquire import generate_synthetic  # noqa: E402
from churn.clean import clean  # noqa: E402
from churn.features import (FeatureBuilder, build_ml_ready, build_preprocessor,  # noqa: E402
                            random_oversample, split_xy)
from churn.validate import assert_valid, validate  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return generate_synthetic(n=1500, seed=7)


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean(raw)[0]


def test_generator_schema(raw):
    for col in [config.ID_COL, config.TARGET, *config.NUMERIC_COLS, *config.CATEGORICAL_COLS]:
        assert col in raw.columns


def test_generator_is_reproducible():
    a, b = generate_synthetic(n=200, seed=1), generate_synthetic(n=200, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_raw_data_fails_validation(raw):
    report = validate(raw)
    assert not report["passed"].all()
    with pytest.raises(ValueError):
        assert_valid(report)


def test_clean_removes_duplicates_and_nulls(cleaned):
    assert cleaned.duplicated().sum() == 0
    assert cleaned.isna().sum().sum() == 0


def test_clean_fixes_types_and_ranges(cleaned):
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["tenure"].min() >= 0
    assert cleaned["MonthlyCharges"].max() <= 120


def test_clean_normalises_text(cleaned):
    assert set(cleaned["gender"]) <= {"Male", "Female"}
    assert set(cleaned["Contract"]) <= config.ALLOWED_VALUES["Contract"]


def test_clean_data_passes_validation(cleaned):
    assert_valid(validate(cleaned))  # must not raise


def test_feature_builder_adds_columns(cleaned):
    X, _ = split_xy(cleaned)
    out = FeatureBuilder().transform(X)
    for col in ["num_addons", "avg_monthly_spend", "tenure_group", "is_auto_pay", "no_support_bundle"]:
        assert col in out.columns
    assert out["num_addons"].between(0, 6).all()


def test_preprocessor_output_is_numeric_and_finite(cleaned):
    X, _ = split_xy(cleaned)
    arr = build_preprocessor().fit_transform(X)
    assert arr.shape[0] == len(X)
    assert np.isfinite(arr).all()


def test_preprocessor_handles_unseen_category(cleaned):
    X, _ = split_xy(cleaned)
    pre = build_preprocessor().fit(X)
    new = X.head(3).copy()
    new["PaymentMethod"] = "UPI (automatic)"  # category never seen in training
    assert pre.transform(new).shape[0] == 3


def test_ml_ready_has_target_and_no_nulls(cleaned):
    ml = build_ml_ready(cleaned)
    assert config.TARGET in ml.columns
    assert ml.isna().sum().sum() == 0


def test_oversample_balances_classes(cleaned):
    X, y = split_xy(cleaned)
    Xb, yb = random_oversample(X, y)
    counts = yb.value_counts()
    assert counts.iloc[0] == counts.iloc[1]
    assert len(Xb) == len(yb)


def test_exported_model_predicts(cleaned):
    """Integration test: needs `python run_pipeline.py` to have been run once."""
    model_path = config.MODELS / "churn_model.joblib"
    if not model_path.exists():
        pytest.skip("run the pipeline first to export the model")
    from churn.train import predict
    proba = predict(cleaned.head(20))
    assert proba.between(0, 1).all()
