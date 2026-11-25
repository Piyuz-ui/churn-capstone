"""Stage 3 - Validation and profiling.

A lightweight rule-based validator (similar in spirit to Great Expectations
or Pandera) so the project has no heavy dependencies. Every rule returns a
pass/fail row; the pipeline stops if any critical rule fails.
"""
from __future__ import annotations

import pandas as pd

from . import config


def validate(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    def rule(name, passed, failing=0, critical=True):
        results.append({"rule": name, "passed": bool(passed),
                        "failing_rows": int(failing), "critical": critical})

    expected = [config.ID_COL, *config.NUMERIC_COLS, *config.BINARY_COLS,
                *config.CATEGORICAL_COLS, config.TARGET]
    missing_cols = [c for c in expected if c not in df.columns]
    rule("all expected columns present", not missing_cols, len(missing_cols))

    nulls = int(df[expected].isna().sum().sum()) if not missing_cols else -1
    rule("no missing values", nulls == 0, max(nulls, 0))

    dup_rows = int(df.duplicated().sum())
    rule("no duplicate rows", dup_rows == 0, dup_rows)

    dup_ids = int(df[config.ID_COL].duplicated().sum())
    rule("customerID unique", dup_ids == 0, dup_ids, critical=False)

    for col, allowed in config.ALLOWED_VALUES.items():
        bad = int((~df[col].isin(allowed)).sum())
        rule(f"{col} in allowed set", bad == 0, bad)

    num = {}
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        rule(f"{col} is numeric type", pd.api.types.is_numeric_dtype(df[col]),
             0 if pd.api.types.is_numeric_dtype(df[col]) else len(df))
        num[col] = pd.to_numeric(df[col], errors="coerce")
        bad = int(((num[col] < lo) | (num[col] > hi)).sum())
        rule(f"{col} within [{lo}, {hi}]", bad == 0, bad)

    # Cross-column business rule: total charges should be roughly tenure x monthly
    expected_total = num["tenure"] * num["MonthlyCharges"]
    off = int(((num["TotalCharges"] - expected_total).abs() > 0.25 * expected_total + 50).sum())
    rule("TotalCharges consistent with tenure x MonthlyCharges", off < 0.05 * len(df), off, critical=False)

    return pd.DataFrame(results)


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level profile: type, missing %, unique count, basic stats."""
    rows = []
    for col in df.columns:
        s = df[col]
        row = {"column": col, "dtype": str(s.dtype),
               "missing_pct": round(s.isna().mean() * 100, 2), "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            row.update(mean=round(s.mean(), 2), std=round(s.std(), 2),
                       min=round(s.min(), 2), max=round(s.max(), 2))
        else:
            row.update(top=str(s.mode().iloc[0]) if s.notna().any() else "")
        rows.append(row)
    return pd.DataFrame(rows)


def assert_valid(report: pd.DataFrame):
    failed = report[(~report["passed"]) & (report["critical"])]
    if len(failed):
        raise ValueError(f"Critical validation failed:\n{failed.to_string(index=False)}")
