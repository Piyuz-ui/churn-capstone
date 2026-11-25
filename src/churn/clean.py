"""Stage 2 - Cleaning.

Each step is a small function and records what it changed in a log,
so the report can show cleaning evidence (rows before/after, values fixed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _log(log: list, step: str, detail: str, affected: int):
    log.append({"step": step, "detail": detail, "rows_affected": int(affected)})


def strip_and_normalise_text(df, log):
    df = df.copy()
    changed = 0
    for col in df.select_dtypes(include=["object", "string"]).columns:
        before = df[col].copy()
        df[col] = df[col].str.strip()
        changed += int((before.fillna("") != df[col].fillna("")).sum())
    if "gender" in df:
        before = df["gender"].copy()
        df["gender"] = df["gender"].str.capitalize()
        changed += int((before.fillna("") != df["gender"].fillna("")).sum())
    _log(log, "normalise_text", "trimmed whitespace and fixed casing", changed)
    return df


def drop_duplicates(df, log):
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    _log(log, "drop_duplicates", "removed exact duplicate rows", n - len(df))
    return df


def fix_types(df, log):
    df = df.copy()
    blanks = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    _log(log, "fix_types", "converted TotalCharges text to numeric (blanks -> NaN)", blanks)
    return df


def fix_invalid_values(df, log):
    """Turn impossible values into NaN so they are imputed, not trusted."""
    df = df.copy()
    bad_tenure = df["tenure"] < 0
    df.loc[bad_tenure, "tenure"] = np.nan
    lo, hi = config.NUMERIC_RANGES["MonthlyCharges"]
    bad_monthly = (df["MonthlyCharges"] < lo) | (df["MonthlyCharges"] > hi)
    df.loc[bad_monthly, "MonthlyCharges"] = np.nan
    _log(log, "fix_invalid_values", "negative tenure / out-of-range MonthlyCharges set to NaN",
         int(bad_tenure.sum() + bad_monthly.sum()))
    return df


def impute(df, log):
    """Business-aware imputation."""
    df = df.copy()
    n_missing = int(df[config.NUMERIC_COLS + ["PaymentMethod"]].isna().sum().sum())
    # tenure: median within the same contract type
    df["tenure"] = df["tenure"].fillna(df.groupby("Contract")["tenure"].transform("median"))
    # MonthlyCharges: median within the same internet service type
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
        df.groupby("InternetService")["MonthlyCharges"].transform("median"))
    # TotalCharges blank means a brand-new customer -> tenure * monthly
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    df["PaymentMethod"] = df["PaymentMethod"].fillna(df["PaymentMethod"].mode()[0])
    df["tenure"] = df["tenure"].round().astype(int)
    _log(log, "impute", "group-median numeric imputation, mode for PaymentMethod", n_missing)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all cleaning steps. Returns (clean_df, cleaning_log_df)."""
    log: list[dict] = []
    _log(log, "input", "raw rows", len(df))
    for step in (strip_and_normalise_text, drop_duplicates, fix_types, fix_invalid_values, impute):
        df = step(df, log)
    _log(log, "output", "clean rows", len(df))
    return df, pd.DataFrame(log)
