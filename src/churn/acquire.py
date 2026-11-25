"""Stage 1 - Data acquisition.

Tries to download the public IBM Telco churn CSV over HTTP. If the request
fails (no internet, blocked URL, bad status), it falls back to a seeded
synthetic generator with the same schema, so the pipeline always runs.
The generator deliberately injects real-world data problems (missing values,
duplicates, inconsistent text, bad types, outliers) for the cleaning stage.
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def fetch_from_url(url: str = config.SOURCE_URL, timeout: int = 15) -> pd.DataFrame:
    """Download a CSV from a URL and return it as a DataFrame."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def generate_synthetic(n: int = 7043, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate telecom customer data with realistic churn behaviour."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    tenure_base = np.where(contract == "Month-to-month", 14, np.where(contract == "One year", 38, 56))
    tenure = np.clip(rng.normal(tenure_base, 15), 0, 72).round().astype(int)

    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])

    def addon(p_yes):
        vals = rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes])
        return np.where(internet == "No", "No internet service", vals)

    online_sec, online_bak = addon(0.29), addon(0.34)
    device, tech = addon(0.34), addon(0.29)
    tv, movies = addon(0.38), addon(0.39)

    multi = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n, p=[0.42, 0.58]))

    monthly = (
        20
        + np.where(internet == "Fiber optic", 50, np.where(internet == "DSL", 25, 0))
        + 5 * sum((a == "Yes") for a in [online_sec, online_bak, device, tech, tv, movies])
        + np.where(multi == "Yes", 5, 0)
        + rng.normal(0, 3, n)
    ).clip(18.25, 118.75).round(2)
    total = (monthly * np.maximum(tenure, 0) * rng.uniform(0.95, 1.05, n)).round(2)

    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])

    # Churn probability driven by known business factors
    logit = (
        -2.0
        + np.where(contract == "Month-to-month", 1.5, np.where(contract == "One year", -0.2, -1.6))
        - 0.035 * tenure
        + np.where(internet == "Fiber optic", 0.7, 0)
        + np.where(payment == "Electronic check", 0.55, 0)
        + np.where(tech == "No", 0.4, 0)
        + np.where(online_sec == "No", 0.35, 0)
        + 0.3 * senior
        + np.where(paperless == "Yes", 0.25, 0)
    )
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": senior,
        "Partner": rng.choice(["Yes", "No"], n, p=[0.48, 0.52]),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total.astype(object),
        "Churn": churn,
    })
    return _inject_dirt(df, rng)


def _inject_dirt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the kinds of problems seen in real exports."""
    df = df.copy()
    n = len(df)
    # TotalCharges stored as text with blanks (as in the real IBM file)
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[rng.choice(n, 11, replace=False), "TotalCharges"] = " "
    # Missing values
    df.loc[rng.choice(n, 60, replace=False), "MonthlyCharges"] = np.nan
    df.loc[rng.choice(n, 45, replace=False), "PaymentMethod"] = np.nan
    # Inconsistent text: casing and whitespace
    idx = rng.choice(n, 80, replace=False)
    df.loc[idx, "gender"] = df.loc[idx, "gender"].str.lower()
    idx = rng.choice(n, 70, replace=False)
    df.loc[idx, "Contract"] = " " + df.loc[idx, "Contract"] + "  "
    # Impossible values / outliers
    df.loc[rng.choice(n, 8, replace=False), "tenure"] = -1
    df.loc[rng.choice(n, 6, replace=False), "MonthlyCharges"] = 999.0
    # Exact duplicate rows
    dups = df.sample(25, random_state=1)
    return pd.concat([df, dups], ignore_index=True)


def acquire(prefer_url: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (raw_dataframe, source_label) and save a raw snapshot."""
    df, source = None, ""
    if prefer_url:
        try:
            df = fetch_from_url()
            source = f"url:{config.SOURCE_URL}"
            log.info("Downloaded %d rows from source URL", len(df))
        except Exception as exc:  # network errors, HTTP errors, parse errors
            log.warning("URL fetch failed (%s); using synthetic generator", exc)
    if df is None:
        df = generate_synthetic()
        source = "synthetic:seed=42"
    df.to_csv(config.DATA_RAW / "telco_churn_raw.csv", index=False)
    df.head(200).to_csv(config.DATA_SAMPLE / "telco_churn_sample.csv", index=False)
    return df, source
