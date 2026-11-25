"""Run the complete churn workflow end to end.

    python run_pipeline.py            # try the public URL, fall back to synthetic data
    python run_pipeline.py --offline  # skip the download, use synthetic data
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn import config  # noqa: E402
from churn.acquire import acquire  # noqa: E402
from churn.clean import clean  # noqa: E402
from churn.features import build_ml_ready  # noqa: E402
from churn.train import feature_importance, run_experiments  # noqa: E402
from churn.validate import assert_valid, profile, validate  # noqa: E402
from churn.visualize import build_dashboard, derive_insights, eda_figures, model_figures  # noqa: E402


def main(offline: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger("pipeline")

    log.info("1/7 Acquiring data")
    raw, source = acquire(prefer_url=not offline)
    log.info("    %d rows from %s", len(raw), source)
    profile(raw).to_csv(config.REPORTS / "profile_raw.csv", index=False)
    validate(raw).to_csv(config.REPORTS / "validation_raw.csv", index=False)

    log.info("2/7 Cleaning")
    df, clean_log = clean(raw)
    clean_log.to_csv(config.REPORTS / "cleaning_log.csv", index=False)

    log.info("3/7 Validating")
    report = validate(df)
    report.to_csv(config.REPORTS / "validation_clean.csv", index=False)
    profile(df).to_csv(config.REPORTS / "profile_clean.csv", index=False)
    assert_valid(report)
    df.to_csv(config.DATA_PROCESSED / "telco_churn_clean.csv", index=False)
    log.info("    %d/%d rules passed", report["passed"].sum(), len(report))

    log.info("4/7 EDA and visualization")
    eda = eda_figures(df)

    log.info("5/7 Feature engineering -> ML-ready dataset")
    ml = build_ml_ready(df)
    ml.to_csv(config.DATA_PROCESSED / "telco_churn_ml_ready.csv", index=False)
    log.info("    ML-ready shape: %s", ml.shape)

    log.info("6/7 Model experiments")
    exp = run_experiments(df)
    exp["table"].to_csv(config.REPORTS / "experiment_results.csv", index=False)
    imp = feature_importance(exp["best"])
    imp.to_csv(config.REPORTS / "feature_importance.csv", index=False)
    log.info("    best model: %s (ROC-AUC %.3f)", exp["best_name"], exp["table"].iloc[0]["roc_auc"])

    log.info("7/7 Dashboard")
    mfigs = model_figures(exp, imp)
    insights = derive_insights(df, imp)
    (config.REPORTS / "insights.txt").write_text("\n".join(insights), encoding="utf-8")
    path = build_dashboard(df, eda, mfigs, exp, report, insights, source)
    log.info("    dashboard written to %s", path)
    return {"source": source, "rows_raw": len(raw), "rows_clean": len(df), "exp": exp}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip download, use synthetic data")
    main(offline=ap.parse_args().offline)
