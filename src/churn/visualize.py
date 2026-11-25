"""Stage 4 & 7 - EDA charts and the HTML dashboard.

All charts are saved as PNGs in reports/figures/. The dashboard is a single
self-contained HTML file (images embedded) that opens in any browser.
"""
from __future__ import annotations

import base64
import html
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

from . import config

CHURN_COLORS = {"No": "#4C78A8", "Yes": "#E45756"}
sns.set_theme(style="whitegrid", font_scale=0.95)


def _save(fig, name):
    path = config.FIGURES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def churn_rate_by(df, col):
    return (df.groupby(col)[config.TARGET].apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False))


def eda_figures(df: pd.DataFrame) -> dict:
    figs = {}

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.TARGET].value_counts()
    ax.bar(counts.index, counts.values, color=[CHURN_COLORS[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.0%})", ha="center", va="bottom")
    ax.set_title("Target balance: churned vs retained")
    ax.set_ylabel("Customers")
    figs["target_balance"] = _save(fig, "01_target_balance")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, config.NUMERIC_COLS):
        sns.histplot(data=df, x=col, hue=config.TARGET, bins=30, palette=CHURN_COLORS,
                     element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(f"{col} by churn")
    figs["numeric_dist"] = _save(fig, "02_numeric_distributions")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, ["Contract", "InternetService", "PaymentMethod", "TechSupport"]):
        rates = churn_rate_by(df, col)
        ax.barh(rates.index, rates.values, color="#E45756")
        ax.axvline((df[config.TARGET] == "Yes").mean() * 100, ls="--", c="grey", label="overall")
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlabel("Churn rate (%)")
        ax.invert_yaxis()
        ax.legend(loc="lower right")
    figs["churn_by_category"] = _save(fig, "03_churn_rate_by_category")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Contract", y="MonthlyCharges", hue=config.TARGET, palette=CHURN_COLORS, ax=ax)
    ax.set_title("Monthly charges by contract and churn")
    figs["charges_box"] = _save(fig, "04_charges_by_contract")

    fig, ax = plt.subplots(figsize=(6, 5))
    num = df[config.NUMERIC_COLS].assign(Churn=(df[config.TARGET] == "Yes").astype(int),
                                         SeniorCitizen=df["SeniorCitizen"])
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric features)")
    figs["correlation"] = _save(fig, "05_correlation")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = pd.cut(df["tenure"], bins=range(0, 78, 6), include_lowest=True)
    rate = df.groupby(bins, observed=True)[config.TARGET].apply(lambda s: (s == "Yes").mean() * 100)
    ax.plot(range(len(rate)), rate.values, marker="o", color="#E45756")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([f"{int(i.left)+1}-{int(i.right)}" for i in rate.index], rotation=45)
    ax.set_title("Churn rate falls as tenure grows")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate (%)")
    figs["tenure_curve"] = _save(fig, "06_churn_vs_tenure")
    return figs


def model_figures(exp: dict, importance: pd.DataFrame) -> dict:
    figs = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    t = exp["table"].set_index("name")[["roc_auc", "f1", "recall", "precision"]]
    t.plot.bar(ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.set_title("Experiment comparison (test set)")
    ax.legend(loc="lower right", ncol=4, fontsize=8)
    figs["experiments"] = _save(fig, "07_experiment_comparison")

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(exp["y_test"], exp["proba"], ax=ax, name=exp["best_name"])
    ax.plot([0, 1], [0, 1], ls="--", c="grey")
    ax.set_title("ROC curve - best model")
    figs["roc"] = _save(fig, "08_roc_curve")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(exp["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix - best model")
    figs["cm"] = _save(fig, "09_confusion_matrix")

    fig, ax = plt.subplots(figsize=(7, 5))
    imp = importance.iloc[::-1]
    ax.barh(imp["feature"], imp["importance"], color="#4C78A8")
    ax.set_title("Top features driving churn")
    figs["importance"] = _save(fig, "10_feature_importance")

    fig, ax = plt.subplots(figsize=(6, 4))
    segs = pd.cut(exp["proba"], [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
    seg = pd.DataFrame({"risk": segs, "actual": exp["y_test"].values})
    summary = seg.groupby("risk", observed=False).agg(customers=("actual", "size"),
                                                      churn_rate=("actual", "mean"))
    ax.bar(summary.index.astype(str), summary["customers"], color=["#54A24B", "#EECA3B", "#E45756"])
    for i, (c, r) in enumerate(zip(summary["customers"], summary["churn_rate"])):
        ax.text(i, c, f"{c} customers\n{r:.0%} churned", ha="center", va="bottom", fontsize=8)
    ax.set_title("Predicted risk segments (test set)")
    ax.set_ylim(0, summary["customers"].max() * 1.3)
    figs["risk_segments"] = _save(fig, "11_risk_segments")
    return figs


def _img(path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}">'


def build_dashboard(df, eda: dict, mfigs: dict, exp: dict, validation: pd.DataFrame,
                    insights: list[str], source: str) -> str:
    churn_rate = (df[config.TARGET] == "Yes").mean()
    best = exp["table"].iloc[0]
    kpis = [
        ("Customers", f"{len(df):,}"),
        ("Churn rate", f"{churn_rate:.1%}"),
        ("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}"),
        ("Revenue at risk / month", f"${df.loc[df[config.TARGET]=='Yes','MonthlyCharges'].sum():,.0f}"),
        ("Best model", best["name"]),
        ("Test ROC-AUC", f"{best['roc_auc']:.3f}"),
        ("Validation rules passed", f"{int(validation['passed'].sum())}/{len(validation)}"),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="v">{html.escape(v)}</div>'
                       f'<div class="l">{html.escape(k)}</div></div>' for k, v in kpis)
    exp_cols = ["name", "cv_roc_auc", "roc_auc", "accuracy", "precision", "recall", "f1"]
    exp_html = exp["table"][exp_cols].to_html(index=False, classes="tbl", float_format="%.3f")
    ins_html = "".join(f"<li>{html.escape(i)}</li>" for i in insights)

    def section(title, keys, figs):
        return (f"<h2>{title}</h2><div class='grid'>" +
                "".join(f"<div class='card'>{_img(figs[k])}</div>" for k in keys) + "</div>")

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Churn Analytics Dashboard</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f5f6f8;color:#222}}
header{{background:#1f2a44;color:#fff;padding:20px 32px}} header p{{margin:4px 0 0;opacity:.8;font-size:14px}}
main{{padding:20px 32px;max-width:1300px;margin:auto}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.kpi{{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px #0001}}
.kpi .v{{font-size:22px;font-weight:600}} .kpi .l{{font-size:12px;color:#666;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:14px}}
.card{{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px #0001}}
.card img{{width:100%}} h2{{margin-top:28px;border-bottom:2px solid #1f2a44;padding-bottom:4px}}
.tbl{{border-collapse:collapse;background:#fff;width:100%}} .tbl td,.tbl th{{padding:6px 10px;border:1px solid #ddd;text-align:left}}
.tbl th{{background:#1f2a44;color:#fff}} li{{margin:6px 0}}
footer{{text-align:center;color:#888;font-size:12px;padding:20px}}
</style></head><body>
<header><h1>Customer Churn Analytics Dashboard</h1>
<p>AI-ready data-engineering &amp; visualization workflow · data source: {html.escape(source)} · generated {datetime.now():%d %b %Y %H:%M}</p></header>
<main><div class="kpis">{kpi_html}</div>
<h2>Key insights</h2><ul>{ins_html}</ul>
{section("Exploratory data analysis", ["target_balance","tenure_curve","churn_by_category","charges_box","numeric_dist","correlation"], eda)}
<h2>Experiment tracking</h2>{exp_html}
{section("Model performance", ["experiments","roc","cm","importance","risk_segments"], mfigs)}
</main><footer>Piyush Agrawal · BCA (AI &amp; DS) · KR Mangalam University · AML001 Capstone</footer></body></html>"""
    out = config.REPORTS / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return str(out)


def derive_insights(df: pd.DataFrame, importance: pd.DataFrame) -> list[str]:
    overall = (df[config.TARGET] == "Yes").mean() * 100
    c = churn_rate_by(df, "Contract")
    p = churn_rate_by(df, "PaymentMethod")
    i = churn_rate_by(df, "InternetService")
    t = churn_rate_by(df, "TechSupport")
    new = (df.loc[df["tenure"] <= 12, config.TARGET] == "Yes").mean() * 100
    old = (df.loc[df["tenure"] > 48, config.TARGET] == "Yes").mean() * 100
    return [
        f"Overall churn is {overall:.1f}%. Month-to-month customers churn at {c['Month-to-month']:.1f}% "
        f"versus {c['Two year']:.1f}% on two-year contracts - contract length is the strongest lever.",
        f"Customers in their first year churn at {new:.1f}%, compared with {old:.1f}% for customers "
        f"with more than 4 years of tenure. Onboarding in the first 12 months matters most.",
        f"Electronic-check payers churn at {p['Electronic check']:.1f}%, the highest of all payment methods; "
        f"nudging them to automatic payment is a low-cost retention action.",
        f"Fiber-optic customers churn at {i['Fiber optic']:.1f}% despite paying the most, which points to "
        f"price or service-quality issues on that product.",
        f"Customers without tech support churn at {t['No']:.1f}% versus {t['Yes']:.1f}% with it; "
        f"bundling support into at-risk plans could reduce churn.",
        f"The model's top drivers are {', '.join(importance['feature'].head(4))}, matching the EDA findings.",
    ]
