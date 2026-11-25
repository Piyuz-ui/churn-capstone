"""Builds docs/Capstone_Report.pdf from the pipeline outputs in reports/.

Run after `python run_pipeline.py` and `pytest`:   python docs/build_report.py
"""
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
R, FIG = ROOT / "reports", ROOT / "reports" / "figures"
OUT = ROOT / "docs" / "Capstone_Report.pdf"

ss = getSampleStyleSheet()
NAVY = colors.HexColor("#1f2a44")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, textColor=NAVY, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, textColor=NAVY, spaceBefore=8, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=11, leading=15.5, spaceAfter=6)
CAP = ParagraphStyle("CAP", parent=B, fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=8.3, leading=10.5, backColor=colors.HexColor("#f4f4f4"),
                      borderPadding=5, leftIndent=4, rightIndent=4, spaceAfter=8)
CELL = ParagraphStyle("CELL", parent=B, fontSize=9, leading=11.5, spaceAfter=0)


def p(t):
    return Paragraph(t, B)


def bullets(items):
    return [Paragraph(f"&bull;&nbsp; {i}", ParagraphStyle("bl", parent=B, leftIndent=12, firstLineIndent=-8))
            for i in items]


def table(df_or_rows, widths=None, header=True):
    rows = df_or_rows
    if isinstance(df_or_rows, pd.DataFrame):
        rows = [list(df_or_rows.columns)] + df_or_rows.astype(str).values.tolist()
    HD = ParagraphStyle("hd", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
    rows = [[Paragraph(str(c), HD if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def fig(name, caption, width=15.5):
    path = FIG / name
    from PIL import Image as PI
    w, h = PI.open(path).size
    img = Image(str(path), width=width * cm, height=width * cm * h / w)
    return KeepTogether([img, Paragraph(caption, CAP), Spacer(1, 8)])


def code(path, start_marker, n_lines):
    lines = (ROOT / path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if start_marker in l)
    return Preformatted("\n".join(lines[i:i + n_lines]), CODE)


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "AML001 Capstone · Customer Churn Workflow · Piyush Agrawal (2601060053)")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}")
    c.restoreState()


def build():
    exp = pd.read_csv(R / "experiment_results.csv")
    clog = pd.read_csv(R / "cleaning_log.csv")
    vraw = pd.read_csv(R / "validation_raw.csv")
    vclean = pd.read_csv(R / "validation_clean.csv")
    imp = pd.read_csv(R / "feature_importance.csv")
    insights = (R / "insights.txt").read_text().splitlines()
    tests = [l for l in (R / "test_results.txt").read_text().splitlines() if "::" in l]
    clean = pd.read_csv(ROOT / "data/processed/telco_churn_clean.csv")
    ml = pd.read_csv(ROOT / "data/processed/telco_churn_ml_ready.csv", nrows=5)
    raw_rows = int(clog.loc[clog.step == "input", "rows_affected"].iloc[0])
    best = exp.iloc[0]
    churn_rate = (clean["Churn"] == "Yes").mean()

    s = []
    # ---------- Cover ----------
    s += [Spacer(1, 3.5 * cm),
          Paragraph("Customer Churn Prediction:<br/>An AI-Ready Data-Engineering &amp; Visualization Workflow",
                    ParagraphStyle("T", parent=H1, fontSize=22, leading=28, alignment=TA_CENTER)),
          Spacer(1, 0.6 * cm),
          Paragraph("Capstone Project Report", ParagraphStyle("st", parent=B, fontSize=14, alignment=TA_CENTER)),
          Spacer(1, 2.5 * cm),
          table([["Student", "Piyush Agrawal"], ["Roll number", "2601060053"], ["Programme", "BCA (AI & Data Science), Section C"],
                 ["Course", "AML001 Data Handling, Visualization & Feature Engineering"],
                 ["Instructor", "Aarti Sharma"], ["Coordinator", "Aman Kumar"],
                 ["University", "KR Mangalam University"], ["Batch", "2026-2030"],
                 ["GitHub repository", "https://github.com/Piyuz-ui/churn-capstone"],
                 ["Date", date.today().strftime("%d %B %Y")]], widths=[4.5 * cm, 11 * cm], header=False),
          PageBreak()]

    # ---------- Contents ----------
    s += [Paragraph("Contents", H1)]
    s += bullets(["1. Problem statement", "2. Objectives", "3. Data", "4. System design",
                  "5. Implementation", "6. Exploratory analysis and dashboard", "7. Feature engineering",
                  "8. Experiments and model selection", "9. Testing", "10. Results and business insights",
                  "11. Conclusions, limitations and future work", "12. How to run", "References"])
    s += [PageBreak()]

    # ---------- 1 Problem ----------
    s += [Paragraph("1. Problem statement", H1),
          p("Telecom companies lose a share of their customers every month (<i>churn</i>). Winning a new customer "
            "usually costs several times more than keeping an existing one. A company that can spot at-risk "
            "customers early can act before they leave, for example with a contract offer, a support bundle "
            "or a switch to automatic payment."),
          p("Customer data in real companies is rarely ready for machine learning. It arrives from different "
            "systems with missing values, wrong data types, inconsistent text and duplicates. This project builds "
            "a complete, reusable workflow that turns raw customer records into three outputs: a validated, "
            "ML-ready dataset, a dashboard that explains <i>why</i> customers leave, and a tracked, exported model "
            "that scores new customers by churn risk.")]

    # ---------- 2 Objectives ----------
    s += [Paragraph("2. Objectives", H1)]
    s += bullets([
        "<b>Acquire</b> data programmatically from an external source (HTTP), with a reliable fallback.",
        "<b>Clean</b> the data with documented, logged steps (types, duplicates, text, invalid values, missing values).",
        "<b>Validate</b> the data with explicit rules and stop the pipeline if critical rules fail.",
        "<b>Explore and visualize</b> the data to find the main churn drivers.",
        "<b>Engineer features</b> inside a reusable preprocessing pipeline that is shared by training and prediction.",
        "<b>Track experiments</b> across several models and imbalance strategies, and export the best model.",
        "<b>Present</b> KPIs, charts and insights in a single dashboard.",
    ])

    # ---------- 3 Data ----------
    s += [Paragraph("3. Data", H1),
          p("The target source is the public <b>IBM Telco Customer Churn</b> dataset: 7,043 customers and 21 columns "
            "covering demographics, subscribed services, account details and a <i>Churn</i> label (Yes/No). "
            "<font face='Courier'>acquire.py</font> downloads it over HTTP with the <font face='Courier'>requests</font> library."),
          p("If the download fails (no internet or a blocked URL), the module falls back to a seeded generator "
            "with the same schema. The churn probability in the generator depends on contract type, tenure, "
            "internet service, payment method, tech support, online security, age group and paperless billing. "
            "The generator also <b>injects real-world defects</b> so the cleaning stage has real work to do: "
            "TotalCharges stored as text with blanks, missing values, inconsistent casing and whitespace, "
            "negative tenure, extreme charge outliers, and 25 duplicate rows."),
          p(f"<b>Note:</b> the results in this report come from the seeded fallback data "
            f"({raw_rows:,} raw rows, {len(clean):,} after cleaning). The pipeline runs unchanged on the real IBM file."),
          table([["Group", "Columns"],
                 ["Identifier", "customerID"],
                 ["Demographics", "gender, SeniorCitizen, Partner, Dependents"],
                 ["Services", "PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
                              "DeviceProtection, TechSupport, StreamingTV, StreamingMovies"],
                 ["Account", "tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges"],
                 ["Target", "Churn (Yes / No)"]], widths=[3.5 * cm, 12 * cm]),
          PageBreak()]

    # ---------- 4 Design ----------
    s += [Paragraph("4. System design", H1),
          p("The workflow is split into seven stages. Each stage is a separate Python module with small, testable "
            "functions. <font face='Courier'>run_pipeline.py</font> runs all of them with one command, and "
            "<font face='Courier'>config.py</font> holds every path, column list and validation rule in one place."),
          Preformatted(
              " [1] ACQUIRE      requests.get(IBM CSV)  --fails-->  seeded synthetic generator\n"
              "        |\n"
              " [2] CLEAN        normalise text > drop duplicates > fix types > invalid->NaN > impute\n"
              "        |\n"
              " [3] VALIDATE     28 rules (schema, nulls, allowed values, ranges, cross-column)\n"
              "        |         critical failure  -->  pipeline stops\n"
              " [4] EDA          6 charts  -> reports/figures/\n"
              "        |\n"
              " [5] FEATURES     FeatureBuilder -> ColumnTransformer (scale / encode / one-hot)\n"
              "        |         -> data/processed/telco_churn_ml_ready.csv\n"
              " [6] EXPERIMENTS  4 models x imbalance strategy, 5-fold CV, tracked runs\n"
              "        |         best pipeline -> models/churn_model.joblib\n"
              " [7] DASHBOARD    KPIs + charts + experiment table + insights -> reports/dashboard.html",
              CODE),
          Paragraph("Design principles", H2)]
    s += bullets([
        "<b>One reusable pipeline.</b> Feature creation, scaling and encoding sit in one scikit-learn "
        "<font face='Courier'>Pipeline</font> saved with the model, so prediction applies the same steps as training.",
        "<b>No data leakage.</b> The train/test split happens first. Scalers, encoders and oversampling are fitted on training data only.",
        "<b>Fail fast.</b> The validation gate stops bad data before it reaches EDA or modelling.",
        "<b>Evidence by default.</b> Every stage writes a CSV or PNG to <font face='Courier'>reports/</font>, so results can be checked without rerunning anything.",
        "<b>Reproducible.</b> A fixed random seed (42) is used for generation, splits and models.",
    ])
    s += [table([["Stage", "Module", "Main outputs"],
                 ["Acquisition", "acquire.py", "data/raw/, data/sample/"],
                 ["Cleaning", "clean.py", "cleaning_log.csv"],
                 ["Validation / profiling", "validate.py", "validation_raw.csv, validation_clean.csv, profile_*.csv"],
                 ["EDA / dashboard", "visualize.py", "figures/*.png, dashboard.html, insights.txt"],
                 ["Feature engineering", "features.py", "telco_churn_ml_ready.csv"],
                 ["Experiments", "train.py, tracking.py", "experiments/runs.csv, models/*.joblib"]],
                widths=[4 * cm, 4 * cm, 7.5 * cm]),
          PageBreak()]

    # ---------- 5 Implementation ----------
    s += [Paragraph("5. Implementation", H1),
          Paragraph("5.1 Acquisition", H2),
          p("The download is wrapped in try/except. Any network, HTTP or parsing error is logged and the generator "
            "is used instead, so the pipeline never breaks because of a network problem."),
          code("src/churn/acquire.py", "def acquire(", 16),
          Paragraph("5.2 Cleaning", H2),
          p("Cleaning is a chain of small functions. Each one records how many rows it changed. Imputation is "
            "business-aware: tenure uses the median of the same contract type, MonthlyCharges uses the median of the same "
            "internet service, and a blank TotalCharges is rebuilt as tenure &times; MonthlyCharges."),
          table(clog.rename(columns={"rows_affected": "rows"}), widths=[3.8 * cm, 9.7 * cm, 2 * cm]),
          Spacer(1, 6),
          code("src/churn/clean.py", "def impute(", 14),
          Paragraph("5.3 Validation", H2),
          p(f"<font face='Courier'>validate.py</font> checks 28 rules covering the schema, missing values, duplicates, allowed "
            f"category values, numeric types and ranges, and a cross-column rule (TotalCharges &asymp; tenure &times; "
            f"MonthlyCharges). On the <b>raw</b> data, {int((~vraw.passed).sum())} rules fail. After cleaning, "
            f"<b>{int(vclean.passed.sum())}/{len(vclean)}</b> pass. The only remaining flag is non-critical: "
            f"{int(vclean.iloc[-1].failing_rows)} rows where the rebuilt TotalCharges differs from tenure &times; MonthlyCharges."),
          table(vraw[~vraw.passed][["rule", "failing_rows", "critical"]].rename(
              columns={"rule": "Rule failing on RAW data", "failing_rows": "rows"}), widths=[10 * cm, 2.5 * cm, 3 * cm]),
          PageBreak()]

    # ---------- 6 EDA ----------
    s += [Paragraph("6. Exploratory analysis and dashboard", H1),
          p(f"After cleaning, {churn_rate:.1%} of customers churned. The classes are imbalanced, which is "
            f"handled in Section 8."),
          fig("01_target_balance.png", "Figure 1 - Target balance", 8.5),
          fig("06_churn_vs_tenure.png", "Figure 2 - Churn rate by tenure band: new customers are the most at risk", 12),
          fig("03_churn_rate_by_category.png", "Figure 3 - Churn rate by contract, internet service, payment method and tech support"),
          fig("04_charges_by_contract.png", "Figure 4 - Monthly charges by contract and churn", 11),
          fig("02_numeric_distributions.png", "Figure 5 - Numeric distributions by churn (density)"),
          fig("05_correlation.png", "Figure 6 - Correlation between numeric features and churn", 10),
          Paragraph("Dashboard", H2),
          p("<font face='Courier'>reports/dashboard.html</font> is a single self-contained web page that opens in any "
            "browser. It shows KPI cards (customers, churn rate, average charge, monthly revenue at risk, best model, "
            "ROC-AUC, validation status), the key insights, all EDA charts, the experiment-tracking table and the model charts."),
          table([["KPI", "Value"],
                 ["Customers", f"{len(clean):,}"], ["Churn rate", f"{churn_rate:.1%}"],
                 ["Average monthly charge", f"${clean.MonthlyCharges.mean():.2f}"],
                 ["Monthly revenue from churned customers", f"${clean.loc[clean.Churn == 'Yes', 'MonthlyCharges'].sum():,.0f}"],
                 ["Best model / test ROC-AUC", f"{best['name']} / {best.roc_auc:.3f}"],
                 ["Validation rules passed", f"{int(vclean.passed.sum())}/{len(vclean)}"]], widths=[8 * cm, 7.5 * cm]),
          p("<i>[Add a screenshot of reports/dashboard.html opened in your browser here before submitting.]</i>"),
          PageBreak()]

    # ---------- 7 Features ----------
    s += [Paragraph("7. Feature engineering", H1),
          p("<font face='Courier'>FeatureBuilder</font> is a custom scikit-learn transformer that adds six business features. "
            "A <font face='Courier'>ColumnTransformer</font> then scales the numeric columns, ordinal-encodes the binary "
            "columns and one-hot encodes the categorical columns. <font face='Courier'>handle_unknown='ignore'</font> "
            "lets the model handle categories it has never seen, such as a new payment method."),
          table([["Feature", "Definition", "Why it helps"],
                 ["num_addons", "count of the 6 add-on services = Yes", "customers with more services are more locked in"],
                 ["avg_monthly_spend", "TotalCharges / tenure", "actual historic spend per month"],
                 ["charge_ratio", "MonthlyCharges / avg_monthly_spend", "detects recent price increases"],
                 ["tenure_group", "0-12, 13-24, 25-48, 49-72 months", "captures the non-linear tenure effect"],
                 ["is_auto_pay", "payment method is automatic", "automatic payers churn less"],
                 ["no_support_bundle", "no TechSupport and no OnlineSecurity", "flags unsupported customers"]],
                widths=[3.5 * cm, 6 * cm, 6 * cm]),
          Spacer(1, 6),
          code("src/churn/features.py", "def build_preprocessor", 13),
          p(f"The exported ML-ready table <font face='Courier'>telco_churn_ml_ready.csv</font> has "
            f"{len(clean):,} rows and {ml.shape[1] - 1} numeric feature columns plus the target, with no missing values."),
          Paragraph("Imbalanced data", H2),
          p("Two strategies were compared: <b>class weighting</b> (<font face='Courier'>class_weight='balanced'</font>) and "
            "<b>random oversampling</b> of the minority class, applied only to the training split. SMOTE (imbalanced-learn) "
            "is a possible extension."),
          PageBreak()]

    # ---------- 8 Experiments ----------
    s += [Paragraph("8. Experiments and model selection", H1),
          p("Each model was trained inside the same preprocessing pipeline on an 80/20 stratified split, with 5-fold "
            "stratified cross-validation on the training data. <font face='Courier'>tracking.py</font> records the "
            "parameters, metrics and duration of every run in <font face='Courier'>experiments/&lt;run_id&gt;/run.json</font> "
            "and <font face='Courier'>experiments/runs.csv</font>. It has an MLflow-style API and also logs to MLflow when MLflow is installed."),
          table(exp[["name", "cv_roc_auc", "roc_auc", "accuracy", "precision", "recall", "f1"]].assign(**{c: exp[c].map("{:.3f}".format) for c in ["cv_roc_auc", "roc_auc", "accuracy", "precision", "recall", "f1"]}),
                widths=[4 * cm] + [1.9 * cm] * 6),
          Spacer(1, 6),
          p(f"All four models score within about 0.01 ROC-AUC of each other. <b>{best['name']}</b> has the best "
            f"test ROC-AUC ({best.roc_auc:.3f}) and is also the easiest to explain, so it was exported as "
            f"<font face='Courier'>models/churn_model.joblib</font> (full pipeline) along with "
            f"<font face='Courier'>models/preprocessor.joblib</font>. Recall of {best.recall:.1%} means the model "
            f"catches most customers who will actually leave, which matters more than precision for retention campaigns."),
          fig("07_experiment_comparison.png", "Figure 7 - Experiment comparison on the test set", 13),
          fig("08_roc_curve.png", "Figure 8 - ROC curve of the selected model", 9),
          fig("09_confusion_matrix.png", "Figure 9 - Confusion matrix of the selected model", 8),
          fig("10_feature_importance.png", "Figure 10 - Top 15 features (absolute coefficient)", 13),
          PageBreak()]

    # ---------- 9 Testing ----------
    s += [Paragraph("9. Testing", H1),
          p("The project has 13 automated tests written with <b>pytest</b>. They cover the generator, cleaning, "
            "validation, feature engineering, the preprocessor, oversampling and the exported model. All 13 pass."),
          table([["Test", "Result"]] + [[t.split("::")[1].split()[0], "PASSED" if "PASSED" in t else "FAILED"] for t in tests],
                widths=[12 * cm, 3.5 * cm]),
          Spacer(1, 6),
          p("Two of these tests matter most: <i>test_raw_data_fails_validation</i> shows the validator catches bad "
            "data, and <i>test_preprocessor_handles_unseen_category</i> shows the deployed pipeline does not crash on a new category.")]

    # ---------- 10 Results ----------
    s += [Paragraph("10. Results and business insights", H1)]
    s += bullets(insights)
    s += [fig("11_risk_segments.png", "Figure 11 - Customers grouped by predicted risk, with the actual churn rate of each group", 11),
          Paragraph("Recommended actions", H2)]
    s += bullets([
        "Offer month-to-month customers in the <b>High</b> risk group a discounted one-year contract.",
        "Run a structured onboarding programme for the first 12 months.",
        "Give a small incentive to move electronic-check payers to automatic payment.",
        "Bundle tech support and online security into fiber-optic plans for at-risk customers.",
        "Re-score all customers monthly with the exported model and track churn in each risk group.",
    ])

    # ---------- 11 Conclusion ----------
    s += [Paragraph("11. Conclusions, limitations and future work", H1),
          p(f"The project delivers a complete, tested and reproducible workflow from raw data to decisions. "
            f"Data with nine kinds of validation failures was cleaned until all critical rules passed, turned into an "
            f"ML-ready dataset through a reusable pipeline, explored in a dashboard, and modelled with four tracked "
            f"experiments. The selected model reaches a test ROC-AUC of {best.roc_auc:.3f}. Its top drivers agree with "
            f"the EDA: contract type, tenure, payment method and service add-ons."),
          Paragraph("Limitations", H2)]
    s += bullets(["The reported numbers come from seeded synthetic data with the IBM schema. Results on the real IBM "
                  "file will differ, and the pipeline should be re-run with internet access.",
                  "The dashboard is static HTML. It does not support filtering yet.",
                  "Probability threshold is fixed at 0.5; it is not yet tuned to retention-campaign cost."])
    s += [Paragraph("Future work", H2)]
    s += bullets(["Interactive dashboard with Streamlit or Plotly Dash, with filters by segment.",
                  "SMOTE and hyper-parameter search (GridSearchCV / Optuna), tracked with MLflow.",
                  "SHAP explanations for individual customer predictions.",
                  "Schedule the pipeline (cron / Airflow) and add data-drift monitoring."])

    # ---------- 12 Run ----------
    s += [Paragraph("12. How to run", H1),
          Preformatted("git clone https://github.com/Piyuz-ui/churn-capstone\ncd churn-capstone\n"
                       "pip install -r requirements.txt\npython run_pipeline.py        # or --offline\n"
                       "pytest -q\n# open reports/dashboard.html", CODE),
          Paragraph("References", H2)]
    s += bullets(["IBM, <i>Telco Customer Churn</i> sample dataset, github.com/IBM/telco-customer-churn-on-icp4d.",
                  "Pedregosa et al., <i>Scikit-learn: Machine Learning in Python</i>, JMLR 12, 2011.",
                  "pandas, matplotlib and seaborn documentation.",
                  "AML001 lab manual, KR Mangalam University, 2026-27."])

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title="Customer Churn Capstone Report",
                            author="Piyush Agrawal")
    doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
