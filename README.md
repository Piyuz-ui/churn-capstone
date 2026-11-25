# Customer Churn: AI-Ready Data-Engineering & Visualization Workflow

**Capstone · AML001 Data Handling, Visualization & Feature Engineering (Sec C)**
Piyush Agrawal · Roll No. 2601060053 · BCA (AI & Data Science) · KR Mangalam University

## Problem

A telecom company loses customers every month, and each lost customer is lost recurring revenue. This project builds a complete, reusable workflow. It takes raw customer data, cleans and validates it, explores it visually, engineers features and trains tracked models. It produces three outputs:

1. a clean, **ML-ready dataset**
2. an **HTML dashboard** with KPIs, EDA charts, model results and business insights
3. an **exported model** that scores new customers by churn risk

## Workflow

```
 acquire ──► clean ──► validate ──► EDA/visualize ──► feature engineering ──► experiments ──► dashboard
 (URL or      (types,    (28 rules,    (6 charts)        (sklearn Pipeline,     (4 tracked      (HTML +
  generator)   dupes,     stops on                        6 new features)        runs, best      insights)
               nulls)     failure)                                               model exported)
```

| Stage | Module | Output |
|---|---|---|
| 1. Acquisition | `src/churn/acquire.py` | `data/raw/telco_churn_raw.csv`, `data/sample/` |
| 2. Cleaning | `src/churn/clean.py` | `reports/cleaning_log.csv` |
| 3. Validation & profiling | `src/churn/validate.py` | `reports/validation_*.csv`, `reports/profile_*.csv` |
| 4. EDA & visualization | `src/churn/visualize.py` | `reports/figures/01-06*.png` |
| 5. Feature engineering | `src/churn/features.py` | `data/processed/telco_churn_ml_ready.csv` |
| 6. Experiments & export | `src/churn/train.py`, `tracking.py` | `experiments/runs.csv`, `models/*.joblib` |
| 7. Dashboard | `src/churn/visualize.py` | `reports/dashboard.html` |

## How to run

```bash
git clone <this-repo-url>
cd churn-capstone
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py              # downloads the IBM Telco CSV; falls back to generated data if offline
python run_pipeline.py --offline    # always use the seeded generator (reproducible)

pytest -q                           # 13 tests
```

Then open `reports/dashboard.html` in a browser.

For a guided, step-by-step version, open `notebooks/churn_walkthrough.ipynb`.

### Score new customers with the exported model

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from churn.train import predict

new = pd.read_csv("data/sample/telco_churn_sample.csv")   # same columns as the raw data, after cleaning
print(predict(new).head())
```

## Data

- **Primary source:** the public [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) CSV (7,043 customers, 21 columns), downloaded over HTTP in `acquire.py`.
- **Fallback:** if the download fails, `generate_synthetic()` creates data with the same schema and realistic churn behaviour (seed 42). It deliberately injects real-world problems so the cleaning stage has work to do: blanks stored as text, missing values, inconsistent casing and whitespace, impossible values and duplicate rows.
- The results in `reports/` were produced with the **synthetic fallback**. Re-run with internet access to use the real IBM data. The code works unchanged on both.

## Key design decisions

- **Reusable preprocessing:** all transforms live in one scikit-learn `Pipeline` (`FeatureBuilder` → `ColumnTransformer`). The exported model applies the same steps at prediction time, so nothing is re-coded by hand.
- **No leakage:** the train/test split happens before fitting, and oversampling is applied to training data only.
- **Validation gate:** the pipeline stops if any critical rule fails, so bad data never reaches the model.
- **Experiment tracking:** a lightweight MLflow-style tracker logs params and metrics for each run to `experiments/`. It also logs to MLflow if that is installed.
- **Imbalance handling:** compared `class_weight="balanced"` with random oversampling.

## Results (synthetic run)

| Model | CV ROC-AUC | Test ROC-AUC | Recall | F1 |
|---|---|---|---|---|
| logreg_balanced (**selected**) | 0.847 | 0.870 | 0.854 | 0.698 |
| logreg_oversampled | 0.847 | 0.870 | 0.852 | 0.693 |
| gradient_boosting | 0.841 | 0.868 | 0.866 | 0.701 |
| random_forest | 0.840 | 0.864 | 0.854 | 0.686 |

The models score within 0.01 ROC-AUC of each other, so the simple, explainable logistic regression is the right pick.

Main findings: month-to-month contracts, short tenure, electronic-check payment, fiber-optic service and having no tech support are the strongest churn signals. See `reports/insights.txt`.

## Repository layout

```
churn-capstone/
├── run_pipeline.py          # one command runs everything
├── src/churn/               # reusable modules
├── tests/                   # pytest suite
├── notebooks/               # walkthrough notebook
├── data/{raw,processed,sample}/
├── models/                  # exported pipeline (.joblib)
├── experiments/             # tracked runs
├── reports/                 # dashboard, figures, logs, CSV evidence
└── docs/                    # capstone report (PDF)
```
