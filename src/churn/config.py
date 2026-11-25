"""Central configuration for the churn capstone pipeline.

Every path and setting lives here so the other modules stay reusable.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
EXPERIMENTS = ROOT / "experiments"

# Public copy of the IBM Telco Customer Churn dataset.
# If the machine is offline, acquire.py falls back to a seeded generator
# that produces data with the same schema.
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling",
]
CATEGORICAL_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# Validation rules: column -> allowed values (categorical) or (min, max) (numeric)
ALLOWED_VALUES = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "PaperlessBilling": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}
NUMERIC_RANGES = {
    "tenure": (0, 72),
    "MonthlyCharges": (18.0, 120.0),
    "TotalCharges": (0.0, 9000.0),
}

for _p in (DATA_RAW, DATA_PROCESSED, DATA_SAMPLE, MODELS, FIGURES, EXPERIMENTS):
    _p.mkdir(parents=True, exist_ok=True)
