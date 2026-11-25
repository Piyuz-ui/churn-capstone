"""Experiment tracking.

A small file-based tracker with an MLflow-like API (start a run, log params,
log metrics, save artifacts). Every run gets a folder under experiments/ and a
row in experiments/runs.csv, so results are reproducible and comparable.
If MLflow is installed it is used in addition.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

import pandas as pd

from . import config

try:
    import mlflow  # optional
except ImportError:  # pragma: no cover
    mlflow = None


class Run:
    def __init__(self, name: str):
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.name = name
        self.params: dict = {}
        self.metrics: dict = {}
        self.dir = config.EXPERIMENTS / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def log_params(self, **params):
        self.params.update(params)

    def log_metrics(self, **metrics):
        self.metrics.update({k: round(float(v), 4) for k, v in metrics.items()})

    def finish(self):
        record = {"run_id": self.id, "name": self.name,
                  "duration_s": round(time.time() - self._t0, 2),
                  **{f"param_{k}": v for k, v in self.params.items()},
                  **self.metrics}
        (self.dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
        index = config.EXPERIMENTS / "runs.csv"
        df = pd.DataFrame([record])
        if index.exists():
            df = pd.concat([pd.read_csv(index), df], ignore_index=True)
        df.to_csv(index, index=False)
        if mlflow is not None:
            with mlflow.start_run(run_name=self.name):
                mlflow.log_params({k: str(v) for k, v in self.params.items()})
                mlflow.log_metrics(self.metrics)
        return record
