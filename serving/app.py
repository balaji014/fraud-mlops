import os
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.schema import load_schema
from serving.model_loader import load_production_model
from serving.schemas import PredictRequest, PredictResponse
from serving.metrics import (
    REQUESTS_TOTAL,
    PREDICTIONS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    ERRORS_TOTAL,
)

app = FastAPI(title="Fraud Anomaly Scoring API", version="1.0")

MODEL_NAME = os.environ.get("MODEL_NAME", "fraud_anomaly_model")
SCHEMA_PATH = "schemas/v1_input_schema.json"

model = None
model_version: Optional[str] = None
threshold: Optional[float] = None
feature_cols = None
schema = None


@app.on_event("startup")
def startup():
    global model, model_version, threshold, feature_cols, schema
    schema = load_schema(SCHEMA_PATH)
    model, model_version, threshold, feature_cols = load_production_model(MODEL_NAME)

    # Helpful startup logs (shows what MLflow actually loaded)
    print(f"[startup] Loaded model type: {type(model)}")
    print(f"[startup] Has named_steps: {hasattr(model, 'named_steps')}")
    print(f"[startup] Model version: {model_version}, threshold: {threshold}")
    print(f"[startup] Feature cols count: {len(feature_cols) if feature_cols else 'N/A'}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "threshold": threshold,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _compute_anomaly_score(m, X: pd.DataFrame) -> float:
    """
    Returns anomaly_score where higher = more anomalous.
    Works with:
      - sklearn Pipeline (named_steps)
      - bare estimators (IsolationForest, OneClassSVM, etc.)
    """
    # Pipeline case
    if hasattr(m, "named_steps"):
        # Prefer calling decision_function on the pipeline directly if available
        if hasattr(m, "decision_function"):
            decision = float(m.decision_function(X)[0])
        else:
            # Fallback: transform then call last estimator
            # (rare case)
            steps = list(m.named_steps.keys())
            # everything except last step transforms
            transformer = m[:-1]
            estimator = m.named_steps[steps[-1]]
            X_t = transformer.transform(X)
            if hasattr(estimator, "decision_function"):
                decision = float(estimator.decision_function(X_t)[0])
            elif hasattr(estimator, "score_samples"):
                decision = float(estimator.score_samples(X_t)[0])
            else:
                raise RuntimeError("Estimator does not support decision_function/score_samples")
    else:
        # Bare estimator case
        if hasattr(m, "decision_function"):
            decision = float(m.decision_function(X)[0])
        elif hasattr(m, "score_samples"):
            decision = float(m.score_samples(X)[0])
        else:
            raise RuntimeError("Model does not support decision_function/score_samples")

    # IsolationForest: higher = more normal; lower = more anomalous
    anomaly_score = float(-decision)
    return anomaly_score


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    endpoint = "/predict"

    try:
        if model is None or feature_cols is None or threshold is None:
            ERRORS_TOTAL.labels(type="model_not_loaded", endpoint=endpoint, model_version=str(model_version)).inc()
            REQUESTS_TOTAL.labels(endpoint=endpoint, http_status="503", model_version=str(model_version)).inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        feats: Dict[str, Any] = req.features

        # Build row in exact training column order
        try:
            row = {c: float(feats[c]) for c in feature_cols}
        except KeyError as e:
            missing = str(e).strip("'")
            ERRORS_TOTAL.labels(type="missing_feature", endpoint=endpoint, model_version=str(model_version)).inc()
            REQUESTS_TOTAL.labels(endpoint=endpoint, http_status="400", model_version=str(model_version)).inc()
            raise HTTPException(status_code=400, detail=f"Missing required feature: {missing}")
        except ValueError:
            ERRORS_TOTAL.labels(type="non_numeric", endpoint=endpoint, model_version=str(model_version)).inc()
            REQUESTS_TOTAL.labels(endpoint=endpoint, http_status="400", model_version=str(model_version)).inc()
            raise HTTPException(status_code=400, detail="All feature values must be numeric")

        X = pd.DataFrame([row], columns=feature_cols)

        # Optional: schema validation if you want it enforced
        # If your validate_dataframe returns (ok, error_msg) or raises, adapt here.
        # Keeping schema loaded for future use, but not enforcing by default to avoid mismatches.
        # validate_dataframe(X, schema)

        try:
            anomaly_score = _compute_anomaly_score(model, X)
        except Exception as ex:
            ERRORS_TOTAL.labels(type="scoring_failed", endpoint=endpoint, model_version=str(model_version)).inc()
            REQUESTS_TOTAL.labels(endpoint=endpoint, http_status="500", model_version=str(model_version)).inc()
            raise HTTPException(status_code=500, detail=f"Scoring failed: {str(ex)}")

        # fraud decision
        # anomaly_score higher => more anomalous
        is_fraud = bool(anomaly_score >= float(threshold))

        # prediction metrics
        PREDICTIONS_TOTAL.labels(
            result="fraud" if is_fraud else "not_fraud",
            model_version=str(model_version),
        ).inc()

        REQUESTS_TOTAL.labels(endpoint=endpoint, http_status="200", model_version=str(model_version)).inc()

        return PredictResponse(
            anomaly_score=float(anomaly_score),
            is_fraud=is_fraud,
            model_name=MODEL_NAME,
            model_version=str(model_version),
            threshold=float(threshold),
        )

    finally:
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint, model_version=str(model_version)).observe(elapsed)
