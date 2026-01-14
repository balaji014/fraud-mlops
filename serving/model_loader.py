import json
import os
from pathlib import Path
from typing import Any, Tuple, List

import mlflow
from mlflow.tracking import MlflowClient


def load_production_model(model_name: str = "fraud_anomaly_model") -> Tuple[Any, str, float, list]:
    """
    Loads model + inference metadata.

    Priority:
      1) If MODEL_URI env is provided (e.g., runs:/<run_id>/model), load direct from that run.
      2) Else load from Model Registry Production stage: models:/<model_name>/Production
         (fallback to latest version if Production not set)

    Returns: (model, model_version_label, threshold, feature_cols)
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # ---------- Mode 1: Direct URI ----------
    model_uri = os.environ.get("MODEL_URI")
    if model_uri:
        run_id = ""
        if model_uri.startswith("runs:/"):
            # runs:/<run_id>/model
            parts = model_uri.split("/")
            if len(parts) >= 3:
                run_id = parts[1]

        if not run_id:
            raise RuntimeError("MODEL_URI must be like: runs:/<RUN_ID>/model")

        cache_dir = Path("artifacts") / "inference_cache" / f"run_{run_id}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        threshold_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="threshold.json", dst_path=str(cache_dir)
        )
        feature_cols_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="feature_cols.json", dst_path=str(cache_dir)
        )

        with open(threshold_path, "r") as f:
            threshold = float(json.load(f)["threshold"])
        with open(feature_cols_path, "r") as f:
            feature_cols = json.load(f)

        model = mlflow.sklearn.load_model(model_uri)
        model_version = os.environ.get("MODEL_VERSION", f"run-{run_id}")
        return model, model_version, threshold, feature_cols

    # ---------- Mode 2: Registry (Production preferred) ----------
    # Try Production stage first
    prod = client.get_latest_versions(model_name, stages=["Production"])
    if prod:
        chosen = prod[0]
        run_id = chosen.run_id
        model_version = f"v{chosen.version}-Production"
        registry_uri = f"models:/{model_name}/Production"
    else:
        # Fallback: latest version of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(
                f"No model versions found for: {model_name}. "
                "Train & register a model first, or set MODEL_URI."
            )
        chosen = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
        run_id = chosen.run_id
        model_version = f"v{chosen.version}"
        registry_uri = f"models:/{model_name}/{chosen.version}"

    cache_dir = Path("artifacts") / "inference_cache" / f"{model_name}_{model_version}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    threshold_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="threshold.json", dst_path=str(cache_dir)
    )
    feature_cols_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="feature_cols.json", dst_path=str(cache_dir)
    )

    with open(threshold_path, "r") as f:
        threshold = float(json.load(f)["threshold"])
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    model = mlflow.sklearn.load_model(registry_uri)
    return model, model_version, threshold, feature_cols
