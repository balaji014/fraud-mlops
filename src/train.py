import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd

# ---- config ----
EXPERIMENT_NAME = "fraud-anomaly-dev"
REGISTERED_MODEL_NAME = "fraud_anomaly_model"

def main():
    # 1) Tracking to local MLflow server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2) Load data
    df = pd.read_csv("data/creditcard.csv")

    # --- your existing feature selection logic ---
    # IMPORTANT: Keep this consistent with your serving schema
    feature_cols = [c for c in df.columns if c != "Class"]  # adjust if needed
    X = df[feature_cols]
    y = df["Class"] if "Class" in df.columns else None

    # 3) Train your model (reuse your existing code here)
    # NOTE: Replace this with your already-working model training pipeline
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(
        n_estimators=200,
        contamination=0.0017,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    # 4) Compute your threshold (reuse your existing threshold logic)
    # Example: use decision_function or score_samples based thresholding
    scores = model.decision_function(X)  # higher = more normal
    threshold = float(scores.mean() - 3 * scores.std())  # example; keep yours if you have

    # 5) Save artifacts locally (then log them to MLflow)
    tmp_dir = Path("artifacts") / "train_artifacts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    (tmp_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}))
    (tmp_dir / "feature_cols.json").write_text(json.dumps(feature_cols))

    # If you already have schema file, log it too
    schema_path = Path("schemas") / "v1_input_schema.json"
    has_schema = schema_path.exists()

    # 6) Log EVERYTHING to MLflow (model + artifacts + params + metrics)
    with mlflow.start_run(run_name="local-train") as run:
        run_id = run.info.run_id

        # params
        mlflow.log_params({
            "model": "IsolationForest",
            "n_estimators": 200,
            "contamination": 0.0017,
        })

        # metrics (add your real metrics if you compute them)
        mlflow.log_metrics({
            "threshold": threshold,
        })

        # artifacts
        mlflow.log_artifact(str(tmp_dir / "threshold.json"))
        mlflow.log_artifact(str(tmp_dir / "feature_cols.json"))
        if has_schema:
            mlflow.log_artifact(str(schema_path))

        # model (THIS is the key upgrade)
        input_example = X.head(2)
        signature = infer_signature(input_example, model.decision_function(input_example))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",   # ✅ ensures runs:/<run_id>/model works
            signature=signature,
            input_example=input_example,
        )

        # Register in Model Registry
        model_uri = f"runs:/{run_id}/model"
        reg = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

        print("\n=== TRAIN COMPLETE ===")
        print("Run ID:", run_id)
        print("Model URI:", model_uri)
        print("Registered:", REGISTERED_MODEL_NAME, "version", reg.version)

if __name__ == "__main__":
    main()
