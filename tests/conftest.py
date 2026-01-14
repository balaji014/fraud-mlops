import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from fastapi.testclient import TestClient

import serving.app as app_module


class DummyModel:
    """
    Minimal sklearn-like model object with decision_function().
    We simulate anomaly scores deterministically.
    """

    def decision_function(self, X):
        # Return a "decision" array. Higher decision => less anomalous in your logic
        # You convert anomaly_score = -decision
        # We'll return fixed decision so response is predictable.
        return np.array([0.5])


@pytest.fixture(autouse=True)
def mock_startup(monkeypatch):
    """
    Override the startup-loaded globals:
      model, model_version, threshold, feature_cols, schema
    """
    app_module.model = DummyModel()
    app_module.model_version = "test"
    app_module.threshold = 0.2  # anomaly_score >= 0.2 => fraud
    # Keep feature cols aligned with your API expectation
    app_module.feature_cols = ["Time", "Amount", "V1", "V2", "V3"]
    # If your code uses schema, keep it minimal or load real one
    app_module.schema = None

    yield


@pytest.fixture()
def client():
    return TestClient(app_module.app)
