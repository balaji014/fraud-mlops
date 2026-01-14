# tests/test_api.py
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_name" in data
    assert "model_version" in data
    assert "threshold" in data


def test_predict_success(client):
    payload = {
        "features": {
            "Time": 1000,
            "Amount": 50,
            "V1": 0.1,
            "V2": -0.2,
            "V3": 0.3
        }
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()

    # Contract checks
    assert "anomaly_score" in data
    assert "is_fraud" in data
    assert data["model_version"] == "test"

    # DummyModel decision_function returns 0.5, anomaly_score = -0.5
    # threshold=0.2 => -0.5 >= 0.2 is False
    assert data["is_fraud"] is False


def test_predict_missing_feature(client):
    payload = {
        "features": {
            "Time": 1000,
            "Amount": 50,
            "V1": 0.1,
            "V2": -0.2
            # Missing V3
        }
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert "Missing required feature" in r.json()["detail"]


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus format is plain text
    assert "python_info" in r.text or "fraud_api_requests_total" in r.text
