from prometheus_client import Counter, Histogram

# Labels help us slice dashboards by model version
REQUESTS_TOTAL = Counter(
    "fraud_api_requests_total",
    "Total number of requests to the fraud scoring API",
    ["endpoint", "http_status", "model_version"],
)

PREDICTIONS_TOTAL = Counter(
    "fraud_api_predictions_total",
    "Total predictions made",
    ["result", "model_version"],  # result: fraud|not_fraud
)

REQUEST_LATENCY_SECONDS = Histogram(
    "fraud_api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "model_version"],
)

ERRORS_TOTAL = Counter(
    "fraud_api_errors_total",
    "Total errors",
    ["type", "endpoint", "model_version"],
)