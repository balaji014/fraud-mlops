from pydantic import BaseModel, Field
from typing import Dict, Any


class PredictRequest(BaseModel):
    # Accept flexible payload. We'll validate required features ourselves.
    features: Dict[str, Any] = Field(..., description="Feature dict matching training schema")


class PredictResponse(BaseModel):
    anomaly_score: float
    is_fraud: bool
    model_name: str
    model_version: str
    threshold: float