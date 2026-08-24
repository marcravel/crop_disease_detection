from typing import List, Optional
from pydantic import BaseModel, Field

class PredictionItem(BaseModel):
    class_idx: int = Field(..., description="Numerical index of the predicted class")
    class_name: str = Field(..., description="Human readable name of the plant disease class")
    confidence: float = Field(..., description="Confidence probability percentage (0-100%)")

class DiseaseDetail(BaseModel):
    disease_id: str
    name_tr: str
    name_en: str
    crop_type: str
    is_healthy: bool
    severity: str
    description: str
    symptoms: List[str]
    organic_treatment: List[str]
    chemical_treatment: List[str]
    prevention: List[str]

class SinglePredictionResponse(BaseModel):
    status: str = "success"
    filename: str
    top_prediction: PredictionItem
    top_k_predictions: List[PredictionItem]
    disease_info: Optional[DiseaseDetail] = None
    latency_ms: float

class BatchPredictionResponse(BaseModel):
    status: str = "success"
    total_images: int
    predictions: List[SinglePredictionResponse]
    total_latency_ms: float

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    version: str
    model_loaded: bool
    model_path: str
    device: str
