from fastapi import APIRouter
from backend.app.config import settings
from backend.app.services.onnx_service import onnx_service
from backend.app.schemas.predict import HealthCheckResponse

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    """
    Returns system status, ONNX model loading status, and active runtime execution provider.
    """
    model_loaded = onnx_service.session is not None
    providers = onnx_service.session.get_providers() if model_loaded else []
    device_type = providers[0] if providers else "None"

    return HealthCheckResponse(
        status="healthy",
        version=settings.VERSION,
        model_loaded=model_loaded,
        model_path=settings.MODEL_PATH,
        device=device_type
    )
