from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, predict, disease

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health Check"])
api_router.include_router(predict.router, tags=["Inference Engine"])
api_router.include_router(disease.router, tags=["Disease Knowledge Base"])
