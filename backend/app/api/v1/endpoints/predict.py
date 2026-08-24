import time
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from backend.app.schemas.predict import SinglePredictionResponse, BatchPredictionResponse
from backend.app.services.onnx_service import onnx_service

router = APIRouter()

@router.post("/predict", response_model=SinglePredictionResponse)
async def predict_single_image(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=15, description="Number of top probability predictions to return")
):
    """
    Accepts an uploaded crop leaf image (JPG/PNG), performs ONNX Runtime inference,
    and returns predicted disease class, confidence scores, and agricultural treatment recommendations.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPG, PNG, JPEG).")

    contents = await file.read()
    try:
        result = onnx_service.predict(image_bytes=contents, filename=file.filename, top_k=top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch_images(
    files: List[UploadFile] = File(...),
    top_k: int = Query(3, ge=1, le=15)
):
    """
    Accepts multiple uploaded leaf images and returns batch predictions.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    start_time = time.time()
    predictions = []

    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        contents = await file.read()
        try:
            res = onnx_service.predict(image_bytes=contents, filename=file.filename, top_k=top_k)
            predictions.append(res)
        except Exception:
            continue

    total_latency_ms = round((time.time() - start_time) * 1000, 2)

    return BatchPredictionResponse(
        status="success",
        total_images=len(predictions),
        predictions=predictions,
        total_latency_ms=total_latency_ms
    )
