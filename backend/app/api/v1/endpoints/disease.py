from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from backend.app.services.disease_db import DISEASE_KNOWLEDGE_BASE, get_disease_info

router = APIRouter()

@router.get("/disease", response_model=List[Dict[str, Any]])
def list_all_diseases():
    """
    Returns a summary list of all 15 plant disease classes in the knowledge base.
    """
    diseases = []
    for class_name, info in DISEASE_KNOWLEDGE_BASE.items():
        diseases.append({
            "disease_id": info["disease_id"],
            "name_tr": info["name_tr"],
            "name_en": info["name_en"],
            "crop_type": info["crop_type"],
            "is_healthy": info["is_healthy"],
            "severity": info["severity"]
        })
    return diseases

@router.get("/disease/{class_name}", response_model=Dict[str, Any])
def get_disease_by_class(class_name: str):
    """
    Returns detailed symptoms, causes, organic/chemical treatments, and prevention protocols for a specific class name.
    """
    if class_name not in DISEASE_KNOWLEDGE_BASE:
        raise HTTPException(status_code=440, detail=f"Disease '{class_name}' not found in database.")
    return get_disease_info(class_name)
