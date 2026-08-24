import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_list_diseases_endpoint():
    response = client.get("/api/v1/disease")
    assert response.status_code == 200
    diseases = response.json()
    assert isinstance(diseases, list)
    assert len(diseases) == 15

def test_predict_single_image_endpoint():
    # Use real image if exists
    test_img_path = "data/PlantVillage/Tomato_healthy/000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG"
    if not os.path.exists(test_img_path):
        pytest.skip(f"Test image '{test_img_path}' not found.")

    with open(test_img_path, "rb") as f:
        files = {"file": ("test_leaf.jpg", f, "image/jpeg")}
        response = client.post("/api/v1/predict?top_k=3", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "top_prediction" in data
    assert data["top_prediction"]["class_name"] == "Tomato_healthy"
    assert len(data["top_k_predictions"]) == 3
    assert "disease_info" in data
    assert "latency_ms" in data
