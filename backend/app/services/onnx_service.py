"""
ONNX Runtime Inference Service.
Loads checkpoints/crop_disease_model.onnx, preprocesses leaf images, runs inference, and computes Softmax probabilities.
"""

import os
import time
import io
import numpy as np
import onnxruntime as ort
from PIL import Image

from backend.app.config import settings
from backend.app.services.disease_db import get_disease_info

# Class mapping matching Staj-I training index order
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

class ONNXInferenceService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_name = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Warning: ONNX model file not found at '{self.model_path}'. ONNX session uninitialized.")
            return

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"ONNX Runtime Session successfully loaded from '{self.model_path}' using providers {self.session.get_providers()}")
        except Exception as e:
            print(f"Error loading ONNX model session: {e}")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Converts raw image bytes to (1, 3, 224, 224) normalized NCHW float32 numpy array.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(settings.IMAGE_SIZE)

        # Convert to numpy float32 array in range [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Apply ImageNet mean and std normalization
        mean = np.array(settings.MEAN, dtype=np.float32)
        std = np.array(settings.STD, dtype=np.float32)
        img_np = (img_np - mean) / std

        # Transpose HWC (224, 224, 3) -> CHW (3, 224, 224)
        img_np = np.transpose(img_np, (2, 0, 1))

        # Add batch dimension -> NCHW (1, 3, 224, 224)
        img_np = np.expand_dims(img_np, axis=0)
        return img_np

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def predict(self, image_bytes: bytes, filename: str = "image.jpg", top_k: int = 3) -> dict:
        """
        Runs inference on image bytes and returns structured prediction payload.
        """
        if self.session is None:
            self.load_model()
            if self.session is None:
                raise RuntimeError("ONNX Runtime session is not initialized. Model file missing.")

        start_time = time.time()
        input_tensor = self.preprocess_image(image_bytes)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]
        probs = self.softmax(logits)

        # Top-k indices
        top_k_idx = np.argsort(probs)[::-1][:top_k]

        top_k_list = []
        for idx in top_k_idx:
            class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
            top_k_list.append({
                "class_idx": int(idx),
                "class_name": class_name,
                "confidence": round(float(probs[idx] * 100), 2)
            })

        top_pred = top_k_list[0]
        disease_info = get_disease_info(top_pred["class_name"])
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "filename": filename,
            "top_prediction": top_pred,
            "top_k_predictions": top_k_list,
            "disease_info": disease_info,
            "latency_ms": latency_ms
        }

onnx_service = ONNXInferenceService()
