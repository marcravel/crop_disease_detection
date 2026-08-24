import os

class Settings:
    PROJECT_NAME: str = "Crop Disease Detector API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Path to ONNX model export from Staj-I
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH: str = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "checkpoints", "crop_disease_model.onnx"))
    CHECKPOINT_PTH_PATH: str = os.getenv("CHECKPOINT_PTH_PATH", os.path.join(BASE_DIR, "checkpoints", "best_crop_model.pth"))
    
    # Model Normalization Constants (ImageNet)
    MEAN: list = [0.485, 0.456, 0.406]
    STD: list = [0.229, 0.224, 0.225]
    IMAGE_SIZE: tuple = (224, 224)
    
    # CORS Configuration
    CORS_ORIGINS: list = ["*"]

settings = Settings()
