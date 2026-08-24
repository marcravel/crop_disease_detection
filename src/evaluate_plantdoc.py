"""
Zero-shot baseline evaluation script for PlantDoc dataset using PlantVillage-trained model.

Evaluates checkpoints/best_crop_model.pth on PlantDoc test split without fine-tuning
to measure the initial lab-to-field generalization gap.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score

from src.model import get_crop_disease_model

# Mapping dictionary between PlantDoc folder names and PlantVillage 15-class standard names
PLANTDOC_TO_PLANTVILLAGE = {
    "Bell_pepper leaf spot": "Pepper__bell___Bacterial_spot",
    "Pepper bell Bacterial spot": "Pepper__bell___Bacterial_spot",
    "Pepper__bell___Bacterial_spot": "Pepper__bell___Bacterial_spot",

    "Bell_pepper leaf": "Pepper__bell___healthy",
    "Pepper bell healthy": "Pepper__bell___healthy",
    "Pepper__bell___healthy": "Pepper__bell___healthy",

    "Potato leaf early blight": "Potato___Early_blight",
    "Potato Early blight": "Potato___Early_blight",
    "Potato___Early_blight": "Potato___Early_blight",

    "Potato leaf late blight": "Potato___Late_blight",
    "Potato Late blight": "Potato___Late_blight",
    "Potato___Late_blight": "Potato___Late_blight",

    "Potato leaf": "Potato___healthy",
    "Potato healthy": "Potato___healthy",
    "Potato___healthy": "Potato___healthy",

    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato Bacterial spot": "Tomato_Bacterial_spot",
    "Tomato_Bacterial_spot": "Tomato_Bacterial_spot",

    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato Early blight": "Tomato_Early_blight",
    "Tomato_Early_blight": "Tomato_Early_blight",

    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato Late blight": "Tomato_Late_blight",
    "Tomato_Late_blight": "Tomato_Late_blight",

    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Leaf Mold": "Tomato_Leaf_Mold",
    "Tomato_Leaf_Mold": "Tomato_Leaf_Mold",

    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato_Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",

    "Tomato two spotted spider mites leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato Two-spotted spider mite": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato_Spider_mites_Two_spotted_spider_mite",

    "Tomato target spot leaf": "Tomato__Target_Spot",
    "Tomato Target Spot": "Tomato__Target_Spot",
    "Tomato__Target_Spot": "Tomato__Target_Spot",

    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato Yellow Leaf Curl Virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",

    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_mosaic_virus": "Tomato__Tomato_mosaic_virus",

    "Tomato leaf": "Tomato_healthy",
    "Tomato healthy": "Tomato_healthy",
    "Tomato_healthy": "Tomato_healthy"
}

def evaluate_plantdoc_zero_shot(
    plantdoc_dir="data/plantdoc/test",
    checkpoint_path="checkpoints/best_crop_model.pth",
    out_metrics_path="results/plantdoc_baseline_metrics.json"
):
    """
    Evaluates PlantVillage checkpoint on PlantDoc test set in a zero-shot manner.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running zero-shot PlantDoc evaluation using device: {device}")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint '{checkpoint_path}' not ready yet. Skipping evaluation until training completes.")
        return {}

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = checkpoint.get("idx_to_class", {v: k for k, v in class_to_idx.items()})
    num_classes = len(class_to_idx)

    model = get_crop_disease_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(plantdoc_dir):
        print(f"Warning: PlantDoc directory '{plantdoc_dir}' not found.")
        return {}

    all_preds = []
    all_targets = []
    valid_images = 0

    for folder_name in os.listdir(plantdoc_dir):
        folder_path = os.path.join(plantdoc_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        target_pv_class = PLANTDOC_TO_PLANTVILLAGE.get(folder_name, folder_name)
        if target_pv_class not in class_to_idx:
            continue

        target_idx = class_to_idx[target_pv_class]

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                tensor_img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor_img)
                    pred_idx = torch.argmax(output, dim=1).item()

                all_preds.append(pred_idx)
                all_targets.append(target_idx)
                valid_images += 1
            except Exception:
                continue

    if valid_images == 0:
        print("No valid PlantDoc images evaluated.")
        return {}

    acc = accuracy_score(all_targets, all_preds)
    print(f"Zero-Shot PlantDoc Accuracy: {acc * 100:.2f}% ({valid_images} images evaluated)")

    results = {
        "dataset": "PlantDoc Test Split (Zero-Shot)",
        "num_samples": valid_images,
        "accuracy": float(acc),
        "note": "Baseline accuracy before fine-tuning on real-world field imagery."
    }

    os.makedirs(os.path.dirname(out_metrics_path), exist_ok=True)
    with open(out_metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved zero-shot PlantDoc metrics to {out_metrics_path}")

    return results

if __name__ == "__main__":
    evaluate_plantdoc_zero_shot()
