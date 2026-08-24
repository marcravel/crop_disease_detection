"""
Evaluation script for PlantVillage crop disease detection model.

Computes:
- Overall test accuracy
- Per-class precision, recall, and F1-score
- Exports evaluation metrics to results/plantvillage_metrics.json
- Generates confusion matrix heatmap saved to results/confusion_matrix.png
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import src.dataset
from src.model import get_crop_disease_model

def evaluate_model(checkpoint_path="checkpoints/best_crop_model.pth",
                   metrics_out_path="results/plantvillage_metrics.json",
                   cm_out_path="results/confusion_matrix.png"):
    """
    Loads trained model checkpoint, evaluates on held-out test split, and exports metrics/confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating PlantVillage model using device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", src.dataset.class_to_idx)
    idx_to_class = checkpoint.get("idx_to_class", {v: k for k, v in class_to_idx.items()})
    num_classes = len(class_to_idx)

    # Initialize model and load weights
    model = get_crop_disease_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Data loader
    _, _, test_loader, _ = src.dataset.get_dataloaders(
        data_dir=src.dataset.DATA_DIR, batch_size=32, num_workers=2, seed=src.dataset.SEED
    )

    all_preds = []
    all_targets = []

    print("Running inference on PlantVillage test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")

    target_names = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

    # Save metrics JSON
    metrics_data = {
        "overall_accuracy": acc,
        "class_to_idx": class_to_idx,
        "per_class_metrics": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1_score": report[name]["f1-score"],
                "support": report[name]["support"]
            }
            for name in target_names
        },
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"]
    }

    os.makedirs(os.path.dirname(metrics_out_path), exist_ok=True)
    with open(metrics_out_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to {metrics_out_path}")

    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('PlantVillage Test Set - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(cm_out_path), exist_ok=True)
    plt.savefig(cm_out_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_out_path}")

    return metrics_data

if __name__ == "__main__":
    evaluate_model()
