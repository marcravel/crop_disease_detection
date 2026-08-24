"""
Production inference script for crop disease detection.

Part 2 Web Application Integration API:
Provides predict_image() function that loads model checkpoint metadata dynamically,
processes input image, and returns top-1 and top-k predicted disease classes with confidence scores.

CLI usage:
python -m src.predict --image data/test_sample.jpg --checkpoint checkpoints/best_crop_model.pth
"""

import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import get_crop_disease_model

def load_model_from_checkpoint(checkpoint_path=None, device=None):
    if checkpoint_path is None:
        if os.path.exists("checkpoints/best_plantdoc_model.pth"):
            checkpoint_path = "checkpoints/best_plantdoc_model.pth"
        else:
            checkpoint_path = "checkpoints/best_crop_model.pth"
    """
    Loads model architecture and state dict from checkpoint payload.

    Returns:
        model (torch.nn.Module): Loaded model in eval mode.
        idx_to_class (dict): Mapping from class index to class name.
        transform (transforms.Compose): Preprocessing transform pipeline matching training setup.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'. Train the model first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 15)
    idx_to_class = checkpoint.get("idx_to_class", {})

    model = get_crop_disease_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform_params = checkpoint.get("transform_params", {
        "resize": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    })

    transform = transforms.Compose([
        transforms.Resize(tuple(transform_params["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_params["mean"], std=transform_params["std"])
    ])

    return model, idx_to_class, transform, device


def predict_image(image_path, checkpoint_path=None, top_k=3, device=None):
    """
    Inference function for single image prediction.

    Args:
        image_path (str): Path to input image file.
        checkpoint_path (str): Path to model checkpoint file.
        top_k (int): Number of top predictions to return.
        device (torch.device, optional): Device to run inference on.

    Returns:
        dict: Prediction results including top prediction, top_k list, and latency.
    """
    start_time = time.time()

    model, idx_to_class, transform, device = load_model_from_checkpoint(checkpoint_path, device)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at '{image_path}'")

    img = Image.open(image_path).convert("RGB")
    tensor_img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = F.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(idx_to_class)))

    top_k_results = []
    for prob, idx in zip(top_probs, top_indices):
        idx_item = idx.item()
        prob_item = prob.item()
        class_name = idx_to_class.get(idx_item, f"Class_{idx_item}")
        top_k_results.append({
            "class_idx": idx_item,
            "class_name": class_name,
            "confidence": round(prob_item * 100, 2)
        })

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "success",
        "image_path": image_path,
        "top_prediction": top_k_results[0],
        "top_k_predictions": top_k_results,
        "latency_ms": latency_ms
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run crop disease inference on an image.")
    parser.add_argument('--image', type=str, required=True, help='Path to image file for prediction.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_crop_model.pth', help='Path to checkpoint.')
    parser.add_argument('--top-k', type=int, default=3, help='Top K predictions to output.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_image(args.image, args.checkpoint, top_k=args.top_k)
    print(json.dumps(result, indent=4))
