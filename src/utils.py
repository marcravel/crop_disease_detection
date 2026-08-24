"""
Utility functions for crop disease detection model training, logging, visualization, and deployment exports.
"""

import os
import csv
import torch
import matplotlib.pyplot as plt

def save_training_log(log_data, log_csv_path="results/training_log.csv"):
    """
    Saves or appends epoch metrics to a CSV file.

    Args:
        log_data (list of dict): List containing epoch dictionaries with keys:
            ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr']
        log_csv_path (str): Path to CSV log file.
    """
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr']

    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_data:
            writer.writerow(row)
    print(f"Training log successfully saved to {log_csv_path}")


def plot_learning_curves(log_csv_path="results/training_log.csv", save_png_path="results/learning_curves.png"):
    """
    Reads epoch metrics from CSV and plots side-by-side loss and accuracy curves.

    Args:
        log_csv_path (str): Path to CSV log file.
        save_png_path (str): Output path for saved plot figure.
    """
    if not os.path.exists(log_csv_path):
        print(f"Warning: Log file {log_csv_path} not found. Skipping plot generation.")
        return

    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with open(log_csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            train_acc.append(float(row['train_acc']))
            val_acc.append(float(row['val_acc']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2, marker='o')
    ax1.plot(epochs, val_loss, label='Val Loss', color='#ff7f0e', linewidth=2, marker='s')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)

    # Accuracy Curve
    ax2.plot(epochs, train_acc, label='Train Accuracy', color='#2ca02c', linewidth=2, marker='o')
    ax2.plot(epochs, val_acc, label='Val Accuracy', color='#d62728', linewidth=2, marker='s')
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
    plt.savefig(save_png_path, dpi=300)
    plt.close()
    print(f"Learning curves plot successfully saved to {save_png_path}")


def export_model_formats(model, device=None, export_dir="checkpoints"):
    """
    Exports trained PyTorch model into TorchScript (.pt) and ONNX (.onnx) formats for Part 2 web app integration.

    Args:
        model (torch.nn.Module): Trained model instance.
        device (torch.device): Device where dummy input should be located.
        export_dir (str): Directory where exported files will be stored.
    """
    os.makedirs(export_dir, exist_ok=True)
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # 1. TorchScript export
    ts_path = os.path.join(export_dir, "crop_disease_model.pt")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        print(f"TorchScript model successfully exported to {ts_path}")
    except Exception as e:
        print(f"Error exporting TorchScript model: {e}")

    # 2. ONNX export
    onnx_path = os.path.join(export_dir, "crop_disease_model.onnx")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model successfully exported to {onnx_path}")
    except Exception as e:
        print(f"Error exporting ONNX model: {e}")
