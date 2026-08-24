"""
Full model training script for crop disease detection.

Handles:
- Command line argument parsing (--epochs, --batch-size, --lr, --patience)
- ResNet18 model setup and dynamic GPU memory management
- Loss logging to CSV (results/training_log.csv)
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping based on validation metrics
- Saving complete checkpoint metadata payload (checkpoints/best_crop_model.pth)
- Automated plot generation (results/learning_curves.png)
- Exporting TorchScript and ONNX formats for Part 2 web app integration
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import src.dataset
from src.model import get_crop_disease_model
from src.utils import save_training_log, plot_learning_curves, export_model_formats

torch.manual_seed(src.dataset.SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 for Crop Disease Detection.")
    parser.add_argument('--epochs', type=int, default=15, help='Maximum number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for Adam optimizer.')
    parser.add_argument('--patience', type=int, default=4, help='Early stopping patience count.')
    return parser.parse_args()

def main():
    args = parse_args()

    # GPU Device detection & VRAM cache management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    # DataLoaders setup
    train_loader, val_loader, test_loader, class_to_idx = src.dataset.get_dataloaders(
        data_dir=src.dataset.DATA_DIR,
        batch_size=args.batch_size,
        num_workers=2,
        seed=src.dataset.SEED
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Model, Loss, Optimizer & LR Scheduler initialization
    model = get_crop_disease_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0

    log_history = []
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"Starting training for up to {args.epochs} epochs (Batch Size: {args.batch_size}, Initial LR: {args.lr})...")

    for epoch in range(1, args.epochs + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", unit="batch")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += (preds == labels).sum().item()
            train_total += labels.size(0)

            train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(train_corrects / train_total):.4f}")

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_corrects / train_total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]  ", unit="batch")
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(val_corrects / val_total):.4f}")

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects / val_total

        current_lr = optimizer.param_groups[0]['lr']

        # Record log
        log_history.append({
            'epoch': epoch,
            'train_loss': f"{epoch_train_loss:.6f}",
            'train_acc': f"{epoch_train_acc:.6f}",
            'val_loss': f"{epoch_val_loss:.6f}",
            'val_acc': f"{epoch_val_acc:.6f}",
            'lr': f"{current_lr:.6f}"
        })

        print(f"Epoch {epoch:02d}/{args.epochs:02d} Summary | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        # Step LR Scheduler based on validation loss
        scheduler.step(epoch_val_loss)

        # --- Checkpoint & Early Stopping Logic ---
        if epoch_val_acc > best_acc:
            print(f" Validation accuracy improved from {best_acc:.4f} to {epoch_val_acc:.4f}. Saving best checkpoint...")
            best_acc = epoch_val_acc
            best_loss = epoch_val_loss
            epochs_no_improve = 0

            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "best_loss": best_loss,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "transform_params": {
                    "resize": (224, 224),
                    "mean": src.dataset.MEAN_VALUE,
                    "std": src.dataset.STD_VALUE
                },
                "architecture": "resnet18"
            }
            torch.save(checkpoint_payload, "checkpoints/best_crop_model.pth")
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve. Early stopping counter: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs of training!")
            break

    # Save final log CSV and generate plots
    save_training_log(log_history, "results/training_log.csv")
    plot_learning_curves("results/training_log.csv", "results/learning_curves.png")

    # Load best checkpoint before exporting models
    print("\nLoading best model checkpoint for deployment export...")
    best_ckpt = torch.load("checkpoints/best_crop_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Export to TorchScript (.pt) and ONNX (.onnx) for Part 2 Web Application Integration
    export_model_formats(model, device=device, export_dir="checkpoints")

    print("\nTraining and model export pipeline completed successfully!")

if __name__ == "__main__":
    main()
