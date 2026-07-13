"""
Initialize and configure a ResNet18 model for multi-class image classification.

This module sets up a pre-trained ResNet18 model with ImageNet weights, replaces
its final fully connected layer to match the number of disease classes, and
configures the training components including loss function and optimizer.

Components:
    - Model: ResNet18 with pre-trained ImageNet1K weights
    - Loss Function: CrossEntropyLoss for multi-class classification
    - Optimizer: Adam optimizer with learning rate of 1e-3
    - Random Seed: Set for reproducibility

The model is set to training mode to enable dropout and batch normalization updates.
"""

import src.dataset
import torch
from tqdm import tqdm
from src.model import get_crop_disease_model
import argparse

torch.manual_seed(src.dataset.SEED)

def parse_args():
    """
    Parses command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a ResNet18 model for crop disease classification.")
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train the model.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    return parser.parse_args()

def main():
    """
    Main function to initialize the model, loss function, optimizer, and start training.

    This function sets up the ResNet18 model for crop disease classification,
    configures the loss function and optimizer, and initiates the training loop
    for a specified number of epochs.
    """
    args = parse_args()

    #Sending to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_crop_disease_model(num_classes=src.dataset.NUM_CLASSES)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    #We use the Adam optimizer with the learning rate from parsed arguments.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize dataloaders using the parsed batch size
    train_dataloader = torch.utils.data.DataLoader(
        dataset=src.dataset.train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=src.dataset.val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    best_acc = 0.0

    #Training Loop
    for epoch in range(args.epochs):
        # 'train' phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0
        
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", unit="batch")
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
            
            train_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(train_corrects/train_total):.4f}")
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_corrects / train_total

        # 'val' phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", unit="batch")
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                val_loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(val_corrects/val_total):.4f}")
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects / val_total

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Check if validation accuracy is strictly greater than best_acc
        if epoch_val_acc > best_acc:
            print(f"Validation accuracy increased from {best_acc:.4f} to {epoch_val_acc:.4f}. Saving checkpoint...")
            best_acc = epoch_val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss
            }
            torch.save(checkpoint, 'checkpoints/best_crop_model.pth')
        
if __name__ == "__main__":
    main()
