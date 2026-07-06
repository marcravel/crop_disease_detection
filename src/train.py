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

EPOCHS = 1

torch.manual_seed(src.dataset.SEED)

def main():
    """
    Main function to initialize the model, loss function, optimizer, and start training.

    This function sets up the ResNet18 model for crop disease classification,
    configures the loss function and optimizer, and initiates the training loop
    for a specified number of epochs.
    """
    #Sending to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_crop_disease_model(num_classes=src.dataset.NUM_CLASSES)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    #We use the Adam optimizer with a learning rate of 1e-3.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #Training
    for epoch in range(EPOCHS):
        total_loss = 0.0
        loop = tqdm(src.dataset.train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")  # updates live on same line
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(src.dataset.train_dataloader)}")
        
if __name__ == "__main__":
    main()
