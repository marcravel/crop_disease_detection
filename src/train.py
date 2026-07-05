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
import torchvision
import torch
from tqdm import tqdm

EPOCHS = 1

torch.manual_seed(src.dataset.SEED)

#Using transfer learning; we use the pre-trained ResNet18 model with ImageNet1K weights.
resnet18_model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
#
resnet18_model.fc = torch.nn.Linear(resnet18_model.fc.in_features,
                                    out_features=src.dataset.NUM_CLASSES) 

criterion = torch.nn.CrossEntropyLoss()
#We use the Adam optimizer with a learning rate of 1e-3.
optimizer = torch.optim.Adam(resnet18_model.parameters(), lr=1e-3)

#Sending to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18_model.to(device)
print(f"Using device: {device}")

#Training
for epoch in range(EPOCHS):
    total_loss = 0.0
    loop = tqdm(src.dataset.train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = resnet18_model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")  # updates live on same line
        
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(src.dataset.train_dataloader)}")
        
