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


torch.manual_seed(src.dataset.SEED)

resnet18_model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
resnet18_model.fc = torch.nn.Linear(resnet18_model.fc.in_features,
                                    out_features=src.dataset.NUM_CLASSES) 

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18_model.parameters(), lr=1e-3)

#one epoch loop

resnet18_model.train()
