"""
Model architecture definition for crop disease detection.
Utilizes ResNet-18 pretrained on ImageNet1K with a custom fully-connected classification head featuring Dropout regularization.
"""

import torch
import torch.nn as nn
import torchvision.models as models

def get_crop_disease_model(num_classes: int = 15, pretrained: bool = True, dropout_prob: float = 0.3):
    """
    Initializes a ResNet18 model for crop disease classification with optional Dropout.

    Args:
        num_classes (int): Number of output classes for the classification task.
        pretrained (bool): Whether to load weights pretrained on ImageNet.
        dropout_prob (float): Dropout probability applied before final linear layer.

    Returns:
        torch.nn.Module: Configured ResNet-18 model.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    in_features = model.fc.in_features
    if dropout_prob > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    
    return model
