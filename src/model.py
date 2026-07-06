import torch
import torchvision

def get_crop_disease_model(num_classes: int):
    """
    Initializes a ResNet18 model for crop disease classification.

    Args:
        num_classes (int): Number of output classes for the classification task.
    """
    
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, out_features=num_classes)
    return model
