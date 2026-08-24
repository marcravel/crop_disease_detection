"""
Dataset preparation and loading module for crop disease detection.
This module handles:
- Loading plant disease images from the PlantVillage dataset
- Applying image preprocessing (resizing, normalization)
- Splitting the dataset into training, validation, and test sets (80% / 10% / 10%)
- Creating data loaders for batch processing during model training and evaluation
"""

import os
import torch
from torchvision import transforms, datasets

SEED = 42
BATCH_SIZE = 32

MEAN_VALUE = [0.485, 0.456, 0.406]
STD_VALUE = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_VALUE, std=STD_VALUE)
])

DATA_DIR = "data/PlantVillage"

if os.path.exists(DATA_DIR):
    plant_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    NUM_CLASSES = len(plant_dataset.classes)
    class_to_idx = plant_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    total_len = len(plant_dataset)
    train_size = int(total_len * 0.80)
    val_size = int(total_len * 0.10)
    test_size = total_len - (train_size + val_size)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_set, val_set, test_set = torch.utils.data.random_split(
        plant_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
else:
    plant_dataset = None
    NUM_CLASSES = 15
    class_to_idx = {}
    idx_to_class = {}
    train_set = val_set = test_set = None
    train_dataloader = val_dataloader = test_dataloader = None


def get_dataloaders(data_dir=DATA_DIR, batch_size=32, num_workers=2, seed=SEED):
    """
    Creates reproducible DataLoaders for train, val, and test splits.
    """
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    total_len = len(dataset)
    train_size = int(total_len * 0.80)
    val_size = int(total_len * 0.10)
    test_size = total_len - (train_size + val_size)

    gen = torch.Generator()
    gen.manual_seed(seed)

    tr_set, va_set, te_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=gen
    )

    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_loader = torch.utils.data.DataLoader(va_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te_loader = torch.utils.data.DataLoader(te_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return tr_loader, va_loader, te_loader, dataset.class_to_idx
