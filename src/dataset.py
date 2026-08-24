"""
Dataset preparation and loading module for crop disease detection.
This module handles:
- Loading plant disease images from the PlantVillage dataset
- Preprocessing and optional field-condition data augmentations (color jitter, random crop, cutout)
- Splitting dataset into training, validation, and test sets (80% / 10% / 10%)
- Creating reproducible DataLoaders for model training and evaluation
"""

import os
import torch
from torchvision import transforms, datasets

SEED = 42
BATCH_SIZE = 32

MEAN_VALUE = [0.485, 0.456, 0.406]
STD_VALUE = [0.229, 0.224, 0.225]

# Standard evaluation transform (Resize & ImageNet Normalization)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_VALUE, std=STD_VALUE)
])

# Field condition simulation transform (Color Jitter, Random Crop, Rotation, Cutout)
field_sim_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_VALUE, std=STD_VALUE),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random')
])

DATA_DIR = "data/PlantVillage"

if os.path.exists(DATA_DIR):
    plant_dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transform)
    NUM_CLASSES = len(plant_dataset.classes)
    class_to_idx = plant_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
else:
    plant_dataset = None
    NUM_CLASSES = 15
    class_to_idx = {}
    idx_to_class = {}


class TransformedSubset(torch.utils.data.Dataset):
    """
    Wraps a subset to apply a specific transform to the underlying ImageFolder sample.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        # Underlying dataset is ImageFolder; x is PIL Image if root dataset has transform=None
        # If root dataset already transformed, apply transform
        if self.transform is not None:
            # Re-fetch PIL image from base dataset
            orig_dataset = self.subset.dataset
            orig_idx = self.subset.indices[index]
            path, target = orig_dataset.samples[orig_idx]
            img = orig_dataset.loader(path)
            x = self.transform(img)
            y = target
        return x, y

    def __len__(self):
        return len(self.subset)


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=2, seed=SEED, use_field_aug=False):
    """
    Creates reproducible DataLoaders for train, val, and test splits.

    Args:
        data_dir (str): Path to PlantVillage dataset root folder.
        batch_size (int): DataLoader batch size.
        num_workers (int): Number of worker processes.
        seed (int): Random seed for split reproducibility.
        use_field_aug (bool): Whether to apply field simulation augmentations to training set.
    """
    # Load raw dataset without transform so subsets apply custom transforms
    raw_dataset = datasets.ImageFolder(data_dir, transform=None)
    total_len = len(raw_dataset)
    train_size = int(total_len * 0.80)
    val_size = int(total_len * 0.10)
    test_size = total_len - (train_size + val_size)

    gen = torch.Generator()
    gen.manual_seed(seed)

    raw_train, raw_val, raw_test = torch.utils.data.random_split(
        raw_dataset, [train_size, val_size, test_size], generator=gen
    )

    tr_transform = field_sim_transform if use_field_aug else eval_transform
    tr_set = TransformedSubset(raw_train, transform=tr_transform)
    va_set = TransformedSubset(raw_val, transform=eval_transform)
    te_set = TransformedSubset(raw_test, transform=eval_transform)

    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_loader = torch.utils.data.DataLoader(va_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te_loader = torch.utils.data.DataLoader(te_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return tr_loader, va_loader, te_loader, raw_dataset.class_to_idx
