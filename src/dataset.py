"""
Dataset preparation and loading module for crop disease detection.
This module handles:
- Loading plant disease images from the PlantVillage dataset
- Applying image preprocessing (resizing, normalization)
- Splitting the dataset into training, validation, and test sets
- Creating data loaders for batch processing during model training and evaluation
Constants:
    SEED (int): Random seed for reproducibility (42)
    BATCH_SIZE (int): Number of samples per batch (16)
    MEAN_VALUE (list): ImageNet normalization mean values
    STD_VALUE (list): ImageNet normalization standard deviation values
    DATA_DIR (str): Path to the PlantVillage dataset directory
Attributes:
    plant_dataset: ImageFolder dataset containing all plant disease images
    NUM_CLASSES (int): Total number of plant disease classes
    train_dataloader: DataLoader for training set with shuffling enabled
    val_dataloader: DataLoader for validation set without shuffling
    test_dataloader: DataLoader for test set without shuffling
Data Split:
    - Training: 80% of total dataset
    - Validation: 10% of total dataset
    - Testing: 10% of total dataset
"""

import torch
from torchvision import transforms
from torchvision import datasets

SEED = 42
BATCH_SIZE = 16

MEAN_VALUE = [0.485, 0.456, 0.406]
STD_VALUE = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_VALUE,
                         std=STD_VALUE)
])

DATA_DIR = "data/PlantVillage"
plant_dataset = datasets.ImageFolder(DATA_DIR, transform)
NUM_CLASSES = len(plant_dataset.classes)
print(f"Class numbers: {NUM_CLASSES}")

#Print class names inside the plant_dataset
# for data in plant_dataset.classes:
#     print(data)
# print()

#Print mapping
for key, value in plant_dataset.class_to_idx.items():
    print(f"{key} | idx={value}")
    
# data_loader = torch.utils.data.DataLoader(dataset=plant_dataset,
#                                           batch_size=16,
#                                           shuffle=True,
#                                           num_workers=2)

total_len = len(plant_dataset)
train_size = int(total_len * 0.80)
val_size = int(total_len * 0.10)
test_size = total_len - (train_size + val_size)
print("-----------------------")
print(f"Total_len: {total_len}")
print(f"Train_size: {train_size}")
print(f"val_size: {val_size}")
print(f"test_size: {test_size}")
print(f"{test_size + train_size + val_size} =? {total_len}")
print("-----------------------")

# Gun 3:
generator = torch.Generator()
generator.manual_seed(SEED)

train_set, val_set, test_set = torch.utils.data.random_split(plant_dataset,
                                                             [train_size, val_size, test_size],
                                                             generator=generator)

train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=2)
val_dataloader = torch.utils.data.DataLoader(dataset=val_set,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=2)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=2)

print(f"Batch count: train={len(train_dataloader)}, val={len(val_dataloader)}, test={len(test_dataloader)}")

