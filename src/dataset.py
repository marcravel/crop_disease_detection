import torch
from torchvision import transforms
from torchvision import datasets

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

DATA_DIR = "data/PlantVillage"
plant_dataset = datasets.ImageFolder(DATA_DIR, transform)

for data in plant_dataset.classes:
    print(data)
print()

for key, value in plant_dataset.class_to_idx.items():
    print(f"{key} | idx={value}")