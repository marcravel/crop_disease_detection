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

#Print class names inside the plant_dataset
# for data in plant_dataset.classes:
#     print(data)
# print()

print(f"Class numbers: {len(plant_dataset.classes)}")
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
generator.manual_seed(2)

train_set, val_set, test_set = torch.utils.data.random_split(plant_dataset,
                                                             [train_size, val_size, test_size],
                                                             generator=generator)

train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=2)
val_dataloader = torch.utils.data.DataLoader(dataset=val_set,
                                               batch_size=32,
                                               shuffle=False,
                                               num_workers=2)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=32,
                                               shuffle=False,
                                               num_workers=2)

print(f"Batch count: train={len(train_dataloader)}, val={len(val_dataloader)}, test={len(test_dataloader)}")