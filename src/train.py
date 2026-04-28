import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# FIX: Use relative paths instead of hardcoded Windows paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from model import CNNModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# FIX: Relative paths - dataset folder should be at project root
train_data = datasets.ImageFolder(os.path.join(ROOT_DIR, "dataset", "train"), transform=train_transform)
val_data   = datasets.ImageFolder(os.path.join(ROOT_DIR, "dataset", "val"),   transform=val_transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=8, shuffle=False)

model     = CNNModel(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0008)

best_acc = 0

# FIX: Save model to correct relative path
saved_models_dir = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(saved_models_dir, exist_ok=True)
model_save_path  = os.path.join(saved_models_dir, "pancreas_model.pth")

for epoch in range(10):
    model.train()
    for img, lbl in train_loader:
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss   = criterion(output, lbl)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img, lbl in val_loader:
            img, lbl = img.to(device), lbl.to(device)
            output   = model(img)
            _, pred  = torch.max(output, 1)
            correct += (pred == lbl).sum().item()
            total   += lbl.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/10 — Val Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), model_save_path)
        print("✅ Model Improved — Saved!")

print("🎯 Training Done")
print(f"Best Val Accuracy: {best_acc:.4f}")
