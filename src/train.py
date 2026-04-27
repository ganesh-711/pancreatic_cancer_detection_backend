import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
    transforms.ToTensor(),  # ✅ REQUIRED
])


train_data = datasets.ImageFolder("C:/Users/sivak/OneDrive/Desktop/PancreasCancerDetection/dataset/train", transform=train_transform)
val_data = datasets.ImageFolder("C:/Users/sivak/OneDrive/Desktop/PancreasCancerDetection/dataset/val", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

model = CNNModel(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0008)

best_acc = 0

for epoch in range(10):  # CPU friendly
    model.train()
    for img, lbl in train_loader:
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, lbl)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img, lbl in val_loader:
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            _, pred = torch.max(output, 1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/10 — Val Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../saved_models/pancreas_model.pth")
        print("✅ Model Improved — Saved!")

print("🎯 Training Done")
