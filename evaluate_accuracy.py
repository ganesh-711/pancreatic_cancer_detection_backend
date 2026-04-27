import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
sys.path.append(r"c:\AI-ML PROJECT")
from src.model import CNNModel

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dir = r"c:\AI-ML PROJECT\dataset\train"
    test_dir = r"c:\AI-ML PROJECT\dataset\test"
    
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    model = CNNModel(num_classes=2).to(device)
    model.load_state_dict(torch.load(r"c:\AI-ML PROJECT\saved_models\pancreas_model.pth", map_location=device))
    model.eval()
    
    def get_accuracy(loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for img, lbl in loader:
                img, lbl = img.to(device), lbl.to(device)
                output = model(img)
                _, pred = torch.max(output, 1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)
        return correct / total * 100.0 if total > 0 else 0.0

    print("Evaluating Train Accuracy...")
    train_acc = get_accuracy(train_loader)
    print(f"Train Accuracy: {train_acc:.2f}%")
    
    print("Evaluating Test Accuracy...")
    test_acc = get_accuracy(test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    evaluate()
