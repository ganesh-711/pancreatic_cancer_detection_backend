import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNNModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_data = datasets.ImageFolder("../dataset/test", transform=test_transform)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

model = CNNModel(num_classes=2).to(device)
model.load_state_dict(torch.load("../saved_models/pancreas_model.pth", map_location=device))
model.eval()

correct = total = 0

with torch.no_grad():
    for img, labels in test_loader:
        img, labels = img.to(device), labels.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"🎯 Test Accuracy: {accuracy:.4f}")
