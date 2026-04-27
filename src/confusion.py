import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNNModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# Same preprocessing as validation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Load test dataset
test_data = datasets.ImageFolder("../dataset/test", transform=test_transform)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# ✅ Load trained model
model = CNNModel(num_classes=2).to(device)
model.load_state_dict(torch.load("../saved_models/pancreas_model.pth", map_location=device))
model.eval()

# Evaluate
y_true = []
y_pred = []

with torch.no_grad():
    for img, labels in test_loader:
        img, labels = img.to(device), labels.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ✅ Plot heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=test_data.classes,
            yticklabels=test_data.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ✅ Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=test_data.classes))
