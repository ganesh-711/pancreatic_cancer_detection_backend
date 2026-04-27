import sys
import os
import torch
from PIL import Image

# Setup sys.path properly to allow imports from src
sys.path.insert(0, os.path.abspath("."))

from src.config import CLASS_NAMES, MODEL_PATH, get_eval_transform
from src.model import CNNModel

device = "cpu"
model = CNNModel(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = get_eval_transform()

img_path = os.path.abspath("dataset/test/normal/1-004.jpg")
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(x)
    probabilities = torch.nn.functional.softmax(out, dim=1)
    confidence, pred = torch.max(probabilities, 1)

pred_idx = pred.item()
prediction = CLASS_NAMES[pred_idx].title()

print(f"CLASS_NAMES: {CLASS_NAMES}")
print(f"out (logits): {out}")
print(f"probabilities: {probabilities}")
print(f"pred_idx: {pred_idx}, prediction string: {prediction}")
