import os
import sys

# Insert the FULL ROOT DIR
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import CLASS_NAMES, MODEL_PATH, get_eval_transform
from src.model import CNNModel

device = "cpu"
model = CNNModel(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = get_eval_transform()

img_path = os.path.join(ROOT_DIR, "dataset", "test", "normal", "1-004.jpg")
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(x)
    probabilities = F.softmax(out, dim=1)
    confidence, pred = torch.max(probabilities, 1)

pred_idx = pred.item()
prediction = CLASS_NAMES[pred_idx].title()

print(f"Prediction: '{prediction}'")
