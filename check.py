import sys, os, torch
sys.path.append(r"c:\AI-ML PROJECT")
from PIL import Image
import torch.nn.functional as F

from src.config import CLASS_NAMES, MODEL_PATH, get_eval_transform
from src.model import CNNModel

model = CNNModel(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
transform = get_eval_transform()

img = Image.open(r'c:\AI-ML PROJECT\dataset\test\normal\1-004.jpg').convert('RGB')
x = transform(img).unsqueeze(0)
conf, pred = torch.max(F.softmax(model(x), dim=1), 1)
print(f'Prediction: {CLASS_NAMES[pred.item()].title()} Confidence: {conf.item()}')
