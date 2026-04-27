import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import CNNModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load Model
model = CNNModel(num_classes=2).to(device)
model.load_state_dict(torch.load(
    os.path.join(ROOT_DIR, "saved_models/pancreas_model.pth"),
    map_location=device
))
model.eval()

# Target Layer
target_layers = [model.model.layer4[-1]]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def generate_heatmap(image_path):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    rgb_img = np.array(img).astype(np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor)[0]

    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save to outputs/heatmaps/
    output_dir = os.path.join(ROOT_DIR, "outputs/heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gradcam_output.jpg")
    cv2.imwrite(output_path, heatmap)

    return output_path
