import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# FIX: Model is NO LONGER loaded here (was causing double load + OOM crash)
# Model is loaded once in app.py and passed into generate_heatmap()

def generate_heatmap(image_path, model, device):
    """
    Generate a GradCAM heatmap for the given image.

    Args:
        image_path: path to the input image
        model:      the already-loaded CNNModel instance (from app.py)
        device:     torch device string ("cpu" or "cuda")

    Returns:
        output_path: path to the saved heatmap image
    """
    # Target the last conv layer of ResNet18
    target_layers = [model.model.layer4[-1]]

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    rgb_img = np.array(img).astype(np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor)[0]

    # show_cam_on_image returns RGB
    heatmap_rgb = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save heatmap
    output_dir = os.path.join(ROOT_DIR, "outputs", "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gradcam_output.jpg")

    # FIX: Convert RGB -> BGR before saving with cv2 (fixes color swap bug)
    cv2.imwrite(output_path, cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))

    return output_path
