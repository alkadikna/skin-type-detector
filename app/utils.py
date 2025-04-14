import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from config import MODEL_SAVE_PATH

def ensure_dir_exists(path):
    """Ensure directory exists at the given path"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(model, name, path=MODEL_SAVE_PATH):
    """Save model to disk"""
    ensure_dir_exists(path)
    model_path = os.path.join(path, f"{name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model, name, path=MODEL_SAVE_PATH):
    """Load model from disk"""
    model_path = os.path.join(path, f"{name}.pth")
    model.load_state_dict(torch.load(model_path))
    return model

def display_image(img_array, title=None):
    """Display an image from a numpy array"""
    plt.figure(figsize=(6, 6))
    
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
    
    plt.imshow(img_array)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()