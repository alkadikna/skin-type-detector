import os
import sys
import glob
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

index_label = {
    0: "combination",
    1: "dry",
    2: "normal",
    3: "oily"
}

IMG_SIZE = 224
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(model, img_path):
    image = Image.open(img_path).convert("RGB")
    image_tensor = inference_transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

if __name__ == "__main__":
    model_files = glob.glob(os.path.join("app", "models", "facial_model_*.pkl"))
    if not model_files:
        print("No model files found. Please train the model first.")
        sys.exit(1)

    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model_path = model_files[0]
    print(f"Loading model from: {latest_model_path}")

    with open(latest_model_path, "rb") as f:
        model = pickle.load(f)

    img_path = input("Enter the path to your image: ").strip()
    if not os.path.exists(img_path):
        print("Image file not found. Please check your path.")
        sys.exit(1)

    predicted_class_idx = predict_image(model, img_path)
    predicted_class_name = index_label.get(predicted_class_idx, "Unknown")
    print(f"Prediction: {predicted_class_name}")