import os
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from tqdm.auto import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

FILEPATH = "app/data"
faces = np.load(f"{FILEPATH}/preprocessed_faces.npy")
labels = np.load(f"{FILEPATH}/preprocessed_labels.npy")

EPOCHS = 75
BATCH = 32
OUT_CLASSES = 4
IMG_SIZE = 224
device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomVerticalFlip(0.6),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

faces_train, faces_temp, labels_train, labels_temp = train_test_split(
    faces, labels, test_size=0.2, random_state=42
)
faces_val, faces_test, labels_val, labels_test = train_test_split(
    faces_temp, labels_temp, test_size=0.5, random_state=42
)

class NPYDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_ds = NPYDataset(faces_train, labels_train, train_transform)
val_ds = NPYDataset(faces_val, labels_val, transform)
test_ds = NPYDataset(faces_test, labels_test, transform)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

effnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
num_ftrs = effnet.classifier[1].in_features
effnet.classifier[1] = nn.Linear(num_ftrs, OUT_CLASSES)

def train_model(model, epochs, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.079603)
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            correct += (outputs.argmax(1) == target).sum().item()
            total += data.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_running, val_correct, val_tot = 0.0, 0, 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_running += loss.item() * data.size(0)
                val_correct += (outputs.argmax(1) == target).sum().item()
                val_tot += data.size(0)

        val_loss = val_running / val_tot
        val_acc = val_correct / val_tot

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Epoch {epoch}/{epochs} => "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.savefig("artifacts/training_artifact.png")
    plt.close()

    return model

if __name__ == "__main__":
    model = deepcopy(effnet).to(device)
    trained_model = train_model(model, EPOCHS, train_dl, val_dl)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/facial_model_{current_time}.pkl"

    print("Training complete. Saving the model as:", model_filename)
    with open(model_filename, "wb") as f:
        pickle.dump(trained_model, f)

    print("Model has been saved successfully.")