import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from config import BATCH, IMG_SIZE, FACES_PATH, LABELS_PATH, LABEL_INDEX

# Data transformations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomVerticalFlip(0.6),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

def create_df(base):
    dd = {"images": [], "labels": []}
    for i in os.listdir(base):
        label_dir = os.path.join(base, i)
        for j in os.listdir(label_dir):
            img_path = os.path.join(label_dir, j)
            dd["images"].append(img_path)
            dd["labels"].append(LABEL_INDEX[i])
    return pd.DataFrame(dd)

def load_and_split_data():
    """Load data and split into train, val, and test sets"""
    faces = np.load(FACES_PATH)
    labels = np.load(LABELS_PATH)
    
    print("Faces shape:", faces.shape)
    print("Labels shape:", labels.shape)
    
    # Split data
    faces_train, faces_temp, labels_train, labels_temp = train_test_split(
        faces, labels, test_size=0.2, random_state=42
    )
    faces_val, faces_test, labels_val, labels_test = train_test_split(
        faces_temp, labels_temp, test_size=0.5, random_state=42
    )
    
    # Create datasets
    train_ds = NPYDataset(faces_train, labels_train, train_transform)
    val_ds = NPYDataset(faces_val, labels_val, val_transform)
    test_ds = NPYDataset(faces_test, labels_test, val_transform)
    
    # Create dataloaders
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
    
    return {
        'dataloaders': (train_dl, val_dl, test_dl),
        'datasets': (train_ds, val_ds, test_ds),
        'data': {
            'train': (faces_train, labels_train),
            'val': (faces_val, labels_val),
            'test': (faces_test, labels_test)
        }
    }

def visualize_samples(faces, labels, n_samples=12):
    """Visualize sample images with their labels"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    for i in range(min(n_samples, len(faces))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(faces[i])
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.show()