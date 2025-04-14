import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
from config import DEVICE, INDEX_LABEL, val_transform

def predict_image(model, img_array):
    """Predict class for a single image"""
    img = val_transform(img_array)
    img = img.unsqueeze(0)
    model.eval()
    
    with torch.no_grad():
        img = img.to(DEVICE)
        out = model(img)
        return out.argmax(1).item()

def evaluate_model(model, test_faces, test_labels):
    """Evaluate model on test data and return metrics"""
    # Generate predictions
    pred = []
    truth = []
    
    for i in range(len(test_faces)):
        pred.append(predict_image(model, test_faces[i]))
        truth.append(test_labels[i])
    
    # Calculate metrics
    score = accuracy_score(truth, pred)
    report = classification_report(truth, pred)
    cm = confusion_matrix(truth, pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"Confusion Matrix - Accuracy: {round(score * 100, 2)}%")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # Print classification report
    print("Classification Report:")
    print(report)
    
    return {
        'accuracy': score,
        'predictions': pred,
        'true_labels': truth,
        'confusion_matrix': cm,
        'classification_report': report
    }

def visualize_predictions(model, test_faces, test_labels, n_rows=5, n_cols=5):
    """Visualize predictions on test data"""
    # Get predictions
    pred = []
    for i in range(min(n_rows * n_cols, len(test_faces))):
        pred.append(predict_image(model, test_faces[i]))
    
    # Plot predictions vs truth
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 12))
    index = 0
    
    for i in range(n_rows):
        for j in range(n_cols):
            if index < len(test_faces):
                img = Image.fromarray(np.uint8(test_faces[index] * 255))
                axes[i][j].imshow(img)
                axes[i][j].set_title(f"Pred: {INDEX_LABEL[pred[index]]}\nTrue: {INDEX_LABEL[test_labels[index]]}")
                axes[i][j].axis("off")
                index += 1
    
    plt.tight_layout()
    plt.show()