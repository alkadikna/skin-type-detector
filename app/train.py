import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from config import EPOCHS, LR, STEP, GAMMA, DEVICE, MODEL_SAVE_PATH

def train_model(model, train_dl, val_dl, save_path=MODEL_SAVE_PATH):
    """Train the model and return training history and best model"""
    
    # Make sure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
    
    # Track best model and metrics
    best_model = deepcopy(model)
    best_acc = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in tqdm(train_dl, desc=f"Epoch {epoch} Training", leave=False):
            optimizer.zero_grad()
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            correct += (outputs.argmax(1) == target).sum().item()
            total += data.size(0)
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_dl, desc=f"Epoch {epoch} Validation", leave=False):
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                outputs = model(data)
                loss = criterion(outputs, target)
                
                running_loss += loss.item() * data.size(0)
                correct += (outputs.argmax(1) == target).sum().item()
                total += data.size(0)
        
        epoch_val_loss = running_loss / total
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model = deepcopy(model)
            # Save the checkpoint
            model_path = os.path.join(save_path, f"best_model_epoch_{epoch}.pth")
            best_model.save(model_path)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Epoch {epoch} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(save_path, "final_model.pth")
    model.save(final_model_path)
    
    # Save best model
    best_model_path = os.path.join(save_path, "best_model.pth")
    best_model.save(best_model_path)
    
    return best_model, history

def plot_training_history(history):
    """Plot training and validation loss and accuracy"""
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'])
    axes[0].plot(epochs, history['val_loss'])
    axes[0].legend(["Training", "Validation"])
    axes[0].set_title("Loss log")
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'])
    axes[1].plot(epochs, history['val_acc'])
    axes[1].legend(["Training", "Validation"])
    axes[1].set_title("Accuracy log")
    
    plt.tight_layout()
    plt.show()