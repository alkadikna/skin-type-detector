import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from config import OUT_CLASSES, DEVICE

class ResNetModel(nn.Module):
    def __init__(self, num_classes=OUT_CLASSES, pretrained=True):
        super(ResNetModel, self).__init__()
        
        # Load pretrained ResNet model
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = resnet50(weights=None)
        
        # Modify the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def save(self, path):
        """Save model state dictionary to path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load model state dictionary from path"""
        self.load_state_dict(torch.load(path, map_location=DEVICE))
        self.to(DEVICE)
        return self