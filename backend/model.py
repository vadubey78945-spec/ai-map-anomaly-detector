import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class MapQAModel(nn.Module):
    """CNN model for detecting missing line features in maps"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(MapQAModel, self).__init__()
        
        # Use EfficientNet-B0 as base
        self.base_model = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace the classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()  # Remove original classifier
        
        # Custom classifier for binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        # Feature map for Grad-CAM
        self.feature_map = None
        self.gradients = None
        
        # Register hooks for Grad-CAM
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for feature map and gradients"""
        def save_feature_map(module, input, output):
            self.feature_map = output
        
        def save_gradients(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        # Hook to the last convolutional layer
        target_layer = self.base_model.features[-1]
        target_layer.register_forward_hook(save_feature_map)
        target_layer.register_backward_hook(save_gradients)
    
    def forward(self, x):
        # Extract features
        features = self.base_model(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_attention_map(self, target_class: int = 1):
        """Generate Grad-CAM attention map"""
        if self.feature_map is None or self.gradients is None:
            return None
        
        # Global average pooling of gradients
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        
        # Weighted combination of feature maps
        attention_map = torch.sum(self.feature_map * weights, dim=1, keepdim=True)
        attention_map = F.relu(attention_map)
        
        # Normalize
        attention_map = F.interpolate(attention_map, size=(224, 224), mode='bilinear', align_corners=False)
        attention_map = attention_map - attention_map.min()
        attention_map = attention_map / (attention_map.max() + 1e-8)
        
        return attention_map.squeeze().cpu().detach().numpy()


class SimpleCNN(nn.Module):
    """Simpler CNN model for smaller datasets"""
    
    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()
        
        # Feature extractor
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


def get_model(model_type: str = "efficientnet", num_classes: int = 2):
    """Factory function to get model"""
    if model_type == "efficientnet":
        return MapQAModel(num_classes=num_classes)
    elif model_type == "simple":
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_model(model, path: str):
    """Save model checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'efficientnet' if isinstance(model, MapQAModel) else 'simple'
    }, path)


def load_model(path: str, device: str = 'cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    if checkpoint.get('model_type') == 'efficientnet':
        model = MapQAModel()
    else:
        model = SimpleCNN()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model