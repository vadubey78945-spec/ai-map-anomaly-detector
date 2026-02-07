import os
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime
from typing import List, Tuple, Optional
import hashlib

def create_directories():
    """Create necessary directories for the application"""
    os.makedirs("dataset/correct", exist_ok=True)
    os.makedirs("dataset/incorrect", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)

def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to target dimensions"""
    return cv2.resize(image, target_size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1]"""
    return image.astype(np.float32) / 255.0

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess image for model inference"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = resize_image(image, target_size)
    
    # Normalize
    image = normalize_image(image)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Convert to channel-first format for PyTorch
    image = np.transpose(image, (0, 3, 1, 2))
    
    return image

def save_uploaded_file(uploaded_file, save_dir: str) -> str:
    """Save uploaded file to directory and return path"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
    ext = uploaded_file.name.split('.')[-1]
    filename = f"{timestamp}_{file_hash}.{ext}"
    
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return filepath

def generate_heatmap(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Generate heatmap overlay on image"""
    # Normalize mask
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    # Convert image to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Blend heatmap with image
    blended = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)
    
    # Convert back to RGB
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    
    return blended_rgb

def draw_bounding_box(image: np.ndarray, box: Tuple[int, int, int, int], 
                     label: str = "Missing Line", color: Tuple[int, int, int] = (255, 0, 0),
                     thickness: int = 2) -> np.ndarray:
    """Draw bounding box on image"""
    x1, y1, x2, y2 = box
    img_copy = image.copy()
    
    # Draw rectangle
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    
    label_bg_top_left = (x1, y1 - label_size[1] - 10)
    label_bg_bottom_right = (x1 + label_size[0] + 10, y1)
    
    cv2.rectangle(img_copy, label_bg_top_left, label_bg_bottom_right, color, -1)
    
    # Draw label text
    cv2.putText(img_copy, label, (x1 + 5, y1 - 5), font, font_scale, 
                (255, 255, 255), thickness)
    
    return img_copy

def get_image_stats(image_path: str) -> dict:
    """Get statistics about an image"""
    image = cv2.imread(image_path)
    if image is None:
        return {}
    
    return {
        "dimensions": f"{image.shape[1]}x{image.shape[0]}",
        "channels": image.shape[2] if len(image.shape) > 2 else 1,
        "size_kb": os.path.getsize(image_path) / 1024
    }