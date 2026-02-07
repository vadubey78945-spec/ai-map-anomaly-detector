import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import os

from .model import load_model, get_model
from .utils import preprocess_image, generate_heatmap, draw_bounding_box


class MapQADetector:
    """Detector for missing line features in maps"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = torch.device(device)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path, device=self.device)
        else:
            print("Using default model")
            self.model = get_model("efficientnet")
            self.model.eval()
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_path: str) -> Dict:
        """Make prediction on a single image"""
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path)
            image_tensor = torch.from_numpy(image_tensor).to(self.device)
            
            # Load original image for visualization
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get class labels
            class_labels = ["Correct", "Missing Line Features"]
            status = class_labels[predicted_class]
            
            # Generate heatmap if error detected
            heatmap_image = None
            if predicted_class == 1:  # Error detected
                # Get attention map (Grad-CAM)
                self.model.zero_grad()
                outputs = self.model(image_tensor)
                loss = outputs[0, 1]  # Focus on error class
                loss.backward()
                
                attention_map = self.model.get_attention_map(target_class=1)
                if attention_map is not None:
                    heatmap_image = generate_heatmap(original_image, attention_map, alpha=0.5)
            
            # Generate bounding box (simulated for demo - in production, use object detection)
            bounding_box = None
            if predicted_class == 1:
                # For demo, create a bounding box in the region with highest attention
                if attention_map is not None:
                    # Find region with highest attention
                    attention_norm = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX)
                    _, thresholded = cv2.threshold(attention_norm.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
                    
                    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Use largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Scale to original image size
                        orig_h, orig_w = original_image.shape[:2]
                        scale_x = orig_w / 224
                        scale_y = orig_h / 224
                        
                        x1 = int(x * scale_x)
                        y1 = int(y * scale_y)
                        x2 = int((x + w) * scale_x)
                        y2 = int((y + h) * scale_y)
                        
                        bounding_box = (x1, y1, x2, y2)
                        
                        # Draw bounding box on original image
                        annotated_image = draw_bounding_box(original_image, bounding_box, 
                                                           f"Missing Line ({confidence:.1%})")
                    else:
                        annotated_image = original_image
                else:
                    annotated_image = original_image
            else:
                annotated_image = original_image
            
            # Determine location quadrant
            location = self._get_location_quadrant(bounding_box, original_image.shape if bounding_box else None)
            
            # Prepare result
            result = {
                "image_path": image_path,
                "status": status,
                "confidence": confidence,
                "predicted_class": predicted_class,
                "location": location,
                "timestamp": datetime.now().isoformat(),
                "has_error": predicted_class == 1,
                "heatmap_available": heatmap_image is not None,
                "bounding_box": bounding_box
            }
            
            # Save visualization images
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save original
            original_path = os.path.join(output_dir, f"{base_name}_original.png")
            cv2.imwrite(original_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
            
            # Save annotated
            annotated_path = os.path.join(output_dir, f"{base_name}_annotated.png")
            cv2.imwrite(annotated_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            # Save heatmap if available
            if heatmap_image is not None:
                heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
                cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR))
                result["heatmap_path"] = heatmap_path
            
            result["original_path"] = original_path
            result["annotated_path"] = annotated_path
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "Error",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """Make predictions on multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results
    
    def _get_location_quadrant(self, bounding_box: Optional[Tuple], image_shape: Optional[Tuple]) -> str:
        """Determine location quadrant of the error"""
        if not bounding_box or not image_shape:
            return "Unknown"
        
        x1, y1, x2, y2 = bounding_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        img_h, img_w = image_shape[:2]
        
        if center_x < img_w / 3:
            horizontal = "Left"
        elif center_x < 2 * img_w / 3:
            horizontal = "Center"
        else:
            horizontal = "Right"
        
        if center_y < img_h / 3:
            vertical = "Top"
        elif center_y < 2 * img_h / 3:
            vertical = "Middle"
        else:
            vertical = "Bottom"
        
        return f"{vertical}-{horizontal}"
    
    def generate_report(self, results: List[Dict], format: str = "csv") -> str:
        """Generate report from detection results"""
        if format == "csv":
            return self._generate_csv_report(results)
        elif format == "json":
            return self._generate_json_report(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_csv_report(self, results: List[Dict]) -> str:
        """Generate CSV report"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Image Name", "Error Detected", "Error Type", 
            "Confidence", "Location", "Timestamp", "Image Path"
        ])
        
        # Write data
        for result in results:
            writer.writerow([
                os.path.basename(result.get("image_path", "")),
                "Yes" if result.get("has_error", False) else "No",
                result.get("status", "Unknown"),
                f"{result.get('confidence', 0):.2%}",
                result.get("location", "Unknown"),
                result.get("timestamp", ""),
                result.get("image_path", "")
            ])
        
        return output.getvalue()
    
    def _generate_json_report(self, results: List[Dict]) -> str:
        """Generate JSON report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_images": len(results),
            "errors_detected": sum(1 for r in results if r.get("has_error", False)),
            "detections": results
        }
        return json.dumps(report, indent=2)


def create_sample_dataset():
    """Create sample dataset for demonstration"""
    import shutil
    
    sample_dir = "sample_dataset"
    os.makedirs(os.path.join(sample_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(sample_dir, "incorrect"), exist_ok=True)
    
    # Create sample images using matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Sample 1: Correct map (with complete road network)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw complete road network
    roads = [
        [(1, 1), (9, 1)],  # Horizontal main road
        [(1, 9), (9, 9)],  # Another horizontal road
        [(5, 1), (5, 9)],  # Vertical main road
        [(2, 3), (8, 3)],  # Secondary horizontal
        [(2, 7), (8, 7)],  # Another secondary
        [(3, 2), (3, 8)],  # Secondary vertical
        [(7, 2), (7, 8)],  # Another secondary
    ]
    
    for road in roads:
        x1, y1 = road[0]
        x2, y2 = road[1]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=3)
        # Add dashed center line
        ax.plot([x1, x2], [y1, y2], 'w--', linewidth=1, alpha=0.7)
    
    # Add some intersections
    intersections = [(5, 5), (3, 3), (7, 7), (3, 7), (7, 3)]
    for x, y in intersections:
        ax.plot(x, y, 'ko', markersize=10)
    
    plt.title("Complete Map: All Roads Present", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "correct", "sample_correct_1.png"), dpi=100)
    plt.close()
    
    # Sample 2: Another correct map
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw different road pattern
    roads = [
        [(2, 2), (8, 2)],
        [(2, 8), (8, 8)],
        [(2, 2), (2, 8)],
        [(8, 2), (8, 8)],
        [(5, 2), (5, 8)],
        [(2, 5), (8, 5)],
    ]
    
    for road in roads:
        x1, y1 = road[0]
        x2, y2 = road[1]
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=3)
        ax.plot([x1, x2], [y1, y2], 'w--', linewidth=1, alpha=0.7)
    
    plt.title("Grid Map: Complete Network", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "correct", "sample_correct_2.png"), dpi=100)
    plt.close()
    
    # Sample 3: Incorrect map (missing line features)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw incomplete road network (missing some segments)
    roads = [
        [(1, 1), (5, 1)],  # Only half of the horizontal road
        # Missing: [(5, 1), (9, 1)]
        [(1, 9), (9, 9)],  # This one is complete
        [(5, 1), (5, 5)],  # Only half of vertical road
        # Missing: [(5, 5), (5, 9)]
        [(2, 3), (6, 3)],  # Partial secondary
        # Missing connection from (6, 3) to (8, 3)
    ]
    
    for road in roads:
        x1, y1 = road[0]
        x2, y2 = road[1]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=3)
        ax.plot([x1, x2], [y1, y2], 'w--', linewidth=1, alpha=0.7)
    
    # Highlight missing segments with red dashed lines
    missing = [
        [(5, 1), (9, 1)],
        [(5, 5), (5, 9)],
        [(6, 3), (8, 3)]
    ]
    
    for segment in missing:
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        ax.plot([x1, x2], [y1, y2], 'r--', linewidth=2, alpha=0.5)
        # Add missing indicator
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, "✗", color='red', fontsize=20, 
                ha='center', va='center', alpha=0.7)
    
    plt.title("Incomplete Map: Missing Road Segments", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "incorrect", "sample_incorrect_1.png"), dpi=100)
    plt.close()
    
    # Sample 4: Another incorrect map
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw map with disconnected roads
    roads = [
        [(1, 1), (4, 1)],
        [(6, 1), (9, 1)],  # Gap in the middle
        [(1, 5), (9, 5)],  # Complete horizontal
        [(5, 1), (5, 4)],  # Doesn't connect to horizontal
        # Missing connecting segment from (5, 4) to (5, 5)
        [(5, 6), (5, 9)],  # Disconnected vertical
    ]
    
    for road in roads:
        x1, y1 = road[0]
        x2, y2 = road[1]
        ax.plot([x1, x2], [y1, y2], 'g-', linewidth=3)
        ax.plot([x1, x2], [y1, y2], 'w--', linewidth=1, alpha=0.7)
    
    # Add red X marks at missing connections
    missing_points = [(4.5, 1), (5, 4.5), (5, 5.5)]
    for x, y in missing_points:
        ax.text(x, y, "✗", color='red', fontsize=24, 
                ha='center', va='center', alpha=0.8)
        ax.plot(x, y, 'ro', markersize=10, alpha=0.5)
    
    plt.title("Disconnected Road Network", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "incorrect", "sample_incorrect_2.png"), dpi=100)
    plt.close()
    
    print(f"Sample dataset created in {sample_dir}")
    return sample_dir