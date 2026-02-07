"""
WKT Processor for handling vector geometry data
Converts WKT to images for AI analysis and performs geometric validation
"""
import os
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, GeometryCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WKTProcessor:
    """Process WKT files for map QA system"""
    
    def __init__(self, output_dir="vector_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "uploaded_wkt"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rendered_images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "processed_geojson"), exist_ok=True)
    
    def load_wkt_file(self, wkt_file_path):
        """Load and parse WKT file"""
        with open(wkt_file_path, 'r') as f:
            content = f.read()
        
        # Determine if it's a single geometry or multiple
        lines = content.strip().split('\n')
        geometries = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    geom = wkt.loads(line)
                    geometries.append(geom)
                except Exception as e:
                    print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
        
        return geometries
    
    def wkt_to_geojson(self, wkt_file_path, output_path=None):
        """Convert WKT file to GeoJSON format"""
        geometries = self.load_wkt_file(wkt_file_path)
        
        # Create GeoDataFrame
        data = []
        for idx, geom in enumerate(geometries):
            if geom and not geom.is_empty:
                data.append({
                    'id': idx,
                    'geometry': geom,
                    'type': geom.geom_type,
                    'length': geom.length if hasattr(geom, 'length') else 0,
                    'is_valid': geom.is_valid
                })
        
        if not data:
            return None
        
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        if output_path:
            gdf.to_file(output_path, driver='GeoJSON')
        
        return gdf
    
    def render_wkt_to_image(self, wkt_file_path, image_size=(224, 224), 
                           line_width=2, line_color='black', 
                           background_color='white', dpi=100):
        """Render WKT geometries to image for AI processing"""
        
        geometries = self.load_wkt_file(wkt_file_path)
        
        if not geometries:
            raise ValueError("No valid geometries found in WKT file")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(image_size[0]/dpi, image_size[1]/dpi), dpi=dpi)
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
        
        # Get bounds from all geometries
        all_bounds = []
        for geom in geometries:
            if geom and not geom.is_empty:
                all_bounds.append(geom.bounds)
        
        if not all_bounds:
            raise ValueError("No valid bounds from geometries")
        
        # Calculate overall bounds
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        
        # Add padding
        padding_x = (maxx - minx) * 0.1
        padding_y = (maxy - miny) * 0.1
        minx -= padding_x
        maxx += padding_x
        miny -= padding_y
        maxy += padding_y
        
        # Set plot limits
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Plot geometries
        for geom in geometries:
            if geom and not geom.is_empty:
                if geom.geom_type == 'LineString':
                    x, y = geom.xy
                    ax.plot(x, y, color=line_color, linewidth=line_width, solid_capstyle='round')
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        x, y = line.xy
                        ax.plot(x, y, color=line_color, linewidth=line_width, solid_capstyle='round')
                elif geom.geom_type == 'GeometryCollection':
                    for sub_geom in geom.geoms:
                        if sub_geom.geom_type in ['LineString', 'MultiLineString']:
                            if sub_geom.geom_type == 'LineString':
                                x, y = sub_geom.xy
                                ax.plot(x, y, color=line_color, linewidth=line_width, solid_capstyle='round')
                            else:
                                for line in sub_geom.geoms:
                                    x, y = line.xy
                                    ax.plot(x, y, color=line_color, linewidth=line_width, solid_capstyle='round')
        
        # Save image
        filename = os.path.basename(wkt_file_path).replace('.wkt', '.png')
        output_path = os.path.join(self.output_dir, "rendered_images", filename)
        
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        return output_path
    
    def detect_missing_lines(self, original_wkt, generalized_wkt, tolerance=0.01):
        """
        Direct geometric comparison to find missing lines
        
        Parameters:
        -----------
        original_wkt: Path to original map WKT
        generalized_wkt: Path to generalized map WKT
        tolerance: Buffer tolerance for matching
        
        Returns:
        --------
        dict with missing lines and statistics
        """
        # Load both WKT files
        original_geoms = self.load_wkt_file(original_wkt)
        generalized_geoms = self.load_wkt_file(generalized_wkt)
        
        if not original_geoms or not generalized_geoms:
            return {"error": "Could not load WKT files"}
        
        # Flatten all geometries to LineStrings
        def extract_lines(geometries):
            lines = []
            for geom in geometries:
                if geom and not geom.is_empty:
                    if geom.geom_type == 'LineString':
                        lines.append(geom)
                    elif geom.geom_type == 'MultiLineString':
                        lines.extend(list(geom.geoms))
                    elif geom.geom_type == 'GeometryCollection':
                        for sub_geom in geom.geoms:
                            if sub_geom.geom_type == 'LineString':
                                lines.append(sub_geom)
                            elif sub_geom.geom_type == 'MultiLineString':
                                lines.extend(list(sub_geom.geoms))
            return lines
        
        original_lines = extract_lines(original_geoms)
        generalized_lines = extract_lines(generalized_geoms)
        
        # Find missing lines (in original but not in generalized)
        missing_lines = []
        
        for orig_line in original_lines:
            found = False
            for gen_line in generalized_lines:
                # Check if lines are similar (within tolerance)
                if orig_line.distance(gen_line) < tolerance:
                    found = True
                    break
            
            if not found:
                missing_lines.append(orig_line)
        
        # Calculate statistics
        stats = {
            "total_original_lines": len(original_lines),
            "total_generalized_lines": len(generalized_lines),
            "missing_lines_count": len(missing_lines),
            "completeness_percentage": (len(generalized_lines) / len(original_lines) * 100) if original_lines else 0,
            "missing_percentage": (len(missing_lines) / len(original_lines) * 100) if original_lines else 0
        }
        
        # Convert missing lines to WKT for output
        missing_wkt = [line.wkt for line in missing_lines]
        
        return {
            "statistics": stats,
            "missing_lines_wkt": missing_wkt,
            "missing_lines_count": len(missing_lines),
            "missing_lines": missing_lines
        }
    
    def create_comparison_visualization(self, original_wkt, generalized_wkt, output_path=None):
        """Create side-by-side visualization of original vs generalized"""
        
        # Render both to images
        orig_image = self.render_wkt_to_image(original_wkt, image_size=(400, 400))
        gen_image = self.render_wkt_to_image(generalized_wkt, image_size=(400, 400))
        
        # Create comparison figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Load and display original
        img1 = plt.imread(orig_image)
        ax1.imshow(img1)
        ax1.set_title('Original Map', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Load and display generalized
        img2 = plt.imread(gen_image)
        ax2.imshow(img2)
        ax2.set_title('Generalized Map', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Detect and highlight missing lines
        missing_data = self.detect_missing_lines(original_wkt, generalized_wkt)
        
        if missing_data.get('missing_lines_count', 0) > 0:
            # Add annotation
            fig.suptitle(f'Missing Line Features Detected: {missing_data["missing_lines_count"]} lines', 
                        fontsize=16, fontweight='bold', color='red')
            
            # Create third subplot for missing lines
            fig2, ax3 = plt.subplots(figsize=(6, 6))
            
            # Plot only missing lines in red
            for line in missing_data.get('missing_lines', []):
                x, y = line.xy
                ax3.plot(x, y, color='red', linewidth=3, linestyle='--', alpha=0.7)
            
            ax3.set_title('Missing Lines (Highlighted)', fontsize=14, fontweight='bold')
            ax3.axis('off')
            ax3.set_aspect('equal')
            
            # Save missing lines plot
            missing_output = os.path.join(self.output_dir, "rendered_images", "missing_lines.png")
            plt.tight_layout()
            plt.savefig(missing_output, dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        else:
            fig.suptitle('No Missing Lines Detected', fontsize=16, fontweight='bold', color='green')
        
        # Save comparison plot
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, "rendered_images", f"comparison_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return output_path, missing_data
    
    def generate_geometric_report(self, original_wkt, generalized_wkt):
        """Generate detailed geometric analysis report"""
        
        missing_data = self.detect_missing_lines(original_wkt, generalized_wkt)
        
        # Load geometries for additional analysis
        original_geoms = self.load_wkt_file(original_wkt)
        generalized_geoms = self.load_wkt_file(generalized_wkt)
        
        report = {
            "analysis_date": datetime.now().isoformat(),
            "original_file": os.path.basename(original_wkt),
            "generalized_file": os.path.basename(generalized_wkt),
            "missing_lines_analysis": missing_data.get("statistics", {}),
            "geometric_metrics": {
                "total_geometries_original": len(original_geoms),
                "total_geometries_generalized": len(generalized_geoms),
                "geometry_types_original": list(set(g.geom_type for g in original_geoms if g)),
                "geometry_types_generalized": list(set(g.geom_type for g in generalized_geoms if g))
            },
            "missing_lines_details": []
        }
        
        # Add details for each missing line
        for idx, line in enumerate(missing_data.get('missing_lines', [])):
            report["missing_lines_details"].append({
                "id": idx,
                "wkt": line.wkt,
                "length": line.length,
                "bounds": line.bounds,
                "centroid": (line.centroid.x, line.centroid.y)
            })
        
        return report
    
    def save_wkt_file(self, uploaded_file, category="upload"):
        """Save uploaded WKT file"""
        
        save_dir = os.path.join(self.output_dir, "uploaded_wkt", category)
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        filepath = os.path.join(save_dir, filename)
        
        # Save file
        if hasattr(uploaded_file, 'read'):
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            with open(filepath, 'w') as f:
                f.write(content)
        else:
            with open(filepath, 'w') as f:
                f.write(uploaded_file)
        
        return filepath


def create_sample_wkt_files():
    """Create sample WKT files for demonstration"""
    
    sample_dir = "sample_wkt_data"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Sample 1: Complete road network (original)
    original_wkt1 = """LINESTRING (0 0, 10 0)
LINESTRING (0 10, 10 10)
LINESTRING (5 0, 5 10)
LINESTRING (0 5, 10 5)
LINESTRING (2 2, 8 2)
LINESTRING (2 8, 8 8)
LINESTRING (2 2, 2 8)
LINESTRING (8 2, 8 8)"""
    
    # Sample 1: Generalized version (missing some lines)
    generalized_wkt1 = """LINESTRING (0 0, 10 0)
LINESTRING (0 10, 10 10)
LINESTRING (5 0, 5 5)  # Missing the second half
# LINESTRING (0 5, 10 5)  # This line is missing
LINESTRING (2 2, 8 2)
# LINESTRING (2 8, 8 8)  # This line is missing
LINESTRING (2 2, 2 8)
LINESTRING (8 2, 8 8)"""
    
    # Save files
    with open(os.path.join(sample_dir, "original_complete.wkt"), 'w') as f:
        f.write(original_wkt1)
    
    with open(os.path.join(sample_dir, "generalized_incomplete.wkt"), 'w') as f:
        f.write(generalized_wkt1)
    
    # Sample 2: More complex network
    original_wkt2 = """MULTILINESTRING ((0 0, 20 0), (0 5, 20 5), (0 10, 20 10))
LINESTRING (10 0, 10 20)
LINESTRING (5 0, 5 15)
LINESTRING (15 0, 15 15)
LINESTRING (2 3, 18 3)
LINESTRING (2 7, 18 7)
LINESTRING (2 12, 18 12)
LINESTRING (2 17, 18 17)"""
    
    generalized_wkt2 = """MULTILINESTRING ((0 0, 20 0), (0 5, 20 5))
# LINESTRING (0 10, 20 10)  # Missing
LINESTRING (10 0, 10 10)  # Shortened
# LINESTRING (5 0, 5 15)  # Missing
LINESTRING (15 0, 15 15)
LINESTRING (2 3, 18 3)
# LINESTRING (2 7, 18 7)  # Missing
LINESTRING (2 12, 18 12)
# LINESTRING (2 17, 18 17)  # Missing"""
    
    with open(os.path.join(sample_dir, "original_complex.wkt"), 'w') as f:
        f.write(original_wkt2)
    
    with open(os.path.join(sample_dir, "generalized_simplified.wkt"), 'w') as f:
        f.write(generalized_wkt2)
    
    print(f"Sample WKT files created in {sample_dir}")
    return sample_dir