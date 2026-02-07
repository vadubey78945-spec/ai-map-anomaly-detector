import sys
from pathlib import Path

file = Path(__file__).resolve()
root = file.parents[1]

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
    

import streamlit as st
import os
import sys
from pathlib import Path
from backend.wkt_processor import WKTProcessor, create_sample_wkt_files
import geopandas as gpd
from shapely import wkt
import tempfile
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import json

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent / "backend"))


try:
    from backend.wkt_processor import WKTProcessor, create_sample_wkt_files
    from backend.detect import MapQADetector, create_sample_dataset
    from backend.train import train_model, evaluate_model
    from backend.utils import save_uploaded_file, get_image_stats, create_directories
    import backend.model as model_module
except ModuleNotFoundError as e:
    import streamlit as st
    st.error(f"Module Discovery Error: {e}")
    st.info(f"System Path currently looking in: {sys.path[:3]}")
    st.stop()

plt.switch_backend('Agg')

# Page configuration
st.set_page_config(
    page_title="AI Map QA: Missing Line Detection",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .error-card {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .success-card {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .confidence-meter {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Create necessary directories
create_directories()

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        try:
            st.session_state.detector = MapQADetector(model_path)
            st.session_state.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    else:
        # Use default detector (will use untrained model)
        st.session_state.detector = MapQADetector()
        st.session_state.model_loaded = False
        st.warning("No trained model found. Using default model. Train a model first for better accuracy.")
        return True
    
   

menu = None # Define menu variable at the top level so it can be used in all functions

def main():
    """Main application"""
    
    # 1. Page Header (Always visible)
    st.markdown('<h1 class="main-header">🗺️ AI Map QA</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p>Detect missing road/line features in generalized map outputs using AI-powered computer vision.</p>
        <p style='color: #666; font-size: 0.9rem;'>Error Type: <strong>Missing Line Features</strong> | Technology: Deep Learning CNN | Framework: PyTorch</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Sidebar Configuration & Menu Definition
    # We define 'menu' HERE so it exists for the logic below
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/map.png", width=80)
        st.title("Navigation")
        
        menu = st.radio(
            "Select Module",
            ["🏠 Dashboard", 
             "🔍 Detection", 
             "📊 Batch Testing", 
             "🗺️ WKT Processing", 
             "🎓 Training", 
             "📈 Model Metrics", 
             "📄 Reports",
             "⚙️ Settings"]
        )
        st.markdown("---")
        
        # Model status display
        st.subheader("Model Status")
        if st.session_state.model_loaded:
            st.success("✓ Model Loaded")
        else:
            st.warning("⚠ Default Model")
        
        st.markdown("---")
        st.subheader("Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload Model"):
                if load_model():
                    st.success("Model reloaded!")
                    st.rerun()
        
        with col2:
            if st.button("🆕 Create Samples"):
                with st.spinner("Creating sample dataset..."):
                    create_sample_dataset()
                    st.success("Sample dataset created!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style='font-size: 0.8rem; color: #666;'>
        <strong>Instructions:</strong><br>
        1. Upload map screenshots<br>
        2. Train model with labeled data<br>
        3. Run detection on new maps<br>
        4. Download reports
        </div>
        """, unsafe_allow_html=True)

    # 3. Main Content Logic
    # Now that 'menu' is defined, we can safely use it
    if menu == "🏠 Dashboard":
        show_dashboard()
    elif menu == "🔍 Detection":
        show_detection()
    elif menu == "📊 Batch Testing":
        show_batch_testing()
    elif menu == "🗺️ WKT Processing":
        show_wkt_processing()
    elif menu == "🎓 Training":
        show_training()
    elif menu == "📈 Model Metrics":
        show_metrics()
    elif menu == "📄 Reports":
        show_reports()
    elif menu == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Dashboard view"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📊 System Overview</h2>', unsafe_allow_html=True)
        
        # Stats cards
        col1a, col2a, col3a = st.columns(3)
        
        with col1a:
            correct_count = len(os.listdir("dataset/correct")) if os.path.exists("dataset/correct") else 0
            st.metric("Correct Samples", correct_count)
        
        with col2a:
            incorrect_count = len(os.listdir("dataset/incorrect")) if os.path.exists("dataset/incorrect") else 0
            st.metric("Incorrect Samples", incorrect_count)
        
        with col3a:
            model_exists = os.path.exists("models/best_model.pth")
            status = "Trained" if model_exists else "Not Trained"
            st.metric("Model Status", status)
        
        # Recent detections
        st.markdown('<h3 class="sub-header">📈 Recent Activity</h3>', unsafe_allow_html=True)
        
        if st.session_state.detection_results:
            recent_results = st.session_state.detection_results[-5:]  # Last 5 results
            for result in reversed(recent_results):
                display_result_card(result)
        else:
            st.info("No recent detections. Upload an image in the Detection tab to get started.")
        
        # Quick start guide
        st.markdown('<h3 class="sub-header">🚀 Quick Start</h3>', unsafe_allow_html=True)
        
        with st.expander("Step-by-Step Guide", expanded=True):
            st.markdown("""
            1. **Prepare Data**: Collect screenshots of correct and incorrect maps
            2. **Train Model**: Go to Training tab and upload your dataset
            3. **Run Detection**: Upload new map screenshots for AI analysis
            4. **Generate Reports**: Export detection results as CSV/JSON
            
            **For Demo**: Click "Create Samples" in sidebar to generate example data.
            """)
    
    with col2:
        st.markdown('<h3 class="sub-header">🎯 Error Type</h3>', unsafe_allow_html=True)
        
        st.info("""
        **Missing Line Features**
        
        The AI detects when expected road/line geometries disappear or are incomplete after map generalization.
        
        **Common Causes:**
        - Over-aggressive simplification
        - Zoom-level changes
        - Data processing errors
        - Rendering issues
        """)
        
        # Sample images
        st.markdown('<h3 class="sub-header">🖼️ Examples</h3>', unsafe_allow_html=True)
        
        col1b, col2b = st.columns(2)
        
        with col1b:
            st.markdown("**Correct Map**")
            st.image("https://via.placeholder.com/150x100/4CAF50/FFFFFF?text=Complete+Roads", 
                    caption="All line features present")
        
        with col2b:
            st.markdown("**Incorrect Map**")
            st.image("https://via.placeholder.com/150x100/F44336/FFFFFF?text=Missing+Lines", 
                    caption="Missing road segments")
        
        # System info
        st.markdown('<h3 class="sub-header">⚙️ System Info</h3>', unsafe_allow_html=True)
        
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Compute Device", device)
        
        # Load model button
        if not st.session_state.model_loaded:
            if st.button("Load Trained Model", type="primary"):
                if load_model():
                    st.success("Model loaded successfully!")
                    st.rerun()

def show_detection():
    """Enhanced detection with WKT support"""
    st.markdown('<h2 class="sub-header">🔍 Multi-Mode Detection</h2>', unsafe_allow_html=True)
    
    # Detection mode selector
    detection_mode = st.radio(
        "Select Detection Mode:",
        ["🖼️ Image-Based Detection", "🗺️ WKT-Based Detection", "🔄 Compare Original vs Generalized"],
        horizontal=True
    )
    
    if detection_mode == "🖼️ Image-Based Detection":
        # Existing image detection code
        show_image_detection()
    
    elif detection_mode == "🗺️ WKT-Based Detection":
        show_wkt_detection()
    
    elif detection_mode == "🔄 Compare Original vs Generalized":
        show_comparison_detection()

def show_image_detection():
    """Image-based detection"""
    st.markdown("### Upload Map Screenshot for Detection")

    # File uploader for single or multiple images
    uploaded_files = st.file_uploader(
        "Upload map screenshot(s)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload one or more map screenshots for AI analysis"
    )

    if uploaded_files:
        # Display uploaded images
        st.markdown(f"**{len(uploaded_files)}** file(s) uploaded")

        cols = st.columns(min(3, len(uploaded_files)))
        for idx, uploaded_file in enumerate(uploaded_files[:3]):
            with cols[idx % 3]:
                st.image(uploaded_file, use_column_width=True)
                st.caption(uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)

        if len(uploaded_files) > 3:
            st.caption(f"... and {len(uploaded_files) - 3} more file(s)")

        # Detection button
        if st.button("🤖 Run AI Detection", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {len(uploaded_files)} image(s)..."):
                # Ensure detector is loaded
                if st.session_state.detector is None:
                    load_model()

                # Save uploaded files temporarily
                temp_dir = tempfile.mkdtemp()
                image_paths = []

                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    image_paths.append(file_path)

                # Run detection on each image
                for idx, image_path in enumerate(image_paths):
                    st.markdown(f"### Result {idx + 1}: {os.path.basename(image_path)}")

                    try:
                        result = st.session_state.detector.predict(image_path)

                        # Add timestamp to result
                        result['timestamp'] = datetime.now().isoformat()

                        # Display results
                        display_detection_results(result)

                        # Save to session results
                        st.session_state.detection_results.append(result)

                    except Exception as e:
                        st.error(f"Error processing {os.path.basename(image_path)}: {str(e)}")

def show_wkt_detection():
    """WKT-specific detection"""
    st.markdown("### Upload WKT File for Detection")
    
    # File upload
    wkt_file = st.file_uploader(
        "Upload a WKT file",
        type=['wkt', 'txt'],
        help="Upload a WKT file of a generalized map"
    )
    
    if wkt_file:
        # Save and process
        if 'wkt_processor' not in st.session_state:
            st.session_state.wkt_processor = WKTProcessor()
        
        processor = st.session_state.wkt_processor
        
        with st.spinner("Processing WKT file..."):
            # Save WKT
            wkt_path = processor.save_wkt_file(wkt_file, "detection")
            
            # Convert to image
            image_path = processor.render_wkt_to_image(wkt_path)
            
            # Load image for display
            st.image(image_path, caption="Rendered Map from WKT", use_column_width=True)
            
            # Run AI detection on rendered image
            if st.session_state.detector:
                if st.button("🤖 Run AI Detection on Rendered Map", type="primary"):
                    with st.spinner("Analyzing with AI..."):
                        result = st.session_state.detector.predict(image_path)
                        display_detection_results(result)
            else:
                st.warning("AI model not loaded. Please train or load a model first.")

def show_comparison_detection():
    """Compare original vs generalized maps"""
    st.markdown("### Compare Original vs Generalized Maps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Map**")
        original_file = st.file_uploader(
            "Upload original map (WKT or Image)",
            type=['wkt', 'txt', 'png', 'jpg', 'jpeg'],
            key="compare_original"
        )
    
    with col2:
        st.markdown("**Generalized Map**")
        generalized_file = st.file_uploader(
            "Upload generalized map (WKT or Image)",
            type=['wkt', 'txt', 'png', 'jpg', 'jpeg'],
            key="compare_generalized"
        )
    
    if original_file and generalized_file:
        # Process based on file type
        if 'wkt_processor' not in st.session_state:
            st.session_state.wkt_processor = WKTProcessor()
        
        processor = st.session_state.wkt_processor
        
        # Determine file types and process accordingly
        original_ext = original_file.name.split('.')[-1].lower()
        generalized_ext = generalized_file.name.split('.')[-1].lower()
        
        if st.button("🔍 Compare Maps", type="primary"):
            with st.spinner("Comparing maps..."):
                # This would implement sophisticated comparison logic
                st.info("Comparison feature would analyze differences between original and generalized maps")
                # Implement geometric comparison, visual diff, etc.

def display_detection_results(result, show_heatmap=True):
    """Display detection results in a formatted way"""
    
    # Result card
    if result.get("has_error", False):
        st.markdown('<div class="result-card error-card">', unsafe_allow_html=True)
        st.error("⚠ Missing Line Features Detected")
    else:
        st.markdown('<div class="result-card success-card">', unsafe_allow_html=True)
        st.success("✓ No Errors Detected")
    
    # Confidence meter
    confidence = result.get("confidence", 0)
    confidence_color = "#f44336" if result.get("has_error", False) else "#4caf50"
    
    st.markdown(f"""
    <div style='margin: 1rem 0;'>
        <strong>Confidence:</strong> {confidence:.1%}
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {confidence * 100}%; background-color: {confidence_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Status", result.get("status", "Unknown"))
        st.metric("Location", result.get("location", "Unknown"))
    
    with col2:
        st.metric("Error Detected", "Yes" if result.get("has_error", False) else "No")
        if "timestamp" in result:
            st.caption(f"Detected at: {result['timestamp']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("### Visualizations")
    
    if result.get("has_error", False) and show_heatmap:
        # Show comparison
        cols = st.columns(2)
        
        with cols[0]:
            if os.path.exists(result.get("original_path", "")):
                st.image(result["original_path"], caption="Original Map", use_column_width=True)
        
        with cols[1]:
            if os.path.exists(result.get("annotated_path", "")):
                st.image(result["annotated_path"], caption="Annotated Result", use_column_width=True)
        
        # Show heatmap if available
        if result.get("heatmap_path") and os.path.exists(result["heatmap_path"]):
            st.image(result["heatmap_path"], caption="AI Attention Heatmap", use_column_width=True)
            
            st.info("""
            **Heatmap Interpretation:**
            - Red/Yellow areas: High AI attention (likely error locations)
            - Blue areas: Low attention
            - Focus on areas where the AI suspects missing line features
            """)
    elif os.path.exists(result.get("original_path", "")):
        st.image(result["original_path"], caption="Analyzed Map", use_column_width=True)
    
    # Export options
    st.markdown("### Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Save to Report"):
            st.session_state.detection_results.append(result)
            st.success("Result saved to report!")
    
    with col2:
        if st.button("🔄 Detect Another"):
            st.session_state.current_result = None
            st.rerun()

def show_batch_testing():
    """Batch testing view"""
    st.markdown('<h2 class="sub-header">📊 Batch Testing</h2>', unsafe_allow_html=True)
    
    # File uploader for multiple images
    uploaded_files = st.file_uploader(
        "Upload multiple map screenshots",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Select multiple images for batch processing"
    )
    
    if uploaded_files:
        # Save uploaded files
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        st.markdown(f"**{len(uploaded_files)}** files uploaded for batch processing")
        
        # Display uploaded files
        cols = st.columns(4)
        for idx, uploaded_file in enumerate(uploaded_files[:8]):  # Show first 8
            with cols[idx % 4]:
                st.image(uploaded_file, use_column_width=True)
                st.caption(uploaded_file.name[:15] + "..." if len(uploaded_file.name) > 15 else uploaded_file.name)
        
        if len(uploaded_files) > 8:
            st.caption(f"... and {len(uploaded_files) - 8} more files")
        
        # Batch processing options
        st.markdown("### Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.slider("Batch Size", min_value=1, max_value=10, value=4, 
                                  help="Number of images to process at once")
            confidence_thresh = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.05)
        
        with col2:
            generate_report = st.checkbox("Generate report after completion", value=True)
            show_individual = st.checkbox("Show individual results", value=True)
        
        # Process button
        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                # Save all files
                image_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    image_paths.append(file_path)
                
                # Ensure detector is loaded
                if st.session_state.detector is None:
                    load_model()
                
                # Process in batches
                results = []
                progress_bar = st.progress(0)
                
                for i in range(0, len(image_paths), batch_size):
                    batch = image_paths[i:i + batch_size]
                    batch_results = st.session_state.detector.batch_predict(batch)
                    results.extend(batch_results)
                    
                    # Update progress
                    progress = (i + len(batch)) / len(image_paths)
                    progress_bar.progress(progress)
                
                # Store results
                st.session_state.batch_results = results
                st.session_state.detection_results.extend(results)
                
                st.success(f"✅ Processed {len(results)} images!")
                
                # Display summary
                display_batch_summary(results)
                
                # Show individual results if requested
                if show_individual and results:
                    st.markdown("### Individual Results")
                    
                    for result in results:
                        with st.expander(f"{os.path.basename(result.get('image_path', 'Unknown'))} - {result.get('status', 'Unknown')}"):
                            display_result_card(result)
                
                # Generate report if requested
                if generate_report and results:
                    st.markdown("### Report Generation")
                    
                    report_format = st.selectbox("Report Format", ["CSV", "JSON"])
                    
                    if st.button(f"📥 Download {report_format} Report"):
                        report_content = st.session_state.detector.generate_report(results, report_format.lower())
                        
                        if report_format == "CSV":
                            st.download_button(
                                label="Download CSV",
                                data=report_content,
                                file_name=f"map_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.download_button(
                                label="Download JSON",
                                data=report_content,
                                file_name=f"map_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

def show_wkt_processing():
    """WKT file processing view"""
    st.markdown('<h2 class="sub-header">🗺️ WKT File Processing</h2>', unsafe_allow_html=True)
    
    # Initialize processor
    if 'wkt_processor' not in st.session_state:
        st.session_state.wkt_processor = WKTProcessor()
    
    processor = st.session_state.wkt_processor
    
    # Tab interface for different WKT operations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload WKT Files",
        "🔄 Convert to Images",
        "🔍 Geometric Analysis",
        "📊 Comparison Report"
    ])
    
    with tab1:
        st.markdown("### Upload WKT Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Map (Complete)**")
            original_wkt = st.file_uploader(
                "Upload original map WKT file",
                type=['wkt', 'txt'],
                key="original_wkt"
            )
        
        with col2:
            st.markdown("**Generalized Map**")
            generalized_wkt = st.file_uploader(
                "Upload generalized map WKT file",
                type=['wkt', 'txt'],
                key="generalized_wkt"
            )
        
        if original_wkt and generalized_wkt:
            # Save uploaded files
            with st.spinner("Saving WKT files..."):
                original_path = processor.save_wkt_file(original_wkt, "original")
                generalized_path = processor.save_wkt_file(generalized_wkt, "generalized")
                
                st.session_state.original_wkt_path = original_path
                st.session_state.generalized_wkt_path = generalized_path
            
            st.success("✅ WKT files uploaded successfully!")
            
            # Preview WKT content
            with st.expander("Preview WKT Content"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original WKT**")
                    with open(original_path, 'r') as f:
                        content = f.read()
                    st.text_area("", content[:500] + "..." if len(content) > 500 else content, 
                               height=200, disabled=True)
                
                with col2:
                    st.markdown("**Generalized WKT**")
                    with open(generalized_path, 'r') as f:
                        content = f.read()
                    st.text_area("", content[:500] + "..." if len(content) > 500 else content, 
                               height=200, disabled=True)
        
        # Create sample WKT files
        st.markdown("---")
        st.markdown("### Sample Data")
        
        if st.button("🆕 Create Sample WKT Files", help="Generate example WKT files for testing"):
            with st.spinner("Creating sample WKT files..."):
                sample_dir = create_sample_wkt_files()
                st.success(f"✅ Sample WKT files created in {sample_dir}!")
                
                # Display sample files
                st.markdown("**Available Sample Files:**")
                import glob
                sample_files = glob.glob(os.path.join(sample_dir, "*.wkt"))
                for file in sample_files:
                    st.code(os.path.basename(file))
    
    with tab2:
        st.markdown("### Convert WKT to Images")
        
        if 'original_wkt_path' in st.session_state and 'generalized_wkt_path' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Map**")
                if st.button("Render Original to Image", key="render_orig"):
                    with st.spinner("Rendering original map..."):
                        image_path = processor.render_wkt_to_image(
                            st.session_state.original_wkt_path,
                            image_size=(400, 400),
                            line_width=2,
                            line_color='blue'
                        )
                        st.session_state.original_image_path = image_path
                        st.success("✅ Original map rendered!")
            
            with col2:
                st.markdown("**Generalized Map**")
                if st.button("Render Generalized to Image", key="render_gen"):
                    with st.spinner("Rendering generalized map..."):
                        image_path = processor.render_wkt_to_image(
                            st.session_state.generalized_wkt_path,
                            image_size=(400, 400),
                            line_width=2,
                            line_color='red'
                        )
                        st.session_state.generalized_image_path = image_path
                        st.success("✅ Generalized map rendered!")
            
            # Display rendered images
            if 'original_image_path' in st.session_state and 'generalized_image_path' in st.session_state:
                st.markdown("### Rendered Images")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(st.session_state.original_image_path, 
                           caption="Original Map (Rendered)", use_column_width=True)
                
                with col2:
                    st.image(st.session_state.generalized_image_path, 
                           caption="Generalized Map (Rendered)", use_column_width=True)
                
                # Option to use these images for AI detection
                st.markdown("---")
                if st.button("🎯 Use for AI Detection", type="primary"):
                    # Set these as current images for detection tab
                    st.session_state.current_detection_images = {
                        'original': st.session_state.original_image_path,
                        'generalized': st.session_state.generalized_image_path
                    }
                    st.success("✅ Images ready for AI detection! Switch to Detection tab.")
        else:
            st.info("Please upload WKT files in the 'Upload WKT Files' tab first.")
    
    with tab3:
        st.markdown("### Geometric Analysis")
        
        if 'original_wkt_path' in st.session_state and 'generalized_wkt_path' in st.session_state:
            # Analysis options
            tolerance = st.slider(
                "Matching Tolerance", 
                min_value=0.001, 
                max_value=0.1, 
                value=0.01, 
                step=0.001,
                help="Distance threshold for considering lines as matching"
            )
            
            if st.button("🔍 Run Geometric Analysis", type="primary"):
                with st.spinner("Analyzing geometric differences..."):
                    missing_data = processor.detect_missing_lines(
                        st.session_state.original_wkt_path,
                        st.session_state.generalized_wkt_path,
                        tolerance=tolerance
                    )
                    
                    st.session_state.geometric_analysis = missing_data
                
                # Display results
                if missing_data and 'statistics' in missing_data:
                    stats = missing_data['statistics']
                    
                    st.markdown("### Analysis Results")
                    
                    # Metrics cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Original Lines", stats.get('total_original_lines', 0))
                    
                    with col2:
                        st.metric("Generalized Lines", stats.get('total_generalized_lines', 0))
                    
                    with col3:
                        missing_count = stats.get('missing_lines_count', 0)
                        st.metric("Missing Lines", missing_count, 
                                 delta=f"-{stats.get('missing_percentage', 0):.1f}%")
                    
                    with col4:
                        completeness = stats.get('completeness_percentage', 0)
                        st.metric("Completeness", f"{completeness:.1f}%")
                    
                    # Detailed report
                    if missing_count > 0:
                        st.error(f"⚠️ Found {missing_count} missing line features!")
                        
                        with st.expander("View Missing Lines Details"):
                            for idx, line_wkt in enumerate(missing_data.get('missing_lines_wkt', [])[:5]):  # Show first 5
                                st.text(f"Missing Line {idx + 1}: {line_wkt[:100]}...")
                        
                        # Create visualization
                        if st.button("📊 Visualize Missing Lines"):
                            comparison_path, _ = processor.create_comparison_visualization(
                                st.session_state.original_wkt_path,
                                st.session_state.generalized_wkt_path
                            )
                            st.image(comparison_path, caption="Comparison: Original vs Generalized", 
                                   use_column_width=True)
                    else:
                        st.success("✅ No missing lines detected!")
        else:
            st.info("Please upload WKT files in the 'Upload WKT Files' tab first.")
    
    with tab4:
        st.markdown("### Generate Comparison Report")
        
        if 'original_wkt_path' in st.session_state and 'generalized_wkt_path' in st.session_state:
            if st.button("📄 Generate Full Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    # Generate geometric report
                    report = processor.generate_geometric_report(
                        st.session_state.original_wkt_path,
                        st.session_state.generalized_wkt_path
                    )
                    
                    # Create visualization
                    comparison_path, missing_data = processor.create_comparison_visualization(
                        st.session_state.original_wkt_path,
                        st.session_state.generalized_wkt_path
                    )
                    
                    # Display report
                    st.markdown("### Complete Analysis Report")
                    
                    # Summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.json(report['missing_lines_analysis'])
                    
                    with col2:
                        st.json(report['geometric_metrics'])
                    
                    # Visualization
                    st.image(comparison_path, caption="Map Comparison", use_column_width=True)
                    
                    # Download options
                    st.markdown("### Export Report")
                    
                    # Convert to JSON for download
                    report_json = json.dumps(report, indent=2)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=report_json,
                            file_name=f"wkt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        # Also offer CSV with missing lines
                        if missing_data and missing_data.get('missing_lines_count', 0) > 0:
                            import pandas as pd
                            df_data = []
                            for idx, line in enumerate(missing_data.get('missing_lines', [])):
                                df_data.append({
                                    'id': idx,
                                    'length': line.length,
                                    'bounds': str(line.bounds),
                                    'centroid_x': line.centroid.x,
                                    'centroid_y': line.centroid.y
                                })
                            
                            if df_data:
                                df = pd.DataFrame(df_data)
                                csv_data = df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📥 Download Missing Lines CSV",
                                    data=csv_data,
                                    file_name=f"missing_lines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
        else:
            st.info("Please upload WKT files in the 'Upload WKT Files' tab first.")

# Update main() function to include WKT processing
with st.sidebar:
    menu = st.selectbox("Menu", ["🏠 Dashboard", "Settings", "Analytics"])

def display_batch_summary(results):
    """Display summary of batch processing results"""
    if not results:
        return
    
    # Calculate statistics
    total = len(results)
    errors = sum(1 for r in results if r.get("has_error", False))
    correct = total - errors
    error_rate = (errors / total * 100) if total > 0 else 0
    
    # Display summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", total)
    
    with col2:
        st.metric("Errors Found", errors)
    
    with col3:
        st.metric("Correct Maps", correct)
    
    with col4:
        st.metric("Error Rate", f"{error_rate:.1f}%")
    
    # Confidence distribution
    confidences = [r.get("confidence", 0) for r in results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Pie chart
    labels = ['Correct', 'Errors']
    sizes = [correct, errors]
    colors = ['#4CAF50', '#F44336']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Error Distribution')
    
    # Confidence histogram
    ax2.hist(confidences, bins=10, color='#2196F3', edgecolor='black', alpha=0.7)
    ax2.axvline(avg_confidence, color='red', linestyle='--', label=f'Avg: {avg_confidence:.2f}')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def show_training():
    """Model training view"""
    st.markdown('<h2 class="sub-header">🎓 Model Training</h2>', unsafe_allow_html=True)
    
    # Check for existing model
    model_exists = os.path.exists("models/best_model.pth")
    
    if model_exists:
        st.info("✅ A trained model already exists. You can retrain with new data.")
    
    # Training data upload
    st.markdown("### Upload Training Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Correct Maps**")
        correct_files = st.file_uploader(
            "Upload correct map screenshots",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="correct_upload"
        )
        
        if correct_files:
            st.success(f"✅ {len(correct_files)} correct samples ready")
    
    with col2:
        st.markdown("**Incorrect Maps**")
        incorrect_files = st.file_uploader(
            "Upload incorrect map screenshots (with missing lines)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="incorrect_upload"
        )
        
        if incorrect_files:
            st.success(f"✅ {len(incorrect_files)} incorrect samples ready")
    
    # Save uploaded files
    if correct_files or incorrect_files:
        if st.button("💾 Save Uploaded Files"):
            saved_count = 0
            
            # Save correct files
            for uploaded_file in correct_files:
                save_path = save_uploaded_file(uploaded_file, "dataset/correct")
                saved_count += 1
            
            # Save incorrect files
            for uploaded_file in incorrect_files:
                save_path = save_uploaded_file(uploaded_file, "dataset/incorrect")
                saved_count += 1
            
            st.success(f"Saved {saved_count} files to dataset!")
            st.rerun()
    
    # Dataset statistics
    st.markdown("### Dataset Statistics")
    
    correct_count = len(os.listdir("dataset/correct")) if os.path.exists("dataset/correct") else 0
    incorrect_count = len(os.listdir("dataset/incorrect")) if os.path.exists("dataset/incorrect") else 0
    total_count = correct_count + incorrect_count
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", total_count)
    
    with col2:
        st.metric("Correct Samples", correct_count)
    
    with col3:
        st.metric("Incorrect Samples", incorrect_count)
    
    # Training configuration
    st.markdown("### Training Configuration")
    
    with st.form("training_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Architecture",
                ["efficientnet", "simple"],
                help="EfficientNet: Better accuracy, slower. Simple CNN: Faster, less accurate"
            )
            
            epochs = st.slider("Training Epochs", 5, 50, 20, help="Number of training iterations")
            
            batch_size = st.slider("Batch Size", 4, 32, 8, help="Images per training batch")
        
        with col2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                value=0.001,
                help="How fast the model learns"
            )
            
            val_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05, 
                                 help="Percentage of data for validation")
            
            use_gpu = st.checkbox("Use GPU if available", value=True)
        
        # Training button
        train_button = st.form_submit_button(
            "🚀 Start Training",
            type="primary",
            disabled=total_count < 10 or st.session_state.training_in_progress
        )
    
    if total_count < 10:
        st.warning("⚠ Need at least 10 total samples (5 correct + 5 incorrect) to start training.")
    
    # Start training
    if train_button and total_count >= 10:
        st.session_state.training_in_progress = True
        
        with st.spinner("Training model... This may take several minutes."):
            try:
                # Train the model
                trained_model, metrics = train_model(
                    correct_dir="dataset/correct",
                    incorrect_dir="dataset/incorrect",
                    model_type=model_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    val_split=val_split,
                    use_gpu=use_gpu
                )
                
                # Load the trained model
                load_model()
                
                st.session_state.training_in_progress = False
                
                # Show success message
                st.success("✅ Training completed successfully!")
                
                # Display metrics
                display_training_metrics(metrics)
                
            except Exception as e:
                st.session_state.training_in_progress = False
                st.error(f"Training failed: {str(e)}")
    
    elif st.session_state.training_in_progress:
        st.info("⏳ Training in progress... Please wait.")
    
    # Sample dataset creation
    st.markdown("### Quick Start: Sample Dataset")
    
    if st.button("🆕 Create Sample Dataset", help="Generate example training data"):
        with st.spinner("Creating sample dataset..."):
            create_sample_dataset()
            st.success("✅ Sample dataset created in 'sample_dataset' folder!")
            st.info("Move these files to 'dataset/correct' and 'dataset/incorrect' to use for training.")

def display_training_metrics(metrics):
    """Display training metrics"""
    st.markdown("### Training Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Val Accuracy", f"{metrics.get('final_val_accuracy', 0):.1f}%")
    
    with col2:
        st.metric("Best Val Accuracy", f"{metrics.get('best_val_accuracy', 0):.1f}%")
    
    with col3:
        st.metric("Final Train Accuracy", f"{metrics.get('final_train_accuracy', 0):.1f}%")
    
    with col4:
        st.metric("Training Samples", metrics.get('dataset_size', 0))
    
    # Show training plots
    if os.path.exists("models/training_plots.png"):
        st.image("models/training_plots.png", caption="Training Progress", use_column_width=True)

def show_metrics():
    """Model metrics view"""
    st.markdown('<h2 class="sub-header">📈 Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists("models/best_model.pth"):
        st.warning("No trained model found. Train a model first to see metrics.")
        
        if st.button("Go to Training"):
            st.switch_page("app.py?menu=🎓 Training")
        return
    
    # Load metrics
    metrics_path = "models/training_metrics.json"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Display key metrics
        st.markdown("### Model Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Accuracy", f"{metrics.get('best_val_accuracy', 0):.1f}%")
        
        with col2:
            st.metric("Training Samples", metrics.get('dataset_size', 0))
        
        with col3:
            st.metric("Training Epochs", metrics.get('epochs', 0))
        
        with col4:
            st.metric("Batch Size", metrics.get('batch_size', 0))
        
        # Training history
        st.markdown("### Training History")
        
        if 'train_accuracies' in metrics and 'val_accuracies' in metrics:
            # Create accuracy plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            epochs_range = range(1, len(metrics['train_accuracies']) + 1)
            
            # Accuracy plot
            ax1.plot(epochs_range, metrics['train_accuracies'], 'b-', label='Training Accuracy', marker='o')
            ax1.plot(epochs_range, metrics['val_accuracies'], 'r-', label='Validation Accuracy', marker='s')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('Accuracy over Epochs')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss plot
            ax2.plot(epochs_range, metrics['train_losses'], 'b-', label='Training Loss', marker='o')
            ax2.plot(epochs_range, metrics['val_losses'], 'r-', label='Validation Loss', marker='s')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss over Epochs')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Show confusion matrix if exists
        if os.path.exists("models/confusion_matrix.png"):
            st.markdown("### Confusion Matrix")
            st.image("models/confusion_matrix.png", use_column_width=True)
        
        # Model details
        st.markdown("### Model Details")
        
        with st.expander("View Training Configuration"):
            st.json({
                "model_type": "EfficientNet-B0" if metrics.get('model_type') == 'efficientnet' else "Simple CNN",
                "final_val_accuracy": f"{metrics.get('final_val_accuracy', 0):.2f}%",
                "best_val_accuracy": f"{metrics.get('best_val_accuracy', 0):.2f}%",
                "training_date": metrics.get('training_date', 'Unknown'),
                "learning_rate": metrics.get('learning_rate', 0.001),
                "batch_size": metrics.get('batch_size', 8),
                "epochs": metrics.get('epochs', 20)
            })
    else:
        st.info("No detailed metrics found. The model was likely trained outside this session.")
    
    # Current model evaluation
    st.markdown("### Evaluate Current Model")
    
    if st.button("📊 Evaluate on Current Dataset", type="secondary"):
        with st.spinner("Evaluating model..."):
            try:
                # This would require loading the dataset and running evaluation
                # For now, show a placeholder
                st.info("Full evaluation requires reloading the dataset. Use the training interface for evaluation.")
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

def show_reports():
    """Reports view"""
    st.markdown('<h2 class="sub-header">📄 Detection Reports</h2>', unsafe_allow_html=True)
    
    # Check for existing results
    if not st.session_state.detection_results:
        st.info("No detection results yet. Run some detections first.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Detection"):
                st.switch_page("app.py?menu=🔍 Detection")
        with col2:
            if st.button("Go to Batch Testing"):
                st.switch_page("app.py?menu=📊 Batch Testing")
        return
    
    # Display results summary
    total_results = len(st.session_state.detection_results)
    errors_found = sum(1 for r in st.session_state.detection_results if r.get("has_error", False))
    
    st.metric("Total Detections", total_results)
    st.metric("Total Errors Found", errors_found)
    
    # Results table
    st.markdown("### Recent Detections")
    
    # Create dataframe for display
    report_data = []
    for result in st.session_state.detection_results[-20:]:  # Last 20 results
        report_data.append({
            "Image": os.path.basename(result.get("image_path", "Unknown")),
            "Status": result.get("status", "Unknown"),
            "Error": "Yes" if result.get("has_error", False) else "No",
            "Confidence": f"{result.get('confidence', 0):.1%}",
            "Location": result.get("location", "Unknown"),
            "Timestamp": result.get("timestamp", "").split('T')[0]  # Date only
        })
    
    if report_data:
        df = pd.DataFrame(report_data)
        st.dataframe(df, use_container_width=True)
    
    # Report generation
    st.markdown("### Generate Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_format = st.selectbox("Format", ["CSV", "JSON", "PDF"])
    
    with col2:
        include_all = st.checkbox("Include all results", value=True)
        if not include_all:
            result_count = st.slider("Number of results", 1, len(st.session_state.detection_results), 
                                    min(10, len(st.session_state.detection_results)))
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("📊 Generate Report", type="primary")
    
    if generate_btn:
        with st.spinner("Generating report..."):
            # Select results to include
            if include_all:
                results_to_export = st.session_state.detection_results
            else:
                results_to_export = st.session_state.detection_results[-result_count:]
            
            # Generate report
            if st.session_state.detector is None:
                load_model()
            
            if report_format == "PDF":
                # Generate PDF report
                pdf_path = generate_pdf_report(results_to_export)
                
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=f"map_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
            else:
                # Generate CSV or JSON
                report_content = st.session_state.detector.generate_report(
                    results_to_export, 
                    report_format.lower()
                )
                
                if report_format == "CSV":
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=report_content,
                        file_name=f"map_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=report_content,
                        file_name=f"map_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # Clear results
    st.markdown("### Manage Results")
    
    if st.button("🗑️ Clear All Results", type="secondary"):
        st.session_state.detection_results = []
        st.success("All results cleared!")
        st.rerun()

def generate_pdf_report(results):
    """Generate PDF report"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    
    # Create PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"reports/report_{timestamp}.pdf"
    os.makedirs("reports", exist_ok=True)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("Map QA: Missing Line Features Detection Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.25*inch))
    
    # Summary
    total = len(results)
    errors = sum(1 for r in results if r.get("has_error", False))
    error_rate = (errors / total * 100) if total > 0 else 0
    
    summary_text = f"""
    <b>Report Summary:</b><br/>
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
    Total Images Analyzed: {total}<br/>
    Errors Detected: {errors}<br/>
    Error Rate: {error_rate:.1f}%<br/>
    """
    summary = Paragraph(summary_text, styles['Normal'])
    elements.append(summary)
    elements.append(Spacer(1, 0.5*inch))
    
    # Results table
    table_data = [["Image", "Status", "Error", "Confidence", "Location", "Date"]]
    
    for result in results:
        table_data.append([
            os.path.basename(result.get("image_path", "Unknown"))[:20],
            result.get("status", "Unknown"),
            "Yes" if result.get("has_error", False) else "No",
            f"{result.get('confidence', 0):.1%}",
            result.get("location", "Unknown"),
            result.get("timestamp", "").split('T')[0]
        ])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    
    # Build PDF
    doc.build(elements)
    
    return pdf_path

def show_settings():
    """Settings view"""
    st.markdown('<h2 class="sub-header">⚙️ Application Settings</h2>', unsafe_allow_html=True)
    
    # Application settings
    st.markdown("### General Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
        st.checkbox("Show tutorial tips", value=True)
        st.checkbox("Auto-load last model", value=True)
    
    with col2:
        image_size = st.selectbox("Default Image Size", ["224x224", "256x256", "384x384"], index=0)
        confidence_default = st.slider("Default Confidence Threshold", 0.5, 0.95, 0.75, 0.05)
        st.checkbox("Enable batch processing", value=True)
    
    # Model settings
    st.markdown("### Model Settings")
    
    current_model = "EfficientNet-B0" if os.path.exists("models/best_model.pth") else "Default (Simple CNN)"
    st.metric("Current Model", current_model)
    
    if st.button("🔄 Reload Model"):
        if load_model():
            st.success("Model reloaded successfully!")
    
    # Data management
    st.markdown("### Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        correct_count = len(os.listdir("dataset/correct")) if os.path.exists("dataset/correct") else 0
        st.metric("Training Images (Correct)", correct_count)
        
        if st.button("Clear Correct Images", type="secondary"):
            # Implementation would go here
            st.info("This would clear all correct training images")
    
    with col2:
        incorrect_count = len(os.listdir("dataset/incorrect")) if os.path.exists("dataset/incorrect") else 0
        st.metric("Training Images (Incorrect)", incorrect_count)
        
        if st.button("Clear Incorrect Images", type="secondary"):
            # Implementation would go here
            st.info("This would clear all incorrect training images")
    
    # System information
    st.markdown("### System Information")
    
    import torch
    import platform
    import sys
    
    sys_info = {
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CPU Cores": os.cpu_count(),
        "Platform": platform.platform(),
        "Streamlit Version": st.__version__
    }
    
    for key, value in sys_info.items():
        st.text(f"{key}: {value}")
    
    # Save settings
    st.markdown("---")
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved! (Note: This is a demo - settings persistence would be implemented in production)")

def display_result_card(result):
    """Display a single result as a card"""
    card_class = "error-card" if result.get("has_error", False) else "success-card"
    
    st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        image_name = os.path.basename(result.get("image_path", "Unknown"))
        st.markdown(f"**{image_name}**")
        st.markdown(f"Status: **{result.get('status', 'Unknown')}**")
        st.markdown(f"Confidence: **{result.get('confidence', 0):.1%}**")
    
    with col2:
        if result.get("has_error", False):
            st.error("Error")
        else:
            st.success("OK")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize model on startup
    if st.session_state.detector is None:
        load_model()
    
    main()