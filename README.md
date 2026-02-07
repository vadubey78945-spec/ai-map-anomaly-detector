# AI Map QA: Missing Line Feature Detection

A web-based application for AI-driven quality assurance on cartographic map outputs using screenshot-based anomaly detection.

## 🎯 Features

- **AI-Powered Detection**: Deep learning model to identify missing line features in maps
- **Interactive Web Interface**: Streamlit-based frontend with drag-and-drop upload
- **Training Module**: Upload your own correct/incorrect examples to train custom models
- **Batch Processing**: Test multiple images simultaneously
- **Visual Results**: Heatmaps, bounding boxes, and side-by-side comparisons
- **Report Generation**: Export results as CSV, JSON, or PDF
- **Sample Dataset**: Includes demo images for immediate testing

## 🏗️ Architecture

- **Frontend**: Streamlit (Python-based web UI)
- **Backend**: FastAPI + PyTorch
- **AI Model**: EfficientNet-B0 with transfer learning (or Simple CNN for smaller datasets)
- **Computer Vision**: OpenCV for image processing
- **Visualization**: Grad-CAM heatmaps, bounding box annotations

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for training)
- 1GB+ free disk space
- Optional: NVIDIA GPU with CUDA support for faster training

## 🚀 Installation

1. **Clone or download the project**

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate