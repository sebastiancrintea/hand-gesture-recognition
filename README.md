# Hand Gesture Recognition
---


A high-performance, real-time hand gesture recognition system built with **MediaPipe**, **PyTorch**, and **ONNX Runtime**.

## Key Features

*   **Real-Time Inference**: Optimized for low latency.
*   **Custom Neural Network**: Lightweight PyTorch model (3-layer MLP).
*   **GPU Acceleration**: Full CUDA support for both training and inference.
*   **ONNX Deployment**: Uses standardized ONNX format for portability and speed.
*   **Confidence Scoring**: Displays prediction probability (e.g., "Like (98%)").
*   **Robust Logging**: Structured logs for easy debugging.

## Installation

This project uses `uv` for dependency management, but standard `pip` works too.

### Prerequisites
*   Python 3.12+ (Recommended `<3.13` for compatibility)
*   NVIDIA GPU (Optional, but recommended for CUDA support)

### Setup
```bash
# 1. Install dependencies
uv sync

# IMPORTANT for GPU Users:
# Ensure you have onnxruntime-gpu installed and compatible NVIDIA drivers.
# The project is configured to look for CUDA 12.4 compatible libraries.
```

## Usage

### 1. Collect Data (`src/collect_data.py`)
Build your own dataset!
```bash
uv run src/collect_data.py
```
*   **Controls**: Press `0-9` to record a "burst" of 100 frames for that class ID.
*   **Tip**: Move your hand around, rotate it, and vary the distance to train a robust model.

### 2. Train Model (`src/train_model.py`)
Train the neural network on your collected data.
```bash
uv run src/train_model.py
```
*   Automatically detects your GPU.
*   Saves PyTorch weights (`.pth`) and exports to ONNX (`.onnx`) in `model/keypoint_classifier/`.

### 3. Run Application (`src/main.py`)
Start the real-time recognizer.
```bash
uv run src/main.py
```
*   **Controls**: Press `q` to quit.

## Configuration

*   **`src/config.py`**: Adjust settings like `FRAME_SKIP`, resolution, and detection confidence.
*   **`model/keypoint_classifier/keypoint_classifier_label.csv`**: Define your gesture names here (Row 0 = ID 0, Row 1 = ID 1, etc.).

## Project Structure

```
gestures/
├── model/                  # Model artifacts and CSV data
├── src/
│   ├── model/              # PyTorch model definition
│   ├── utils/              # HandTracker, Logger, WebCam
│   ├── collect_data.py     # Data collection script
│   ├── train_model.py      # Training script
│   ├── main.py             # Main application entry point
│   └── config.py           # Configuration settings
├── pyproject.toml          # Dependencies
└── README.md
```
