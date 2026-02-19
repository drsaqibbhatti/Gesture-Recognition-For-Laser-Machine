# Gesture Recognition for Laser Machine Safety (ToF + Time-Series)

Two-stage safety system for laser machine control:
1) **Human detection** (safety gate)
2) **3D time-series gesture classification** using **Time-of-Flight (ToF)** sensor data

Designed for **real-time / low-latency** safety control and robust operation in production environments.

> Project page (demo + overview):  
> https://drsaqibbhatti.com/projects/gesture-recognition-laser.html

---

## Highlights
- **ToF-based gesture recognition** (depth/time-series)
- **Two-stage pipeline** (human detection → gesture classification)
- **6 gesture classes** (safety/control gestures)
- Real-time inference with latency-focused implementation
- Includes **PyTorch training** and **ONNX export / ONNXRuntime inference**

---

## Pipeline (High Level)
**Stage A — Human Detection**
- Detect presence/valid ROI for gesture recognition
- Reduces false positives in safety-critical workflow

**Stage B — Gesture Classification (Time-Series)**
- Uses a short clip of ToF frames (time window)
- Predicts gesture class from a sequence instead of a single frame

---

## Tech Stack
- **Python**
- **PyTorch**
- **OpenCV**
- **ONNX / ONNXRuntime** (deployment)
- **ToF Camera** (sensor input)

---

## Repository Contents (Key Files)
- `train.py` — model training (paths/config may need editing)
- `eval.py` — evaluation utilities (if used in your workflow)
- `inference.py` — PyTorch inference (webcam / video style loop)
- `Export_Onnx.py` — export trained model to ONNX
- `InferenceV2_Onnx.py` — ONNXRuntime inference (ToF time-series pipeline)
- `Dataset/` — dataset utilities / loaders
- `Model/` — network definitions

