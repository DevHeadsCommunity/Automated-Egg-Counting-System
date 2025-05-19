# Automated Egg Counting System

This project implements an Automated Egg Counting System that detects and counts eggs in real time on a conveyor belt.  
It uses **YOLOv5** for object detection and a **Raspberry Pi** for on-device inference, giving egg-packing facilities an accurate, low-overhead way to track production.

---

## 1 · Project Overview
- **Goal:** Reduce human error and speed up counting in egg-packaging lines.  
- **Approach:** Train a custom YOLOv5 model, deploy it on a Raspberry Pi, and log counts as eggs pass a checkpoint on the belt.

---

## 2 · Key Features
| Feature | Description |
|---------|-------------|
| **Real-time counting** | Counts eggs as they cross the camera’s checkpoint. |
| **High accuracy** | YOLOv5’s detection keeps false positives/negatives low. |
| **Data logging** | Live on-screen count plus CSV logging for batch analysis. |

---

## 3 · Hardware
- Raspberry Pi 3 or 4  
- Pi Camera Module (or USB webcam)  
- Consistent LED lighting over the conveyor  

---

## 4 · Software
| Layer | Requirement |
|-------|-------------|
| **Model training** | Google Colab + YOLOv5 notebook |
| **Runtime** | Raspberry Pi OS, Python 3.8+ |
| **Python libs** | `torch`, `torchvision`, `opencv-python`, `numpy`, `pandas`, `matplotlib` |

---

## 5 · Setup & Workflow

### 5.1 Train the Model
1. Label egg images (e.g., Roboflow or CVAT).  
2. In Google Colab, train YOLOv5 and export `best.pt`.

### 5.2 Prepare the Raspberry Pi
```bash
# Clone your repo
git clone https://github.com/<your-org>/egg-counter.git
cd egg-counter

# Install Python deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
