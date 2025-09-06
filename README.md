```python
%%writefile README.md
# Intelligent Traffic and Safety Monitoring System

A real-time computer vision system that detects vehicles, monitors driver drowsiness, and identifies traffic anomalies using YOLO and OpenCV.

##  Features

- **Vehicle Detection & Counting**: Identifies cars, buses, trucks, and motorcycles using YOLO
- **Drowsiness Detection**: Simulates driver fatigue detection for visible drivers
- **Anomaly Detection**: Flags slow-moving vehicles that might indicate hazards
- **Real-time Processing**: Works with both video files and webcam feed
- **Detailed Reporting**: Generates JSON reports and annotated video output

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy

##  Installation

1. **Install dependencies**
   ```bash
   pip install opencv-python numpy
```
