# Road Object Detection for Autonomous Driving

![Python](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

This repository presents the results of my master's thesis in Artificial Intelligence, focused on real-time object detection for autonomous vehicles. The project compares YOLOv8n and RT-DETR models trained on the BDD100K dataset (road object detection).

## Objective

Train and compare lightweight convolutional (YOLOv8n) and transformer-based (RT-DETR-R101-VD) object detection models on 8 classes from the BDD100K dataset.

## Object Classes

The original label set from the BDD100K dataset contained 10 classes. For this project, it was reduced to 8 classes with the following modifications:
- The **Train** class was removed due to its rare occurrence and low relevance to the task.
- The **Motorcycle** and **Bicycle** classes were combined into a single class called **Two-wheeler**, as they share similar visual features and behavior.

The final set of 8 classes used in this project is:
1. Passenger car  
2. Traffic sign  
3. Traffic light  
4. Pedestrian  
5. Truck  
6. Bus  
7. Two-wheeler (bicycle or motorcycle)  
8. Rider on two-wheeler

This reduction helps improve training efficiency and model performance by minimizing class imbalance and category overlap.

## Repository Structure

| File | Description |
|------|-------------|
| [preprocessing.ipynb](preprocessing.ipynb) | Cleans BDD100K annotations, applies balancing techniques, and prepares formats for YOLOv8 and RT-DETR training |
| [yolov8_training.ipynb](yolov8_training.ipynb) | Fine-tunes YOLOv8n on BDD100K using Ultralytics interface. Tracks training metrics (mAP, recall, precision) and saves model checkpoints |
| [rtdetr_training.ipynb](rtdetr_training.ipynb) | Fine-tuning RT-DETR using Hugging Face and PyTorch and saves model checkpoints |
| [yolov8_inference.ipynb](yolov8_inference.ipynb) | Inference and visualization for YOLOv8 |
| [rtdetr_inference.ipynb](rtdetr_inference.ipynb) | Inference and visualization for RT-DETR, latency measurement |
| [rtdetr_evaluation.ipynb](rtdetr_evaluation.ipynb) | Evaluation of RT-DETR model (mAP, precision, recall) |
| [images/](images/) | Example output images from models |
| [requirements.txt](requirements.txt) | Project dependencies |
| [ПаранинаДМ 2025 Распознавание объектов на дороге в реальном времени Защита.pptx](ПаранинаДМ%202025%20Распознавание%20объектов%20на%20дороге%20в%20реальном%20времени%20Защита.pptx) | PowerPoint presentation from the project defense (available for download) |

## Training Details

**Environment:**
- Python, PyTorch
- Development platform: Jupyter Notebook
- GPU: NVIDIA GTX 1630
- Training type: Full fine-tuning (no frozen layers)
- Auxiliary libraries: Pillow, OpenCV, Matplotlib, PyTorch
- Frameworks: Ultralytics YOLO, Hugging Face Transformers
- Dataset: BDD100K (road object detection)

**YOLOv8n:**
- Epochs: 25  
- Optimizer: SGD  
- Batch size: 16  
- Image size: 768×768 px  
- Training time: ~72 hours  

**RT-DETR-R101-VD:**
- Epochs: 10  
- Optimizer: AdamW  
- Batch size: 2  
- Image size: 512×512 px  
- Training time: ~90 hours  

## Example Results (more in /images)

![Результат инференса модели yolo8n](images/output_yolo.jpg)

![Результат инференса модели rt-detr](images/output_rtdetr.jpg)

## Model Comparison Results

| Model                     | Size     | Resolution | mAP@0.5 | mAP@[.5:.95] | Recall | Inference Time | FPS     |
|--------------------------|----------|------------|---------|--------------|--------|----------------|---------|
| YOLOv8n                  | 5.94 MB  | 768×768    | 0.569   | 0.324        | 0.511  | 9.6 ms         | >100 FPS |
| RT-DETR-R101-VD (HF)     | 293.1 MB | 512×512    | 0.537   | 0.307        | 0.454  | 158.7 ms       | 6.3 FPS  |

Both models were fine-tuned on 8 object classes from the BDD100K dataset.  
YOLOv8n demonstrates faster inference and slightly better accuracy on this setup, making it more suitable for real-time applications.  
RT-DETR, although slower, is a promising transformer-based alternative with comparable performance.

## About

Project by **Darya Paranina**, MSc in Artificial Intelligence (Siberian Federal University, 2025)
PowerPoint presentation from the project defense (available for download): [ПаранинаДМ 2025 Распознавание объектов на дороге в реальном времени Защита.pptx](ПаранинаДМ%202025%20Распознавание%20объектов%20на%20дороге%20в%20реальном%20времени%20Защита.pptx) 
[GitHub](https://github.com/odarapara-ml)
