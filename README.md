# Road Object Detection for Autonomous Driving

This repository presents the results of my master's thesis in Artificial Intelligence, focused on real-time object detection for autonomous vehicles. The project compares YOLOv8n and RT-DETR models trained on the BDD100K dataset (road object detection).

## Objective

Train and compare lightweight convolutional (YOLOv8n) and transformer-based (RT-DETR) object detection models on 8 classes from the BDD100K dataset.

## Repository Structure

| File | Description |
|------|-------------|
| `preprocessing.ipynb` | Cleans and converts BDD100K annotations to YOLO and COCO formats |
| `yolov8_training.ipynb` | Fine-tuning YOLOv8n using Ultralytics framework |
| `rtdetr_training.ipynb` | Fine-tuning RT-DETR using Hugging Face and PyTorch |
| `yolov8_inference.ipynb` | Inference and visualization for YOLOv8 |
| `rtdetr_inference.ipynb` | Inference and visualization for RT-DETR |
| `rtdetr_evaluation.ipynb` | Evaluation of RT-DETR model (mAP, precision, recall, FPS, latency) |
| `images/` | Example output images from models |
| `requirements.txt` | Project dependencies |

## Tools and Frameworks

- Python, PyTorch
- Jupyter Notebook
- Ultralytics YOLO, Hugging Face Transformers
- Dataset: BDD100K (road object detection)

## Example Results



## About

Project by **Darya Paranina**, MSc in Artificial Intelligence (Siberian Federal University, 2025)  
[GitHub](https://github.com/odarapara-ml)
