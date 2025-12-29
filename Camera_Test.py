import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLO11 model
model = YOLO("./Jetson_yolov11n-kitti-Cam-only-4/train/weights/best.engine")

# Define paths
image_dataset_path = './Dataset/testing/image_2'
output_path = "result/Camera-M4"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Run inference on the image dataset and save results
results = model.predict(source=image_dataset_path, save=True, project="result", name="Camera")
