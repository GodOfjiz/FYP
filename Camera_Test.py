import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

# Load the YOLO11 model
model = YOLO("./Jetson_yolov11n-kitti-Cam-only-5/train/weights/best.engine")

# Define paths
image_dataset_path = './Dataset/testing/image_2'
output_path = "result/Camera"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Get all image files
image_files = list(Path(image_dataset_path).glob('*.jpg')) + \
              list(Path(image_dataset_path).glob('*.png')) + \
              list(Path(image_dataset_path).glob('*.jpeg'))

# Process each image
for image_path in image_files:
    results = model.predict(source=str(image_path), save=True, project="result", name="Camera")
    print(f"Processed: {image_path.name}")

print(f"All results saved to {output_path}")
