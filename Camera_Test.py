import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLO11 model
model = YOLO("./Jetson_yolov11n-kitti-Cam-only-8/train/weights/last.pt")

# Define paths
image_dataset_path = './Dataset/testing/image_2'

# Run inference on the image dataset and save results
results = model.predict(source=image_dataset_path, save=True, project="result", name="Camera-M8", device="0")
