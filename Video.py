import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("./Jetson_yolov11n-kitti-Cam-only-7/train/weights/best.engine")

# Open the video file
video_path = "/home/jasminder/Downloads/Test.mp4"
results = model.predict(source=video_path, show=True)
