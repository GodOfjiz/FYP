import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

def main():
    # Load the YOLO11 model
    model = YOLO("./Jetson_yolov11n-kitti-Cam-only-8/train/weights/last.pt", task="detect")

    # Define paths
    image_dataset_path = './Dataset/testing/image_2'
    output_path = "Camera-M8-FP16"
    
    # Run inference on the image dataset and save results
    results = model.predict(
        image_dataset_path * 8,
        save=False,
        project="result",
        name=output_path,
        device=0,
        stream=True,
        verbose=True
    )

    out_dir = os.path.join("result", output_path)
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()
    for i, r in enumerate(results):
        img = r.plot()      
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), img)
    time_elapsed = time.time() - start_time
    print(f"Processed {i+1} images in {time_elapsed:.2f} seconds, average {time_elapsed/(i+1):.2f} seconds per image.")

if __name__ == "__main__":
    main()
