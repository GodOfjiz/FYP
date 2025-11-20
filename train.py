import ultralytics
import torch
from ultralytics import YOLO

# Load model 
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

if __name__ == '__main__':
    # Train the model 
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        batch=8,
        workers=8,
        amp=True,
        mixup=0.1,
        project='Jetson_yolov11n-kitti-Cam-only-3',
        device=0,
    )
    print("Training completed.")
    
    # Validation
    valid_results = model.val()
    print("Validation completed.")