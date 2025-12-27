import ultralytics
import torch
from ultralytics import YOLO

model = YOLO("yolo11n.yaml").load("yolo11n.pt")

Version = 'Jetson_yolov11n-kitti-LIDARBEV-only-3'

custom_transforms = []

if __name__ == '__main__':
    results = model.train(
        data="BEVLidardataset.yaml",
        epochs=100,
        batch=8,   
        workers=8,          
        amp=True,           
        device=0,
        imgsz=1024,
        augmentations=custom_transforms,        
        
        # ===== AUGMENTATION =====
        hsv_h=0.0,          # No hue changes
        hsv_s=0.0,          # No saturation changes
        hsv_v=0.0,          # No brightness changes
        erasing=0.4,        # Occlusion simulation
        fliplr=0.5,         # Left/right symmetry
        flipud=0.0,         # No up/down symmetry
        # ===== TRAINING =====
        project=Version,
        augment=False,
        save=True,
        plots=True,
        patience=100,
    )
    
    print("Training completed.")
