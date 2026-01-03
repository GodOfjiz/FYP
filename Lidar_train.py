import ultralytics
import torch
from ultralytics import YOLO

model = YOLO("yolo11n.yaml").load("yolo11n.pt")

Version = 'Jetson_yolov11n-kitti-LIDARBEV-only-4'

custom_transforms = []

if __name__ == '__main__':
    results = model.train(
        data="BEVLidardataset.yaml",
        epochs=120,
        batch=20,   
        workers=16,          
        amp=True,           
        device=0,
        imgsz=640,
        augmentations=custom_transforms,        
        
        # ===== AUGMENTATION =====
        hsv_h=0.0,          # No hue changes
        hsv_s=0.0,          # No saturation changes
        hsv_v=0.0,          # No brightness changes
        erasing=0.4,        # Occlusion simulation
        fliplr=0.0,         # Left/right symmetry
        flipud=0.0,         # No up/down symmetry
        mosaic=1.0,         # Mosaic augmentation
        # ===== TRAINING =====
        project=Version,
        augment=True,
        save=True,
        plots=True,
        patience=100,
    )
    
    print("Training completed.")
