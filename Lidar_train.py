import ultralytics
import torch
from ultralytics import YOLO

model = YOLO("yolo11n.yaml").load("yolo11n.pt")

if __name__ == '__main__':
    results = model.train(
        data="BEVLidardataset.yaml",
        epochs=100,
        batch=20,   
        workers=8,          
        amp=True,           
        device=0,           
        
        # ===== AUGMENTATION =====
        hsv_h=0.0,          # No hue changes
        hsv_s=0.0,          # No saturation changes
        hsv_v=0.0,          # No brightness changes
        erasing=0.4,        # Occlusion simulation
        fliplr=0.5,         # Left/right symmetry
        flipud=0.0,         # No up/down symmetry
        # ===== TRAINING =====
        project='Jetson_yolov11n-kitti-LIDARBEV-only-2',
        augment=False,
        save=True,
        plots=True,
        patience=100,
    )
    
    print("Training completed.")
    
    best_model = YOLO('./Jetson_yolov11n-kitti-LIDARBEV-only-2/train/weights/best.pt')
    valid_results = best_model.val()
    print("Validation completed.")
