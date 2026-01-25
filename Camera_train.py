import ultralytics
import torch
from ultralytics import YOLO
import albumentations as A

# Load model 
model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Custom augmentation using Albumentations
custom_transforms = [
    A.ISONoise(
    color_shift=[0.01, 0.05],
    intensity=[0, 0.5],
    p=0.5,
    ),
]

Version = 'Jetson_yolov26n-kitti-Cam-only-1'

if __name__ == '__main__':
    # Train the model 
    results = model.train(
        data="dataset.yaml",
        epochs=300,
        batch=20,
        workers=16,
        amp=True,
        project=Version,
        device=0,
        augmentations=custom_transforms,
        
        # ==================== AUGMENTATION PARAMETERS ====================
        
        # --- Photometric Augmentations (Color/Lighting) ---
        hsv_h=0.015,        # Hue shift - subtle color variations
        hsv_s=0.6,          # Saturation - weather (sunny/foggy)
        hsv_v=0.4,          # Brightness/Value - time of day (sunny/night)
        
        # --- Geometric Augmentations ---
        scale=0.5,          # Scale - far/near objects (0.5x to 1.5x zoom)
        mosaic=1.0,         # Mosaic 
        
        # --- Occlusion/Dropout Augmentations ---
        erasing=0.4,        # Random erasing - occlusion 
        
        # ================================================================
        
        # Training settings
        augment=True,       # Enable augmentation
        save=True,          # Save checkpoints
        plots=True,         # Save training plots
        patience=200,      # Early stopping patience
    )
    print("Training completed.")
    
