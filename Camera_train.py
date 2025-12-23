import ultralytics
import torch
from ultralytics import YOLO
import albumentations as A

# Load model 
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Custom augmentation using Albumentations
custom_transforms = [
    A.ZoomBlur(
    max_factor=[1, 1.2],
    step_factor=[0.01, 0.03],
    p=0.3,
    ),
    A.ISONoise(
    color_shift=[0.01, 0.05],
    intensity=[0, 0.5],
    p=0.5,
    ),
    A.RandomRain(
    slant_range=[-15, 15],
    drop_length=50,
    drop_width=1,
    drop_color=[200, 200, 200],
    blur_value=7,
    brightness_coefficient=0.6,
    rain_type="default",
    p =0.4,
    )
]


if __name__ == '__main__':
    # Train the model 
    results = model.train(
        data="dataset.yaml",
        epochs=120,
        batch=8,
        workers=8,
        amp=True,
        project='Jetson_yolov11n-kitti-Cam-only-4',
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
    )
    print("Training completed.")
    
    # Validation
    valid_results = model.val()
    print("Validation completed.")