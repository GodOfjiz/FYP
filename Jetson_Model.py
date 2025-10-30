from ultralytics import YOLO
import torch

# Clear memory
torch.cuda.empty_cache()

# Load the .pt file
model = YOLO('Jetson_yolov11n-kitti/train2/weights/best.pt')

# Export ON JETSON
model.export(
    format='engine',
    half=True,
    imgsz=640,  # Or 416 if memory constrained
    simplify=True,
    device=0,   # Use Jetson GPU for export
    workspace=2  # 2GB workspace
)

print("TensorRT engine created on Jetson!")
print("TensorRT engine created: best.engine")