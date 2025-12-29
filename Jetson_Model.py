from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-3/train/weights/best.pt")

# Export the model to TensorRT
model.export(format="engine", device="0")  # creates 'yolo11n.engine'

print("TensorRT engine created on Jetson!")
print("TensorRT engine created: best.engine")
