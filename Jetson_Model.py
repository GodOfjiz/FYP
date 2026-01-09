from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("./Jetson_yolov11n-kitti-Cam-only-8/train/weights/last.pt")

# Export the model to TensorRT
model.export(format="onnx")  # creates 'yolo11n.engine'

print("TensorRT engine created on Jetson!")
print("TensorRT engine created: best.engine")
