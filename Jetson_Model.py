from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("./Jetson_yolov26n-kitti-LIDARBEV-only-1/train/weights/last.pt")

# Export the model to TensorRT
model.export(format="engine",
             device="0", 
             half=True,
             batch=1
             ) # creates 'yolo11n.engine'  with FP16 format engine model

print("TensorRT engine created on Jetson!")
print("TensorRT engine created: best.engine")
