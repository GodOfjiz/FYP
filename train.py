import ultralytics
import torch
from ultralytics import YOLO

#Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model 
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="dataset.yaml",
    epochs=10,
    patience=3,
    mixup=0.1,
    project='yolov11n-kitti',
    device=0
    )

# Validation
valid_results = model.val()

# Export model to TensorRT engine
model.export(format="engine", 
             device="dla:0",
             name="FYP_yolo11n_engine"
             )