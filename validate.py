from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-5/train/weights/last.pt")  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list containing mAP50-95 for each category