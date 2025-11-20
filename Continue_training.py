from ultralytics import YOLO


def continue_training():
# Load a model
    model = YOLO("./Jetson_yolov11n-kitti-Cam-only-3/train/weights/last.pt")  # load a partially trained model

# Resume training
    results = model.train(resume=True)  

if __name__ == "__main__":
    continue_training()
