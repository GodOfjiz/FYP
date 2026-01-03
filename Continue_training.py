from ultralytics import YOLO

def continue_training():
    # Resume training
    model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-4/train/weights/last.pt")
    results = model.train(resume=True)
    return results

if __name__ == "__main__":
    continue_training()
