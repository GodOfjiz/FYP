from ultralytics import YOLO

def continue_training():
    # Resume training
    model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-2/train/weights/last.pt")
    results = model.train(resume=True)
    
    # Validate best model
    best_model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-2/train/weights/best.pt")
    validation_results = best_model.val()
    
    return results, validation_results

if __name__ == "__main__":
    continue_training()