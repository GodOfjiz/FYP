import os
import cv2
from ultralytics import YOLO

model = YOLO("./Jetson_yolov11n-kitti-Cam-only-8/train/weights/last.engine", task="detect")
video_path = "/home/jasminder/Downloads/Test.mp4"

if os.environ.get("DISPLAY", "") == "":
    print("No DISPLAY found — cannot show cv2 windows. Run on a desktop or use X forwarding.")
else:
    results = model.predict(source=video_path, batch=24, device=0, stream=True)
    for r in results:
        img = r.plot()  # annotated frame as numpy array
        cv2.imshow("YOLO", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
