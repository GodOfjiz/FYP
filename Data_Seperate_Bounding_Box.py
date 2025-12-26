import cv2
import os
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def visualize_bbox(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        print(f"Found {len(lines)} objects in label file")
    except:
        print(f"Could not read label file: {label_path}")
        return
    
    for line in lines:
        data = line.strip().split()
        if len(data) == 5:  # YOLO format
            class_id = int(data[0])
            if class_id < len(class_names):
                x_center = float(data[1])
                y_center = float(data[2])
                w = float(data[3])
                h = float(data[4])
                
                x_center_px = int(x_center * width)
                y_center_px = int(y_center * height)
                w_px = int(w * width)
                h_px = int(h * height)
                
                x1 = max(0, int(x_center_px - w_px/2))
                y1 = max(0, int(y_center_px - h_px/2))
                x2 = min(width, int(x_center_px + w_px/2))
                y2 = min(height, int(y_center_px + h_px/2))
                
                plot_one_box([x1, y1, x2, y2], image, label=class_names[class_id])
                print(f"Drew {class_names[class_id]} at ({x1},{y1},{x2},{y2})")
    
    cv2.imshow('Bounding Box Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===== UPDATED: 6 classes matching data_convert.py =====
# 0: Car, 1: Van, 2: Truck, 3: Pedestrian (includes Person_sitting), 4: Cyclist, 5: Tram
class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Tram']

# YOLO dataset paths
val_images = './data/val/images'
val_labels = './data/val/labels'

image_name = '000005.png'
image_path = os.path.join(val_images, image_name)
label_path = os.path.join(val_labels, image_name.replace('.png', '.txt'))

if not os.path.exists(image_path):
    print(f"Image not found: {image_path}")
    exit(1)
if not os.path.exists(label_path):
    print(f"Label not found: {label_path}")
    exit(1)

visualize_bbox(image_path, label_path, class_names)