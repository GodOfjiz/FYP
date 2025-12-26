import cv2
import os
import numpy as np
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def visualize_bbox(image_path, label_path, class_names):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Read label file
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except:
        print(f"Could not read label file: {label_path}")
        return
    
    # Draw each bounding box
    for line in lines:
        data = line.strip().split()
        if len(data) >= 15:  # KITTI format has 15 columns
            class_name = data[0]  # In KITTI, first column is class name
            if class_name in class_names:
                class_id = class_names.index(class_name)
                
                # KITTI format: left, top, right, bottom
                x1 = float(data[4])  # left
                y1 = float(data[5])  # top
                x2 = float(data[6])  # right
                y2 = float(data[7])  # bottom
                
                # Draw box
                plot_one_box([x1, y1, x2, y2], image, label=class_name)
                print(f"Drew {class_names[class_id]} at ({x1},{y1},{x2},{y2})")
    
    # Show image
    cv2.imshow('Bounding Box Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define class names (same as in your training)
class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

# Use KITTI dataset paths
kitti_images = './Dataset/training/image_2'
kitti_labels = './Dataset/training/label_2'

# Example usage with KITTI paths
image_name = '000527.png'  # Replace with your image name
image_path = os.path.join(kitti_images, image_name)
label_path = os.path.join(kitti_labels, image_name.replace('.png', '.txt'))

# Verify paths exist
if not os.path.exists(image_path):
    print(f"Image not found: {image_path}")
    exit(1)
if not os.path.exists(label_path):
    print(f"Label not found: {label_path}")
    exit(1)

visualize_bbox(image_path, label_path, class_names)