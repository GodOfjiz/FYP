import numpy as np
import pandas as pd
import os
import shutil
import cv2
from sklearn.model_selection import train_test_split

# Define the image and label paths
yolo_base_path = './data'
images_path = './Dataset/training/image_2'
labels_path = './Dataset/training/label_2'
yolo_images_path = os.path.join(yolo_base_path, 'images')
yolo_labels_path = os.path.join(yolo_base_path, 'labels')

# create the yolo data folders
os.makedirs(yolo_base_path, exist_ok=True)
os.makedirs(yolo_images_path, exist_ok=True)
os.makedirs(yolo_labels_path, exist_ok=True)

### Label:
### class_index x_center y_center width height
kitti_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']


# Covert KITTI bbox to YOLO bbox format
def convert_bbox(size, box):
    dw = 1. / size[0] #Scaling factor width = 1/image_width
    dh = 1. / size[1] #Scaling factor height = 1/image_height
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


for label_file in os.listdir(labels_path):
    if not label_file.endswith('.txt'):
        continue
    # Corresponding image file name
    image_file = label_file.replace('.txt', '.png')

    # Read image to get dimensions
    image = cv2.imread(os.path.join(images_path, image_file))
    image_height, image_width, _ = image.shape

    with open(os.path.join(labels_path, label_file), 'r') as labelfile:
        lines = labelfile.readlines()
    yolo_labels = []
    for line in lines:
        annotations = line.strip().split(' ')
        class_index = annotations[0]
        if class_index in kitti_classes:
            class_index = kitti_classes.index(class_index)  # In KITTI, the first element is the class
            xmin = float(annotations[4])
            ymin = float(annotations[5])
            xmax = float(annotations[6])
            ymax = float(annotations[7])
            bbox = convert_bbox((image_width, image_height), (xmin, xmax, ymin, ymax))
            bbox_str = ' '.join([f"{value}" for value in bbox])
            yolo_labels.append(f"{class_index} {bbox_str}\n")

    with open(os.path.join(yolo_labels_path, label_file), 'w') as yf:
        yf.writelines(yolo_labels)
    # Copy image to YOLO directory
    shutil.copy(os.path.join(images_path, image_file), yolo_images_path)

# Split dataset into train, val, test sets
all_images = [f for f in os.listdir(yolo_images_path) if f.endswith('.png')]
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=69)
val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=69)

# Function to move files to appropriate directories
def move_files(file_list, dest_dir):
    os.makedirs(os.path.join(dest_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels'), exist_ok=True)
    for file_name in file_list:
        shutil.move(os.path.join(yolo_images_path, file_name), os.path.join(dest_dir, 'images', file_name))
        label_file = file_name.replace('.png', '.txt')
        shutil.move(os.path.join(yolo_labels_path, label_file), os.path.join(dest_dir, 'labels', label_file))

# Move files to train, val, test directories
move_files(train_images, os.path.join(yolo_base_path, 'train'))
move_files(val_images, os.path.join(yolo_base_path, 'val'))
move_files(test_images, os.path.join(yolo_base_path, 'test'))

# Clean up empty directories
if os.path.exists('./data/images'):
    shutil.rmtree('./data/images')
if os.path.exists('./data/labels'):
    shutil.rmtree('./data/labels')

print("Conversion and splitting complete.")
