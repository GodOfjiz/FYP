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

# ===== UPDATED: 6 classes with Person_sitting merged into Pedestrian =====
kitti_to_yolo_mapping = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 3,  
    'Cyclist': 4,
    'Tram': 5
}

# YOLO class names (for reference)
yolo_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Tram']

print(f"Converting KITTI to YOLO format...")
print(f"Target classes: {yolo_classes}")
print(f"Mapping: Car→0, Van→1, Truck→2, Pedestrian→3, Person_sitting→3, Cyclist→4, Tram→5")
print(f"Skipping: Misc, DontCare, and other classes\n")


# Convert KITTI bbox to YOLO bbox format
def convert_bbox(size, box):
    dw = 1. / size[0]  # Scaling factor width = 1/image_width
    dh = 1. / size[1]  # Scaling factor height = 1/image_height
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# Statistics tracking
stats = {
    'Car': 0,
    'Van': 0,
    'Truck': 0,
    'Pedestrian': 0,
    'Person_sitting': 0,
    'Cyclist': 0,
    'Tram': 0,
    'Skipped': 0,
    'Total_images': 0,
    'Images_with_objects': 0
}

for label_file in os.listdir(labels_path):
    if not label_file.endswith('.txt'):
        continue
    
    stats['Total_images'] += 1
    
    # Corresponding image file name
    image_file = label_file.replace('.txt', '.png')

    # Check if image exists
    image_path = os.path.join(images_path, image_file)
    if not os.path.exists(image_path):
        print(f"Warning: Image not found for {label_file}, skipping...")
        continue

    # Read image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_file}, skipping...")
        continue
    image_height, image_width, _ = image.shape

    with open(os.path.join(labels_path, label_file), 'r') as labelfile:
        lines = labelfile.readlines()
    
    yolo_labels = []
    for line in lines:
        annotations = line.strip().split(' ')
        class_name = annotations[0]
        
        # ===== FILTER: Only keep specified classes =====
        if class_name in kitti_to_yolo_mapping:
            # Get YOLO class index
            yolo_class_index = kitti_to_yolo_mapping[class_name]
            
            # Track statistics
            stats[class_name] += 1
            
            # Parse bounding box coordinates
            xmin = float(annotations[4])
            ymin = float(annotations[5])
            xmax = float(annotations[6])
            ymax = float(annotations[7])
            
            # Convert to YOLO format
            bbox = convert_bbox((image_width, image_height), (xmin, xmax, ymin, ymax))
            bbox_str = ' '.join([f"{value}" for value in bbox])
            yolo_labels.append(f"{yolo_class_index} {bbox_str}\n")
        else:
            # Skip other classes (Misc, DontCare, etc.)
            stats['Skipped'] += 1

    # Only save label file and copy image if there are valid objects
    if yolo_labels:
        stats['Images_with_objects'] += 1
        with open(os.path.join(yolo_labels_path, label_file), 'w') as yf:
            yf.writelines(yolo_labels)
        # Copy image to YOLO directory
        shutil.copy(image_path, yolo_images_path)

# Print statistics
print("\n" + "="*60)
print("CONVERSION STATISTICS")
print("="*60)
print(f"Total images processed: {stats['Total_images']}")
print(f"Images with valid objects: {stats['Images_with_objects']}")
print(f"\nObjects converted:")
print(f"  Car:            {stats['Car']}")
print(f"  Van:            {stats['Van']}")
print(f"  Truck:          {stats['Truck']}")
print(f"  Pedestrian:     {stats['Pedestrian']}")
print(f"  Person_sitting: {stats['Person_sitting']} (→ merged into Pedestrian)")
print(f"  Cyclist:        {stats['Cyclist']}")
print(f"  Tram:           {stats['Tram']}")
print(f"\nTotal Pedestrians (combined): {stats['Pedestrian'] + stats['Person_sitting']}")
print(f"Objects skipped (other classes): {stats['Skipped']}")
print("="*60 + "\n")

# Split dataset into train, val, test sets
all_images = [f for f in os.listdir(yolo_images_path) if f.endswith('.png')]

if len(all_images) == 0:
    print("ERROR: No images found after conversion! Check your dataset paths.")
    exit(1)

print(f"Splitting {len(all_images)} images into train/val/test...")

train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=69)
val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=69)

print(f"Train: {len(train_images)} images ({len(train_images)/len(all_images)*100:.1f}%)")
print(f"Val:   {len(val_images)} images ({len(val_images)/len(all_images)*100:.1f}%)")
print(f"Test:  {len(test_images)} images ({len(test_images)/len(all_images)*100:.1f}%)\n")


# Function to move files to appropriate directories
def move_files(file_list, dest_dir):
    os.makedirs(os.path.join(dest_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels'), exist_ok=True)
    for file_name in file_list:
        # Move image
        src_img = os.path.join(yolo_images_path, file_name)
        dst_img = os.path.join(dest_dir, 'images', file_name)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        
        # Move label
        label_file = file_name.replace('.png', '.txt')
        src_label = os.path.join(yolo_labels_path, label_file)
        dst_label = os.path.join(dest_dir, 'labels', label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)


# Move files to train, val, test directories
print("Moving files to train/val/test directories...")
move_files(train_images, os.path.join(yolo_base_path, 'train'))
move_files(val_images, os.path.join(yolo_base_path, 'val'))
move_files(test_images, os.path.join(yolo_base_path, 'test'))

# Clean up empty directories
if os.path.exists('./data/images'):
    shutil.rmtree('./data/images')
if os.path.exists('./data/labels'):
    shutil.rmtree('./data/labels')

print("\n" + "="*60)
print("CONVERSION AND SPLITTING COMPLETE!")
print("="*60)
print(f"Dataset saved to: {yolo_base_path}/")
print(f"  - train/images/ and train/labels/")
print(f"  - val/images/ and val/labels/")
print(f"  - test/images/ and test/labels/")
print(f"\nClass mapping for YAML file:")
print(f"names:")
print(f"  0: Car")
print(f"  1: Van")
print(f"  2: Truck")
print(f"  3: Pedestrian including Person_sitting)")
print(f"  4: Cyclist")
print(f"  5: Tram")
print("="*60)