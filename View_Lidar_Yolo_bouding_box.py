import cv2
import os
import numpy as np

# Set environment variables
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

def read_yolo_label(label_path, img_width, img_height):
    """
    Read YOLO format labels and convert to pixel coordinates
    YOLO format: class_id center_x center_y width height (normalized 0-1)
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            if len(data) < 5:
                continue
            
            class_id = int(data[0])
            center_x = float(data[1]) * img_width
            center_y = float(data[2]) * img_height
            width = float(data[3]) * img_width
            height = float(data[4]) * img_height
            
            # Convert to corner coordinates
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)
            
            boxes.append((class_id, x1, y1, x2, y2))
    
    return boxes

def visualize_bboxes(image_path, label_path, class_names=None):
    """
    Visualize bounding boxes on image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # Read labels
    boxes = read_yolo_label(label_path, img_width, img_height)
    
    # Define colors for different classes
    colors = [
        (255, 0, 0),      # Car - Red
        (0, 255, 0),      # Van - Green
        (0, 0, 255),      # Truck - Blue
        (255, 255, 0),    # Pedestrian - Cyan
        (255, 0, 255),    # Person_sitting - Magenta
        (0, 255, 255),    # Cyclist - Yellow
        (128, 0, 128),    # Tram - Purple
        (128, 128, 128)   # Misc - Gray
    ]
    
    # Draw bounding boxes
    for box in boxes:
        class_id, x1, y1, x2, y2 = box
        color = colors[class_id % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if class_names is not None and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img

def main():
    image_dir = "./Dataset/training/lidar_bev/"
    label_dir = "./Dataset/training/lidar_bev_labels/"
    
    # KITTI dataset class names
    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    
    # Get all images
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No images found in the directory")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process images
    current_idx = 0
    
    while current_idx < len(image_files):
        image_file = image_files[current_idx]
        image_path = os.path.join(image_dir, image_file)
        
        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        # Visualize
        img = visualize_bboxes(image_path, label_path, class_names)
        
        if img is not None:
            # Display image
            cv2.imshow('KITTI YOLO Bounding Boxes', img)
            print(f"Showing: {image_file} ({current_idx + 1}/{len(image_files)})")
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('n') or key == 83:  # Next (or right arrow)
                current_idx = min(current_idx + 1, len(image_files) - 1)
            elif key == ord('p') or key == 81:  # Previous (or left arrow)
                current_idx = max(current_idx - 1, 0)
            else:  # Any other key - next image
                current_idx += 1
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()