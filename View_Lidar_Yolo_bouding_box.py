import cv2
import os
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

def read_yolo_label_4corners(label_path, img_width, img_height):
    """
    Read YOLO-OBB format labels with 4 corners
    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates 0-1)
    
    Args:
        label_path: Path to label file
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List of (class_id, corners) tuples with denormalized pixel coordinates
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            if len(data) < 9:
                continue
            
            class_id = int(data[0])
            
            # Read normalized coordinates (0-1 range)
            x1_norm, y1_norm = float(data[1]), float(data[2])
            x2_norm, y2_norm = float(data[3]), float(data[4])
            x3_norm, y3_norm = float(data[5]), float(data[6])
            x4_norm, y4_norm = float(data[7]), float(data[8])
            
            # Denormalize to pixel coordinates
            x1 = x1_norm * img_width
            y1 = y1_norm * img_height
            x2 = x2_norm * img_width
            y2 = y2_norm * img_height
            x3 = x3_norm * img_width
            y3 = y3_norm * img_height
            x4 = x4_norm * img_width
            y4 = y4_norm * img_height
            
            corners = np.array([
                [x1, y1],  # rear-left
                [x2, y2],  # rear-right
                [x3, y3],  # front-right
                [x4, y4]   # front-left
            ], dtype=np.float32)
            
            boxes.append((class_id, corners))
    
    return boxes

def visualize_bboxes_4corners(image_path, label_path, class_names=None, show_corners=False, show_direction=True):
    """Visualize 4-corner oriented bounding boxes on image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # Read boxes with denormalized coordinates
    boxes = read_yolo_label_4corners(label_path, img_width, img_height)
    
    colors = [
        (0, 0, 255),      # Car - Red (BGR)
        (0, 255, 0),      # Van - Green
        (255, 0, 0),      # Truck - Blue
        (0, 255, 255),    # Pedestrian - Yellow
        (255, 0, 255),    # Cyclist - Magenta
        (255, 255, 0),    # Tram - Cyan
        (128, 0, 128),    # Purple
        (128, 128, 128)   # Gray
    ]
    
    corner_labels = ['RL', 'RR', 'FR', 'FL']  # Rear-left, Rear-right, Front-right, Front-left
    
    for box in boxes:
        class_id, corners = box
        color = colors[class_id % len(colors)]
        
        # Convert corners to integer pixel coordinates
        corners_int = corners.astype(np.int32)
        
        # Draw filled polygon (semi-transparent)
        overlay = img.copy()
        cv2.fillPoly(overlay, [corners_int], color)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        # Draw outline
        cv2.polylines(img, [corners_int], isClosed=True, color=color, thickness=2)
        
        # Draw corner points
        for i, corner in enumerate(corners_int):
            cv2.circle(img, tuple(corner), 3, color, -1)
            
            # Optionally label corners
            if show_corners:
                cv2.putText(img, corner_labels[i], 
                           (corner[0] + 5, corner[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw direction arrow (rear center → front center)
        if show_direction:
            # Calculate rear center (midpoint of rear-left and rear-right)
            rear_center = ((corners_int[0] + corners_int[1]) / 2).astype(np.int32)
            # Calculate front center (midpoint of front-right and front-left)
            front_center = ((corners_int[2] + corners_int[3]) / 2).astype(np.int32)
            
            # Draw arrow from rear to front
            cv2.arrowedLine(img, tuple(rear_center), tuple(front_center), 
                           color, thickness=2, tipLength=0.3)
        
        # Add class label at the first corner (rear-left)
        if class_names is not None and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
        
        label_pos = (corners_int[0][0], corners_int[0][1] - 10)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw label background
        cv2.rectangle(img, 
                     (label_pos[0], label_pos[1] - label_size[1] - 4),
                     (label_pos[0] + label_size[0], label_pos[1]),
                     color, -1)
        cv2.putText(img, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img, len(boxes)

def main():
    # Updated paths for 4-corner labels
    image_dir = "./Dataset/training/lidar_bev_improved/"
    label_dir = "./Dataset/training/lidar_bev_improved_labels_4corners/"
    
    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Tram']
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No images found in the directory")
        return
    
    print("="*70)
    print("YOLO-OBB BEV LiDAR Dataset Viewer")
    print("="*70)
    print(f"Found {len(image_files)} images")
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    print("\nControls:")
    print("  'n' / Right Arrow = Next image")
    print("  'p' / Left Arrow  = Previous image")
    print("  'c'              = Toggle corner labels")
    print("  'd'              = Toggle direction arrows")
    print("  'q' / ESC        = Quit")
    print("\nVisualization:")
    print("  - Format: YOLO-OBB (normalized coordinates)")
    print("  - Oriented bounding boxes (4 corners)")
    print("  - Arrow shows object direction (rear → front)")
    print("  - Corner order: Rear-Left, Rear-Right, Front-Right, Front-Left")
    print("\nColor Coding:")
    for i, name in enumerate(class_names):
        print(f"  {name}: Class {i}")
    print("="*70)
    
    current_idx = 0
    show_corners = False
    show_direction = True
    
    while current_idx < len(image_files):
        image_file = image_files[current_idx]
        image_path = os.path.join(image_dir, image_file)
        
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        result = visualize_bboxes_4corners(image_path, label_path, class_names, show_corners, show_direction)
        
        if result is not None:
            img, num_boxes = result
            
            # Add status text
            status_lines = [
                f"Image: {image_file} ({current_idx + 1}/{len(image_files)})",
                f"Objects: {num_boxes}",
                f"Corner Labels: {'ON' if show_corners else 'OFF'}",
                f"Direction Arrows: {'ON' if show_direction else 'OFF'}"
            ]
            
            y_offset = 30
            for i, status in enumerate(status_lines):
                # Black background for better visibility
                text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (5, y_offset - 20 + i*30), 
                            (15 + text_size[0], y_offset + 5 + i*30),
                            (0, 0, 0), -1)
                cv2.putText(img, status, (10, y_offset + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display image
            cv2.imshow('BEV 4-Corner Bounding Boxes (YOLO-OBB)', img)
            print(f"\r{status_lines[0]} | {status_lines[1]} | {status_lines[2]} | {status_lines[3]}", end='', flush=True)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('n') or key == 83:  # 'n' or Right arrow
                current_idx = min(current_idx + 1, len(image_files) - 1)
            elif key == ord('p') or key == 81:  # 'p' or Left arrow
                current_idx = max(current_idx - 1, 0)
            elif key == ord('c'):  # Toggle corner labels
                show_corners = not show_corners
                continue  # Redraw same image with toggle
            elif key == ord('d'):  # Toggle direction arrows
                show_direction = not show_direction
                continue  # Redraw same image with toggle
            else:
                current_idx += 1
        else:
            print(f"\nSkipping {image_file} (cannot read image)")
            current_idx += 1
    
    print("\n" + "="*70)
    print("Viewer closed")
    print("="*70)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()