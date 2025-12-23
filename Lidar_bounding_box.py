import numpy as np
import glob
import os

def load_calib(calib_file):
    """Load calibration matrices from KITTI calibration file"""
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])
    
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)
    
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    # Inverse transformation: camera to velodyne
    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    R0_rect_inv = np.linalg.inv(R0_rect)
    
    return R0_rect, Tr_velo_to_cam, R0_rect_inv, Tr_cam_to_velo

def load_labels(label_file):
    """Load 3D object labels from KITTI label file"""
    objects = []
    
    if not os.path.exists(label_file):
        return objects
    
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            if len(parts) < 15:
                continue
            
            obj = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z in camera
                'rotation_y': float(parts[14])
            }
            objects.append(obj)
    
    return objects

def get_3d_box_corners(obj):
    """
    Get 3D bounding box corners in camera coordinate system
    Returns 8 corners of the 3D box
    """
    h, w, l = obj['dimensions']
    x, y, z = obj['location']
    ry = obj['rotation_y']
    
    # 3D bounding box corners (in object coordinate system)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    # Rotation matrix around Y-axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotate and translate
    corners_3d = np.dot(R, np.array([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    return corners_3d.T

def camera_to_velodyne(corners_cam, R0_rect_inv, Tr_cam_to_velo):
    """Convert corners from camera coordinate to velodyne coordinate"""
    # Add homogeneous coordinate
    corners_cam_hom = np.hstack([corners_cam, np.ones((corners_cam.shape[0], 1))])
    
    # Transform from rectified camera to camera 0
    corners_cam0 = (R0_rect_inv @ corners_cam_hom.T).T
    
    # Transform from camera 0 to velodyne
    corners_velo = (Tr_cam_to_velo @ corners_cam0.T).T
    
    return corners_velo[:, :3]

def velodyne_to_bev_coords(points_velo, y_range=(-40, 40), x_range=(0, 70), resolution=0.1):
    """Convert velodyne coordinates to BEV pixel coordinates"""
    x = points_velo[:, 0]  # Forward
    y = points_velo[:, 1]  # Left-right
    
    # Convert to pixel coordinates
    x_img = ((y_range[1] - y) / resolution).astype(int)
    y_img = ((x_range[1] - x) / resolution).astype(int)
    
    return x_img, y_img

def get_bev_bbox(corners_velo, img_width=800, img_height=700, y_range=(-40, 40), x_range=(0, 70), resolution=0.1):
    """
    Get axis-aligned bounding box in BEV image coordinates
    Returns normalized YOLO format: x_center, y_center, width, height (all 0-1)
    """
    # Get BEV coordinates for bottom 4 corners
    bottom_corners = corners_velo[:4]
    x_img, y_img = velodyne_to_bev_coords(bottom_corners, y_range, x_range, resolution)
    
    # Get min/max to form axis-aligned bounding box
    x_min = np.min(x_img)
    x_max = np.max(x_img)
    y_min = np.min(y_img)
    y_max = np.max(y_img)
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize to 0-1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clip to valid range
    x_center_norm = np.clip(x_center_norm, 0, 1)
    y_center_norm = np.clip(y_center_norm, 0, 1)
    width_norm = np.clip(width_norm, 0, 1)
    height_norm = np.clip(height_norm, 0, 1)
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def generate_yolo_labels():
    """Generate YOLO format labels for BEV images"""
    bev_path = "./Dataset/training/lidar_bev/"
    label_path = "./Dataset/training/label_2/"
    calib_path = "./Dataset/training/calib/"
    output_path = "./Dataset/training/lidar_bev_labels/"
    
    os.makedirs(output_path, exist_ok=True)
    
    bev_files = sorted(glob.glob(os.path.join(bev_path, "*.png")))
    
    if not bev_files:
        print(f"No BEV images found in {bev_path}")
        return
    
    # Class mapping (removed Misc from the class map)
    class_map = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6
    }
    
    # BEV image parameters (should match your BEV generation settings)
    y_range = (-40, 40)
    x_range = (0, 70)
    resolution = 0.1
    img_width = int((y_range[1] - y_range[0]) / resolution)  # 800
    img_height = int((x_range[1] - x_range[0]) / resolution)  # 700
    
    print(f"Found {len(bev_files)} BEV images")
    print(f"BEV image size: {img_width}x{img_height}")
    print(f"\nClass mapping:")
    for class_name, class_id in class_map.items():
        print(f"  {class_id}: {class_name}")
    print(f"\nIgnoring: DontCare, Misc")
    print("\n" + "="*60)
    print("Processing files...")
    
    total_objects = 0
    class_counts = {class_name: 0 for class_name in class_map.keys()}
    skipped_counts = {'DontCare': 0, 'Misc': 0, 'Other': 0}
    
    for idx, bev_file in enumerate(bev_files):
        file_id = os.path.splitext(os.path.basename(bev_file))[0]
        
        label_file = os.path.join(label_path, file_id + ".txt")
        calib_file = os.path.join(calib_path, file_id + ".txt")
        output_file = os.path.join(output_path, file_id + ".txt")
        
        try:
            # Check if calibration file exists
            if not os.path.exists(calib_file):
                print(f"  [{idx+1}/{len(bev_files)}] Skipping {file_id}: Calibration file not found")
                # Create empty label file
                with open(output_file, 'w') as f:
                    pass
                continue
            
            # Load calibration
            R0_rect, Tr_velo_to_cam, R0_rect_inv, Tr_cam_to_velo = load_calib(calib_file)
            
            # Load labels
            objects = load_labels(label_file)
            
            # Generate YOLO labels
            yolo_labels = []
            obj_count = 0
            
            for obj in objects:
                # Skip DontCare and Misc objects
                if obj['type'] in ['DontCare', 'Misc']:
                    skipped_counts[obj['type']] += 1
                    continue
                
                # Skip objects not in our class map
                if obj['type'] not in class_map:
                    skipped_counts['Other'] += 1
                    continue
                
                # Get 3D box corners in camera coordinates
                corners_cam = get_3d_box_corners(obj)
                
                # Convert to velodyne coordinates
                corners_velo = camera_to_velodyne(corners_cam, R0_rect_inv, Tr_cam_to_velo)
                
                # Get YOLO format bounding box
                x_center, y_center, width, height = get_bev_bbox(
                    corners_velo, img_width, img_height, y_range, x_range, resolution
                )
                
                # Check if box is valid (within image bounds and has valid size)
                if width > 0 and height > 0 and width <= 1 and height <= 1:
                    class_id = class_map[obj['type']]
                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    obj_count += 1
                    total_objects += 1
                    class_counts[obj['type']] += 1
            
            # Write YOLO label file
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
                if yolo_labels:
                    f.write('\n')
            
            print(f"  [{idx+1}/{len(bev_files)}] {file_id}: {obj_count} objects")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(bev_files)}] Error processing {file_id}: {str(e)}")
            # Create empty label file on error
            with open(output_file, 'w') as f:
                pass
    
    print("\n" + "="*60)
    print(f"Completed!")
    print(f"YOLO labels saved to: {output_path}")
    print(f"\nTotal objects: {total_objects}")
    print(f"\nObjects per class:")
    for class_name, count in class_counts.items():
        if count > 0:
            print(f"  {class_name}: {count}")
    
    print(f"\nSkipped objects:")
    for skip_type, count in skipped_counts.items():
        if count > 0:
            print(f"  {skip_type}: {count}")
    
    # Generate classes.txt file
    classes_file = os.path.join(output_path, "classes.txt")
    with open(classes_file, 'w') as f:
        for class_name in sorted(class_map.keys(), key=lambda x: class_map[x]):
            f.write(f"{class_name}\n")
    print(f"\nClass names saved to: {classes_file}")

def main():
    generate_yolo_labels()

if __name__ == "__main__":
    main()