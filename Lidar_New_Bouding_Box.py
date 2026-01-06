import numpy as np
import glob
import os

def load_velodyne_points(bin_file):
    """Load LiDAR point cloud from KITTI .bin file"""
    with open(bin_file, 'rb') as f:
        points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    return points  # [N, 4] where columns are [x, y, z, intensity]

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

def check_points_in_box_3d(points_velo, corners_velo, min_points=3):
    """
    Check if bounding box contains at least min_points LiDAR points
    
    Args:
        points_velo: [N, 3] LiDAR points in velodyne coordinates
        corners_velo: [8, 3] 3D box corners in velodyne coordinates
        min_points: Minimum number of points required (default: 3)
    
    Returns:
        tuple: (has_points: bool, num_points: int)
    """
    # Get bottom 4 corners (they define the horizontal extent)
    bottom_corners = corners_velo[:4]
    
    # Get x and y ranges from bottom corners
    x_min = np.min(bottom_corners[:, 0])
    x_max = np.max(bottom_corners[:, 0])
    y_min = np.min(bottom_corners[:, 1])
    y_max = np.max(bottom_corners[:, 1])
    
    # Get z range from all 8 corners (full height)
    z_min = np.min(corners_velo[:, 2])
    z_max = np.max(corners_velo[:, 2])
    
    # Count points inside the 3D bounding box (axis-aligned approximation)
    mask = (
        (points_velo[:, 0] >= x_min) & (points_velo[:, 0] <= x_max) &
        (points_velo[:, 1] >= y_min) & (points_velo[:, 1] <= y_max) &
        (points_velo[:, 2] >= z_min) & (points_velo[:, 2] <= z_max)
    )
    
    num_points = np.sum(mask)
    return num_points >= min_points, num_points

def velodyne_to_bev_coords(points_velo, y_range=(-20, 20), x_range=(0, 50), resolution=0.08):
    """
    Convert velodyne coordinates to BEV pixel coordinates
    Accounts for vertical flip only (no horizontal flip, no padding)
    
    Args:
        points_velo: [N, 3] array of points in velodyne coordinates
        y_range: left-right range in meters (-20, 20)
        x_range: forward range in meters (0, 50)
        resolution: meters per pixel (0.08m = 8cm per pixel)
    
    Returns:
        x_img: column indices (lateral position)
        y_img: row indices (forward position, after flip)
    """
    x = points_velo[:, 0]  # Forward
    y = points_velo[:, 1]  # Left-right
    
    # Grid size (no padding)
    H = int(np.ceil((x_range[1] - x_range[0]) / resolution))  # 625 pixels
    W = int(np.ceil((y_range[1] - y_range[0]) / resolution))  # 500 pixels
    
    # Convert to grid indices (before any flips)
    ix = ((x - x_range[0]) / resolution).astype(int)
    iy = ((y - y_range[0]) / resolution).astype(int)
    
    # Apply vertical flip only (flipud): row 0 becomes row (H-1)
    ix_flipped = (H - 1) - ix
    
    # No horizontal flip, no padding
    y_img = ix_flipped  # Final row in image
    x_img = iy          # Final column in image
    
    return x_img, y_img

def get_bev_4corners(corners_velo, y_range=(-20, 20), x_range=(0, 50), resolution=0.08, image_size=(500, 625)):
    """
    Get 4 corners of bounding box in BEV image pixel coordinates
    
    Args:
        corners_velo: [8, 3] 3D box corners in velodyne coordinates
        y_range: left-right range in meters (-20, 20)
        x_range: forward range in meters (0, 50)
        resolution: meters per pixel (0.08m = 8cm per pixel)
        image_size: (width, height) of image (500, 625)
    
    Returns:
        corners_2d: [4, 2] array of corner pixels [x, y] in image coordinates
                    Order: [rear-left, rear-right, front-right, front-left]
    """
    # Get BEV coordinates for bottom 4 corners
    bottom_corners = corners_velo[:4]
    x_img, y_img = velodyne_to_bev_coords(
        bottom_corners, y_range, x_range, resolution
    )
    
    # Stack into [4, 2] array
    corners_2d = np.stack([x_img, y_img], axis=1)
    
    # [0]: front-right, [1]: front-left, [2]: rear-left, [3]: rear-right
    # [rear-left, rear-right, front-right, front-left]
    # Reorder: [2, 3, 0, 1]
    ordered_corners = corners_2d[[2, 3, 0, 1]]
    
    return ordered_corners

def generate_yolo_labels():
    """Generate YOLO-OBB format labels for BEV images with 4 corner coordinates"""
    bev_path = "./Dataset/training/lidar_bev_improved/"
    label_path = "./Dataset/training/label_2/"
    calib_path = "./Dataset/training/calib/"
    velodyne_path = "./Dataset/training/velodyne/"  # LiDAR point clouds
    output_path = "./Dataset/training/lidar_bev_improved_labels_4corners/"
    
    os.makedirs(output_path, exist_ok=True)
    
    bev_files = sorted(glob.glob(os.path.join(bev_path, "*.png")))
    
    if not bev_files:
        print(f"No BEV images found in {bev_path}")
        return
    
    # Updated class mapping - Person_sitting merged with Pedestrian
    class_map = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 3,  # Merged with Pedestrian
        'Cyclist': 4,
        'Tram': 5
    }
    
    # Classes to include in output
    output_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Tram']
    
    # BEV parameters with 0.08m resolution (8cm per pixel)
    y_range = (-20, 20)      # 40m total left-right
    x_range = (0, 50)        # 50m forward
    resolution = 0.08        # 0.08m per pixel
    
    # Image size calculation:
    # H = 50 / 0.08 = 625 pixels
    # W = 40 / 0.08 = 500 pixels
    image_size = (500, 625)  # width × height
    
    # Minimum LiDAR points required inside bounding box
    min_points_threshold = 3  # At least 3 points to keep the box
    
    print("="*70)
    print("YOLO-OBB LABEL GENERATION - BEV LiDAR with 4 Corner Format")
    print("="*70)
    print(f"Found {len(bev_files)} BEV images")
    print(f"\nBEV Configuration:")
    print(f"  Image size: {image_size[0]}×{image_size[1]} (width×height)")
    print(f"  Resolution: {resolution}m per pixel (8cm detail!)")
    print(f"  Coverage: {x_range[1]}m forward, ±{y_range[1]}m left-right")
    print(f"  Grid size: {image_size[0]}×{image_size[1]} pixels (no padding)")
    print(f"\nFiltering Settings:")
    print(f"  Minimum LiDAR points per box: {min_points_threshold}")
    print(f"  Empty boxes will be: IGNORED ✗")
    print(f"\nTransformations:")
    print(f"  Horizontal flip: Disabled")
    print(f"  Vertical flip: Enabled")
    print(f"\nOutput Format (YOLO-OBB):")
    print(f"  <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>")
    print(f"  Where (x, y) are NORMALIZED coordinates (0-1) of 4 corners")
    print(f"  Corner order: rear-left, rear-right, front-right, front-left")
    print(f"\nClass Mapping (YOLO Format):")
    for i, class_name in enumerate(output_classes):
        print(f"  {i}: {class_name}")
    print(f"\nNote: Person_sitting merged into Pedestrian (class 3)")
    print(f"Ignoring: DontCare, Misc, and other unlisted classes")
    print("="*70)
    print("Processing files...\n")
    
    total_objects = 0
    empty_boxes_filtered = 0
    class_counts = {class_name: 0 for class_name in output_classes}
    person_sitting_count = 0
    skipped_counts = {'DontCare': 0, 'Misc': 0, 'Other': 0}
    files_with_empty_boxes = 0
    
    for idx, bev_file in enumerate(bev_files):
        file_id = os.path.splitext(os.path.basename(bev_file))[0]
        
        label_file = os.path.join(label_path, file_id + ".txt")
        calib_file = os.path.join(calib_path, file_id + ".txt")
        velodyne_file = os.path.join(velodyne_path, file_id + ".bin")
        output_file = os.path.join(output_path, file_id + ".txt")
        
        try:
            # Check if calibration file exists
            if not os.path.exists(calib_file):
                print(f"  [{idx+1}/{len(bev_files)}] Skipping {file_id}: Calibration file not found")
                with open(output_file, 'w') as f:
                    pass
                continue
            
            # Check if velodyne file exists
            if not os.path.exists(velodyne_file):
                print(f"  [{idx+1}/{len(bev_files)}] Skipping {file_id}: Velodyne file not found")
                with open(output_file, 'w') as f:
                    pass
                continue
            
            # Load LiDAR point cloud
            points_velo = load_velodyne_points(velodyne_file)
            
            # Load calibration
            R0_rect, Tr_velo_to_cam, R0_rect_inv, Tr_cam_to_velo = load_calib(calib_file)
            
            # Load labels
            objects = load_labels(label_file)
            
            # Generate YOLO labels
            yolo_labels = []
            obj_count = 0
            empty_count = 0
            
            for obj in objects:
                # Skip DontCare and Misc objects
                if obj['type'] in ['DontCare', 'Misc']:
                    skipped_counts[obj['type']] += 1
                    continue
                
                # Skip objects not in our class map
                if obj['type'] not in class_map:
                    skipped_counts['Other'] += 1
                    continue
                
                # Track Person_sitting separately
                if obj['type'] == 'Person_sitting':
                    person_sitting_count += 1
                
                # Get 3D box corners in camera coordinates
                corners_cam = get_3d_box_corners(obj)
                
                # Convert to velodyne coordinates
                corners_velo = camera_to_velodyne(corners_cam, R0_rect_inv, Tr_cam_to_velo)
                
                # *** CHECK IF BOX CONTAINS LIDAR POINTS ***
                has_points, num_points = check_points_in_box_3d(
                    points_velo[:, :3], corners_velo, min_points_threshold
                )
                
                if not has_points:
                    empty_count += 1
                    empty_boxes_filtered += 1
                    continue  # IGNORE this bounding box
                
                # Get 4 corners in BEV pixel coordinates
                corners_2d = get_bev_4corners(
                    corners_velo, y_range, x_range, resolution, image_size
                )
                
                # Verify corners are within image bounds
                img_w, img_h = image_size
                if np.all((corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < img_w) &
                         (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < img_h)):
                    
                    class_id = class_map[obj['type']]
                    
                    # Normalize coordinates for YOLO-OBB format (0-1 range)
                    x1_norm = corners_2d[0, 0] / img_w  # rear-left
                    y1_norm = corners_2d[0, 1] / img_h
                    x2_norm = corners_2d[1, 0] / img_w  # rear-right
                    y2_norm = corners_2d[1, 1] / img_h
                    x3_norm = corners_2d[2, 0] / img_w  # front-right
                    y3_norm = corners_2d[2, 1] / img_h
                    x4_norm = corners_2d[3, 0] / img_w  # front-left
                    y4_norm = corners_2d[3, 1] / img_h
                    
                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                    yolo_labels.append(
                        f"{class_id} {x1_norm:.6f} {y1_norm:.6f} {x2_norm:.6f} {y2_norm:.6f} "
                        f"{x3_norm:.6f} {y3_norm:.6f} {x4_norm:.6f} {y4_norm:.6f}"
                    )
                    
                    obj_count += 1
                    total_objects += 1
                    # Count under the output class name (Person_sitting counted as Pedestrian)
                    if obj['type'] == 'Person_sitting':
                        class_counts['Pedestrian'] += 1
                    else:
                        class_counts[obj['type']] += 1
            
            # Write YOLO label file
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
                if yolo_labels:
                    f.write('\n')
            
            if empty_count > 0:
                files_with_empty_boxes += 1
                print(f"  [{idx+1}/{len(bev_files)}] {file_id}: {obj_count} objects kept, {empty_count} empty boxes filtered")
            else:
                print(f"  [{idx+1}/{len(bev_files)}] {file_id}: {obj_count} objects")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(bev_files)}] ERROR processing {file_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create empty label file on error
            with open(output_file, 'w') as f:
                pass
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"YOLO-OBB labels saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  Total valid objects: {total_objects}")
    print(f"  Empty boxes filtered: {empty_boxes_filtered}")
    print(f"  Files with empty boxes: {files_with_empty_boxes}")
    if (total_objects + empty_boxes_filtered) > 0:
        print(f"  Filtering rate: {empty_boxes_filtered/(total_objects+empty_boxes_filtered)*100:.1f}%")
    print(f"\nObjects per Class:")
    for class_name in output_classes:
        count = class_counts[class_name]
        if count > 0:
            print(f"  {class_name}: {count}")
    
    if person_sitting_count > 0:
        print(f"\n  (includes {person_sitting_count} Person_sitting objects merged into Pedestrian)")
    
    print(f"\nSkipped Objects:")
    for skip_type, count in skipped_counts.items():
        if count > 0:
            print(f"  {skip_type}: {count}")
    
    # Generate classes.txt file with only the output classes
    classes_file = os.path.join(output_path, "classes.txt")
    with open(classes_file, 'w') as f:
        for class_name in output_classes:
            f.write(f"{class_name}\n")
    print(f"\nClass names saved to: {classes_file}")
    print(f"\nOutput Format: <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>")
    print(f"Corner order: rear-left, rear-right, front-right, front-left")
    print(f"Coordinates are NORMALIZED (0-1 range) for YOLO-OBB")
    print("="*70)

def main():
    generate_yolo_labels()

if __name__ == "__main__":
    main()