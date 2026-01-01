import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from PIL import Image

# ============================================================================
# File Loading Functions
# ============================================================================

def load_lidar_data(bin_file):
    """Load LiDAR point cloud from binary file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points

def load_calibration(calib_file):
    """Load KITTI calibration file"""
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])
    
    P2 = calib['P2'].reshape(3, 4)
    
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)
    
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    return P2, R0_rect, Tr_velo_to_cam

# ============================================================================
# BEV Generation (Grayscale - 1 Channel)
# ============================================================================

def lidar_to_bev_array(
    points,
    x_range=(0, 70),
    y_range=(-40, 40),
    z_range=(-2.5, 1.5),
    resolution=0.1,
    min_x=0.0,
):
    """
    Convert LiDAR points to 1-channel grayscale BEV image array (height only)
    
    Returns:
        bev_u8: Grayscale BEV image (H, W) - single channel
        height_map: Raw height map (H, W) for extracting actual height values
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Filter ROI
    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1]) &
        (z >= z_range[0]) & (z <= z_range[1]) &
        (x >= min_x)
    )
    x, y, z = x[mask], y[mask], z[mask]

    H = int(np.ceil((x_range[1] - x_range[0]) / resolution))  # 700 pixels
    W = int(np.ceil((y_range[1] - y_range[0]) / resolution))  # 800 pixels

    ix = ((x - x_range[0]) / resolution).astype(np.int32)
    iy = ((y - y_range[0]) / resolution).astype(np.int32)

    valid = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    ix, iy, z = ix[valid], iy[valid], z[valid]

    # Aggregate height map only
    height_map = np.full((H, W), -np.inf, dtype=np.float32)
    np.maximum.at(height_map, (ix, iy), z)

    # Normalize height
    height_map[~np.isfinite(height_map)] = z_range[0]
    height_norm = (np.clip(height_map, *z_range) - z_range[0]) / (z_range[1] - z_range[0])

    # Convert to uint8 (single channel)
    bev_u8 = (height_norm * 255.0).astype(np.uint8)
    
    # Flip vertically (forward points upward)
    bev_u8 = np.flipud(bev_u8)

    return bev_u8, height_map  # Return both display image (H, W) and raw height map

# ============================================================================
# LiDAR Height Extraction
# ============================================================================

def get_Lidar_Height_BEV_image(bev_image, bbox, z_range=(-2.5, 1.5)):
    """
    Extract maximum height and ground level from BEV bounding box
    
    Args:
        bev_image: Grayscale BEV image (H, W) - single channel
        bbox: YOLO bbox [x_center, y_center, width, height] in normalized coords
        z_range: Height range in meters
    
    Returns:
        max_height_m: Maximum object height in meters
        ground_height_m: Estimated ground height in meters
        object_height_m: Object height above ground
    """
    # bev_image is already 2D (H, W)
    img_h, img_w = bev_image.shape
    
    # Convert bbox to pixels
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width/2) * img_w)
    x2 = int((x_center + width/2) * img_w)
    y1 = int((y_center - height/2) * img_h)
    y2 = int((y_center + height/2) * img_h)
    
    # Clamp to image bounds
    x1, x2 = max(0, x1), min(img_w, x2)
    y1, y2 = max(0, y1), min(img_h, y2)
    
    # Extract bbox region
    bbox_region = bev_image[y1:y2, x1:x2]
    
    # Get max height in bbox
    max_gray = np.max(bbox_region) if bbox_region.size > 0 else 0
    
    # Estimate ground using percentile from expanded region
    margin = 20
    y1_exp = max(0, y1 - margin)
    y2_exp = min(img_h, y2 + margin)
    x1_exp = max(0, x1 - margin)
    x2_exp = min(img_w, x2 + margin)
    
    expanded_region = bev_image[y1_exp:y2_exp, x1_exp:x2_exp]
    valid_pixels = expanded_region[expanded_region > 0]
    
    if len(valid_pixels) > 0:
        ground_gray = np.percentile(valid_pixels, 5)  # 5th percentile as ground
    else:
        ground_gray = 0
    
    # Convert to meters
    max_height_m = (max_gray / 255.0) * (z_range[1] - z_range[0]) + z_range[0]
    ground_height_m = (ground_gray / 255.0) * (z_range[1] - z_range[0]) + z_range[0]
    object_height_m = max_height_m - ground_height_m
    
    return max_height_m, ground_height_m, object_height_m

# ============================================================================
# BEV Corner Extraction
# ============================================================================

def get_Lidar_Corners_BEV_image(bbox, bev_shape, x_range=(0, 70), y_range=(-40, 40)):
    """
    Get 4 corners of BEV bounding box in LiDAR coordinates (meters)
    
    Args:
        bbox: YOLO bbox [x_center, y_center, width, height] in normalized coords
        bev_shape: (height, width) of BEV image
        x_range: Forward range in meters
        y_range: Left-right range in meters
    
    Returns:
        corners_3d: Array of shape (4, 2) with [x, y] in LiDAR coords (meters)
                    Order: [front-left, front-right, rear-right, rear-left]
    """
    # Handle both 2D and 3D shapes
    if len(bev_shape) == 3:
        img_h, img_w = bev_shape[:2]
    else:
        img_h, img_w = bev_shape
    
    # Convert normalized bbox to pixels
    x_center, y_center, width, height = bbox
    x_center_px = x_center * img_w
    y_center_px = y_center * img_h
    width_px = width * img_w
    height_px = height * img_h
    
    # Get 4 corners in pixel coordinates
    x1 = x_center_px - width_px / 2
    x2 = x_center_px + width_px / 2
    y1 = y_center_px - height_px / 2
    y2 = y_center_px + height_px / 2
    
    # Remember: BEV is flipped (flipud), so y-axis is inverted
    # Top of image (y=0) = far forward (x=70m)
    # Bottom of image (y=H) = near (x=0m)
    
    # Convert pixels to LiDAR coordinates
    resolution = (x_range[1] - x_range[0]) / img_h
    
    # Top-left corner (front-left in real world)
    fl_x = x_range[1] - (y1 * resolution)  # Inverted due to flipud
    fl_y = y_range[0] + (x1 * resolution * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))
    
    # Top-right corner (front-right)
    fr_x = x_range[1] - (y1 * resolution)
    fr_y = y_range[0] + (x2 * resolution * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))
    
    # Bottom-right corner (rear-right)
    rr_x = x_range[1] - (y2 * resolution)
    rr_y = y_range[0] + (x2 * resolution * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))
    
    # Bottom-left corner (rear-left)
    rl_x = x_range[1] - (y2 * resolution)
    rl_y = y_range[0] + (x1 * resolution * (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))
    
    corners_2d = np.array([
        [fl_x, fl_y],  # Front-left
        [fr_x, fr_y],  # Front-right
        [rr_x, rr_y],  # Rear-right
        [rl_x, rl_y]   # Rear-left
    ])
    
    return corners_2d

# ============================================================================
# Coordinate Transformation
# ============================================================================

def Lidar_Coords_to_Camera_Coords(lidar_points, R0_rect, Tr_velo_to_cam):
    """
    Transform LiDAR coordinates to camera coordinates
    
    Args:
        lidar_points: (N, 2) or (N, 3) array of [x, y] or [x, y, z] in LiDAR coords
        R0_rect: 4x4 rectification matrix
        Tr_velo_to_cam: 4x4 velodyne to camera transformation
    
    Returns:
        cam_points: (N, 3) array of [x, y, z] in camera coordinates
    """
    # If 2D points (x, y), add z=0 for ground plane
    if lidar_points.shape[1] == 2:
        lidar_points_3d = np.hstack([
            lidar_points,
            np.zeros((lidar_points.shape[0], 1))
        ])
    else:
        lidar_points_3d = lidar_points
    
    # Convert to homogeneous coordinates
    N = lidar_points_3d.shape[0]
    lidar_hom = np.hstack([lidar_points_3d, np.ones((N, 1))])  # (N, 4)
    
    # Transform: LiDAR -> Camera -> Rectified Camera
    cam_hom = (R0_rect @ Tr_velo_to_cam @ lidar_hom.T).T  # (N, 4)
    
    # Convert back to 3D
    cam_points = cam_hom[:, :3]
    
    return cam_points

# ============================================================================
# 3D to 2D Projection
# ============================================================================

def project_Lidar_Converted_Points_to_Camera_Image(cam_points_3d, P2):
    """
    Project 3D camera coordinates to 2D image coordinates
    
    Args:
        cam_points_3d: (N, 3) array of [x, y, z] in camera coords
        P2: 3x4 camera projection matrix
    
    Returns:
        image_points: (N, 2) array of [u, v] pixel coordinates
        valid_mask: Boolean mask of points in front of camera (z > 0)
    """
    # Convert to homogeneous coordinates
    N = cam_points_3d.shape[0]
    cam_hom = np.hstack([cam_points_3d, np.ones((N, 1))])  # (N, 4)
    
    # Project to image plane
    img_hom = (P2 @ cam_hom.T).T  # (N, 3)
    
    # Normalize by depth
    valid_mask = img_hom[:, 2] > 0  # Points in front of camera
    
    image_points = np.zeros((N, 2))
    image_points[valid_mask] = img_hom[valid_mask, :2] / img_hom[valid_mask, 2:3]
    
    return image_points, valid_mask

# ============================================================================
# Detection Matching
# ============================================================================

def Combine_Camera_and_Lidar_Detections(camera_boxes, lidar_boxes, iou_threshold=0.1):
    """
    Match camera and LiDAR detections using 2D IoU
    
    Args:
        camera_boxes: List of camera detections (YOLO results)
        lidar_boxes: List of LiDAR detections (YOLO results)
        iou_threshold: Minimum IoU for matching
    
    Returns:
        matches: List of (camera_idx, lidar_idx) tuples
    """
    def box_iou(box1, box2):
        """Calculate IoU between two boxes [x_center, y_center, width, height]"""
        # Convert to [x1, y1, x2, y2]
        b1_x1 = box1[0] - box1[2] / 2
        b1_y1 = box1[1] - box1[3] / 2
        b1_x2 = box1[0] + box1[2] / 2
        b1_y2 = box1[1] + box1[3] / 2
        
        b2_x1 = box2[0] - box2[2] / 2
        b2_y1 = box2[1] - box2[3] / 2
        b2_x2 = box2[0] + box2[2] / 2
        b2_y2 = box2[1] + box2[3] / 2
        
        # Intersection
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    matches = []
    used_lidar = set()
    
    for cam_idx, cam_box in enumerate(camera_boxes):
        best_iou = 0
        best_lidar_idx = -1
        
        for lidar_idx, lidar_box in enumerate(lidar_boxes):
            if lidar_idx in used_lidar:
                continue
            
            # Project LiDAR bbox to camera view for comparison
            # For simplicity, we'll use spatial proximity heuristic
            # In practice, you'd project LiDAR bbox corners to camera
            iou = box_iou(cam_box[:4], lidar_box[:4])
            
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_lidar_idx = lidar_idx
        
        if best_lidar_idx >= 0:
            matches.append((cam_idx, best_lidar_idx))
            used_lidar.add(best_lidar_idx)
    
    return matches

# ============================================================================
# 3D Box Drawing
# ============================================================================

def draw_3d_box(image, corners_2d, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box on image
    
    Args:
        image: Input image
        corners_2d: (8, 2) array of projected corners
                    Order: [fbl, fbr, frr, frl, rbl, rbr, rrr, rrl]
                           (front-bottom-left, etc.)
        color: Box color (BGR)
        thickness: Line thickness
    """
    corners_2d = corners_2d.astype(np.int32)
    
    # Front face (bottom 4 points)
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Rear face (top 4 points)
    for i in range(4, 8):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[4 + (i+1)%4]), color, thickness)
    
    # Vertical edges
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    return image

def save_fused_detections(image, corners_3d_list, output_path, labels=None):
    """
    Save image with projected 3D boxes
    
    Args:
        image: Camera image
        corners_3d_list: List of (8, 2) corner arrays
        output_path: Output file path
        labels: Optional list of label strings
    """
    result_img = image.copy()
    
    for idx, corners_2d in enumerate(corners_3d_list):
        # Draw 3D box
        result_img = draw_3d_box(result_img, corners_2d, color=(0, 255, 0), thickness=2)
        
        # Add label if provided
        if labels and idx < len(labels):
            label_pos = tuple(corners_2d[0].astype(np.int32))
            cv2.putText(result_img, labels[idx], label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, result_img)
    return result_img

# ============================================================================
# Main Processing
# ============================================================================

def main():
    # Load models
    print("Loading models...")
    camera_model = YOLO("./Jetson_yolov11n-kitti-Cam-only-4/train/weights/best.pt")
    lidar_model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-4/train/weights/best.pt")
    
    # Define paths
    camera_path = "./Dataset/testing/image_2"
    velodyne_path = "./Dataset/testing/velodyne/"
    calib_path = "./Dataset/testing/calib/"
    output_path = "result/Fusion_3D"
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/camera", exist_ok=True)
    os.makedirs(f"{output_path}/lidar", exist_ok=True)
    os.makedirs(f"{output_path}/fused_3d", exist_ok=True)
    
    # Get file lists
    camera_files = sorted(glob.glob(os.path.join(camera_path, "*.png")))
    bin_files = sorted(glob.glob(os.path.join(velodyne_path, "*.bin")))
    
    if not camera_files or not bin_files:
        print("Error: Missing camera or LiDAR files")
        return
    
    # Get class names
    class_names = camera_model.names
    
    print(f"\nProcessing {len(camera_files)} samples...")
    
    # Process each sample
    for idx, (cam_file, bin_file) in enumerate(zip(camera_files, bin_files)):
        file_id = os.path.splitext(os.path.basename(cam_file))[0]
        calib_file = os.path.join(calib_path, f"{file_id}.txt")
        
        try:
            # ================================================================
            # 1. Load data
            # ================================================================
            camera_image = cv2.imread(cam_file)
            points = load_lidar_data(bin_file)
            P2, R0_rect, Tr_velo_to_cam = load_calibration(calib_file)
            
            # ================================================================
            # 2. Camera inference
            # ================================================================
            cam_results = camera_model.predict(source=camera_image, save=False, verbose=False)
            cam_detections = cam_results[0].boxes
            
            # Save camera result
            cam_viz = cam_results[0].plot()
            cv2.imwrite(f"{output_path}/camera/{file_id}.png", cam_viz)
            
            # ================================================================
            # 3. LiDAR inference (1-channel grayscale BEV)
            # ================================================================
            bev_image, height_map = lidar_to_bev_array(points)
            
            # Verify it's single channel
            assert bev_image.ndim == 2, f"Expected 2D array, got shape {bev_image.shape}"
            
            lidar_results = lidar_model.predict(source=bev_image, save=False, verbose=False)
            lidar_detections = lidar_results[0].boxes
            
            # Save LiDAR result
            lidar_viz = lidar_results[0].plot()
            cv2.imwrite(f"{output_path}/lidar/{file_id}.png", lidar_viz)
            
            # ================================================================
            # 4. Process each LiDAR detection
            # ================================================================
            corners_3d_list = []
            labels_list = []
            
            for det in lidar_detections:
                # Get bbox in normalized coords
                bbox = det.xywhn[0].cpu().numpy()  # [x_center, y_center, width, height]
                cls = int(det.cls[0])
                conf = float(det.conf[0])
                
                # Extract height
                max_h, ground_h, obj_h = get_Lidar_Height_BEV_image(bev_image, bbox)
                
                # Get 2D corners in LiDAR coords
                corners_2d = get_Lidar_Corners_BEV_image(bbox, bev_image.shape)
                
                # Create 8 corners (4 ground + 4 top)
                corners_ground = np.hstack([corners_2d, np.full((4, 1), ground_h)])
                corners_top = np.hstack([corners_2d, np.full((4, 1), max_h)])
                corners_3d_lidar = np.vstack([corners_ground, corners_top])
                
                # Transform to camera coordinates
                corners_3d_cam = Lidar_Coords_to_Camera_Coords(
                    corners_3d_lidar, R0_rect, Tr_velo_to_cam
                )
                
                # Project to image
                corners_2d_img, valid = project_Lidar_Converted_Points_to_Camera_Image(
                    corners_3d_cam, P2
                )
                
                # Only keep if all corners are valid and in image bounds
                img_h, img_w = camera_image.shape[:2]
                in_bounds = np.all(
                    (corners_2d_img[:, 0] >= 0) & 
                    (corners_2d_img[:, 0] < img_w) &
                    (corners_2d_img[:, 1] >= 0) & 
                    (corners_2d_img[:, 1] < img_h)
                )
                
                if np.all(valid) and in_bounds:
                    corners_3d_list.append(corners_2d_img)
                    label = f"{class_names[cls]} {conf:.2f} H:{obj_h:.2f}m"
                    labels_list.append(label)
            
            # ================================================================
            # 5. Save fused result
            # ================================================================
            fused_img = save_fused_detections(
                camera_image,
                corners_3d_list,
                f"{output_path}/fused_3d/{file_id}.png",
                labels_list
            )
            
            print(f"  [{idx+1}/{len(camera_files)}] {file_id}: "
                  f"Cam={len(cam_detections)} LiDAR={len(lidar_detections)} "
                  f"Fused={len(corners_3d_list)}")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(camera_files)}] Error processing {file_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Completed! Results saved to: {output_path}")
    print(f"  - Camera detections (2D): {output_path}/camera/")
    print(f"  - LiDAR BEV detections (1-channel grayscale): {output_path}/lidar/")
    print(f"  - Fused 3D boxes: {output_path}/fused_3d/")


if __name__ == "__main__":
    main()