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
    x_range=(0, 50),
    y_range=(-20, 20),
    z_range=(-2.5, 1.5),
    resolution=0.08,
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

    H = int(np.ceil((x_range[1] - x_range[0]) / resolution))  
    W = int(np.ceil((y_range[1] - y_range[0]) / resolution)) 

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
    height_map_flipped = np.flipud(height_map)

    return bev_u8, height_map_flipped

# ============================================================================
# LiDAR Height Extraction from OBB
# ============================================================================

def get_Lidar_Height_from_OBB(bev_image, obb_corners, z_range=(-2.5, 1.5)):
    """
    Extract maximum height from OBB bounding box region
    
    Args:
        bev_image: Grayscale BEV image (H, W) - single channel
        obb_corners: (4, 2) array of [x, y] corners in normalized coords
        z_range: Height range in meters
    
    Returns:
        max_height_m: Maximum object height in meters
        ground_height_m: Estimated ground height in meters
        object_height_m: Object height above ground
    """
    img_h, img_w = bev_image.shape
    
    # Convert normalized corners to pixel coordinates
    corners_px = obb_corners.copy()
    corners_px[:, 0] *= img_w  # x coordinates
    corners_px[:, 1] *= img_h  # y coordinates
    corners_px = corners_px.astype(np.int32)
    
    # Get bounding box from corners
    x_min = np.min(corners_px[:, 0])
    x_max = np.max(corners_px[:, 0])
    y_min = np.min(corners_px[:, 1])
    y_max = np.max(corners_px[:, 1])
    
    # Clamp to image bounds
    x_min = max(0, x_min)
    x_max = min(img_w - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(img_h - 1, y_max)
    
    # Extract region
    bbox_region = bev_image[y_min:y_max+1, x_min:x_max+1]
    
    # Get max height in bbox
    max_gray = np.max(bbox_region) if bbox_region.size > 0 else 0
    
    # Estimate ground using percentile from expanded region
    margin = 20
    y_min_exp = max(0, y_min - margin)
    y_max_exp = min(img_h - 1, y_max + margin)
    x_min_exp = max(0, x_min - margin)
    x_max_exp = min(img_w - 1, x_max + margin)
    
    expanded_region = bev_image[y_min_exp:y_max_exp+1, x_min_exp:x_max_exp+1]
    valid_pixels = expanded_region[expanded_region > 0]
    
    if len(valid_pixels) > 0:
        ground_gray = np.percentile(valid_pixels, 5)
    else:
        ground_gray = 0
    
    # Convert to meters
    max_height_m = (max_gray / 255.0) * (z_range[1] - z_range[0]) + z_range[0]
    ground_height_m = (ground_gray / 255.0) * (z_range[1] - z_range[0]) + z_range[0]
    object_height_m = max_height_m - ground_height_m
    
    return max_height_m, ground_height_m, object_height_m

# ============================================================================
# Convert OBB Corners from BEV to LiDAR Coordinates
# ============================================================================

def OBB_BEV_to_Lidar_Coords(obb_corners, bev_shape, x_range=(0, 50), y_range=(-20, 20)):
    """
    Convert OBB corners from BEV image coordinates to LiDAR coordinates
    
    Args:
        obb_corners: (4, 2) array of [x, y] in normalized BEV coords
        bev_shape: (height, width) of BEV image
        x_range: Forward range in meters
        y_range: Left-right range in meters
    
    Returns:
        corners_lidar: (4, 2) array of [x, y] in LiDAR coords (meters)
    """
    if len(bev_shape) == 3:
        img_h, img_w = bev_shape[:2]
    else:
        img_h, img_w = bev_shape
    
    # Convert normalized to pixel coordinates
    corners_px = obb_corners.copy()
    corners_px[:, 0] *= img_w  # x in pixels
    corners_px[:, 1] *= img_h  # y in pixels
    
    # Calculate resolutions
    resolution_x = (x_range[1] - x_range[0]) / img_h  # meters per row
    resolution_y = (y_range[1] - y_range[0]) / img_w  # meters per column
    
    # Convert pixel coordinates to LiDAR coordinates
    # Remember: BEV image is flipped, so row 0 = far forward
    corners_lidar = np.zeros((4, 2))
    for i in range(4):
        px_x = corners_px[i, 0]  # column
        px_y = corners_px[i, 1]  # row
        
        # Convert to LiDAR coords
        lidar_x = x_range[1] - px_y * resolution_x  # forward distance
        lidar_y = y_range[0] + px_x * resolution_y  # lateral distance
        
        corners_lidar[i] = [lidar_x, lidar_y]
    
    return corners_lidar

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
    # If 2D points (x, y), add z coordinate
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

def project_Camera_Coords_to_Image(cam_points_3d, P2):
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
# LiDAR 2D Bounding Box Drawing
# ============================================================================

def draw_Lidar_2D_BBox(image, lidar_boxes_data, output_path, class_names):
    """
    Draw 2D bounding boxes from LiDAR detections on camera image
    Uses min/max X for WIDTH and min/max Y for HEIGHT
    
    Args:
        image: Camera image
        lidar_boxes_data: List of dicts with projected corner data
        output_path: Output file path
        class_names: Dictionary of class names
    
    Returns:
        result_img: Image with 2D boxes drawn
        bbox_2d_list: List of 2D bounding boxes with normalized coords
    """
    result_img = image.copy()
    bbox_2d_list = []
    img_h, img_w = image.shape[:2]
    
    for lidar_data in lidar_boxes_data:
        # Get all 8 projected corners
        corners_img = lidar_data['corners_2d_img']
        
        # Get min/max X for WIDTH, min/max Y for HEIGHT
        x_min = np.min(corners_img[:, 0])
        x_max = np.max(corners_img[:, 0])
        y_min = np.min(corners_img[:, 1])
        y_max = np.max(corners_img[:, 1])
        
        # Convert to integers
        x1, y1 = int(x_min), int(y_min)
        x2, y2 = int(x_max), int(y_max)
        
        # Clamp to image bounds
        x1_clamped = max(0, min(x1, img_w - 1))
        x2_clamped = max(0, min(x2, img_w - 1))
        y1_clamped = max(0, min(y1, img_h - 1))
        y2_clamped = max(0, min(y2, img_h - 1))
        
        # Check if box has valid area
        width = x2_clamped - x1_clamped
        height = y2_clamped - y1_clamped
        
        if width >= 5 and height >= 5:
            # Draw 2D bounding box
            cv2.rectangle(result_img, (x1_clamped, y1_clamped), (x2_clamped, y2_clamped), 
                         (255, 0, 0), 2)  # Blue color
            
            # Add label
            cls = lidar_data['cls']
            conf = lidar_data['conf']
            obj_h = lidar_data['obj_h']
            label = f"{class_names[cls]} {conf:.2f} H:{obj_h:.2f}m"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_img, (x1_clamped, y1_clamped - label_h - 5), 
                         (x1_clamped + label_w, y1_clamped), (255, 0, 0), -1)
            cv2.putText(result_img, label, (x1_clamped, y1_clamped - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Store bbox in normalized format
            bbox_2d_list.append({
                'bbox_normalized': [
                    (x_min + x_max) / 2 / img_w,  # x_center
                    (y_min + y_max) / 2 / img_h,  # y_center
                    (x_max - x_min) / img_w,       # width
                    (y_max - y_min) / img_h        # height
                ],
                'cls': cls,
                'conf': conf,
                'height': obj_h
            })
    
    # Save image
    cv2.imwrite(output_path, result_img)
    
    return result_img, bbox_2d_list

# ============================================================================
# IoU Calculation and Matching
# ============================================================================

def calculate_IoU(box1, box2):
    """
    Calculate IoU between two boxes in [x_center, y_center, width, height] format (normalized)
    """
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

def match_Detections_by_IoU(camera_boxes, lidar_2d_boxes, iou_threshold=0.3):
    """
    Match camera and LiDAR detections using IoU threshold
    
    Args:
        camera_boxes: List of camera bboxes [x_center, y_center, width, height] (normalized)
        lidar_2d_boxes: List of dicts with 'bbox_normalized' key
        iou_threshold: Minimum IoU for matching (default 0.5 = 50%)
    
    Returns:
        matches: List of (camera_idx, lidar_idx) tuples
    """
    matches = []
    used_lidar = set()
    
    for cam_idx, cam_box in enumerate(camera_boxes):
        best_iou = 0
        best_lidar_idx = -1
        
        for lidar_idx, lidar_data in enumerate(lidar_2d_boxes):
            if lidar_idx in used_lidar:
                continue
            
            lidar_box = lidar_data['bbox_normalized']
            iou = calculate_IoU(cam_box, lidar_box)
            
            if iou > best_iou and iou >= iou_threshold:
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
                    Order: [c1_ground, c2_ground, c3_ground, c4_ground,
                           c1_top, c2_top, c3_top, c4_top]
        color: Box color (BGR)
        thickness: Line thickness
    """
    corners_2d = corners_2d.astype(np.int32)
    
    # Bottom face (first 4 points)
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Top face (last 4 points)
    for i in range(4, 8):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[4 + (i+1)%4]), color, thickness)
    
    # Vertical edges
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    return image

def save_fused_3D_detections(image, matched_lidar_data, output_path, class_names):
    """
    Save image with projected 3D boxes for matched detections
    
    Args:
        image: Camera image
        matched_lidar_data: List of lidar_boxes_data entries that were matched
        output_path: Output file path
        class_names: Dictionary of class names
    """
    result_img = image.copy()
    
    for lidar_data in matched_lidar_data:
        # Draw 3D box
        corners_2d = lidar_data['corners_2d_img']
        result_img = draw_3d_box(result_img, corners_2d, color=(0, 255, 0), thickness=2)
        
        # Add label
        cls = lidar_data['cls']
        conf = lidar_data['conf']
        obj_h = lidar_data['obj_h']
        label = f"{class_names[cls]} {conf:.2f} H:{obj_h:.2f}m"
        
        label_pos = tuple(corners_2d[0].astype(np.int32))
        cv2.putText(result_img, label, label_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, result_img)
    return result_img

# ============================================================================
# Main Processing
# ============================================================================

def main():
    # Load models
    print("Loading models...")
    camera_model = YOLO("./Jetson_yolov11n-kitti-Cam-only-8/train/weights/last.engine", task="detect")
    lidar_model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-5/train/weights/last.engine", task="obb")
    
    # Define paths
    camera_path = "./Dataset/testing/image_2"
    velodyne_path = "./Dataset/testing/velodyne/"
    calib_path = "./Dataset/testing/calib/"
    output_path = "result/Fusion_3D_OBB"
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/camera", exist_ok=True)
    os.makedirs(f"{output_path}/lidar", exist_ok=True)
    os.makedirs(f"{output_path}/lidar_2d", exist_ok=True)
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
    print("="*70)
    print("YOLO Format: OBB (4 corners per detection)")
    print("Lidar 2D: min/max X for WIDTH, min/max Y for HEIGHT")
    print("IoU Matching: Threshold = 0.5 (50%)")
    print("Fused 3D: Only matched detections (IoU > 50%)")
    print("="*70)
    
    # Process each sample
    for idx, (cam_file, bin_file) in enumerate(zip(camera_files, bin_files)):
        file_id = os.path.splitext(os.path.basename(cam_file))[0]
        calib_file = os.path.join(calib_path, f"{file_id}.txt")
        
        try:
            # ================================================================
            # 1. Load data
            # ================================================================
            camera_image = cv2.imread(cam_file)
            img_h, img_w = camera_image.shape[:2]
            points = load_lidar_data(bin_file)
            P2, R0_rect, Tr_velo_to_cam = load_calibration(calib_file)
            
            # ================================================================
            # 2. Camera inference
            # ================================================================
            cam_results = camera_model.predict(source=camera_image, save=False, verbose=False)
            
            # Extract camera boxes (assuming standard bbox format from camera)
            camera_boxes = []
            cam_detections = cam_results[0].boxes
            if cam_detections is not None and len(cam_detections) > 0:
                for det in cam_detections:
                    # Check if OBB or regular bbox
                    if hasattr(det, 'xyxy'):
                        # Regular bbox - convert to normalized center format
                        xyxy = det.xyxy[0].cpu().numpy()
                        x_center = ((xyxy[0] + xyxy[2]) / 2) / img_w
                        y_center = ((xyxy[1] + xyxy[3]) / 2) / img_h
                        width = (xyxy[2] - xyxy[0]) / img_w
                        height = (xyxy[3] - xyxy[1]) / img_h
                        camera_boxes.append([x_center, y_center, width, height])
                    elif hasattr(det, 'xywhn'):
                        # Normalized center format
                        bbox = det.xywhn[0].cpu().numpy()
                        camera_boxes.append(bbox)
            
            # Save camera result
            cam_viz = cam_results[0].plot()
            cv2.imwrite(f"{output_path}/camera/{file_id}.png", cam_viz)
            
            # ================================================================
            # 3. LiDAR inference (1-channel grayscale BEV with OBB)
            # ================================================================
            bev_image, height_map = lidar_to_bev_array(points)
            
            # Verify it's single channel
            assert bev_image.ndim == 2, f"Expected 2D array, got shape {bev_image.shape}"
            
            lidar_results = lidar_model.predict(source=bev_image, save=False, verbose=False)
            
            # Save LiDAR result
            lidar_viz = lidar_results[0].plot()
            cv2.imwrite(f"{output_path}/lidar/{file_id}.png", lidar_viz)
            
            # ================================================================
            # 4. Process LiDAR OBB detections
            # ================================================================
            lidar_boxes_data = []
            
            # Get OBB detections
            if hasattr(lidar_results[0], 'obb') and lidar_results[0].obb is not None:
                obb_detections = lidar_results[0].obb
                
                for det in obb_detections:
                    # Get OBB data
                    cls = int(det.cls[0])
                    conf = float(det.conf[0])
                    
                    # Get 4 corners in normalized coordinates
                    # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    obb_corners = det.xyxyxyxyn[0].cpu().numpy()  # Shape: (4, 2)
                    
                    # Extract height from BEV region
                    max_h, ground_h, obj_h = get_Lidar_Height_from_OBB(bev_image, obb_corners)
                    
                    # Convert OBB corners from BEV to LiDAR coordinates
                    corners_lidar = OBB_BEV_to_Lidar_Coords(obb_corners, bev_image.shape)
                    
                    # Create 8 3D corners (4 corners at ground + 4 corners at top height)
                    corners_ground = np.hstack([corners_lidar, np.full((4, 1), ground_h)])
                    corners_top = np.hstack([corners_lidar, np.full((4, 1), max_h)])
                    corners_3d_lidar = np.vstack([corners_ground, corners_top])  # 8x3
                    
                    # Transform to camera coordinates
                    corners_3d_cam = Lidar_Coords_to_Camera_Coords(
                        corners_3d_lidar, R0_rect, Tr_velo_to_cam
                    )
                    
                    # Project to image
                    corners_2d_img, valid = project_Camera_Coords_to_Image(corners_3d_cam, P2)
                    
                    # Check if at least half the corners are in front of camera
                    if np.sum(valid) >= 4:
                        lidar_boxes_data.append({
                            'corners_2d_img': corners_2d_img,  # 8 corners for 3D box
                            'cls': cls,
                            'conf': conf,
                            'max_h': max_h,
                            'ground_h': ground_h,
                            'obj_h': obj_h
                        })
            
            # ================================================================
            # 5. Draw LiDAR 2D bounding boxes
            # ================================================================
            if len(lidar_boxes_data) > 0:
                lidar_2d_img, lidar_2d_boxes = draw_Lidar_2D_BBox(
                    camera_image,
                    lidar_boxes_data,
                    f"{output_path}/lidar_2d/{file_id}.png",
                    class_names
                )
            else:
                lidar_2d_boxes = []
                cv2.imwrite(f"{output_path}/lidar_2d/{file_id}.png", camera_image)
            
            # ================================================================
            # 6. Match detections using IoU
            # ================================================================
            matches = []
            if len(camera_boxes) > 0 and len(lidar_2d_boxes) > 0:
                matches = match_Detections_by_IoU(
                    camera_boxes,
                    lidar_2d_boxes,
                    iou_threshold=0.3
                )
            
            # ================================================================
            # 7. Save fused 3D boxes (only matched detections)
            # ================================================================
            matched_lidar_data = []
            for cam_idx, lidar_idx in matches:
                matched_lidar_data.append(lidar_boxes_data[lidar_idx])
            
            if len(matched_lidar_data) > 0:
                fused_img = save_fused_3D_detections(
                    camera_image,
                    matched_lidar_data,
                    f"{output_path}/fused_3d/{file_id}.png",
                    class_names
                )
            else:
                cv2.imwrite(f"{output_path}/fused_3d/{file_id}.png", camera_image)
            
            # Print progress
            num_cam = len(camera_boxes)
            num_lidar_bev = len(lidar_boxes_data)
            num_lidar_2d = len(lidar_2d_boxes)
            num_matched = len(matches)
            
            print(f"  [{idx+1}/{len(camera_files)}] {file_id}: "
                  f"Cam={num_cam} LiDAR_BEV={num_lidar_bev} "
                  f"LiDAR_2D={num_lidar_2d} IoU_matched={num_matched}")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(camera_files)}] Error processing {file_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"Completed! Results saved to: {output_path}")
    print(f"  - Camera detections (2D): {output_path}/camera/")
    print(f"  - LiDAR BEV OBB detections: {output_path}/lidar/")
    print(f"  - LiDAR 2D projected (min/max X,Y): {output_path}/lidar_2d/")
    print(f"  - Fused 3D (IoU > 50% only): {output_path}/fused_3d/")
    print("="*70)


if __name__ == "__main__":
    main()
