import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from PIL import Image

# Import functions from Lidar_BEV_Image.py
from Lidar_BEV_Image import load_lidar_data

# BEV parameters (must match your BEV creation)
BEV_PARAMS = {
    'x_range': (0.0, 100.0),
    'y_range': (-82.5, 82.5),
    'z_range': (-2.5, 1.5),
    'resolution': 0.1,
    'min_x': 2.0
}

def lidar_to_bev_array(
    points,
    x_range=(0.0, 100.0),
    y_range=(-82.5, 82.5),
    z_range=(-2.5, 1.5),
    resolution=0.1,
    min_x=2.0,
):
    """Convert LiDAR point cloud to BEV image"""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    i = points[:, 3] if points.shape[1] > 3 else np.zeros_like(z)

    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1]) &
        (z >= z_range[0]) & (z <= z_range[1]) &
        (x >= min_x)
    )
    x, y, z, i = x[mask], y[mask], z[mask], i[mask]

    H = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    W = int(np.ceil((y_range[1] - y_range[0]) / resolution))

    ix = ((x - x_range[0]) / resolution).astype(np.int32)
    iy = ((y - y_range[0]) / resolution).astype(np.int32)

    valid = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    ix, iy, z, i = ix[valid], iy[valid], z[valid], i[valid]

    height_map = np.full((H, W), -np.inf, dtype=np.float32)
    intensity_map = np.zeros((H, W), dtype=np.float32)
    density_map = np.zeros((H, W), dtype=np.float32)

    np.maximum.at(height_map, (ix, iy), z)
    np.maximum.at(intensity_map, (ix, iy), i)
    np.add.at(density_map, (ix, iy), 1.0)

    height_map[~np.isfinite(height_map)] = z_range[0]
    height_norm = (np.clip(height_map, *z_range) - z_range[0]) / (z_range[1] - z_range[0])

    p99 = np.percentile(intensity_map[intensity_map > 0], 99) if np.any(intensity_map > 0) else 1.0
    intensity_norm = np.clip(intensity_map / (p99 + 1e-6), 0.0, 1.0)

    density_norm = np.clip(np.log1p(density_map) / np.log(1.0 + 32.0), 0.0, 1.0)

    bev = np.stack([density_norm, height_norm, intensity_norm], axis=-1)
    bev_u8 = (bev * 255.0).astype(np.uint8)
    bev_u8 = np.flipud(bev_u8)
    bev_u8 = np.fliplr(bev_u8)

    target_h = ((H + 31) // 32) * 32
    target_w = ((W + 31) // 32) * 32
    
    pad_h = (target_h - H) // 2
    pad_w = (target_w - W) // 2
    
    if H < target_h or W < target_w:
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[pad_h:pad_h+H, pad_w:pad_w+W] = bev_u8
        bev_u8 = padded

    return bev_u8, pad_h, pad_w, H, W


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


def bev_box_to_realworld(bev_box, pad_h, pad_w, H, W, bev_params):
    """
    Convert BEV bounding box from pixels to real-world coordinates
    Returns: (x_real, y_real, width_real, length_real) in meters
    """
    x1_bev, y1_bev, x2_bev, y2_bev = bev_box
    
    # Get center
    cx_bev = (x1_bev + x2_bev) / 2
    cy_bev = (y1_bev + y2_bev) / 2
    
    # Remove padding
    cx_bev_unpadded = cx_bev - pad_w
    cy_bev_unpadded = cy_bev - pad_h
    
    # Reverse horizontal flip (fliplr)
    cx_bev_original = W - cx_bev_unpadded
    
    # Reverse vertical flip (flipud)
    cy_bev_original = H - cy_bev_unpadded
    
    # Convert to real-world coordinates
    # Note: In BEV creation, row (y) corresponds to X (forward), col (x) corresponds to Y (left-right)
    x_real = bev_params['x_range'][0] + (cy_bev_original * bev_params['resolution'])
    y_real = bev_params['y_range'][0] + (cx_bev_original * bev_params['resolution'])
    
    # Convert box dimensions
    width_bev = x2_bev - x1_bev
    height_bev = y2_bev - y1_bev
    
    width_real = width_bev * bev_params['resolution']
    length_real = height_bev * bev_params['resolution']
    
    return x_real, y_real, width_real, length_real


def project_point_to_camera(point_3d_lidar, P2, R0_rect, Tr_velo_to_cam):
    """
    Project a 3D point from LiDAR coordinates to camera image
    Returns: (pixel_x, pixel_y) or None if behind camera
    """
    # Convert to homogeneous coordinates
    point_hom = np.array([point_3d_lidar[0], point_3d_lidar[1], point_3d_lidar[2], 1.0])
    
    # Transform to camera coordinates
    point_cam = Tr_velo_to_cam @ point_hom
    point_cam = R0_rect @ point_cam
    
    # Check if in front of camera
    if point_cam[2] <= 0:
        return None
    
    # Project to image
    point_img = P2 @ point_cam
    pixel_x = point_img[0] / point_img[2]
    pixel_y = point_img[1] / point_img[2]
    
    return pixel_x, pixel_y


def project_bev_detection_to_camera(bev_box, pad_h, pad_w, H, W, bev_params, P2, R0_rect, Tr_velo_to_cam, class_id):
    """
    Project BEV detection to camera image space for matching
    Returns: [x1, y1, x2, y2] in camera image pixels, and 3D position
    """
    # Convert BEV box to real-world coordinates
    x_real, y_real, width_real, length_real = bev_box_to_realworld(
        bev_box, pad_h, pad_w, H, W, bev_params
    )
    
    # Default heights based on class
    class_heights = {
        0: 1.5,   # Car
        1: 1.7,   # Pedestrian
        2: 1.7,   # Cyclist
    }
    default_height = class_heights.get(class_id, 1.5)
    
    # Assume object sits on ground (z = -1.65m is typical camera height in KITTI)
    z_bottom = -1.65
    z_top = z_bottom + default_height
    
    # Project 8 corners of approximate 3D box to get 2D bounding box
    corners_3d = []
    for dx in [-length_real/2, length_real/2]:
        for dy in [-width_real/2, width_real/2]:
            for dz in [z_bottom, z_top]:
                corners_3d.append([x_real + dx, y_real + dy, dz])
    
    # Project all corners
    projected_points = []
    for corner in corners_3d:
        pixel = project_point_to_camera(corner, P2, R0_rect, Tr_velo_to_cam)
        if pixel is not None:
            projected_points.append(pixel)
    
    if len(projected_points) < 4:
        return None, None
    
    # Get 2D bounding box
    projected_points = np.array(projected_points)
    x1 = np.min(projected_points[:, 0])
    y1 = np.min(projected_points[:, 1])
    x2 = np.max(projected_points[:, 0])
    y2 = np.max(projected_points[:, 1])
    
    return [x1, y1, x2, y2], (x_real, y_real, z_bottom + default_height/2)


def estimate_3d_from_fusion(cam_box, lidar_position, P2, class_id):
    """
    Estimate 3D bounding box from matched camera + LiDAR detection
    cam_box: [x1, y1, x2, y2] in camera image
    lidar_position: (x, y, z) in camera coordinates
    Returns: [h, w, l, x, y, z, rotation_y]
    """
    x1, y1, x2, y2 = cam_box
    x_cam, y_cam, z_cam = lidar_position
    
    # Get focal lengths from projection matrix
    focal_length_x = P2[0, 0]
    focal_length_y = P2[1, 1]
    
    # Depth is the forward distance (Z in camera coordinates)
    depth = z_cam
    
    # Estimate 3D height from 2D box height
    pixel_height = y2 - y1
    height_3d = (pixel_height * depth) / focal_length_y
    
    # Estimate 3D width from 2D box width
    pixel_width = x2 - x1
    width_3d = (pixel_width * depth) / focal_length_x
    
    # Default lengths based on class
    class_lengths = {
        0: 3.9,   # Car
        1: 0.8,   # Pedestrian
        2: 1.7,   # Cyclist
    }
    length_3d = class_lengths.get(class_id, 3.9)
    
    # Clamp dimensions to reasonable ranges
    height_3d = np.clip(height_3d, 0.5, 3.0)
    width_3d = np.clip(width_3d, 0.5, 2.5)
    
    # Position
    x_center = x_cam
    y_center = y_cam
    z_center = z_cam
    
    return [height_3d, width_3d, length_3d, x_center, y_center, z_center, 0.0]


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def fuse_detections(camera_results, lidar_results, pad_h, pad_w, H, W, bev_params, 
                    P2, R0_rect, Tr_velo_to_cam, fusion_method='weighted'):
    """
    Fuse camera and LiDAR detections (now properly in same coordinate system)
    Returns: list of [x1, y1, x2, y2, conf, cls, source, box_3d]
    """
    camera_dets = []
    lidar_dets_camera_space = []
    lidar_positions_3d = []
    
    # Extract camera detections (already in camera image space)
    if camera_results and len(camera_results[0].boxes) > 0:
        boxes = camera_results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            camera_dets.append([x1, y1, x2, y2, conf, cls, 'camera', None])
    
    # Convert LiDAR BEV detections to camera image space
    if lidar_results and len(lidar_results[0].boxes) > 0:
        boxes = lidar_results[0].boxes
        for i in range(len(boxes)):
            x1_bev, y1_bev, x2_bev, y2_bev = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            # Project to camera image space
            cam_box, position_3d = project_bev_detection_to_camera(
                [x1_bev, y1_bev, x2_bev, y2_bev],
                pad_h, pad_w, H, W, bev_params,
                P2, R0_rect, Tr_velo_to_cam, cls
            )
            
            if cam_box is not None:
                lidar_dets_camera_space.append(cam_box + [conf, cls, 'lidar', None])
                lidar_positions_3d.append(position_3d)
    
    # Now both are in camera image space - can do proper matching
    fused = []
    used_lidar = set()
    
    for cam_det in camera_dets:
        best_match = None
        best_match_3d = None
        best_iou = 0.3
        best_idx = -1
        
        for idx, lid_det in enumerate(lidar_dets_camera_space):
            if idx in used_lidar:
                continue
            if cam_det[5] != lid_det[5]:  # Different class
                continue
            
            iou = compute_iou(cam_det[:4], lid_det[:4])
            if iou > best_iou:
                best_iou = iou
                best_match = lid_det
                best_match_3d = lidar_positions_3d[idx]
                best_idx = idx
        
        if best_match:
            used_lidar.add(best_idx)
            
            # Create fused 2D box
            if fusion_method == 'weighted':
                fused_box = [(cam_det[i] + best_match[i]) / 2 for i in range(4)]
                fused_conf = (cam_det[4] + best_match[4]) / 2
            elif fusion_method == 'max':
                if cam_det[4] > best_match[4]:
                    fused_box = cam_det[:4]
                    fused_conf = cam_det[4]
                else:
                    fused_box = best_match[:4]
                    fused_conf = best_match[4]
            else:
                fused_box = cam_det[:4]
                fused_conf = cam_det[4]
            
            # Estimate 3D box using camera height + LiDAR position
            box_3d = estimate_3d_from_fusion(
                fused_box, best_match_3d, P2, cam_det[5]
            )
            
            fused.append(fused_box + [fused_conf, cam_det[5], 'fused', box_3d])
        else:
            # Camera only - estimate 3D using default depth
            fused.append(cam_det)
    
    # Add unmatched LiDAR detections
    for idx, lid_det in enumerate(lidar_dets_camera_space):
        if idx not in used_lidar:
            # Create 3D box from LiDAR position
            position_3d = lidar_positions_3d[idx]
            box_3d = estimate_3d_from_fusion(
                lid_det[:4], position_3d, P2, lid_det[5]
            )
            lid_det[7] = box_3d
            fused.append(lid_det)
    
    return fused


def draw_3d_box(image, box_3d, P2, R0_rect, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box on image
    box_3d: [h, w, l, x, y, z, rotation_y]
    """
    if box_3d is None:
        return image
    
    h, w, l, x, y, z, ry = box_3d
    
    # 3D bounding box corners in camera coordinate system
    corners_3d = np.array([
        [l/2, 0, w/2],      # Front-right-bottom
        [l/2, 0, -w/2],     # Front-left-bottom
        [-l/2, 0, -w/2],    # Back-left-bottom
        [-l/2, 0, w/2],     # Back-right-bottom
        [l/2, -h, w/2],     # Front-right-top
        [l/2, -h, -w/2],    # Front-left-top
        [-l/2, -h, -w/2],   # Back-left-top
        [-l/2, -h, w/2],    # Back-right-top
    ])
    
    # Rotation matrix around Y-axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotate and translate
    corners_3d = corners_3d @ R.T
    corners_3d[:, 0] += x
    corners_3d[:, 1] += y
    corners_3d[:, 2] += z
    
    # Convert to homogeneous coordinates
    corners_3d_hom = np.hstack([corners_3d, np.ones((8, 1))])
    
    # Project to image plane
    corners_img = corners_3d_hom @ R0_rect.T @ P2.T
    corners_img[:, 0] /= corners_img[:, 2]
    corners_img[:, 1] /= corners_img[:, 2]
    corners_2d = corners_img[:, :2].astype(np.int32)
    
    # Draw the 12 edges of the 3D box
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for start, end in edges:
        cv2.line(image, tuple(corners_2d[start]), tuple(corners_2d[end]), color, thickness)
    
    # Draw front face differently (thicker)
    front_edges = [(0, 1), (1, 5), (5, 4), (4, 0)]
    for start, end in front_edges:
        cv2.line(image, tuple(corners_2d[start]), tuple(corners_2d[end]), color, thickness + 1)
    
    return image


def visualize_fused_detections(image, detections, P2, R0_rect, class_names):
    """Draw fused detections with 3D boxes on camera image"""
    result_img = image.copy()
    
    colors = {
        'camera': (255, 0, 0),     # Blue
        'lidar': (0, 0, 255),      # Red
        'fused': (0, 255, 0)       # Green
    }
    
    for det in detections:
        x1, y1, x2, y2, conf, cls, source, box_3d = det
        
        color = colors.get(source, (255, 255, 255))
        
        # Draw 3D box if available
        if box_3d is not None:
            result_img = draw_3d_box(result_img, box_3d, P2, R0_rect, color, thickness=2)
        
        # Draw 2D box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names[int(cls)]} {conf:.2f}"
        if box_3d is not None:
            depth = box_3d[5]  # z coordinate
            label += f" {depth:.1f}m"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
        cv2.putText(result_img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_img


def main():
    # Load models
    print("Loading models...")
    camera_model = YOLO("./Jetson_yolov11n-kitti-Cam-only-4/train/weights/best.pt")
    lidar_model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-3/train/weights/best.pt")
    
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
    
    # Process each sample
    for idx, (cam_file, bin_file) in enumerate(zip(camera_files, bin_files)):
        file_id = os.path.splitext(os.path.basename(cam_file))[0]
        calib_file = os.path.join(calib_path, f"{file_id}.txt")
        
        try:
            # Load calibration
            if os.path.exists(calib_file):
                P2, R0_rect, Tr_velo_to_cam = load_calibration(calib_file)
            else:
                print(f"Warning: Calibration file not found for {file_id}, skipping")
                continue
            
            # Load camera image
            camera_image = cv2.imread(cam_file)
            
            # Load LiDAR points
            points = load_lidar_data(bin_file)
            
            # Convert LiDAR to BEV for detection
            bev_image, pad_h, pad_w, H, W = lidar_to_bev_array(
                points,
                **BEV_PARAMS
            )
            
            # Run inference
            camera_results = camera_model.predict(source=camera_image, save=False, verbose=False)
            lidar_results = lidar_model.predict(source=bev_image, save=False, verbose=False)
            
            # Fuse detections (now with proper coordinate transformation)
            fused_detections = fuse_detections(
                camera_results, lidar_results,
                pad_h, pad_w, H, W, BEV_PARAMS,
                P2, R0_rect, Tr_velo_to_cam,
                fusion_method='weighted'
            )
            
            # Save individual results
            camera_result_img = camera_results[0].plot()
            cv2.imwrite(f"{output_path}/camera/{file_id}.png", camera_result_img)
            
            lidar_result_img = lidar_results[0].plot()
            cv2.imwrite(f"{output_path}/lidar/{file_id}.png", lidar_result_img)
            
            # Visualize fused 3D boxes
            fused_3d_img = visualize_fused_detections(
                camera_image, fused_detections, P2, R0_rect, class_names
            )
            cv2.imwrite(f"{output_path}/fused_3d/{file_id}.png", fused_3d_img)
            
            # Count detections with 3D boxes
            num_3d = sum(1 for det in fused_detections if det[7] is not None)
            
            print(f"[{idx+1}/{len(camera_files)}] Processed: {file_id} - "
                  f"Cam: {len(camera_results[0].boxes)}, "
                  f"LiDAR: {len(lidar_results[0].boxes)}, "
                  f"Fused: {len(fused_detections)} ({num_3d} with 3D)")
            
        except Exception as e:
            print(f"[{idx+1}/{len(camera_files)}] Error processing {file_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\nCompleted! Results saved to: {output_path}")
    print(f"  - Camera detections (2D): {output_path}/camera/")
    print(f"  - LiDAR BEV detections: {output_path}/lidar/")
    print(f"  - Fused 3D boxes: {output_path}/fused_3d/")


if __name__ == "__main__":
    main()