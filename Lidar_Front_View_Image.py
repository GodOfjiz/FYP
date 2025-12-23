import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
from PIL import Image

def load_lidar_data(bin_file):
    """Load LiDAR point cloud from binary file"""
    # KITTI velodyne data is stored as float32
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    # Returns (x, y, z, intensity) for each point
    return points

def load_calib(calib_file):
    """Load calibration matrices from KITTI calibration file"""
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])
    
    # Projection matrix from camera 2 (left color camera)
    P2 = calib['P2'].reshape(3, 4)
    
    # Rotation and translation from velodyne to camera 0
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)
    
    # Transformation from velodyne to camera 0
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    return P2, R0_rect, Tr_velo_to_cam

def project_lidar_to_image(points, P2, R0_rect, Tr_velo_to_cam):
    """Project 3D LiDAR points to 2D image coordinates"""
    # Convert to homogeneous coordinates
    points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    
    # Transform from velodyne to camera 0
    points_cam0 = (Tr_velo_to_cam @ points_hom.T).T
    
    # Apply rectification
    points_rect = (R0_rect @ points_cam0.T).T
    
    # Only keep points in front of the camera
    front_mask = points_rect[:, 2] > 0
    points_rect = points_rect[front_mask]
    points_intensity = points[front_mask, 3]
    
    # Project to image
    points_rect_hom = points_rect[:, :3]
    points_2d = (P2 @ np.hstack([points_rect_hom, np.ones((points_rect_hom.shape[0], 1))]).T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    
    # Get depth for coloring
    depth = points_rect[:, 2]
    
    return points_2d, depth, points_intensity, front_mask, points_rect

def save_lidar_front_image(output_path, points_2d, depth, img_width, img_height):
    """
    Save only the LiDAR points as an image (no plot elements)
    
    Args:
        output_path: path to save the image
        points_2d: projected 2D points in image coordinates
        depth: depth values for projected points
        img_width: image width in pixels
        img_height: image height in pixels
    """
    # Filter points that fall within image bounds
    valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width) & \
                 (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)
    
    points_2d_valid = points_2d[valid_mask]
    depth_valid = depth[valid_mask]
    
    # Create blank image (black background)
    img_array = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Normalize depth to 0-1 range for colormap
    depth_min, depth_max = 0, 50  # meters
    depth_normalized = np.clip((depth_valid - depth_min) / (depth_max - depth_min), 0, 1)
    
    # Get colors from jet colormap
    colormap = cm.get_cmap('jet')
    colors = colormap(depth_normalized)[:, :3]  # RGB only, no alpha
    colors = (colors * 255).astype(np.uint8)
    
    # Convert points to integer pixel coordinates
    pixel_coords = points_2d_valid.astype(int)
    
    # Draw each point on the image
    for i, (x, y) in enumerate(pixel_coords):
        if 0 <= y < img_height and 0 <= x < img_width:
            img_array[y, x] = colors[i]
    
    # Save image using PIL
    img_pil = Image.fromarray(img_array)
    img_pil.save(output_path)

def process_all_files():
    """
    Process all bin files and save LiDAR front view images
    """
    # Paths to data
    velodyne_path = "./Dataset/training/velodyne/"
    image_path = "./Dataset/training/image_2/"
    calib_path = "./Dataset/training/calib/"
    output_path = "./Dataset/training/lidar_image_2/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Get all bin files
    bin_files = sorted(glob.glob(os.path.join(velodyne_path, "*.bin")))
    
    if not bin_files:
        print(f"No .bin files found in {velodyne_path}")
        return
    
    print(f"Found {len(bin_files)} LiDAR files")
    print("Processing files...")
    
    # Process each file
    for idx, bin_file in enumerate(bin_files):
        file_id = os.path.splitext(os.path.basename(bin_file))[0]
        
        # Construct corresponding file paths
        image_file = os.path.join(image_path, file_id + ".png")
        calib_file = os.path.join(calib_path, file_id + ".txt")
        output_file = os.path.join(output_path, file_id + ".png")
        
        # Check if files exist
        if not os.path.exists(image_file):
            print(f"  [{idx+1}/{len(bin_files)}] Skipping {file_id}: Image file not found")
            continue
        if not os.path.exists(calib_file):
            print(f"  [{idx+1}/{len(bin_files)}] Skipping {file_id}: Calibration file not found")
            continue
        
        try:
            # Load data
            points = load_lidar_data(bin_file)
            
            # Load calibration
            P2, R0_rect, Tr_velo_to_cam = load_calib(calib_file)
            
            # Project LiDAR to image
            points_2d, depth, intensity, front_mask, points_rect = project_lidar_to_image(points, P2, R0_rect, Tr_velo_to_cam)
            
            # Get image dimensions
            img = Image.open(image_file)
            img_width, img_height = img.size
            
            # Save LiDAR front view image
            save_lidar_front_image(output_file, points_2d, depth, img_width, img_height)
            
            print(f"  [{idx+1}/{len(bin_files)}] Saved: {file_id}.png ({points.shape[0]} points)")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(bin_files)}] Error processing {file_id}: {str(e)}")
    
    print(f"\nCompleted! All images saved to: {output_path}")

def main():
    process_all_files()

if __name__ == "__main__":
    main()