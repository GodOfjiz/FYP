import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
from PIL import Image

def load_lidar_data(bin_file):
    """Load LiDAR point cloud from binary file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points

def load_calib(calib_file):
    """Load calibration matrices from KITTI calibration file"""
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

def save_lidar_bev_image(output_path, points, y_range=(-40, 40), x_range=(0, 70), resolution=0.1):
    """
    Save Bird's Eye View (BEV) with HEIGHT-based coloring - OPTIMIZED VERSION
    Height encoding is better for object detection than distance
    """
    # Extract coordinates
    x = points[:, 0]  # Forward
    y = points[:, 1]  # Left-right
    z = points[:, 2]  # Up (HEIGHT)
    
    # Filter points in front
    front_mask = x > 0
    x_front = x[front_mask]
    y_front = y[front_mask]
    z_front = z[front_mask]
    
    # Calculate image dimensions
    width = int((y_range[1] - y_range[0]) / resolution)
    height = int((x_range[1] - x_range[0]) / resolution)
    
    # Create blank image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert to pixel coordinates
    x_img = ((y_range[1] - y_front) / resolution).astype(int)
    y_img = ((x_range[1] - x_front) / resolution).astype(int)
    
    # Filter valid pixels
    valid_mask = (x_img >= 0) & (x_img < width) & (y_img >= 0) & (y_img < height)
    x_img = x_img[valid_mask]
    y_img = y_img[valid_mask]
    z_valid = z_front[valid_mask]
    
    # HEIGHT-BASED COLORING (better for object detection)
    # Normalize height: ground level (-2m) to vehicle roof (~2m)
    height_normalized = np.clip((z_valid + 2.0) / 4.0, 0, 1)
    
    # Apply colormap: Blue (ground) -> Cyan -> Green -> Yellow -> Red (high)
    colormap = cm.get_cmap('jet')
    colors = (colormap(height_normalized)[:, :3] * 255).astype(np.uint8)
    
    # OPTIMIZED: Direct array indexing
    img_array[y_img, x_img] = colors
    
    # Save image
    Image.fromarray(img_array).save(output_path)

def process_all_files():
    """
    Process all bin files and save LiDAR images
    """
    velodyne_path = "./Dataset/training/velodyne/"
    calib_path = "./Dataset/training/calib/"
    output_bev_path = "./Dataset/training/lidar_bev/"
    
    os.makedirs(output_bev_path, exist_ok=True)
    
    bin_files = sorted(glob.glob(os.path.join(velodyne_path, "*.bin")))
    
    if not bin_files:
        print(f"No .bin files found in {velodyne_path}")
        return
    
    print(f"Found {len(bin_files)} LiDAR files")
    print("  Blue = Ground level")
    print("  Green/Yellow = Medium height (cars)")
    print("  Red = High objects")
    print("\nProcessing files...")
    
    for idx, bin_file in enumerate(bin_files):
        file_id = os.path.splitext(os.path.basename(bin_file))[0]
        output_bev_file = os.path.join(output_bev_path, file_id + ".png")
        
        try:
            points = load_lidar_data(bin_file)

            save_lidar_bev_image(output_bev_file, points)
            
            
            print(f"  [{idx+1}/{len(bin_files)}] Saved: {file_id}.png ({points.shape[0]} points)")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(bin_files)}] Error processing {file_id}: {str(e)}")
    
    print(f"\nCompleted!")
    print(f"BEV images saved to: {output_bev_path}")

def main():
    process_all_files()


if __name__ == "__main__":
    main()