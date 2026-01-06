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
                calib[key] = np.array([float(x) 
                                       for x in value.split()])
    
    P2 = calib['P2'].reshape(3, 4)
    
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)
    
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    return P2, R0_rect, Tr_velo_to_cam

def save_lidar_bev_image(
    output_path,
    points,
    x_range=(0, 50),         # forward (m)
    y_range=(-20, 20),       # left-right (m)
    z_range=(-2.5, 1.5),     # height clip (m)
    resolution=0.08,          
    min_x=0.0,
):
    """
    BEV height map for detection:
      Grayscale = max height
      Ground points appear dark, elevated points appear bright
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

    # Grid indices: row ~ x (forward), col ~ y (left-right)
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

    # Convert to uint8
    height_u8 = (height_norm * 255.0).astype(np.uint8)

    # Create grayscale image
    bev = np.stack([height_u8, height_u8, height_u8], axis=-1)

    # Flip vertically (forward points upward)
    bev = np.flipud(bev)

    Image.fromarray(bev).save(output_path)

def process_all_files():
    """
    Process all bin files and save LiDAR height images
    """
    velodyne_path = "./Dataset/training/velodyne/"
    calib_path = "./Dataset/training/calib/"
    output_bev_path = "./Dataset/training/lidar_bev_improved/"
    
    os.makedirs(output_bev_path, exist_ok=True)
    
    bin_files = sorted(glob.glob(os.path.join(velodyne_path, "*.bin")))
    
    if not bin_files:
        print(f"No .bin files found in {velodyne_path}")
        return
    
    print(f"Found {len(bin_files)} LiDAR files")

    
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
    print(f"BEV height images saved to: {output_bev_path}")

def main():
    process_all_files()


if __name__ == "__main__":
    main()