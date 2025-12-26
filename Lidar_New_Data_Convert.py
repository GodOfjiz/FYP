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
    x_range=(0.0, 100.0),    # forward (m) - extended for better aspect ratio
    y_range=(-82.5, 82.5),   # left-right (m) - ±82.5m = 165m total
    z_range=(-2.5, 1.5),     # height clip (m)
    resolution=0.1,          
    min_x=2.0,               # remove ego-close points
):
    """
    BEV feature map for detection with camera-matching aspect ratio:
      R = density (log scaled)
      G = max height (clipped+normalized)
      B = max intensity (robust normalized)
      
    Output: ~825×500 → padded to 832×512 for 32-divisibility
    Aspect ratio: ~1.625:1 (closer to camera's 3.31:1)
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    i = points[:, 3] if points.shape[1] > 3 else np.zeros_like(z)

    # Filter ROI
    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1]) &
        (z >= z_range[0]) & (z <= z_range[1]) &
        (x >= min_x)
    )
    x, y, z, i = x[mask], y[mask], z[mask], i[mask]

    H = int(np.ceil((x_range[1] - x_range[0]) / resolution))  # 500 pixels
    W = int(np.ceil((y_range[1] - y_range[0]) / resolution))  # 825 pixels

    # Grid indices: row ~ x (forward), col ~ y (left-right)
    ix = ((x - x_range[0]) / resolution).astype(np.int32)
    iy = ((y - y_range[0]) / resolution).astype(np.int32)

    valid = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    ix, iy, z, i = ix[valid], iy[valid], z[valid], i[valid]

    # Aggregate maps
    height_map = np.full((H, W), -np.inf, dtype=np.float32)
    intensity_map = np.zeros((H, W), dtype=np.float32)
    density_map = np.zeros((H, W), dtype=np.float32)

    np.maximum.at(height_map, (ix, iy), z)
    np.maximum.at(intensity_map, (ix, iy), i)
    np.add.at(density_map, (ix, iy), 1.0)

    # Normalize height
    height_map[~np.isfinite(height_map)] = z_range[0]
    height_norm = (np.clip(height_map, *z_range) - z_range[0]) / (z_range[1] - z_range[0])

    # Normalize intensity robustly (per frame)
    p99 = np.percentile(intensity_map[intensity_map > 0], 99) if np.any(intensity_map > 0) else 1.0
    intensity_norm = np.clip(intensity_map / (p99 + 1e-6), 0.0, 1.0)

    # Normalize density (log scale)
    # Adjusted for 0.2m resolution (fewer points per pixel than 0.1m)
    density_norm = np.clip(np.log1p(density_map) / np.log(1.0 + 32.0), 0.0, 1.0)

    bev = np.stack([
        density_norm,   # R
        height_norm,    # G
        intensity_norm  # B
    ], axis=-1)

    bev_u8 = (bev * 255.0).astype(np.uint8)

    # Flip vertically (forward points upward) AND horizontally (mirror left-right)
    bev_u8 = np.flipud(bev_u8)
    bev_u8 = np.fliplr(bev_u8)  # NEW: Horizontal flip

    # Pad to make dimensions divisible by 32 (832×512)
    target_h = ((H + 31) // 32) * 32  # 512
    target_w = ((W + 31) // 32) * 32  # 832
    
    if H < target_h or W < target_w:
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_h = (target_h - H) // 2
        pad_w = (target_w - W) // 2
        padded[pad_h:pad_h+H, pad_w:pad_w+W] = bev_u8
        bev_u8 = padded

    Image.fromarray(bev_u8).save(output_path)

def process_all_files():
    """
    Process all bin files and save LiDAR images
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
    print("\n" + "="*60)
    print("BEV Generation Settings:")
    print("  Resolution: 0.2m per pixel")
    print("  Forward range: 0-100m")
    print("  Left-right range: ±82.5m (165m total)")
    print("  Output size: 832×512 pixels (padded)")
    print("  Aspect ratio: 1.625:1 (matches camera better)")
    print("  Horizontal flip: Enabled (mirrors left-right)")
    print("="*60)
    print("\nColor encoding:")
    print("  Red = Point density (log scale)")
    print("  Green = Height (ground=dark, elevated=bright)")
    print("  Blue = Intensity")
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