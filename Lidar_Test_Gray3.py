import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from PIL import Image

# Import functions from Lidar_BEV_Image.py
from Lidar_BEV_Image import load_lidar_data

def lidar_to_bev_array(
    points,
    x_range=(0, 50),         # forward (m) 
    y_range=(-20, 20),       # left-right (m)
    z_range=(-2.5, 1.5),     # height clip (m)
    resolution=0.08,
    min_x=0.0,               # no minimum x filter
):
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

    H = int(np.ceil((x_range[1] - x_range[0]) / resolution))  # 625 pixels
    W = int(np.ceil((y_range[1] - y_range[0]) / resolution))  # 500 pixels

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

    # Convert to uint8 (single channel)
    bev_u8 = (height_norm * 255.0).astype(np.uint8)

    # Flip vertically (forward points upward)
    bev_u8 = np.flipud(bev_u8)
    
    return bev_u8

def main():
    # Load the YOLO11 model
    print("Loading YOLO model...")
    model = YOLO("./Jetson_yolov11n-kitti-LIDARBEV-only-test/train/weights/best.engine")
    
    # Define paths
    velodyne_path = "./Dataset/testing/velodyne/"
    output_path = "result/Lidar-M5-fp16"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all bin files
    bin_files = sorted(glob.glob(os.path.join(velodyne_path, "*.bin")))
    
    if not bin_files:
        print(f"No .bin files found in {velodyne_path}")
        return
    
    print(f"\nProcessing {len(bin_files)} LiDAR files...")
    
    # Process each file individually
    for idx, bin_file in enumerate(bin_files):
        file_id = os.path.splitext(os.path.basename(bin_file))[0]
        
        try:
            # Load LiDAR point cloud
            points = load_lidar_data(bin_file)
            
            # Convert to 1-channel grayscale BEV image (H, W)
            bev_image = lidar_to_bev_array(points)
            
            # Convert grayscale to 3-channel RGB
            bev_image_3ch = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2RGB)
            
            # Verify it's 3 channels
            assert bev_image_3ch.ndim == 3 and bev_image_3ch.shape[2] == 3, \
                f"Expected 3-channel image, got shape {bev_image_3ch.shape}"
            
            print(f"Processing {file_id}...")

            # Run inference on the 3-channel image
            results = model.predict(
                source=bev_image_3ch,
                save=False,
                verbose=False,
                task='obb',
                device=0,
                half=True
            )
            
            # Save result with proper filename
            for result in results:
                # Get the annotated image
                annotated_img = result.plot()
                
                # Save with original filename
                output_file = os.path.join(output_path, f"{file_id}.jpg")
                cv2.imwrite(output_file, annotated_img)
                
                print(f"  Saved: {output_file}")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(bin_files)}] Error processing {file_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Completed! Results saved to: {output_path}")

if __name__ == "__main__":
    main()