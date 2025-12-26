import os
import shutil
import glob
from sklearn.model_selection import train_test_split
import random

def organize_yolo_dataset():
    # Paths
    images_path = "./Dataset/training/lidar_bev_improved/"
    labels_path = "./Dataset/training/lidar_bev_improved_labels/"
    output_path = "./Lidar Data/" 
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(images_path, "*.png")))
    
    # Extract file IDs
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    
    print(f"Total images found: {len(file_ids)}")
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    # First split: separate test set
    train_val_ids, test_ids = train_test_split(
        file_ids, 
        test_size=test_ratio, 
        random_state=42
    )
    
    # Second split: separate train and val
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio/(train_ratio + val_ratio),  # Adjust ratio
        random_state=42
    )
    
    print(f"Train: {len(train_ids)} images")
    print(f"Val: {len(val_ids)} images")
    print(f"Test: {len(test_ids)} images")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)
    
    # Copy files
    def copy_files(file_ids, split_name):
        for file_id in file_ids:
            # Copy image
            src_img = os.path.join(images_path, f"{file_id}.png")
            dst_img = os.path.join(output_path, split_name, 'images', f"{file_id}.png")
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_label = os.path.join(labels_path, f"{file_id}.txt")
            dst_label = os.path.join(output_path, split_name, 'labels', f"{file_id}.txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                # Create empty label file if doesn't exist
                with open(dst_label, 'w') as f:
                    pass
    
    print("\nCopying files...")
    copy_files(train_ids, 'train')
    print("✓ Train set copied")
    copy_files(val_ids, 'val')
    print("✓ Val set copied")
    copy_files(test_ids, 'test')
    print("✓ Test set copied")
    
    print(f"\n✓ Dataset organized in: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    for split in ['train', 'val', 'test']:
        img_count = len(glob.glob(os.path.join(output_path, split, 'images', '*.png')))
        label_count = len(glob.glob(os.path.join(output_path, split, 'labels', '*.txt')))
        print(f"{split.capitalize():5s}: {img_count} images, {label_count} labels")

if __name__ == "__main__":
    organize_yolo_dataset()