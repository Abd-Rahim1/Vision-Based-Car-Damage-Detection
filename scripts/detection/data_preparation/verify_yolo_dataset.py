import yaml
from pathlib import Path
import random
import cv2
import numpy as np

def verify_dataset_structure():
    """Verify the YOLO dataset structure and annotations"""
    
    dataset_path = Path("C:/Users/Document/OneDrive/Desktop/Car_Damage_Detection/dataset/Dataset_Prepared/detection")
    
    print("=== Dataset Structure Verification ===")
    
    # Check dataset.yaml
    yaml_file = dataset_path / "dataset.yaml"
    if yaml_file.exists():
        with open(yaml_file, 'r') as f:
            dataset_config = yaml.safe_load(f)
        print("✓ dataset.yaml found and valid")
        print(f"  - Classes: {dataset_config['names']}")
        print(f"  - Number of classes: {dataset_config['nc']}")
    else:
        print("✗ dataset.yaml not found")
        return
    
    # Check each split
    for split in ['train', 'val', 'test']:
        print(f"\n=== {split.upper()} ===")
        
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        # Count files
        image_files = list(images_dir.glob('*.*'))
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"Images: {len(image_files)}")
        print(f"Labels: {len(label_files)}")
        
        # Check correspondence
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        
        if image_stems == label_stems:
            print("✓ All images have corresponding labels")
        else:
            missing_images = label_stems - image_stems
            missing_labels = image_stems - label_stems
            if missing_images:
                print(f"✗ Missing images for {len(missing_images)} labels")
            if missing_labels:
                print(f"✗ Missing labels for {len(missing_labels)} images")
        
        # Check annotation format
        if label_files:
            sample_label = random.choice(label_files)
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"Sample annotation from {sample_label.name}:")
                    for line in lines[:3]:  # Show first 3 annotations
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = parts
                            print(f"  Class: {class_id}, BBox: [{x_center}, {y_center}, {width}, {height}]")
        
        # Check image dimensions
        if image_files:
            sample_image = random.choice(image_files)
            img = cv2.imread(str(sample_image))
            if img is not None:
                print(f"Sample image: {sample_image.name} - Size: {img.shape[1]}x{img.shape[0]}")

def visualize_sample_annotations():
    """Visualize a few samples to verify annotations are correct"""
    
    dataset_path = Path("C:/Users/Document/OneDrive/Desktop/Car_Damage_Detection/dataset/Dataset_Prepared/detection")
    
    # Load class names
    with open(dataset_path / "dataset.yaml", 'r') as f:
        config = yaml.safe_load(f)
    class_names = config['names']
    
    # Colors for different classes
    colors = [
        (255, 0, 0),    # red - dent
        (0, 255, 0),    # green - scratch  
        (0, 0, 255),    # blue - crack
        (255, 255, 0),  # cyan - glass shatter
        (255, 0, 255),  # magenta - lamp broken
        (0, 255, 255),  # yellow - tire flat
    ]
    
    # Visualize a few samples from each split
    for split in ['train', 'val']:
        print(f"\n=== Visualizing {split} samples ===")
        
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        image_files = list(images_dir.glob('*.jpg'))[:3]  # Check first 3 images
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
                
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Load and draw annotations
            with open(label_path, 'r') as f:
                annotations = f.readlines()
            
            for ann in annotations:
                parts = ann.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    
                    # Convert from normalized to pixel coordinates
                    x_center_px = int(x_center * img_width)
                    y_center_px = int(y_center * img_height)
                    width_px = int(width * img_width)
                    height_px = int(height * img_height)
                    
                    # Calculate bounding box coordinates
                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)
                    x2 = int(x_center_px + width_px / 2)
                    y2 = int(y_center_px + height_px / 2)
                    
                    # Draw bounding box
                    color = colors[int(class_id) % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw class label
                    class_name = class_names.get(int(class_id), f"Class_{class_id}")
                    cv2.putText(img, class_name, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Resize for display if too large
            display_img = img
            if max(img_height, img_width) > 800:
                scale = 800 / max(img_height, img_width)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                display_img = cv2.resize(img, (new_width, new_height))
            
            # Show image
            cv2.imshow(f"{split} - {img_path.name}", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_dataset_structure()
    print("\n" + "="*50)
    print("Would you like to visualize sample annotations? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        visualize_sample_annotations()