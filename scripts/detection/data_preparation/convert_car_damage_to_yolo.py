import os
import json
import shutil
import yaml
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def load_params():
    """Load parameters from YAML file"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def convert_coco_to_yolo(coco_annotations, output_dir, dataset_type):
    """
    Convert COCO format annotations to YOLO format
    """
    params = load_params()
    
    # Create directories
    images_dir = Path(output_dir) / dataset_type / 'images'
    labels_dir = Path(output_dir) / dataset_type / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load categories
    categories = coco_annotations['categories']
    category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    
    print(f"Converting {dataset_type} set...")
    print(f"Categories mapping: {category_id_to_idx}")
    
    # Process images
    image_id_to_info = {img['id']: img for img in coco_annotations['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    processed_count = 0
    skipped_count = 0
    
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_id_to_info:
            skipped_count += 1
            continue
            
        image_info = image_id_to_info[image_id]
        image_width = image_info['width']
        image_height = image_info['height']
        file_name = image_info['file_name']
        
        # Create YOLO format annotation file
        label_file = labels_dir / f"{Path(file_name).stem}.txt"
        
        with open(label_file, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id']
                
                # Skip if category not in mapping
                if category_id not in category_id_to_idx:
                    continue
                    
                # Get bbox in COCO format [x, y, width, height]
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Convert to YOLO format [x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                w_norm = w / image_width
                h_norm = h / image_height
                
                # Validate bbox coordinates
                if (x_center <= 0 or x_center >= 1 or 
                    y_center <= 0 or y_center >= 1 or
                    w_norm <= 0 or w_norm >= 1 or
                    h_norm <= 0 or h_norm >= 1):
                    skipped_count += 1
                    continue
                
                # Get class index for YOLO (0-indexed)
                class_idx = category_id_to_idx[category_id]
                
                # Write to label file
                f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        processed_count += 1
    
    print(f"Processed {processed_count} images, skipped {skipped_count} annotations")
    return processed_count

def prepare_yolo_dataset():
    """
    Main function to prepare YOLO dataset from CarDD
    """
    params = load_params()
    
    # Paths
    base_dir = Path("C:/Users/Document/OneDrive/Desktop/Car_Damage_Detection/dataset")
    coco_dir = base_dir / "CarDD_release" / "CarDD_COCO"
    output_dir = base_dir / "Dataset_Prepared" / "detection"
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = ['train2017', 'val2017', 'test2017']
    
    for split in splits:
        # Annotation file path
        ann_file = coco_dir / "annotations" / f"instances_{split}.json"
        
        if not ann_file.exists():
            print(f"Warning: {ann_file} not found, skipping...")
            continue
            
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            coco_annotations = json.load(f)
        
        # Convert to YOLO format
        convert_coco_to_yolo(coco_annotations, output_dir, split.replace('2017', ''))
        
        # Copy images
        src_images_dir = coco_dir / split
        dst_images_dir = output_dir / split.replace('2017', '') / 'images'
        
        if src_images_dir.exists():
            print(f"Copying images from {src_images_dir} to {dst_images_dir}")
            
            # Create destination directory
            dst_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all images
            for img_file in src_images_dir.glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy2(img_file, dst_images_dir / img_file.name)
    
    # Create dataset.yaml file for YOLO
    create_yolo_dataset_yaml(output_dir, coco_annotations['categories'])

def create_yolo_dataset_yaml(output_dir, categories):
    """
    Create dataset.yaml file for YOLO training
    """
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': len(categories),
        'names': {idx: cat['name'] for idx, cat in enumerate(categories)}
    }
    
    yaml_file = output_dir / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        f.write(f"path: {yaml_content['path']}\n")
        f.write(f"train: {yaml_content['train']}\n")
        f.write(f"val: {yaml_content['val']}\n")
        f.write(f"test: {yaml_content['test']}\n")
        f.write(f"nc: {yaml_content['nc']}\n")
        f.write("names:\n")
        for idx, name in yaml_content['names'].items():
            f.write(f"  {idx}: '{name}'\n")
    
    print(f"Created dataset.yaml at {yaml_file}")
    print(f"Dataset structure:")
    print(f"  - Classes: {yaml_content['names']}")
    print(f"  - Train: {yaml_content['train']}")
    print(f"  - Val: {yaml_content['val']}")
    print(f"  - Test: {yaml_content['test']}")

def verify_yolo_dataset(output_dir):
    """
    Verify the generated YOLO dataset
    """
    output_path = Path(output_dir)
    
    print("\nVerifying dataset structure...")
    
    for split in ['train', 'val', 'test']:
        images_dir = output_path / split / 'images'
        labels_dir = output_path / split / 'labels'
        
        if images_dir.exists():
            image_files = list(images_dir.glob('*.*'))
            label_files = list(labels_dir.glob('*.txt'))
            
            print(f"\n{split}:")
            print(f"  Images: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            
            # Check if every image has a corresponding label
            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}
            
            missing_labels = image_stems - label_stems
            if missing_labels:
                print(f"  Warning: {len(missing_labels)} images missing labels")
            
            # Sample check of label files
            if label_files:
                sample_label = label_files[0]
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        sample_line = lines[0].strip()
                        print(f"  Sample label: {sample_line}")

if __name__ == "__main__":
    print("Starting CarDD to YOLO conversion...")
    prepare_yolo_dataset()
    
    output_dir = Path("C:/Users/Document/OneDrive/Desktop/Car_Damage_Detection/dataset/Dataset_Prepared/detection")
    verify_yolo_dataset(output_dir)
    
    print("\nConversion completed successfully!")