import os
import json
import cv2
import numpy as np
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict
from utils import load_config, create_directory, load_coco_annotations

class DataExplorer:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_config(config_path)
        self.exploration_config = self.config['data_preparation']['exploration']
        
    def analyze_dataset(self, dataset_path: str) -> Dict:
        """Analyze dataset structure and compute statistics"""
        print(" Analyzing dataset structure...")
        
        stats = {
            'coco': {},
            'sod': {},
            'total_images': 0,
            'total_annotations': 0,
            'damage_categories': {}
        }
        
        # Analyze COCO dataset
        coco_path = os.path.join(dataset_path, "CarDD_COCO")
        if os.path.exists(coco_path):
            stats['coco'] = self._analyze_coco_dataset(coco_path)
            stats['total_images'] += stats['coco']['total_images']
            stats['total_annotations'] += stats['coco']['total_annotations']
            
            # Update damage categories
            for split_data in stats['coco']['splits'].values():
                for cat_id, count in split_data['category_counts'].items():
                    cat_name = split_data['categories'].get(cat_id, f"category_{cat_id}")
                    if cat_name not in stats['damage_categories']:
                        stats['damage_categories'][cat_name] = 0
                    stats['damage_categories'][cat_name] += count
        
        # Analyze SOD dataset
        sod_path = os.path.join(dataset_path, "CarDD_SOD")
        if os.path.exists(sod_path):
            stats['sod'] = self._analyze_sod_dataset(sod_path)
            stats['total_images'] += stats['sod']['total_images']
        
        return stats
    
    def _analyze_coco_dataset(self, coco_path: str) -> Dict:
        """Analyze COCO format dataset"""
        coco_stats = {
            'splits': {},
            'total_images': 0,
            'total_annotations': 0
        }
        
        splits = ['train2017', 'val2017', 'test2017']
        
        for split in splits:
            try:
                coco_data = load_coco_annotations(coco_path, split)
                
                # Get category mapping
                categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
                
                # Count annotations per category
                category_counts = {}
                for ann in coco_data['annotations']:
                    cat_id = ann['category_id']
                    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
                
                split_stats = {
                    'images': len(coco_data['images']),
                    'annotations': len(coco_data['annotations']),
                    'category_counts': category_counts,
                    'categories': categories
                }
                
                coco_stats['splits'][split] = split_stats
                coco_stats['total_images'] += len(coco_data['images'])
                coco_stats['total_annotations'] += len(coco_data['annotations'])
                
            except FileNotFoundError:
                print(f"  Skipping {split} - annotations not found")
                continue
        
        return coco_stats
    
    def _analyze_sod_dataset(self, sod_path: str) -> Dict:
        """Analyze SOD dataset"""
        sod_stats = {
            'splits': {},
            'total_images': 0
        }
        
        splits = ['CarDD-TR', 'CarDD-VAL', 'CarDD-TE']
        
        for split in splits:
            image_dir = os.path.join(sod_path, split, f"{split}-Image")
            
            if not os.path.exists(image_dir):
                continue
            
            images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            split_stats = {
                'images': len(images)
            }
            
            sod_stats['splits'][split] = split_stats
            sod_stats['total_images'] += len(images)
        
        return sod_stats
    
    def create_visualizations(self, dataset_path: str, output_dir: str):
        """Create sample visualizations"""
        print(" Creating sample visualizations...")
        
        create_directory(output_dir)
        
        # Visualize COCO samples
        coco_path = os.path.join(dataset_path, "CarDD_COCO")
        if os.path.exists(coco_path):
            self._visualize_coco_samples(coco_path, os.path.join(output_dir, "coco_samples"))
    
    def _visualize_coco_samples(self, coco_path: str, output_dir: str):
        """Visualize COCO dataset samples"""
        create_directory(output_dir)
        
        splits = ['train2017', 'val2017', 'test2017']
        sample_size = self.exploration_config.get('sample_size', 30) // len(splits)
        
        for split in splits:
            try:
                coco_data = load_coco_annotations(coco_path, split)
                images_dir = os.path.join(coco_path, split)
                
                # Select random samples
                sample_images = random.sample(coco_data['images'], min(sample_size, len(coco_data['images'])))
                
                for image_info in tqdm(sample_images, desc=f"Visualizing {split}"):
                    image_path = os.path.join(images_dir, image_info['file_name'])
                    output_path = os.path.join(output_dir, f"{split}_{image_info['file_name']}")
                    
                    if os.path.exists(image_path):
                        self._create_annotation_visualization(image_path, image_info, coco_data, output_path)
                        
            except FileNotFoundError:
                continue
    
    def _create_annotation_visualization(self, image_path: str, image_info: Dict, coco_data: Dict, output_path: str):
        """Create visualization for a single image with annotations"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # Resize image for visualization
        target_size = self.exploration_config.get('output_size', (640, 480))
        scale_x = target_size[0] / image_info['width']
        scale_y = target_size[1] / image_info['height']
        image = cv2.resize(image, target_size)
        
        # Get annotations for this image
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Draw bounding boxes
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = categories.get(category_id, f"Unknown_{category_id}")
            
            x = int(bbox[0] * scale_x)
            y = int(bbox[1] * scale_y)
            w = int(bbox[2] * scale_x)
            h = int(bbox[3] * scale_y)
            
            color = self.exploration_config.get('colors', [(0,255,0)])[i % 1]
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = f"{category_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        cv2.imwrite(output_path, image)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="dataset_stats.json", help="Output statistics JSON file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    explorer = DataExplorer(args.config)
    
    dataset_path = r"C:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\dataset\CarDD_release"
    output_stats_path = args.output or "dataset_stats.json"
    
    # Analyze dataset
    stats = explorer.analyze_dataset(dataset_path)
    
    # Create visualizations
    explorer.create_visualizations(dataset_path, "exploration_visualizations")
    
    # Save statistics
    create_directory(os.path.dirname(output_stats_path) or ".")
    with open(output_stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Damage categories: {list(stats['damage_categories'].keys())}")
    print(f"Statistics saved to: {output_stats_path}")
    print(f"Visualizations saved to: exploration_visualizations/")

if __name__ == "__main__":
    main()
