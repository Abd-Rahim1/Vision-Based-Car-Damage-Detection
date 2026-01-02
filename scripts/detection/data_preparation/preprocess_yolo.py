import os
import json
import yaml
import argparse
from typing import Dict
from utils import load_config, create_directory, parse_arguments

class DatasetStatistics:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_config(config_path)
        
    def generate_statistics(self, dataset_path: str) -> Dict:
        """Generate comprehensive dataset statistics"""
        print("📊 Generating dataset statistics...")
        
        stats = {
            'overview': {},
            'yolo': {},
            'damage_categories': {},
            'preprocessing_info': {}
        }
        
        # Overview statistics
        stats['overview'] = self._get_overview_statistics(dataset_path)
        
        # YOLO statistics
        stats['yolo'] = self._get_yolo_statistics(dataset_path)
        
        # Damage categories summary
        stats['damage_categories'] = self._get_damage_categories_summary(dataset_path)
        
        # Preprocessing information
        stats['preprocessing_info'] = self.config['data_preparation']['yolo']
        
        return stats
    
    def _get_overview_statistics(self, dataset_path: str) -> Dict:
        """Get overview statistics"""
        overview = {
            'total_images': 0,
            'total_annotations': 0
        }
        
        # Count COCO images and annotations
        coco_path = os.path.join(dataset_path, "CarDD_COCO")
        if os.path.exists(coco_path):
            splits = ['train2017', 'val2017', 'test2017']
            for split in splits:
                annotations_path = os.path.join(coco_path, "annotations", f"instances_{split}.json")
                if os.path.exists(annotations_path):
                    with open(annotations_path, 'r') as f:
                        coco_data = json.load(f)
                    overview['total_images'] += len(coco_data['images'])
                    overview['total_annotations'] += len(coco_data['annotations'])
        
        return overview
    
    def _get_yolo_statistics(self, dataset_path: str) -> Dict:
        """Get YOLO dataset statistics"""
        yolo_stats = {
            'train': {'images': 0, 'labels': 0},
            'val': {'images': 0, 'labels': 0},
            'test': {'images': 0, 'labels': 0},
            'total_images': 0,
            'total_labels': 0
        }
        
        yolo_path = os.path.join(dataset_path, "yolo")
        if os.path.exists(yolo_path):
            for split in ['train', 'val', 'test']:
                images_dir = os.path.join(yolo_path, "images", split)
                labels_dir = os.path.join(yolo_path, "labels", split)
                
                if os.path.exists(images_dir):
                    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    yolo_stats[split]['images'] = len(images)
                    yolo_stats['total_images'] += len(images)
                
                if os.path.exists(labels_dir):
                    labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                    yolo_stats[split]['labels'] = len(labels)
                    yolo_stats['total_labels'] += len(labels)
        
        return yolo_stats
    
    def _get_damage_categories_summary(self, dataset_path: str) -> Dict:
        """Get summary of damage categories"""
        categories_summary = {}
        
        # Get categories from YOLO dataset.yaml
        yolo_path = os.path.join(dataset_path, "yolo", "dataset.yaml")
        if os.path.exists(yolo_path):
            with open(yolo_path, 'r') as f:
                yolo_config = yaml.safe_load(f)
            categories = yolo_config.get('names', [])
            for i, category in enumerate(categories):
                categories_summary[category] = {
                    'yolo_class_id': i,
                    'instances': 0
                }
        
        # Count instances from YOLO label files
        yolo_labels_path = os.path.join(dataset_path, "yolo", "labels")
        if os.path.exists(yolo_labels_path):
            for split in ['train', 'val', 'test']:
                labels_dir = os.path.join(yolo_labels_path, split)
                if os.path.exists(labels_dir):
                    for label_file in os.listdir(labels_dir):
                        if label_file.endswith('.txt'):
                            label_path = os.path.join(labels_dir, label_file)
                            with open(label_path, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) == 5:
                                        class_id = int(parts[0])
                                        if class_id < len(categories):
                                            category_name = categories[class_id]
                                            if category_name in categories_summary:
                                                categories_summary[category_name]['instances'] += 1
        
        return categories_summary

def main():
    args = parse_arguments()
    stats_generator = DatasetStatistics(args.config)
    
    input_path = args.input
    output_path = args.output
    
    # Generate statistics
    stats = stats_generator.generate_statistics(input_path)
    
    # Save statistics
    create_directory(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n Final Dataset Statistics:")
    print(f" Overview: {stats['overview']}")
    print(f" YOLO: {stats['yolo']}")
    print(f" Damage Categories: {len(stats['damage_categories'])} categories")
    print(f" Statistics saved to: {output_path}")

if __name__ == "__main__":
    main()