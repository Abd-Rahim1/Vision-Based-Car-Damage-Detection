import os
import json
from typing import Dict
from utils import load_config, create_directory

class DatasetStatistics:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_config(config_path)
        
    def generate_final_statistics(self, dataset_path: str) -> Dict:
        """Generate final dataset statistics after all preprocessing"""
        print("📊 Generating final dataset statistics...")
        
        stats = {
            'overview': {},
            'yolo': {},
            'detectron2': {},
            'classification': {},
            'damage_categories': {}
        }
        
        # Overview statistics
        stats['overview'] = self._get_overview_statistics(dataset_path)
        
        # YOLO statistics
        stats['yolo'] = self._get_yolo_statistics(dataset_path)
        
        # Detectron2 statistics
        stats['detectron2'] = self._get_detectron2_statistics(dataset_path)
        
        # Classification statistics
        stats['classification'] = self._get_classification_statistics(dataset_path)
        
        # Damage categories summary
        stats['damage_categories'] = self._get_damage_categories_summary(stats)
        
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
            'test': {'images': 0, 'labels': 0}
        }
        
        yolo_path = os.path.join(dataset_path, "yolo")
        if os.path.exists(yolo_path):
            for split in ['train', 'val', 'test']:
                images_dir = os.path.join(yolo_path, "images", split)
                labels_dir = os.path.join(yolo_path, "labels", split)
                
                if os.path.exists(images_dir):
                    yolo_stats[split]['images'] = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                
                if os.path.exists(labels_dir):
                    yolo_stats[split]['labels'] = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
        
        return yolo_stats
    
    def _get_detectron2_statistics(self, dataset_path: str) -> Dict:
        """Get Detectron2 dataset statistics"""
        detectron2_stats = {
            'train': {'images': 0},
            'val': {'images': 0},
            'test': {'images': 0}
        }
        
        detectron2_path = os.path.join(dataset_path, "detectron2")
        if os.path.exists(detectron2_path):
            for split in ['train', 'val', 'test']:
                images_dir = os.path.join(detectron2_path, split)
                if os.path.exists(images_dir):
                    detectron2_stats[split]['images'] = len(os.listdir(images_dir))
        
        return detectron2_stats
    
    def _get_classification_statistics(self, dataset_path: str) -> Dict:
        """Get classification dataset statistics"""
        classification_stats = {
            'train': {'crops': 0, 'categories': {}},
            'val': {'crops': 0, 'categories': {}},
            'test': {'crops': 0, 'categories': {}}
        }
        
        classification_path = os.path.join(dataset_path, "classification", "damage_crops")
        if os.path.exists(classification_path):
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(classification_path, split)
                if os.path.exists(split_path):
                    categories = os.listdir(split_path)
                    total_crops = 0
                    category_counts = {}
                    
                    for category in categories:
                        category_path = os.path.join(split_path, category)
                        crop_count = len([f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                        category_counts[category] = crop_count
                        total_crops += crop_count
                    
                    classification_stats[split]['crops'] = total_crops
                    classification_stats[split]['categories'] = category_counts
        
        return classification_stats
    
    def _get_damage_categories_summary(self, stats: Dict) -> Dict:
        """Get summary of damage categories across all tasks"""
        categories_summary = {}
        
        # Get categories from YOLO dataset.yaml
        yolo_path = "CarDD_release/yolo/dataset.yaml"
        if os.path.exists(yolo_path):
            import yaml
            with open(yolo_path, 'r') as f:
                yolo_config = yaml.safe_load(f)
            categories = yolo_config.get('names', [])
            for i, category in enumerate(categories):
                categories_summary[category] = {
                    'yolo_class_id': i,
                    'total_instances': 0
                }
        
        # Count instances from classification data
        classification_stats = stats.get('classification', {})
        for split in ['train', 'val', 'test']:
            split_categories = classification_stats.get(split, {}).get('categories', {})
            for category, count in split_categories.items():
                if category not in categories_summary:
                    categories_summary[category] = {'total_instances': 0}
                categories_summary[category]['total_instances'] += count
        
        return categories_summary

def main():
    config = load_config()
    stats_generator = DatasetStatistics()
    
    dataset_path = r"C:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\dataset\CarDD_release"
    
    # Generate final statistics
    final_stats = stats_generator.generate_final_statistics(dataset_path)
    
    # Save statistics
    with open("dataset_stats.json", 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"\nFinal Dataset Statistics:")
    print(f"Overview: {final_stats['overview']}")
    print(f"YOLO: {final_stats['yolo']}")
    print(f"Detectron2: {final_stats['detectron2']}")
    print(f"Classification: {final_stats['classification']}")
    print(f"Damage Categories: {list(final_stats['damage_categories'].keys())}")
    print(f"Statistics saved to: dataset_stats.json")

if __name__ == "__main__":
    main()