import os
import json
import shutil
from tqdm import tqdm
from typing import Dict
import cv2
from utils import load_config, create_directory, resize_image, save_image

class OtherTasksPreparer:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_config(config_path)
        self.tasks_config = self.config['data_preparation']['other_tasks']
        
    def prepare_other_tasks(self, dataset_path: str) -> Dict:
        """Prepare dataset for other tasks (Detectron2 and Classification)"""
        print("🔄 Preparing for other tasks...")
        
        stats = {
            'detectron2': {},
            'classification': {}
        }
        
        # Prepare Detectron2 data
        if self.tasks_config['detectron2']:
            stats['detectron2'] = self._prepare_detectron2_data(dataset_path)
        
        # Prepare Classification data
        if self.tasks_config['classification']:
            stats['classification'] = self._prepare_classification_data(dataset_path)
        
        return stats
    
    def _prepare_detectron2_data(self, dataset_path: str) -> Dict:
        """Prepare data for Detectron2 segmentation"""
        print("📋 Preparing Detectron2 data...")
        
        detectron2_path = os.path.join(dataset_path, "detectron2")
        create_directory(os.path.join(detectron2_path, "train"))
        create_directory(os.path.join(detectron2_path, "val"))
        create_directory(os.path.join(detectron2_path, "test"))
        create_directory(os.path.join(detectron2_path, "annotations"))
        
        stats = {
            'train_images': 0,
            'val_images': 0,
            'test_images': 0
        }
        
        coco_path = os.path.join(dataset_path, "CarDD_COCO")
        splits = ['train2017', 'val2017', 'test2017']
        split_mapping = {'train2017': 'train', 'val2017': 'val', 'test2017': 'test'}
        
        for coco_split in splits:
            detectron_split = split_mapping[coco_split]
            annotations_path = os.path.join(coco_path, "annotations", f"instances_{coco_split}.json")
            images_dir = os.path.join(coco_path, coco_split)
            output_dir = os.path.join(detectron2_path, detectron_split)
            
            if not os.path.exists(annotations_path):
                continue
            
            # Copy annotation file
            shutil.copy2(annotations_path, os.path.join(detectron2_path, "annotations", f"{detectron_split}.json"))
            
            # Copy images
            if os.path.exists(images_dir):
                for image_name in tqdm(os.listdir(images_dir), desc=f"Copying {detectron_split}"):
                    src_path = os.path.join(images_dir, image_name)
                    dst_path = os.path.join(output_dir, image_name)
                    shutil.copy2(src_path, dst_path)
                    stats[f'{detectron_split}_images'] += 1
        
        print(f"✅ Detectron2 data prepared: {stats}")
        return stats
    
    def _prepare_classification_data(self, dataset_path: str) -> Dict:
        """Prepare cropped damage regions for classification"""
        print("📋 Preparing classification data...")
        
        classification_path = os.path.join(dataset_path, "classification", "damage_crops")
        create_directory(os.path.join(classification_path, "train"))
        create_directory(os.path.join(classification_path, "val"))
        create_directory(os.path.join(classification_path, "test"))
        
        stats = {
            'train_crops': 0,
            'val_crops': 0,
            'test_crops': 0,
            'categories': {}
        }
        
        coco_path = os.path.join(dataset_path, "CarDD_COCO")
        splits = ['train2017', 'val2017', 'test2017']
        split_mapping = {'train2017': 'train', 'val2017': 'val', 'test2017': 'test'}
        crop_size = (self.tasks_config['classification_crop_size'], self.tasks_config['classification_crop_size'])
        
        for coco_split in splits:
            classification_split = split_mapping[coco_split]
            annotations_path = os.path.join(coco_path, "annotations", f"instances_{coco_split}.json")
            images_dir = os.path.join(coco_path, coco_split)
            
            if not os.path.exists(annotations_path):
                continue
            
            with open(annotations_path, 'r') as f:
                coco_data = json.load(f)
            
            category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}
            image_mapping = {img['id']: img for img in coco_data['images']}
            
            crop_count = 0
            for ann in tqdm(coco_data['annotations'], desc=f"Cropping {classification_split}"):
                image_info = image_mapping.get(ann['image_id'])
                if not image_info:
                    continue
                
                image_path = os.path.join(images_dir, image_info['file_name'])
                if not os.path.exists(image_path):
                    continue
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Extract bbox coordinates
                x, y, w, h = map(int, ann['bbox'])
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                # Crop damage region
                crop = image[y:y+h, x:x+w]
                if crop.size == 0:
                    continue
                
                # Resize crop
                crop = resize_image(crop, crop_size)
                
                # Get category name
                category_name = category_mapping.get(ann['category_id'], "unknown")
                if category_name not in stats['categories']:
                    stats['categories'][category_name] = 0
                
                # Create category directory
                category_dir = os.path.join(classification_path, classification_split, category_name)
                create_directory(category_dir)
                
                # Save crop
                crop_filename = f"{image_info['id']}_{ann['id']}.jpg"
                crop_path = os.path.join(category_dir, crop_filename)
                save_image(crop, crop_path)
                
                stats[f'{classification_split}_crops'] += 1
                stats['categories'][category_name] += 1
                crop_count += 1
            
            print(f"✅ {classification_split}: {crop_count} damage crops")
        
        return stats

def main():
    config = load_config()
    preparer = OtherTasksPreparer()
    
    dataset_path = "CarDD_release"
    
    # Prepare for other tasks
    stats = preparer.prepare_other_tasks(dataset_path)
    
    print(f"\n🎉 Other tasks preparation completed!")
    print(f"Detectron2: {stats.get('detectron2', {})}")
    print(f"Classification: {stats.get('classification', {})}")

if __name__ == "__main__":
    main()