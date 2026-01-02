import os
import json
import argparse
from tqdm import tqdm
from typing import Dict
from typing import List, Tuple
from utils import load_config, create_directory, calculate_bbox_area, calculate_bbox_aspect_ratio, calculate_iou, load_coco_annotations, save_coco_annotations, parse_arguments

class DataCleaner:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_config(config_path)
        self.cleaning_config = self.config['data_preparation']['cleaning']
        
    def clean_dataset(self, input_path: str, output_path: str) -> Dict:
        """Clean annotations in the dataset"""
        print(" Cleaning dataset annotations...")
        
        # Copy dataset to output
        if input_path != output_path:
            create_directory(output_path)
            os.system(f"cp -r {input_path}/* {output_path}/")
        
        stats = {
            'removed_annotations': 0,
            'removed_small_bboxes': 0,
            'removed_invalid_bboxes': 0,
            'removed_duplicate_bboxes': 0,
            'splits': {}
        }
        
        # Clean COCO dataset
        coco_path = os.path.join(output_path, "CarDD_COCO")
        if os.path.exists(coco_path):
            coco_stats = self._clean_coco_dataset(coco_path)
            stats.update(coco_stats)
        
        return stats
    
    def _clean_coco_dataset(self, coco_path: str) -> Dict:
        """Clean COCO dataset annotations in-place"""
        stats = {
            'removed_annotations': 0,
            'removed_small_bboxes': 0,
            'removed_invalid_bboxes': 0,
            'removed_duplicate_bboxes': 0,
            'splits': {}
        }
        
        splits = ['train2017', 'val2017', 'test2017']
        
        for split in splits:
            try:
                coco_data = load_coco_annotations(coco_path, split)
                
                # Clean annotations
                cleaned_annotations = []
                removed_small = 0
                removed_invalid = 0
                removed_duplicate = 0
                
                # Group annotations by image
                image_annotations = {}
                for ann in coco_data['annotations']:
                    image_id = ann['image_id']
                    if image_id not in image_annotations:
                        image_annotations[image_id] = []
                    image_annotations[image_id].append(ann)
                
                # Clean annotations for each image
                for image_id, annotations in image_annotations.items():
                    cleaned_image_annotations, image_stats = self._clean_image_annotations(annotations)
                    cleaned_annotations.extend(cleaned_image_annotations)
                    removed_small += image_stats['removed_small']
                    removed_invalid += image_stats['removed_invalid']
                    removed_duplicate += image_stats['removed_duplicate']
                
                total_removed = len(coco_data['annotations']) - len(cleaned_annotations)
                
                # Update COCO data
                coco_data['annotations'] = cleaned_annotations
                
                # Save updated annotations
                save_coco_annotations(coco_data, coco_path, split)
                
                stats['splits'][split] = {
                    'original_annotations': len(coco_data['annotations']) + total_removed,
                    'cleaned_annotations': len(cleaned_annotations),
                    'removed_small_bboxes': removed_small,
                    'removed_invalid_bboxes': removed_invalid,
                    'removed_duplicate_bboxes': removed_duplicate
                }
                
                stats['removed_annotations'] += total_removed
                stats['removed_small_bboxes'] += removed_small
                stats['removed_invalid_bboxes'] += removed_invalid
                stats['removed_duplicate_bboxes'] += removed_duplicate
                
                print(f"✅ {split}: Removed {total_removed} annotations "
                      f"({removed_small} small, {removed_invalid} invalid, {removed_duplicate} duplicate)")
                
            except FileNotFoundError:
                print(f"⚠️  Skipping {split} - annotations not found")
                continue
        
        return stats
    
    def _clean_image_annotations(self, annotations: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Clean annotations for a single image"""
        valid_annotations = []
        stats = {
            'removed_small': 0,
            'removed_invalid': 0,
            'removed_duplicate': 0
        }
        
        # First pass: remove invalid bboxes
        for ann in annotations:
            bbox = ann['bbox']
            
            # Check bbox area
            area = calculate_bbox_area(bbox)
            if area < self.cleaning_config['min_bbox_area']:
                stats['removed_small'] += 1
                continue
            
            # Check bbox dimensions
            x, y, w, h = bbox
            if (w < self.cleaning_config['min_bbox_dimension'] or 
                h < self.cleaning_config['min_bbox_dimension']):
                stats['removed_invalid'] += 1
                continue
            
            # Check aspect ratio
            aspect_ratio = calculate_bbox_aspect_ratio(bbox)
            if aspect_ratio > self.cleaning_config['max_bbox_aspect_ratio']:
                stats['removed_invalid'] += 1
                continue
            
            valid_annotations.append(ann)
        
        # Second pass: remove duplicate bboxes
        if self.cleaning_config['remove_duplicate_bboxes']:
            valid_annotations, removed_duplicates = self._remove_duplicate_bboxes(valid_annotations)
            stats['removed_duplicate'] += removed_duplicates
        
        return valid_annotations, stats
    
    def _remove_duplicate_bboxes(self, annotations: List[Dict]) -> Tuple[List[Dict], int]:
        """Remove duplicate bounding boxes based on IoU"""
        if len(annotations) <= 1:
            return annotations, 0
        
        # Sort by area (largest first)
        annotations.sort(key=lambda x: calculate_bbox_area(x['bbox']), reverse=True)
        
        filtered_annotations = []
        removed_count = 0
        
        for i in range(len(annotations)):
            keep_annotation = True
            
            for j in range(len(filtered_annotations)):
                iou = calculate_iou(annotations[i]['bbox'], filtered_annotations[j]['bbox'])
                if iou > self.cleaning_config['iou_threshold']:
                    keep_annotation = False
                    break
            
            if keep_annotation:
                filtered_annotations.append(annotations[i])
            else:
                removed_count += 1
        
        return filtered_annotations, removed_count

def main():
    args = parse_arguments()
    cleaner = DataCleaner(args.config)
    
    input_path = args.input
    output_path = args.output
    
    # Clean dataset
    stats = cleaner.clean_dataset(input_path, output_path)
    
    print(f"\n Cleaning completed!")
    print(f" Removed annotations: {stats['removed_annotations']}")
    print(f" Removed small bboxes: {stats['removed_small_bboxes']}")
    print(f" Removed invalid bboxes: {stats['removed_invalid_bboxes']}")
    print(f" Removed duplicate bboxes: {stats['removed_duplicate_bboxes']}")

if __name__ == "__main__":
    main()