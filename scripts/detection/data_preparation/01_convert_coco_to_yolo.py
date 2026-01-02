import json
import os
import shutil
import yaml
from pathlib import Path
from utils import setup_logging, save_metrics, create_data_yaml, track_with_dvc, get_dvc_file_path, get_data_path_for_step, get_step_info, get_project_root

class COCOToYOLOConverter:
    def __init__(self, config_path):
        self.config_path = config_path
        self.logger = setup_logging("COCO_to_YOLO_Conversion")  # Initialize logger FIRST
        self.load_config()
        self.setup_paths()
        
    def load_config(self):
        """Load configuration from params.yaml"""
        self.logger.info(f"Loading config from: {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_prep_config = self.config['data_preparation']
        self.damage_categories = self.config['damage_categories']
        self.logger.info(f"Loaded {len(self.damage_categories)} damage categories: {self.damage_categories}")
        
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = get_project_root()
        self.raw_data_dir = self.base_dir / self.data_prep_config['paths']['raw_data']
        self.processed_dir = self.base_dir / self.data_prep_config['paths']['processed_data']
        
        # Log paths for verification
        self.logger.info(f"Project root: {self.base_dir}")
        self.logger.info(f"Raw data dir: {self.raw_data_dir}")
        self.logger.info(f"Processed dir: {self.processed_dir}")
        
        # Verify raw data exists
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")
        else:
            self.logger.info("   Raw data directory exists")
    
    def map_category_names(self, coco_categories):
        """Map COCO category names to config category names"""
        category_mapping = {
            'glass shatter': 'glass_shatter',
            'lamp broken': 'lamp_broken', 
            'tire flat': 'tire_flat'
        }
        
        categories = {}
        for cat in coco_categories:
            original_name = cat['name']
            mapped_name = category_mapping.get(original_name, original_name)
            
            if mapped_name in self.damage_categories:
                categories[cat['id']] = self.damage_categories.index(mapped_name)
                self.logger.info(f"Category mapping: {original_name} -> {mapped_name} -> {categories[cat['id']]}")
            else:
                self.logger.warning(f"Category '{original_name}' (mapped to '{mapped_name}') not in damage_categories list")
        
        return categories
        
    def convert_split(self, split_name, json_file, images_dir):
        """Convert a single split from COCO to YOLO format"""
        self.logger.info(f"Converting {split_name} split...")
        
        coco_json_path = self.raw_data_dir / 'annotations' / json_file
        images_source_dir = self.raw_data_dir / images_dir
        
        self.logger.info(f"Looking for COCO JSON: {coco_json_path}")
        self.logger.info(f"Looking for images: {images_source_dir}")
        
        if not coco_json_path.exists():
            self.logger.warning(f"COCO JSON file not found: {coco_json_path}")
            return {'images_converted': 0, 'images_skipped': 0, 'annotations_converted': 0, 'categories_found': set()}
        
        if not images_source_dir.exists():
            self.logger.warning(f"Images directory not found: {images_source_dir}")
            return {'images_converted': 0, 'images_skipped': 0, 'annotations_converted': 0, 'categories_found': set()}
        
        output_split_dir = self.processed_dir / split_name
        
        # Create output directories
        images_output_dir = output_split_dir / 'images'
        labels_output_dir = output_split_dir / 'labels'
        images_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Loading COCO annotations from: {coco_json_path}")
        # Load COCO annotations
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create mappings
        images = {img['id']: img for img in coco_data['images']}
        self.logger.info(f"Found {len(images)} images in COCO annotations")
        
        # Map category names to indices with name mapping
        categories = self.map_category_names(coco_data['categories'])
        
        # Group annotations by image id
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        self.logger.info(f"Grouped {len(coco_data['annotations'])} annotations into {len(annotations_by_image)} images")
        
        # Process each image
        stats = {
            'images_converted': 0,
            'images_skipped': 0,
            'annotations_converted': 0,
            'categories_found': set()
        }
        
        processed_count = 0
        for image_id, annotations in annotations_by_image.items():
            if image_id not in images:
                stats['images_skipped'] += 1
                continue
            
            image_info = images[image_id]
            image_file = image_info['file_name']
            image_width = image_info['width']
            image_height = image_info['height']
            
            # Source image path
            src_image_path = images_source_dir / image_file
            
            if not src_image_path.exists():
                stats['images_skipped'] += 1
                if stats['images_skipped'] <= 5:  # Log first 5 missing files
                    self.logger.debug(f"Image not found: {src_image_path}")
                continue
            
            # Copy image
            dst_image_path = images_output_dir / image_file
            shutil.copy2(src_image_path, dst_image_path)
            
            # Create YOLO format labels
            label_file = Path(image_file).stem + '.txt'
            label_path = labels_output_dir / label_file
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    if ann['category_id'] not in categories:
                        continue
                    
                    class_id = categories[ann['category_id']]
                    bbox = ann['bbox']
                    stats['categories_found'].add(class_id)
                    
                    # Convert to YOLO format
                    x_center = (bbox[0] + bbox[2] / 2) / image_width
                    y_center = (bbox[1] + bbox[3] / 2) / image_height
                    width = bbox[2] / image_width
                    height = bbox[3] / image_height
                    
                    # Validate coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        self.logger.warning(f"Invalid bbox coordinates for image {image_file}: {bbox}")
                        continue
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    stats['annotations_converted'] += 1
            
            stats['images_converted'] += 1
            processed_count += 1
            
            if processed_count % 100 == 0:
                self.logger.info(f"Processed {processed_count} images in {split_name}...")
        
        self.logger.info(f"   {split_name} conversion completed:")
        self.logger.info(f"   - Images converted: {stats['images_converted']}")
        self.logger.info(f"   - Images skipped: {stats['images_skipped']}")
        self.logger.info(f"   - Annotations converted: {stats['annotations_converted']}")
        self.logger.info(f"   - Categories found: {len(stats['categories_found'])}")
        
        return stats
    
    def run(self):
        """Run the conversion process"""
        step_name = "converted"
        step_info = get_step_info(step_name, self.config)
        version = step_info['version']
        
        self.logger.info(f"   Starting COCO to YOLO conversion (Version {version})...")
        
        splits = [
            ('train', 'instances_train2017.json', 'train2017'),
            ('val', 'instances_val2017.json', 'val2017'),
            ('test', 'instances_test2017.json', 'test2017')
        ]
        
        total_stats = {
            'total_images_converted': 0,
            'total_annotations_converted': 0,
            'all_categories_found': set()
        }
        
        for split_name, json_file, images_dir in splits:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing {split_name} split...")
            self.logger.info(f"{'='*50}")
            
            stats = self.convert_split(split_name, json_file, images_dir)
            
            total_stats['total_images_converted'] += stats['images_converted']
            total_stats['total_annotations_converted'] += stats['annotations_converted']
            total_stats['all_categories_found'].update(stats['categories_found'])
        
        # Create data.yaml file
        self.logger.info("\nCreating data.yaml file...")
        data_yaml_path = create_data_yaml(self.processed_dir, self.damage_categories)
        self.logger.info(f"   Created data.yaml: {data_yaml_path}")
        
        # Save metrics for DVC
        self.logger.info("Saving metrics for DVC tracking...")
        metrics = {
            'conversion': {
                'version': version,
                'step': step_name,
                'images_converted': total_stats['total_images_converted'],
                'annotations_converted': total_stats['total_annotations_converted'],
                'categories_found': len(total_stats['all_categories_found']),
                'data_yaml': str(data_yaml_path.relative_to(self.base_dir)),
                'timestamp': str(Path.cwd())
            }
        }
        metrics_file = self.base_dir / self.config['dvc']['metrics_file']
        save_metrics(metrics, metrics_file)
        self.logger.info(f"   Metrics saved to: {metrics_file}")
        
        # Track with DVC (MANUAL PUSHING)
        self.logger.info("Starting DVC tracking...")
        dvc_file_path = get_dvc_file_path(step_name, self.base_dir, self.config)
        data_path = get_data_path_for_step(step_name, self.base_dir, self.config)
        
        track_success = track_with_dvc(
            data_path=data_path,
            dvc_file_path=dvc_file_path,
            step_name=step_name,
            version=version,
            metrics_file=metrics_file,
            push_to_remote=False  # MANUAL PUSHING
        )
        
        if track_success:
            self.logger.info(f"  COCO to YOLO conversion completed! (Version {version})")
            self.logger.info(f"   Final Statistics:")
            self.logger.info(f"   - Total images converted: {total_stats['total_images_converted']}")
            self.logger.info(f"   - Total annotations converted: {total_stats['total_annotations_converted']}")
            self.logger.info(f"   - Categories found: {len(total_stats['all_categories_found'])}")
            self.logger.info("")
            self.logger.info("💡 MANUAL DVC STEPS REQUIRED:")
            self.logger.info(f"   1. git add {dvc_file_path} .gitignore {metrics_file}")
            self.logger.info(f"   2. git commit -m 'detection_{version}_{step_name}: COCO to YOLO format conversion'")
            self.logger.info("   3. dvc push")
            self.logger.info("   4. git push origin main")
        else:
            self.logger.warning(f"✅ Conversion completed but DVC tracking had issues (Version {version})")
        
        return total_stats

def main():
    try:
        # Use the project root to find config file
        project_root = get_project_root()
        config_path = project_root / 'configs' / 'params.yaml'
        
        # Check if config file exists
        if not config_path.exists():
            print(f"ERROR: Config file not found at: {config_path}")
            print("Please make sure the file exists and the path is correct.")
            return
        
        print(f"Using config file: {config_path}")
        
        converter = COCOToYOLOConverter(config_path)
        converter.run()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your configuration and file paths.")

if __name__ == "__main__":
    main()