import yaml
from pathlib import Path
import shutil
from utils import setup_logging, save_metrics, track_with_dvc, get_dvc_file_path, get_data_path_for_step, get_step_info, create_data_yaml, get_project_root

class YOLOPreprocessor:
    def __init__(self, config_path):
        self.config_path = config_path
        self.logger = setup_logging("YOLO_Preprocessing")  # Initialize logger FIRST
        self.load_config()
        self.setup_paths()
        
    def load_config(self):
        """Load configuration from params.yaml"""
        self.logger.info(f"Loading config from: {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.yolo_config = self.config['data_preparation']['yolo']
        self.damage_categories = self.config['damage_categories']
        self.logger.info(f"Loaded YOLO config: {self.yolo_config}")
        self.logger.info(f"Loaded {len(self.damage_categories)} damage categories")
        
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = get_project_root()
        self.cleaned_dir = self.base_dir / "data" / "intermediate" / "cleaned" / "detection"
        self.final_dir = self.base_dir / self.config['data_preparation']['paths']['processed_data']
        
        # Log paths for verification
        self.logger.info(f"Project root: {self.base_dir}")
        self.logger.info(f"Cleaned data: {self.cleaned_dir}")
        self.logger.info(f"Final data: {self.final_dir}")
        
        # Verify cleaned data exists
        if not self.cleaned_dir.exists():
            raise FileNotFoundError(f"Cleaned data directory not found: {self.cleaned_dir}")
        else:
            self.logger.info("   Cleaned data directory exists")
        
    def prepare_final_dataset(self):
        """Prepare final dataset structure"""
        self.logger.info("Preparing final dataset structure...")
        
        # Copy cleaned data to final directory
        splits = ['train', 'val', 'test']
        
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'splits_processed': []
        }
        
        for split_name in splits:
            self.logger.info(f"Processing {split_name} split...")
            
            source_split_dir = self.cleaned_dir / split_name
            final_split_dir = self.final_dir / split_name
            
            # Check if source split exists
            if not source_split_dir.exists():
                self.logger.warning(f"Source split directory not found: {source_split_dir}")
                stats['splits_processed'].append({
                    'split': split_name,
                    'images': 0,
                    'annotations': 0
                })
                continue
            
            # Create final directories
            final_images_dir = final_split_dir / 'images'
            final_labels_dir = final_split_dir / 'labels'
            final_images_dir.mkdir(parents=True, exist_ok=True)
            final_labels_dir.mkdir(parents=True, exist_ok=True)
            
            source_images_dir = source_split_dir / 'images'
            
            # Check if source images directory exists
            if not source_images_dir.exists():
                self.logger.warning(f"Source images directory not found: {source_images_dir}")
                stats['splits_processed'].append({
                    'split': split_name,
                    'images': 0,
                    'annotations': 0
                })
                continue
            
            split_stats = {
                'split': split_name,
                'images': 0,
                'annotations': 0
            }
            
            processed_count = 0
            for image_path in source_images_dir.glob('*.*'):
                label_path = source_split_dir / 'labels' / f"{image_path.stem}.txt"
                
                if label_path.exists():
                    shutil.copy2(image_path, final_images_dir / image_path.name)
                    shutil.copy2(label_path, final_labels_dir / label_path.name)
                    
                    # Count annotations
                    with open(label_path, 'r') as f:
                        annotations = f.readlines()
                    
                    stats['total_images'] += 1
                    stats['total_annotations'] += len(annotations)
                    
                    split_stats['images'] += 1
                    split_stats['annotations'] += len(annotations)
                    
                    processed_count += 1
                    
                    # Log progress every 100 images
                    if processed_count % 100 == 0:
                        self.logger.info(f"Processed {processed_count} images in {split_name}...")
            
            stats['splits_processed'].append(split_stats)
            self.logger.info(f"   {split_name}: {split_stats['images']} images, {split_stats['annotations']} annotations")
        
        # Create dataset YAML
        self.logger.info("Creating data.yaml file...")
        data_yaml_path = create_data_yaml(
            self.final_dir,
            self.damage_categories,
            'car_damage'
        )
        
        self.logger.info("    Final dataset preparation completed:")
        self.logger.info(f"   - Total images: {stats['total_images']}")
        self.logger.info(f"   - Total annotations: {stats['total_annotations']}")
        self.logger.info(f"   - Data YAML: {data_yaml_path}")
        
        # Log split-wise statistics
        for split_stat in stats['splits_processed']:
            self.logger.info(f"   - {split_stat['split']}: {split_stat['images']} images, {split_stat['annotations']} annotations")
        
        return stats
    
    def run(self):
        """Run the preprocessing process"""
        step_name = "preprocessed"
        step_info = get_step_info(step_name, self.config)
        version = step_info['version']
        
        self.logger.info(f"   Starting YOLO preprocessing (Version {version})...")
        
        stats = self.prepare_final_dataset()
        
        # Save metrics for DVC
        self.logger.info("Saving metrics for DVC tracking...")
        metrics = {
            'preprocessing': {
                'version': version,
                'step': step_name,
                'final_images': stats['total_images'],
                'final_annotations': stats['total_annotations'],
                'image_size': self.yolo_config['image_size'],
                'categories': len(self.damage_categories),
                'splits': stats['splits_processed'],
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
            self.logger.info(f"   YOLO preprocessing completed! (Version {version})")
            self.logger.info(f"   Final Dataset Statistics:")
            self.logger.info(f"   - Total images: {stats['total_images']}")
            self.logger.info(f"   - Total annotations: {stats['total_annotations']}")
            self.logger.info(f"   - Image size: {self.yolo_config['image_size']}")
            self.logger.info(f"   - Categories: {len(self.damage_categories)}")
            self.logger.info("")
            self.logger.info("  MANUAL DVC STEPS REQUIRED:")
            self.logger.info(f"   1. git add {dvc_file_path} .gitignore {metrics_file}")
            self.logger.info(f"   2. git commit -m 'detection_{version}_{step_name}: Final YOLO preprocessing'")
            self.logger.info("   3. dvc push  # Skip if using DVC locally without remote")
            self.logger.info("   4. git push origin main")
            self.logger.info("")
            self.logger.info("  Your dataset is now ready for YOLO training!")
            self.logger.info(f"   Training command: yolo train data={self.final_dir}/data.yaml model=yolov8n.pt epochs=100 imgsz={self.yolo_config['image_size']}")
        else:
            self.logger.warning(f"  Preprocessing completed but DVC tracking had issues (Version {version})")
        
        return stats

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
        
        preprocessor = YOLOPreprocessor(config_path)
        preprocessor.run()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your configuration and file paths.")

if __name__ == "__main__":
    main()