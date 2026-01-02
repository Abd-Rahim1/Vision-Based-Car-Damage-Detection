import yaml
from pathlib import Path
import shutil
from utils import setup_logging, save_metrics, track_with_dvc, get_dvc_file_path, get_data_path_for_step, get_step_info, get_project_root

class DatasetFilter:
    def __init__(self, config_path):
        self.config_path = config_path
        self.logger = setup_logging("Dataset_Filtering")  # Initialize logger FIRST
        self.load_config()
        self.setup_paths()
        
    def load_config(self):
        """Load configuration from params.yaml"""
        self.logger.info(f"Loading config from: {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.filter_config = self.config['data_preparation']['filtering']
        self.logger.info(f"Loaded filtering config: {self.filter_config}")
        
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = get_project_root()
        self.processed_dir = self.base_dir / self.config['data_preparation']['paths']['processed_data']
        self.filtered_dir = self.base_dir / "data" / "intermediate" / "filtered" / "detection"
        
        # Log paths for verification
        self.logger.info(f"Project root: {self.base_dir}")
        self.logger.info(f"Source data: {self.processed_dir}")
        self.logger.info(f"Filtered data: {self.filtered_dir}")
        
        # Verify source data exists
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Source data directory not found: {self.processed_dir}")
        else:
            self.logger.info("   Source data directory exists")
        
    def filter_image(self, image_path, label_path):
        """Filter individual image based on criteria"""
        # Check file extension
        ext = image_path.suffix.lower()
        if ext not in self.filter_config['allowed_extensions']:
            return False, "Invalid file extension"
        
        # Check file size
        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.filter_config['max_file_size_mb']:
            return False, f"File too large: {file_size_mb:.2f}MB"
        
        # Check if label file exists and has annotations
        if label_path.exists():
            with open(label_path, 'r') as f:
                annotations = f.readlines()
            
            if len(annotations) < self.filter_config['min_annotations_per_image']:
                return False, f"Insufficient annotations: {len(annotations)}"
        else:
            return False, "Label file missing"
        
        return True, "Passed"
    
    def filter_split(self, split_name):
        """Filter a single split"""
        self.logger.info(f"Filtering {split_name} split...")
        
        source_split_dir = self.processed_dir / split_name
        filtered_split_dir = self.filtered_dir / split_name
        
        # Check if source split exists
        if not source_split_dir.exists():
            self.logger.warning(f"Source split directory not found: {source_split_dir}")
            return {'images_processed': 0, 'images_passed': 0, 'images_filtered': 0, 'filter_reasons': {}}
        
        # Create filtered directories
        filtered_images_dir = filtered_split_dir / 'images'
        filtered_labels_dir = filtered_split_dir / 'labels'
        filtered_images_dir.mkdir(parents=True, exist_ok=True)
        filtered_labels_dir.mkdir(parents=True, exist_ok=True)
        
        source_images_dir = source_split_dir / 'images'
        
        # Check if source images directory exists
        if not source_images_dir.exists():
            self.logger.warning(f"Source images directory not found: {source_images_dir}")
            return {'images_processed': 0, 'images_passed': 0, 'images_filtered': 0, 'filter_reasons': {}}
        
        stats = {
            'images_processed': 0,
            'images_passed': 0,
            'images_filtered': 0,
            'filter_reasons': {}
        }
        
        for image_path in source_images_dir.glob('*.*'):
            if image_path.suffix.lower() not in self.filter_config['allowed_extensions']:
                continue
            
            label_path = source_split_dir / 'labels' / f"{image_path.stem}.txt"
            
            stats['images_processed'] += 1
            is_valid, reason = self.filter_image(image_path, label_path)
            
            if is_valid:
                # Copy to filtered directory
                shutil.copy2(image_path, filtered_images_dir / image_path.name)
                shutil.copy2(label_path, filtered_labels_dir / label_path.name)
                stats['images_passed'] += 1
            else:
                stats['images_filtered'] += 1
                stats['filter_reasons'][reason] = stats['filter_reasons'].get(reason, 0) + 1
            
            # Log progress every 100 images
            if stats['images_processed'] % 100 == 0:
                self.logger.info(f"Processed {stats['images_processed']} images in {split_name}...")
        
        self.logger.info(f"   {split_name} filtering completed:")
        self.logger.info(f"   - Processed: {stats['images_processed']}")
        self.logger.info(f"   - Passed: {stats['images_passed']}")
        self.logger.info(f"   - Filtered: {stats['images_filtered']}")
        if stats['filter_reasons']:
            self.logger.info(f"   - Filter reasons: {stats['filter_reasons']}")
        
        return stats
    
    def run(self):
        """Run the filtering process"""
        step_name = "filtered"
        step_info = get_step_info(step_name, self.config)
        version = step_info['version']
        
        self.logger.info(f"   Starting dataset filtering (Version {version})...")
        
        splits = ['train', 'val', 'test']
        
        total_stats = {
            'total_processed': 0,
            'total_passed': 0,
            'total_filtered': 0,
            'all_filter_reasons': {}
        }
        
        for split_name in splits:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing {split_name} split...")
            self.logger.info(f"{'='*50}")
            
            stats = self.filter_split(split_name)
            
            total_stats['total_processed'] += stats['images_processed']
            total_stats['total_passed'] += stats['images_passed']
            total_stats['total_filtered'] += stats['images_filtered']
            
            for reason, count in stats['filter_reasons'].items():
                total_stats['all_filter_reasons'][reason] = total_stats['all_filter_reasons'].get(reason, 0) + count
        
        # Save metrics for DVC
        self.logger.info("\nSaving metrics for DVC tracking...")
        metrics = {
            'filtering': {
                'version': version,
                'step': step_name,
                'images_processed': total_stats['total_processed'],
                'images_passed': total_stats['total_passed'],
                'images_filtered': total_stats['total_filtered'],
                'filter_reasons': total_stats['all_filter_reasons'],
                'timestamp': str(Path.cwd())
            }
        }
        metrics_file = self.base_dir / self.config['dvc']['metrics_file']
        save_metrics(metrics, metrics_file)
        self.logger.info(f"  Metrics saved to: {metrics_file}")
        
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
            self.logger.info(f"   Dataset filtering completed! (Version {version})")
            self.logger.info(f"   Final Statistics:")
            self.logger.info(f"   - Total images processed: {total_stats['total_processed']}")
            self.logger.info(f"   - Total images passed: {total_stats['total_passed']}")
            self.logger.info(f"   - Total images filtered: {total_stats['total_filtered']}")
            self.logger.info(f"   - Filter reasons: {total_stats['all_filter_reasons']}")
            self.logger.info("")
            self.logger.info("  MANUAL DVC STEPS REQUIRED:")
            self.logger.info(f"   1. git add {dvc_file_path} .gitignore {metrics_file}")
            self.logger.info(f"   2. git commit -m 'detection_{version}_{step_name}: Dataset filtering and validation'")
            self.logger.info("   3. dvc push  # Skip if using DVC locally without remote")
            self.logger.info("   4. git push origin main")
        else:
            self.logger.warning(f"  Filtering completed but DVC tracking had issues (Version {version})")
        
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
        
        filter = DatasetFilter(config_path)
        filter.run()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your configuration and file paths.")

if __name__ == "__main__":
    main()