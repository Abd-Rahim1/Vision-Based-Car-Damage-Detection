import yaml
from pathlib import Path
import cv2
from utils import setup_logging, save_metrics, get_step_info

class DatasetVerifier:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_paths()
        self.logger = setup_logging("Dataset_Verification")
        
    def load_config(self):
        """Load configuration from params.yaml"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.damage_categories = self.config['damage_categories']
        
    def setup_paths(self):
        """Setup all necessary paths"""
        # Use the config file location to determine base directory
        self.base_dir = self.config_path.parents[1]  # Go up from configs/ to project root
        self.final_dir = self.base_dir / self.config['data_preparation']['paths']['processed_data']
        
    def verify_split(self, split_name):
        """Verify a single split"""
        self.logger.info(f"Verifying {split_name} split...")
        
        split_dir = self.final_dir / split_name
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        # Check if directories exist
        if not split_dir.exists():
            self.logger.warning(f"Split directory not found: {split_dir}")
            return self._get_empty_stats(split_name)
            
        if not images_dir.exists():
            self.logger.warning(f"Images directory not found: {images_dir}")
            return self._get_empty_stats(split_name)
            
        if not labels_dir.exists():
            self.logger.warning(f"Labels directory not found: {labels_dir}")
            return self._get_empty_stats(split_name)
        
        stats = {
            'images_found': 0,
            'labels_found': 0,
            'images_with_issues': 0,
            'labels_with_issues': 0,
            'annotation_stats': {},
            'image_sizes': {},
            'issues': []
        }
        
        # Count images
        for image_path in images_dir.glob('*.*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                stats['images_found'] += 1
                
                # Verify image can be loaded
                try:
                    img = cv2.imread(str(image_path))
                    if img is None:
                        stats['images_with_issues'] += 1
                        stats['issues'].append(f"Cannot load image: {image_path.name}")
                    else:
                        img_size = f"{img.shape[1]}x{img.shape[0]}"
                        stats['image_sizes'][img_size] = stats['image_sizes'].get(img_size, 0) + 1
                except Exception as e:
                    stats['images_with_issues'] += 1
                    stats['issues'].append(f"Error loading image {image_path.name}: {e}")
        
        # Count labels and verify annotations
        for label_path in labels_dir.glob('*.txt'):
            stats['labels_found'] += 1
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        stats['labels_with_issues'] += 1
                        stats['issues'].append(f"Invalid annotation format in {label_path.name}, line {line_num}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate values
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 <= width <= 1 and 0 <= height <= 1):
                            stats['labels_with_issues'] += 1
                            stats['issues'].append(f"Invalid coordinates in {label_path.name}, line {line_num}")
                        
                        if class_id >= len(self.damage_categories):
                            stats['labels_with_issues'] += 1
                            stats['issues'].append(f"Invalid class ID {class_id} in {label_path.name}, line {line_num}")
                        
                        # Update annotation statistics
                        class_name = self.damage_categories[class_id] if class_id < len(self.damage_categories) else f"class_{class_id}"
                        stats['annotation_stats'][class_name] = stats['annotation_stats'].get(class_name, 0) + 1
                        
                    except ValueError:
                        stats['labels_with_issues'] += 1
                        stats['issues'].append(f"Invalid number format in {label_path.name}, line {line_num}")
                        
            except Exception as e:
                stats['labels_with_issues'] += 1
                stats['issues'].append(f"Error reading label file {label_path.name}: {e}")
        
        self.logger.info(f"{split_name} verification completed:")
        self.logger.info(f"   - Images: {stats['images_found']} found, {stats['images_with_issues']} with issues")
        self.logger.info(f"   - Labels: {stats['labels_found']} found, {stats['labels_with_issues']} with issues")
        self.logger.info(f"   - Annotations by class: {stats['annotation_stats']}")
        
        return stats
    
    def _get_empty_stats(self, split_name):
        """Return empty statistics for missing splits"""
        self.logger.warning(f"Returning empty stats for missing split: {split_name}")
        return {
            'images_found': 0,
            'labels_found': 0,
            'images_with_issues': 0,
            'labels_with_issues': 0,
            'annotation_stats': {},
            'image_sizes': {},
            'issues': [f"Split directory not found: {split_name}"]
        }
    
    def run(self):
        """Run the verification process"""
        step_name = "verified"
        
        self.logger.info("Starting dataset verification...")
        
        splits = ['train', 'val', 'test']
        
        total_stats = {
            'total_images': 0,
            'total_labels': 0,
            'total_images_with_issues': 0,
            'total_labels_with_issues': 0,
            'all_annotation_stats': {},
            'all_image_sizes': {},
            'all_issues': []
        }
        
        for split_name in splits:
            stats = self.verify_split(split_name)
            
            total_stats['total_images'] += stats['images_found']
            total_stats['total_labels'] += stats['labels_found']
            total_stats['total_images_with_issues'] += stats['images_with_issues']
            total_stats['total_labels_with_issues'] += stats['labels_with_issues']
            total_stats['all_issues'].extend(stats['issues'])
            
            for class_name, count in stats['annotation_stats'].items():
                total_stats['all_annotation_stats'][class_name] = total_stats['all_annotation_stats'].get(class_name, 0) + count
            
            for img_size, count in stats['image_sizes'].items():
                total_stats['all_image_sizes'][img_size] = total_stats['all_image_sizes'].get(img_size, 0) + count
        
        # Calculate overall quality metrics
        image_quality = 1 - (total_stats['total_images_with_issues'] / max(1, total_stats['total_images']))
        label_quality = 1 - (total_stats['total_labels_with_issues'] / max(1, total_stats['total_labels']))
        overall_quality = (image_quality + label_quality) / 2
        
        # Save metrics for DVC
        metrics = {
            'verification': {
                'step': step_name,
                'total_images': total_stats['total_images'],
                'total_labels': total_stats['total_labels'],
                'image_quality_score': round(image_quality, 4),
                'label_quality_score': round(label_quality, 4),
                'overall_quality_score': round(overall_quality, 4),
                'annotation_distribution': total_stats['all_annotation_stats'],
                'image_size_distribution': total_stats['all_image_sizes'],
                'issues_found': len(total_stats['all_issues']),
                'timestamp': str(Path.cwd())
            }
        }
        metrics_file = self.base_dir / self.config['dvc']['metrics_file']
        save_metrics(metrics, metrics_file)
        
        self.logger.info("Dataset verification completed:")
        self.logger.info(f"   - Overall quality score: {overall_quality:.4f}")
        self.logger.info(f"   - Image quality score: {image_quality:.4f}")
        self.logger.info(f"   - Label quality score: {label_quality:.4f}")
        self.logger.info(f"   - Total issues found: {len(total_stats['all_issues'])}")
        self.logger.info(f"   - Annotation distribution: {total_stats['all_annotation_stats']}")
        
        if total_stats['all_issues']:
            self.logger.warning("Some issues were found during verification:")
            for issue in total_stats['all_issues'][:10]:  # Show first 10 issues
                self.logger.warning(f"   - {issue}")
            if len(total_stats['all_issues']) > 10:
                self.logger.warning(f"   - ... and {len(total_stats['all_issues']) - 10} more issues")
        
        return total_stats

def main():
    # More robust path handling
    current_script = Path(__file__)
    project_root = current_script.parents[2]  # Adjust based on your actual structure
    
    # Try multiple possible locations for the config file
    possible_config_paths = [
        project_root / 'configs' / 'params.yaml',
        project_root / 'params.yaml',
        current_script.parents[3] / 'configs' / 'params.yaml',  # Original attempt
    ]
    
    config_path = None
    for path in possible_config_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path is None:
        # If no config found, try to find it by searching
        config_files = list(project_root.rglob('params.yaml'))
        if config_files:
            config_path = config_files[0]
        else:
            raise FileNotFoundError("Could not find params.yaml config file")
    
    print(f"Using config file: {config_path}")
    verifier = DatasetVerifier(config_path)
    verifier.run()

if __name__ == "__main__":
    main()