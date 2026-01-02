import yaml
from pathlib import Path
import numpy as np
import shutil
from utils import setup_logging, save_metrics, track_with_dvc, get_dvc_file_path, get_data_path_for_step, get_step_info, calculate_iou, get_project_root

class AnnotationCleaner:
    def __init__(self, config_path):
        self.config_path = config_path
        self.logger = setup_logging("Annotation_Cleaning")  # Initialize logger FIRST
        self.load_config()
        self.setup_paths()
        
    def load_config(self):
        """Load configuration from params.yaml"""
        self.logger.info(f"Loading config from: {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.clean_config = self.config['data_preparation']['cleaning']
        self.logger.info(f"Loaded cleaning config: {self.clean_config}")
        
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = get_project_root()
        self.filtered_dir = self.base_dir / "data" / "intermediate" / "filtered" / "detection"
        self.cleaned_dir = self.base_dir / "data" / "intermediate" / "cleaned" / "detection"
        
        # Log paths for verification
        self.logger.info(f"Project root: {self.base_dir}")
        self.logger.info(f"Filtered data: {self.filtered_dir}")
        self.logger.info(f"Cleaned data: {self.cleaned_dir}")
        
        # Verify filtered data exists
        if not self.filtered_dir.exists():
            raise FileNotFoundError(f"Filtered data directory not found: {self.filtered_dir}")
        else:
            self.logger.info("   Filtered data directory exists")
        
    def clean_bbox(self, bbox, image_width, image_height):
        """Clean individual bounding box"""
        x_center, y_center, width, height = bbox
        
        # Convert to pixel coordinates for validation
        x1 = (x_center - width/2) * image_width
        y1 = (y_center - height/2) * image_height
        w_pixels = width * image_width
        h_pixels = height * image_height
        
        area = w_pixels * h_pixels
        
        # Check minimum area
        if area < self.clean_config['min_bbox_area']:
            return False, "BBox area too small"
        
        # Check minimum dimensions
        if (w_pixels < self.clean_config['min_bbox_dimension'] or 
            h_pixels < self.clean_config['min_bbox_dimension']):
            return False, "BBox dimensions too small"
        
        # Check aspect ratio
        aspect_ratio = max(w_pixels, h_pixels) / min(w_pixels, h_pixels) if min(w_pixels, h_pixels) > 0 else float('inf')
        if aspect_ratio > self.clean_config['max_bbox_aspect_ratio']:
            return False, f"BBox aspect ratio too extreme: {aspect_ratio:.2f}"
        
        # Check if bbox is within image boundaries
        if (x1 < 0 or y1 < 0 or 
            (x1 + w_pixels) > image_width or 
            (y1 + h_pixels) > image_height):
            if self.clean_config['remove_out_of_bounds']:
                return False, "BBox outside image boundaries"
        
        return True, "Passed"
    
    def remove_duplicate_bboxes(self, annotations):
        """Remove duplicate bounding boxes using NMS"""
        if not self.clean_config['remove_duplicate_bboxes'] or len(annotations) <= 1:
            return annotations, 0
        
        # Convert to pixel coordinates for NMS
        bboxes = []
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            x1 = (x_center - width/2)
            y1 = (y_center - height/2)
            x2 = (x_center + width/2)
            y2 = (y_center + height/2)
            bboxes.append([x1, y1, x2, y2, class_id])
        
        bboxes = np.array(bboxes)
        
        # Simple NMS implementation
        keep_indices = []
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        order = areas.argsort()[::-1]
        
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)
            
            if order.size == 1:
                break
                
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
            yy1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
            xx2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
            yy2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= self.clean_config['iou_threshold'])[0]
            order = order[inds + 1]
        
        cleaned_annotations = [annotations[i] for i in keep_indices]
        duplicates_removed = len(annotations) - len(cleaned_annotations)
        
        return cleaned_annotations, duplicates_removed
    
    def clean_split(self, split_name):
        """Clean a single split"""
        self.logger.info(f"Cleaning {split_name} split...")
        
        source_split_dir = self.filtered_dir / split_name
        cleaned_split_dir = self.cleaned_dir / split_name
        
        # Check if source split exists
        if not source_split_dir.exists():
            self.logger.warning(f"Source split directory not found: {source_split_dir}")
            return {
                'images_processed': 0, 
                'images_cleaned': 0, 
                'bboxes_processed': 0, 
                'bboxes_removed': 0, 
                'duplicates_removed': 0, 
                'clean_reasons': {}
            }
        
        # Create cleaned directories
        cleaned_images_dir = cleaned_split_dir / 'images'
        cleaned_labels_dir = cleaned_split_dir / 'labels'
        cleaned_images_dir.mkdir(parents=True, exist_ok=True)
        cleaned_labels_dir.mkdir(parents=True, exist_ok=True)
        
        source_images_dir = source_split_dir / 'images'
        
        # Check if source images directory exists
        if not source_images_dir.exists():
            self.logger.warning(f"Source images directory not found: {source_images_dir}")
            return {
                'images_processed': 0, 
                'images_cleaned': 0, 
                'bboxes_processed': 0, 
                'bboxes_removed': 0, 
                'duplicates_removed': 0, 
                'clean_reasons': {}
            }
        
        stats = {
            'images_processed': 0,
            'images_cleaned': 0,
            'bboxes_processed': 0,
            'bboxes_removed': 0,
            'duplicates_removed': 0,
            'clean_reasons': {}
        }
        
        processed_count = 0
        for image_path in source_images_dir.glob('*.*'):
            label_path = source_split_dir / 'labels' / f"{image_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            stats['images_processed'] += 1
            processed_count += 1
            
            # Get actual image dimensions
            try:
                import cv2
                img = cv2.imread(str(image_path))
                if img is None:
                    self.logger.warning(f"Could not read image: {image_path}")
                    continue
                image_height, image_width = img.shape[:2]
            except Exception as e:
                self.logger.warning(f"Could not get dimensions for {image_path}: {e}")
                # Use default dimensions as fallback
                image_width, image_height = 640, 480
            
            # Read original annotations
            with open(label_path, 'r') as f:
                original_annotations = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        annotation = [int(parts[0])] + [float(x) for x in parts[1:]]
                        original_annotations.append(annotation)
            
            stats['bboxes_processed'] += len(original_annotations)
            
            # Clean annotations with actual image dimensions
            cleaned_annotations = []
            for ann in original_annotations:
                class_id, x_center, y_center, width, height = ann
                is_valid, reason = self.clean_bbox([x_center, y_center, width, height], image_width, image_height)
                
                if is_valid:
                    cleaned_annotations.append(ann)
                else:
                    stats['bboxes_removed'] += 1
                    stats['clean_reasons'][reason] = stats['clean_reasons'].get(reason, 0) + 1
            
            # Remove duplicates
            if cleaned_annotations:
                cleaned_annotations, duplicates_removed = self.remove_duplicate_bboxes(cleaned_annotations)
                stats['duplicates_removed'] += duplicates_removed
            
            # Save cleaned annotations if any remain
            if cleaned_annotations:
                with open(cleaned_labels_dir / label_path.name, 'w') as f:
                    for ann in cleaned_annotations:
                        class_id, x_center, y_center, width, height = ann
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # Copy image
                shutil.copy2(image_path, cleaned_images_dir / image_path.name)
                stats['images_cleaned'] += 1
            
            # Log progress every 100 images
            if processed_count % 100 == 0:
                self.logger.info(f"Processed {processed_count} images in {split_name}...")
        
        self.logger.info(f"✅ {split_name} cleaning completed:")
        self.logger.info(f"   - Images processed: {stats['images_processed']}")
        self.logger.info(f"   - Images cleaned: {stats['images_cleaned']}")
        self.logger.info(f"   - BBoxes processed: {stats['bboxes_processed']}")
        self.logger.info(f"   - BBoxes removed: {stats['bboxes_removed']}")
        self.logger.info(f"   - Duplicates removed: {stats['duplicates_removed']}")
        if stats['clean_reasons']:
            self.logger.info(f"   - Clean reasons: {stats['clean_reasons']}")
        
        return stats
    
    def run(self):
        """Run the cleaning process"""
        step_name = "cleaned"
        step_info = get_step_info(step_name, self.config)
        version = step_info['version']
        
        self.logger.info(f"   Starting annotation cleaning (Version {version})...")
        
        splits = ['train', 'val', 'test']
        
        total_stats = {
            'total_images_processed': 0,
            'total_images_cleaned': 0,
            'total_bboxes_processed': 0,
            'total_bboxes_removed': 0,
            'total_duplicates_removed': 0,
            'all_clean_reasons': {}
        }
        
        for split_name in splits:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing {split_name} split...")
            self.logger.info(f"{'='*50}")
            
            stats = self.clean_split(split_name)
            
            for key in ['images_processed', 'images_cleaned', 'bboxes_processed', 'bboxes_removed', 'duplicates_removed']:
                total_stats[f'total_{key}'] += stats[key]
            
            for reason, count in stats['clean_reasons'].items():
                total_stats['all_clean_reasons'][reason] = total_stats['all_clean_reasons'].get(reason, 0) + count
        
        # Save metrics for DVC
        self.logger.info("\nSaving metrics for DVC tracking...")
        metrics = {
            'cleaning': {
                'version': version,
                'step': step_name,
                'images_processed': total_stats['total_images_processed'],
                'images_cleaned': total_stats['total_images_cleaned'],
                'bboxes_processed': total_stats['total_bboxes_processed'],
                'bboxes_removed': total_stats['total_bboxes_removed'],
                'duplicates_removed': total_stats['total_duplicates_removed'],
                'clean_reasons': total_stats['all_clean_reasons'],
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
            self.logger.info(f"   Annotation cleaning completed! (Version {version})")
            self.logger.info(f"   Final Statistics:")
            self.logger.info(f"   - Total images processed: {total_stats['total_images_processed']}")
            self.logger.info(f"   - Total images cleaned: {total_stats['total_images_cleaned']}")
            self.logger.info(f"   - Total bboxes processed: {total_stats['total_bboxes_processed']}")
            self.logger.info(f"   - Total bboxes removed: {total_stats['total_bboxes_removed']}")
            self.logger.info(f"   - Total duplicates removed: {total_stats['total_duplicates_removed']}")
            self.logger.info(f"   - Clean reasons: {total_stats['all_clean_reasons']}")
            self.logger.info("")
            self.logger.info("   MANUAL DVC STEPS REQUIRED:")
            self.logger.info(f"   1. git add {dvc_file_path} .gitignore {metrics_file}")
            self.logger.info(f"   2. git commit -m 'detection_{version}_{step_name}: Annotation cleaning and deduplication'")
            self.logger.info("   3. dvc push  # Skip if using DVC locally without remote")
            self.logger.info("   4. git push origin main")
        else:
            self.logger.warning(f"  Cleaning completed but DVC tracking had issues (Version {version})")
        
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
        
        cleaner = AnnotationCleaner(config_path)
        cleaner.run()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your configuration and file paths.")

if __name__ == "__main__":
    main()