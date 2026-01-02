import logging
import json
import yaml
from pathlib import Path
import numpy as np
import subprocess
import sys
from datetime import datetime

def setup_logging(name):
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def save_metrics(metrics, metrics_file):
    """Save metrics for DVC tracking"""
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics if any
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = {}
    
    # Update with new metrics
    existing_metrics.update(metrics)
    
    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(existing_metrics, f, indent=2)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union

def create_data_yaml(output_dir, class_names, dataset_name="car_damage"):
    """Create YOLO data.yaml file with proper structure"""
    # Use relative paths for better portability
    yaml_content = {
        'path': str(Path(output_dir).relative_to(Path(output_dir).parents[2])),  # Relative to project root
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path

def get_project_root():
    """Get the project root directory reliably"""
    current_file = Path(__file__)
    # Navigate up from utils.py to project root
    # utils.py -> data_preparation -> detection -> scripts -> Car_Damage_Detection
    project_root = current_file.parents[4]
    return project_root

def get_project_root():
    """Get the project root directory reliably"""
    current_file = Path(__file__)
    
    # Based on debug output, project root is at parents[3]
    # debug_paths.py -> data_preparation -> detection -> scripts -> Car_Damage_Detection (level 3)
    project_root = current_file.parents[3]
    
    # Verify it's the correct root by checking for configs directory
    config_dir = project_root / 'configs'
    if config_dir.exists():
        print(f"   Found project root: {project_root}")
        return project_root
    else:
        # Fallback: search upwards
        for parent in current_file.parents:
            config_dir = parent / 'configs'
            if config_dir.exists():
                print(f"   Found project root by searching: {parent}")
                return parent
        
        # Last resort
        print(f"⚠️ Using calculated root: {project_root}")
        return project_root
def get_step_info(step_name, config):
    """Get step information including version and description"""
    step_info = {
        'converted': {
            'version': config['data_preparation']['versions']['converted'],
            'description': 'COCO to YOLO format conversion'
        },
        'filtered': {
            'version': config['data_preparation']['versions']['filtered'],
            'description': 'Dataset filtering and validation'
        },
        'cleaned': {
            'version': config['data_preparation']['versions']['cleaned'],
            'description': 'Annotation cleaning and deduplication'
        },
        'preprocessed': {
            'version': config['data_preparation']['versions']['preprocessed'],
            'description': 'Final YOLO preprocessing'
        }
    }
    return step_info.get(step_name, {'version': 'v0', 'description': 'Unknown step'})

def get_dvc_file_path(step_name, base_dir, config):
    """Get DVC file path for a specific step"""
    step_info = get_step_info(step_name, config)
    version = step_info['version']
    
    # DVC creates files next to the data, so use that location
    data_path = get_data_path_for_step(step_name, base_dir, config)
    dvc_file_name = f"{data_path.name}.dvc"
    
    return data_path.parent / dvc_file_name

def get_data_path_for_step(step_name, base_dir, config):
    """Get data path for a specific processing step"""
    paths_config = config['data_preparation']['paths']
    
    if step_name == "converted":
        return base_dir / paths_config['processed_data']
    elif step_name == "filtered":
        return base_dir / paths_config['intermediate_data'] / "filtered" / "detection"
    elif step_name == "cleaned":
        return base_dir / paths_config['intermediate_data'] / "cleaned" / "detection"
    elif step_name == "preprocessed":
        return base_dir / paths_config['processed_data']
    else:
        return base_dir / paths_config['processed_data']

def track_with_dvc(data_path, dvc_file_path, step_name, version, metrics_file=None, push_to_remote=False):
    """
    Track dataset with DVC - simplified version for manual pushing
    
    Args:
        data_path: Path to the data directory to track
        dvc_file_path: Path for the .dvc file
        step_name: Name of the processing step
        version: Version tag for this dataset
        metrics_file: Path to metrics file to track
        push_to_remote: Set to False for manual pushing
    """
    logger = setup_logging("DVC_Tracking")
    
    try:
        # Ensure paths are Path objects
        data_path = Path(data_path)
        dvc_file_path = Path(dvc_file_path)
        
        # Create directory for .dvc file if it doesn't exist
        dvc_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Tracking {step_name} dataset (Version {version}) with DVC...")
        
        # Add data to DVC tracking - SIMPLIFIED
        dvc_add_cmd = ["dvc", "add", str(data_path)]
        
        logger.info(f"Running: {' '.join(dvc_add_cmd)}")
        result = subprocess.run(dvc_add_cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"DVC add failed: {result.stderr}")
            return False
        
        logger.info("   DVC add completed successfully")
        
        # Move the .dvc file to our desired location
        generated_dvc_file = Path(f"{data_path}.dvc")
        if generated_dvc_file.exists():
            generated_dvc_file.rename(dvc_file_path)
            logger.info(f"   Moved .dvc file to: {dvc_file_path}")
        
        # Track metrics if provided
        if metrics_file and Path(metrics_file).exists():
            logger.info(f"Tracking metrics: {metrics_file}")
            # Just log the metrics location - user can manually track if needed
            logger.info(f"Metrics available at: {metrics_file}")
        
        logger.info(f"   DVC tracking prepared for {step_name} (Version {version})")
        logger.info("   Manual steps required:")
        logger.info(f"   1. git add {dvc_file_path} .gitignore")
        if metrics_file:
            logger.info(f"   1. git add {metrics_file}")
        logger.info(f"   2. git commit -m 'detection_{version}_{step_name}: Dataset {step_name}'")
        logger.info("   3. dvc push")
        logger.info("   4. git push origin main")
        
        return True
        
    except Exception as e:
        logger.error(f"DVC tracking failed for {step_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_version_report(config, base_dir):
    """Generate a version report for all processing steps"""
    logger = setup_logging("Version_Report")
    
    versions = config['data_preparation']['versions']
    dvc_tracking_base = base_dir / config['data_preparation']['paths']['dvc_tracking']
    
    logger.info("Dataset Version Report:")
    logger.info("=" * 50)
    
    steps = ['converted', 'filtered', 'cleaned', 'preprocessed']
    for step in steps:
        version = versions[step]
        dvc_file = dvc_tracking_base / f"detection_{version}_{step}.dvc"
        status = "Tracked" if dvc_file.exists() else "❌ Missing"
        
        logger.info(f"Version {version} - {step:12} {status}")
    
    logger.info("=" * 50)