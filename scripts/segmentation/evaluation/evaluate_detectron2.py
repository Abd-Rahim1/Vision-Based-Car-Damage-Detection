import os
import sys
import argparse
import cv2
import torch
import random
import numpy as np
import mlflow
import json
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.io import load_params, get_files, read_mask, save_json, ensure_dir
from src.utils.paths import ProjectPaths

def get_damage_dicts(img_dir, mask_dir):
    dataset_dicts = []
    images = get_files(img_dir)
    
    for idx, img_path in enumerate(images):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        mask_filename = filename.replace(os.path.splitext(filename)[1], ".png")
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            continue

        mask = read_mask(mask_path)
        if mask is None:
            continue
            
        objs = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                 continue
                 
            flattened = contour.flatten().tolist()
            if len(flattened) < 6:
                continue
                
            obj = {
                "bbox": cv2.boundingRect(contour),
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [flattened],
                "category_id": 0,
            }
            objs.append(obj)
            
        if objs:
            record = {}
            record["file_name"] = img_path
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            record["annotations"] = objs
            dataset_dicts.append(record)
        
    return dataset_dicts

def setup_cfg(params, model_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(params['training']['model_config']))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params['training']['num_classes']
    
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"
        
    return cfg

def evaluate_trial(trial_name, trial_dir, params):
    print(f"\n{'='*20} Evaluating {trial_name} {'='*20}")
    
    model_path = os.path.join(trial_dir, "model_final.pth")
    if not os.path.exists(model_path):
        print(f"Skipping {trial_name}: model_final.pth not found at {model_path}")
        return

    # Setup MLflow run for this trial
    run_name = f"eval_{trial_name}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log Parameters
        mlflow.log_param("trial_id", trial_name)
        mlflow.log_param("model_path", model_path)
        
        # Setup Config & Predictor
        cfg = setup_cfg(params, model_path)
        mlflow.log_param("score_threshold", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
        
        predictor = DefaultPredictor(cfg)
        
        # Define output directory for this trial's evaluation
        eval_output_dir = os.path.join(ProjectPaths.DETECTRON2_EVAL_DIR, trial_name)
        os.makedirs(eval_output_dir, exist_ok=True)
        
        # Run Evaluation
        evaluator = COCOEvaluator("car_damage_test", cfg, False, output_dir=eval_output_dir)
        val_loader = build_detection_test_loader(cfg, "car_damage_test")
        results = inference_on_dataset(predictor.model, val_loader, evaluator)
        
        print(f"Results for {trial_name}: {results}")
        
        # Log Metrics
        # Flatten the nested structure (e.g. {'segm': {'AP': ...}})
        if "segm" in results:
            for k, v in results["segm"].items():
                mlflow.log_metric(f"segm_{k}", v)
        
        # Save Results locally nicely
        results_file = os.path.join(eval_output_dir, "coco_results.json")
        # Note: COCOEvaluator already saves coco_instances_results.json etc, 
        # but let's save our parsed summary as well.
        import copy
        serializable_results = copy.deepcopy(results)
        # Handle cases where values might not be serializable (though usually float)
        def convert_numpy(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
            
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, default=convert_numpy, indent=2)
            
        # Log Artifacts
        mlflow.log_artifact(results_file)
        # Also log the standard COCO outputs if they exist
        for f_name in os.listdir(eval_output_dir):
            full_path = os.path.join(eval_output_dir, f_name)
            if os.path.isfile(full_path):
                mlflow.log_artifact(full_path)
                
        print(f"Completed evaluation for {trial_name}")

def main():
    params = load_params()
    
    # 1. Setup Data Registry (Global)
    # processed_path = params['data']['processed_path']
    test_img_dir = ProjectPaths.PROCESSED_DATA_DIR / "segmentation" / "test" / "images"
    test_mask_dir = ProjectPaths.PROCESSED_DATA_DIR / "segmentation" / "test" / "masks"
    
    # Ensure paths are strings for cv2/D2 compatibility if needed, though pathlib usually works. 
    # Detectron2 often expects strings.
    test_img_dir = str(test_img_dir)
    test_mask_dir = str(test_mask_dir)
    
    # Check if already registered to avoid errors when re-running or in notebooks
    if "car_damage_test" in DatasetCatalog.list():
        DatasetCatalog.remove("car_damage_test")
        
    DatasetCatalog.register("car_damage_test", lambda: get_damage_dicts(test_img_dir, test_mask_dir))
    MetadataCatalog.get("car_damage_test").set(thing_classes=["damage"])
    
    # 2. Setup MLflow
    # Use the same tracking URI as training
    mlflow.set_tracking_uri(ProjectPaths.MLFLOW_TRACKING_URI)
    
    experiment_name = "detectron2_segmentation" # Same experiment as training
    mlflow.set_experiment(experiment_name)
    
    # 3. Find Trials
    # 3. Find Trials
    output_root = ProjectPaths.DETECTRON2_ARTIFACTS_DIR
    if not os.path.exists(output_root):
        print(f"Output directory {output_root} not found.")
        return

    # Look for trial folders
    trial_dirs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d)) and d.startswith("trial_")]
    trial_dirs.sort() # Ensure trial_0, trial_1, ...
    
    if not trial_dirs:
        print("No trial directories found in output/")
        return
        
    print(f"Found trials: {trial_dirs}")
    
    # 4. Loop & Evaluate
    for trial_name in trial_dirs:
        trial_path = os.path.join(output_root, trial_name)
        evaluate_trial(trial_name, trial_path, params)

if __name__ == "__main__":
    main()