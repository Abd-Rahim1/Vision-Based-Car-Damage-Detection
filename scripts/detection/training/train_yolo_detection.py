# train_yolo_detection.py
import yaml
from ultralytics import YOLO
from pathlib import Path
import torch
import time
import sys
import shutil
import mlflow
import os

# Add project root to path to import src
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths

def main():
    print("=== CAR DAMAGE DETECTION - GPU TRAINING ===")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = 0  # Use GPU
    else:
        print("GPU not available, using CPU")
        device = 'cpu'
    
    start_time = time.time()
    
    # Clear GPU cache if using GPU
    if device == 0:
        torch.cuda.empty_cache()
    
    # CONFIGURATION
    dataset_path = ProjectPaths.PROCESSED_DATA_DIR / "detection" / "data.yaml"
    
    # Define paths for models and artifacts using centralized paths
    experiments_path = ProjectPaths.YOLO_ARTIFACTS_DIR
    models_path = ProjectPaths.YOLO_ARTIFACTS_DIR # Use same dir for consistency or subdir
    
    # Create directories if they don't exist
    experiments_path.mkdir(parents=True, exist_ok=True)
    ProjectPaths.MLFLOW_TRACKING_PATH.mkdir(parents=True, exist_ok=True)
    
    experiment_name = f"car_damage_detection_{int(time.time())}"
    
    # MLflow Setup
    mlflow.set_tracking_uri(ProjectPaths.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("yolo_detection")

    # Verify dataset
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please make sure you've run the data preparation pipeline first.")
        return
    
    # Load dataset config
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"Dataset Info:")
    print(f"Path: {dataset_path}")
    print(f"Classes: {dataset_config['names']}")
    print(f"Number of classes: {dataset_config['nc']}")
    
    # Fixed image counting function
    def count_images(split_path):
        """Properly count images in dataset splits"""
        base_path = Path(dataset_config['path'])
        if not base_path.is_absolute():
            base_path = dataset_path.parent / base_path
        
        images_path = base_path / split_path
        
        if images_path.exists():
            image_files = [f for f in images_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            return len(image_files)
        return 0
    
    train_images = count_images(dataset_config['train'])
    val_images = count_images(dataset_config['val']) 
    test_images = count_images(dataset_config['test'])
    
    print(f"Train images: {train_images}")
    print(f"Val images: {val_images}")
    print(f"Test images: {test_images}")
    
    # GPU optimized parameters for GTX 1650 (4GB VRAM)
    model_name = "yolov8n.pt"
    epochs = 50
    img_size = 640
    batch_size = 8 
    workers = 2      
    patience = 15
    
    params = {
        "model": model_name,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "patience": patience,
        "device": device,
        "workers": workers
    }

    print(f"Training Parameters: {params}")
    
    with mlflow.start_run(run_name=experiment_name) as run:
        mlflow.log_params(params)
        mlflow.log_param("train_images", train_images)
        mlflow.log_param("val_images", val_images)

        # Load model
        print("Loading model...")
        try:
            model = YOLO(model_name)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
        
        # Start training
        print("Starting training...")
        
        try:
            results = model.train(
                data=str(dataset_path),
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=patience,
                workers=workers,
                patience=patience,
                workers=workers,
                project=str(models_path),
                name=experiment_name,
                exist_ok=True,
                device=device,
                amp=False,
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                verbose=False,
                cos_lr=True,
                close_mosaic=10,
                save=True,
                save_period=10,
                val=True,
                plots=True,
                cache=False,
                optimizer='SGD', 
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time/60:.1f} minutes!")
            mlflow.log_metric("training_time_min", training_time/60)
            
            # Model weights are saved in: models/yolo/car_damage_detection_[timestamp]/
            model_weights_dir = models_path / experiment_name
            print(f"Model weights saved to: {model_weights_dir}")
            
            # Log Best Model Checkpoint
            best_model = model_weights_dir / "weights" / "best.pt"
            if best_model.exists():
                mlflow.log_artifact(str(best_model), artifact_path="model")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU Out of Memory! Try reducing batch size to 4.")
                return
            else:
                print(f"Training failed: {e}")
                return
        except Exception as e:
            print(f"Unexpected error during training: {e}")
            return
        
        # Process training artifacts after training completes
        process_training_artifacts(models_path, experiments_path, experiment_name, 
                                  dataset_config, train_images, val_images, test_images, 
                                  model_name, epochs, batch_size, device, training_time, model)

def process_training_artifacts(models_path, experiments_path, experiment_name,
                              dataset_config, train_images, val_images, test_images,
                              model_name, epochs, batch_size, device, training_time, model):
    """Move and organize training artifacts and log to MLflow"""
    print("Processing training artifacts...")
    
    try:
        # Create experiment directory
        experiment_dir = experiments_path / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Source directory where YOLO saved training artifacts
        training_artifacts_dir = models_path / experiment_name
        
        if training_artifacts_dir.exists():
            # Copy all training artifacts except weights
            for item in training_artifacts_dir.iterdir():
                if item.is_file():
                    if item.suffix in ['.png', '.jpg', '.txt', '.csv', '.json']:
                        shutil.copy2(item, experiment_dir)
                        mlflow.log_artifact(str(item), artifact_path="results")
                        print(f"Copied: {item.name}")
                elif item.is_dir() and item.name != 'weights':
                    # Copy entire directory (like 'labels', 'predictions')
                    dest_dir = experiment_dir / item.name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(item, dest_dir)
                    mlflow.log_artifacts(str(item), artifact_path=f"results/{item.name}")
                    print(f"Copied directory: {item.name}")
            
            print(f"Training artifacts saved to: {experiment_dir}")
        else:
            print("Training artifacts directory not found!")
            
    except Exception as e:
        print(f"Error moving training artifacts: {e}")
    
    # Validate the model and save metrics
    print("Validating model and saving metrics...")
    try:
        # Use valid model instance if passed or reload
        metrics = model.val()
        
        print(f"Final Validation Results:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("mAP50", metrics.box.map50)
        mlflow.log_metric("mAP50-95", metrics.box.map)
        mlflow.log_metric("precision", metrics.box.p)
        mlflow.log_metric("recall", metrics.box.r)

        # Save metrics file (Keeping original logic for local file)
        save_training_metrics(experiment_dir, experiment_name, training_time, 
                            model_name, 640, epochs, batch_size,
                            device, train_images, val_images,
                            test_images, dataset_config, metrics,
                            models_path)
            
    except Exception as e:
        print(f"Validation failed: {e}")

def save_training_metrics(experiment_dir, experiment_name, training_time, model_name, 
                         img_size, epochs, batch_size, device, train_images, val_images,
                         test_images, dataset_config, metrics, models_path):
    """Save comprehensive training metrics"""
    metrics_file = experiment_dir / 'training_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"=== CAR DAMAGE DETECTION TRAINING METRICS ===\n\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Training completed: {time.ctime()}\n")
        
        f.write(f"VALIDATION RESULTS:\n")
        f.write(f"mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.p:.4f}\n")
        f.write(f"Recall: {metrics.box.r:.4f}\n")
        
    print(f"Metrics saved to: {metrics_file}")

def check_environment():
    """Check if all required packages are installed"""
    try:
        import ultralytics
        import torch
        import yaml
        import mlflow
        print("All required packages are installed!")
        return True
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install required packages: pip install ultralytics torch pyyaml mlflow")
        return False

if __name__ == "__main__":
    if check_environment():
        main()
    else:
        sys.exit(1)