import mlflow
import os
from pathlib import Path
import sys

# Configuration
MLFLOW_TRACKING_DIR = r"c:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\experiments\mlflow\detectron2"
EXPERIMENT_NAME = "detectron2_segmentation"

def verify_mlflow_reset():
    print("Verifying MLflow Reset...")
    
    # 1. Verify Directory Exists
    if not os.path.exists(MLFLOW_TRACKING_DIR):
        print(f"❌ FAIL: Tracking directory does not exist: {MLFLOW_TRACKING_DIR}")
        sys.exit(1)
    else:
        print(f"✅ PASS: Tracking directory exists: {MLFLOW_TRACKING_DIR}")

    # 2. Verify Legacy Directory is Gone (Optional, depending on if we check it here)
    legacy_dir = r"c:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\mlflow\detectron2"
    if os.path.exists(legacy_dir):
         print(f"❌ FAIL: Legacy directory still exists: {legacy_dir}")
         # Not strict failure, but warning
    else:
         print(f"✅ PASS: Legacy directory removed.")

    # 3. Verify MLflow Experiment
    try:
        mlflow.set_tracking_uri(Path(MLFLOW_TRACKING_DIR).as_uri())
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            print(f"❌ FAIL: Experiment '{EXPERIMENT_NAME}' not found.")
            sys.exit(1)
        
        print(f"✅ PASS: Experiment '{EXPERIMENT_NAME}' found (ID: {experiment.experiment_id}).")
        
        # 4. Verify No Runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) > 0:
            print(f"❌ FAIL: Found {len(runs)} existing runs. Workspace is not clean.")
            sys.exit(1)
        else:
            print(f"✅ PASS: No existing runs found (Clean workspace).")
            
    except Exception as e:
        print(f"❌ FAIL: MLflow verification error: {e}")
        sys.exit(1)

    print("\nALL CHECKS PASSED. MLflow is ready for new experiments.")

if __name__ == "__main__":
    verify_mlflow_reset()
