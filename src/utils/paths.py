import os
from pathlib import Path

class ProjectPaths:
    # Root directory (assuming this script is in src/utils/paths.py)
    # ROOT/src/utils/paths.py -> parents[2] = ROOT
    ROOT_DIR = Path(__file__).resolve().parents[2]

    # Data Directories
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    LEGACY_ARCHIVE_DIR = DATA_DIR / "legacy_archive"

    # Experiments & Outputs
    EXPERIMENTS_DIR = ROOT_DIR / "experiments"
    
    # MLflow
    # Using sqlite database for tracking metadata
    MLFLOW_DATABASE_PATH = ROOT_DIR / "mlflow.db"
    MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DATABASE_PATH.as_posix()}" # SQLite URI

    # Model Outputs & Artifacts
    TRAINING_ARTIFACTS_DIR = EXPERIMENTS_DIR / "training_artifacts"
    DETECTRON2_ARTIFACTS_DIR = TRAINING_ARTIFACTS_DIR / "detectron2"
    YOLO_ARTIFACTS_DIR = TRAINING_ARTIFACTS_DIR / "yolo"
    
    # Evaluation Outputs
    EVALUATION_DIR = EXPERIMENTS_DIR / "evaluation"
    DETECTRON2_EVAL_DIR = EVALUATION_DIR / "detectron2"
    YOLO_EVAL_DIR = EVALUATION_DIR / "yolo"
    
    # Visualization outputs
    VISUALIZATION_DIR = EXPERIMENTS_DIR / "visualization"
    SEGMENTATION_VIS_DIR = VISUALIZATION_DIR / "segmentation"

    # Configs
    CONFIGS_DIR = ROOT_DIR / "configs"
    PARAMS_PATH = CONFIGS_DIR / "params.yaml"

    @staticmethod
    def ensure_directories():
        """Creates basic directory structure if it doesn't exist."""
        dirs = [
            ProjectPaths.RAW_DATA_DIR,
            ProjectPaths.INTERMEDIATE_DATA_DIR,
            ProjectPaths.PROCESSED_DATA_DIR,
            ProjectPaths.LEGACY_ARCHIVE_DIR,
            ProjectPaths.MLFLOW_TRACKING_PATH,
            ProjectPaths.DETECTRON2_ARTIFACTS_DIR,
            ProjectPaths.YOLO_ARTIFACTS_DIR,
            ProjectPaths.SEGMENTATION_VIS_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project path root: {ProjectPaths.ROOT_DIR}")
    print(f"MLflow URI: {ProjectPaths.MLFLOW_TRACKING_URI}")
    ProjectPaths.ensure_directories()
