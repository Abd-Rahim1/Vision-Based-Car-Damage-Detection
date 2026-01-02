import mlflow
import os
from pathlib import Path

# Paths
new_mlflow_dir = r"c:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\experiments\mlflow\detectron2"
experiment_name = "detectron2_segmentation"

print(f"Initializing MLflow Experiment: {experiment_name}")
print(f"Tracking URI: {new_mlflow_dir}")

# Ensure directory exists
os.makedirs(new_mlflow_dir, exist_ok=True)

# Set Tracking URI
tracking_uri = Path(new_mlflow_dir).as_uri()
mlflow.set_tracking_uri(tracking_uri)
print(f"Tracking URI set to: {tracking_uri}")

# Create Experiment
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=tracking_uri
    )
    print(f"Successfully created experiment: {experiment_name} with ID: {experiment_id}")
except Exception as e:
    # Check if experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        print(f"Experiment {experiment_name} already exists with ID: {experiment.experiment_id}")
    else:
        print(f"Error creating experiment: {e}")
