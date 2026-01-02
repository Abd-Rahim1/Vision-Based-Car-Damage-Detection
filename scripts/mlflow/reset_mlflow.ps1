$ErrorActionPreference = "Stop"

# Define paths
$LegacyMlflowDir = "c:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\mlflow\detectron2"
$NewMlflowDir = "c:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\experiments\mlflow\detectron2"
$ExperimentName = "detectron2_segmentation"

Write-Host "Starting MLflow Workspace Cleanup..." -ForegroundColor Cyan

# 1. Clean Legacy Directory
if (Test-Path $LegacyMlflowDir) {
    Write-Host "Removing legacy MLflow directory: $LegacyMlflowDir" -ForegroundColor Yellow
    Remove-Item -Path $LegacyMlflowDir -Recurse -Force
} else {
    Write-Host "Legacy MLflow directory not found (Clean)." -ForegroundColor Green
}

# 2. Clean New Experiment Directory (for fresh start)
if (Test-Path $NewMlflowDir) {
    Write-Host "Cleaning new MLflow experiment directory: $NewMlflowDir" -ForegroundColor Yellow
    Remove-Item -Path $NewMlflowDir -Recurse -Force
}
# Re-create the directory
New-Item -ItemType Directory -Force -Path $NewMlflowDir | Out-Null
Write-Host "Re-created empty directory: $NewMlflowDir" -ForegroundColor Green

# 3. Re-initialize MLflow Experiment
Write-Host "Re-initializing MLflow experiment..." -ForegroundColor Cyan

# Run the python initialization
Write-Host "Running initialization script..." -ForegroundColor Cyan
python "c:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\scripts\mlflow\init_mlflow.py"

Write-Host "MLflow Workspace Reset Complete." -ForegroundColor Green
