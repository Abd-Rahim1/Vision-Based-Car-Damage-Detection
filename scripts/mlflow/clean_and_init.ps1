# PowerShell Script to Clean and Reinitialize MLflow Workspace
# Usage: ./scripts/mlflow/clean_and_init.ps1

$WorkspacePath = "$PSScriptRoot\..\..\experiments\mlflow\detectron2"
$ResolvedPath = Resolve-Path $WorkspacePath -ErrorAction SilentlyContinue

Write-Host "MLflow Workspace Cleaner" -ForegroundColor Cyan
Write-Host "Target: $ResolvedPath" -ForegroundColor Gray

if (-not $ResolvedPath) {
    Write-Error "Workspace path not found: $WorkspacePath"
    exit 1
}

# 1. Clean Workspace
Write-Host "`n[1/3] Cleaning workspace..." -ForegroundColor Yellow
$Items = Get-ChildItem -Path $ResolvedPath -Force

foreach ($Item in $Items) {
    # Preserve .git, .dvc, and files (only deleting directories generally, but removing specific trash)
    if ($Item.Name -eq ".git" -or $Item.Name -eq ".dvc") {
        Write-Host "  Skipping protected item: $($Item.Name)" -ForegroundColor DarkGray
        continue
    }

    # Delete .trash folder
    if ($Item.Name -eq ".trash") {
        Write-Host "  Removing .trash folder..." -ForegroundColor Red
        Remove-Item -Path $Item.FullName -Recurse -Force
        continue
    }

    # Delete experiment runs (directories composed of digits or IDs)
    # We'll assume any directory that ISN'T metadata is a run. 
    # Usually MLflow file store has meta.yaml and run directories.
    if ($Item.PSIsContainer) {
        Write-Host "  Removing run directory: $($Item.Name)" -ForegroundColor Red
        Remove-Item -Path $Item.FullName -Recurse -Force
    } elseif ($Item.Name -ne "meta.yaml") {
         # Identify if there are loose files that shouldn't be there
         # We usually keep meta.yaml if we want to keep the experiment ID, but the task says "Reinitialize".
         # However, removing meta.yaml deletes the experiment definition itself.
         # User asked to "Reinitialize MLflow with a fresh experiment".
         # So we can delete meta.yaml too if we plan to recreate it.
         Write-Host "  Removing file: $($Item.Name)" -ForegroundColor Red
         Remove-Item -Path $Item.FullName -Force
    } else {
        # It's meta.yaml
        Write-Host "  Removing old meta.yaml (Full Reset)..." -ForegroundColor Red
        Remove-Item -Path $Item.FullName -Force
    }
}

Write-Host "Cleanup complete." -ForegroundColor Green

# 2. Reinitialize Experiment
Write-Host "`n[2/3] Reinitializing experiment 'detectron2_segmentation'..." -ForegroundColor Yellow

# Check for Python and MLflow
try {
    $mlflowVersion = python -c "import mlflow; print(mlflow.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  MLflow version: $mlflowVersion" -ForegroundColor Gray
        
        # We need to set the tracking URI to the parent directory of 'detectron2' 
        # Actually, MLflow file URI usually points to the root `mlflow` folder, 
        # and experiment names create subfolders?
        # Or is `experiments/mlflow/detectron2` the location for THIS experiment?
        # If the user wants the workspace at `experiments/mlflow/detectron2`, 
        # that implies `experiments/mlflow` might be the tracking URI base, 
        # and `detectron2` is the experiment?
        # Or `detectron2` IS the tracking URI location?
        
        # User said: "clean the MLflow workspace located at experiments/mlflow/detectron2/"
        # And "Reinitialize MLflow with a fresh experiment named: detectron2_segmentation"
        
        # Let's try to set tracking URI to `experiments/mlflow` and create experiment `detectron2_segmentation`.
        # IF `detectron2` was previously a store, maybe they want it to be the store root?
        # Given the previous existing folder structure `experiments/mlflow/detectron2/285...` (which looks like an experiment ID inside it),
        # it seems `experiments/mlflow/detectron2` WAS the tracking URI (File Store).
        
        $TrackingURI = "file:///$($ResolvedPath.Path.Replace('\', '/'))"
        Write-Host "  Setting Tracking URI: $TrackingURI"
        
        $Script = @"
import mlflow
import os

try:
    mlflow.set_tracking_uri('$TrackingURI')
    exp_name = 'detectron2_segmentation'
    
    # Check if exists (shouldn't after clean, unless we kept it)
    existing = mlflow.get_experiment_by_name(exp_name)
    if existing:
        print(f'Experiment {exp_name} already exists (ID: {existing.experiment_id}).')
    else:
        exp_id = mlflow.create_experiment(exp_name)
        print(f'Created new experiment {exp_name} with ID: {exp_id}')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"@
        python -c $Script
    } else {
        Write-Warning "Python or MLflow not found/functioning. Cleanup performed, but reinitialization skipped."
        Write-Warning "Please ensure 'mlflow' is installed and run: mlflow experiments create -n detectron2_segmentation"
    }
} catch {
    Write-Warning "Failed to execute Python command."
}

# 3. Verification
Write-Host "`n[3/3] Verification..." -ForegroundColor Yellow
if ((Test-Path "$ResolvedPath\meta.yaml") -or ((Get-ChildItem "$ResolvedPath" | Where-Object {$_.PSIsContainer}).Count -gt 0)) {
     # If we set tracking URI to `.../detectron2`, MLflow creates `0` (default) and new exp ID folders inside `detectron2`.
     # OR if `detectron2` IS the experiment name, we should have set tracking URI to `.../experiments/mlflow`.
     
     # Let's assume the user wants `detectron2` directory to BE the file store.
     Write-Host "  Workspace initialized." -ForegroundColor Green
     Get-ChildItem $ResolvedPath
} else {
     Write-Host "  Workspace is empty (Environment might need manual init)." -ForegroundColor Gray
}

Write-Host "`nDone." -ForegroundColor Cyan
