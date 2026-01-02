# visualize_training.py
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.utils.paths import ProjectPaths

def plot_training_results():
    """Plot training metrics and losses"""
    
    # Updated path to look in YOLO artifacts
    training_dir = ProjectPaths.YOLO_ARTIFACTS_DIR
    
    # Find the most recent experiment or specific one? 
    # Let's search for any folder starting with car_damage_detection
    experiments = [d for d in training_dir.iterdir() if d.is_dir() and "car_damage_detection" in d.name]
    
    if not experiments:
        print(f"No experiments found in {training_dir}")
        return

    # Pick the latest one
    latest_exp = sorted(experiments, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Analyzing latest experiment: {latest_exp.name}")
    
    results_file = latest_exp / "results.csv"
    
    if not results_file.exists():
        print(f"Results file not found at: {results_file}")
        print("Available training runs:")
        for item in experiments:
            print(f"  - {item.name}")
        return
    
    # Output dir
    eval_output_dir = ProjectPaths.YOLO_EVAL_DIR / latest_exp.name
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Print available columns for debugging
    print("Available columns in results.csv:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Car Damage Detection - Training Results', fontsize=16, fontweight='bold')
    
    # Plot losses
    loss_columns = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
    colors = ['blue', 'red', 'green']
    titles = ['Box Loss', 'Class Loss', 'DFL Loss']
    
    for i, col in enumerate(loss_columns):
        if col in df.columns:
            axes[0,i].plot(df[col], label=col, color=colors[i], linewidth=2)
            axes[0,i].set_title(titles[i])
            axes[0,i].set_xlabel('Epoch')
            axes[0,i].set_ylabel('Loss')
            axes[0,i].grid(True, alpha=0.3)
            axes[0,i].legend()
        else:
            axes[0,i].text(0.5, 0.5, f'{col} not found', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[0,i].transAxes)
            axes[0,i].set_title(titles[i])
    
    # Plot metrics - try different column naming patterns
    metric_patterns = [
        ['metrics/precision(B)', 'metrics/precision', 'precision(B)'],
        ['metrics/recall(B)', 'metrics/recall', 'recall(B)'], 
        ['metrics/mAP50(B)', 'metrics/mAP50', 'mAP50(B)']
    ]
    metric_titles = ['Precision', 'Recall', 'mAP50']
    metric_colors = ['purple', 'orange', 'brown']
    
    for i, patterns in enumerate(metric_patterns):
        found = False
        for pattern in patterns:
            if pattern in df.columns:
                axes[1,i].plot(df[pattern], label=pattern, color=metric_colors[i], linewidth=2)
                axes[1,i].set_title(metric_titles[i])
                axes[1,i].set_xlabel('Epoch')
                axes[1,i].set_ylabel('Score')
                axes[1,i].grid(True, alpha=0.3)
                axes[1,i].legend()
                found = True
                break
        
        if not found:
            axes[1,i].text(0.5, 0.5, 'Metric not found', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[1,i].transAxes)
            axes[1,i].set_title(metric_titles[i])
    
    plt.tight_layout()
    plt.tight_layout()
    output_plot = eval_output_dir / 'training_results.png'
    plt.savefig(str(output_plot), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {output_plot}")
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Loss summary
    print("\nLOSS PROGRESS:")
    loss_info = [
        ('Box Loss', 'train/box_loss'),
        ('Class Loss', 'train/cls_loss'), 
        ('DFL Loss', 'train/dfl_loss')
    ]
    
    for name, col in loss_info:
        if col in df.columns:
            initial = df[col].iloc[0] if len(df) > 0 else 'N/A'
            final = df[col].iloc[-1] if len(df) > 0 else 'N/A'
            improvement = initial - final if isinstance(initial, (int, float)) and isinstance(final, (int, float)) else 'N/A'
            print(f"  {name}: {initial:.3f} → {final:.3f} (Δ: {improvement:.3f})")
    
    # Metrics summary
    print("\nFINAL METRICS:")
    metric_info = [
        ('Precision', 'metrics/precision(B)', 'metrics/precision', 'precision(B)'),
        ('Recall', 'metrics/recall(B)', 'metrics/recall', 'recall(B)'),
        ('mAP50', 'metrics/mAP50(B)', 'metrics/mAP50', 'mAP50(B)'),
        ('mAP50-95', 'metrics/mAP50-95(B)', 'metrics/mAP50-95', 'mAP50-95(B)')
    ]
    
    for name, *patterns in metric_info:
        value = None
        for pattern in patterns:
            if pattern in df.columns and len(df) > 0:
                value = df[pattern].iloc[-1]
                break
        if value is not None:
            print(f"  {name}: {value:.3f}")
        else:
            print(f"  {name}: Not available")

def plot_training_artifacts():
    """Plot all available training artifacts"""
    training_dir = ProjectPaths.YOLO_ARTIFACTS_DIR
    
    experiments = [d for d in training_dir.iterdir() if d.is_dir() and "car_damage_detection" in d.name]
    if not experiments: return
    
    latest_exp = sorted(experiments, key=lambda x: x.stat().st_mtime)[-1]
    
    # We want to check artifacts INSIDE the experiment folder
    # Note: YOLO structure is: experiment_folder/{weights, plots, etc}
    # But in our train_yolo.py we move artifacts to: experiments/training_artifacts/yolo/{experiment_name}
    
    # So we look in latest_exp
    search_dir = latest_exp
    
    if not search_dir.exists():
        print(f"Training directory not found: {search_dir}")
        return
    
    # List available files
    print("Available training artifacts:")
    for file in training_dir.iterdir():
        if file.is_file():
            print(f"  - {file.name}")
    
    # Check for specific plot files
    plot_files = {
        'results.png': 'Training Results',
        'confusion_matrix.png': 'Confusion Matrix', 
        'confusion_matrix_normalized.png': 'Normalized Confusion Matrix',
        'labels.jpg': 'Dataset Labels Distribution',
        'val_batch0_pred.jpg': 'Validation Predictions'
    }
    
    print("\nKey Artifacts:")
    for file, description in plot_files.items():
        if (search_dir / file).exists():
            print(f"  {description}: {search_dir / file}")
        else:
            print(f"  {description}: {file} (not found)")

if __name__ == "__main__":
    print("Analyzing Training Results...")
    plot_training_results()
    print("\n" + "="*60)
    plot_training_artifacts()