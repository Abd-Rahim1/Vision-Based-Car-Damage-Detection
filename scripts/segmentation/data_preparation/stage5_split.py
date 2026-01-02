import os
import sys
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.io import load_params, get_files, ensure_dir, read_mask, save_json
from src.utils.visualization import plot_bar
from src.utils.paths import ProjectPaths

def process_splits():
    params = load_params()
    pool_path = ProjectPaths.PROCESSED_DATA_DIR / 'segmentation' / 'pool'
    processed_root = ProjectPaths.PROCESSED_DATA_DIR / 'segmentation'
    vis_path = ProjectPaths.SEGMENTATION_VIS_DIR / 'stage5_split'
    ensure_dir(vis_path)

    img_pool = os.path.join(pool_path, 'images')
    mask_pool = os.path.join(pool_path, 'masks')
    
    images = get_files(img_pool)
    
    # Track stats
    split_stats = {
        'train': {'count': 0, 'damage_pixels': 0},
        'val': {'count': 0, 'damage_pixels': 0},
        'test': {'count': 0, 'damage_pixels': 0}
    }
    
    print("Organizing splits...")
    for img_path in tqdm(images):
        filename = os.path.basename(img_path)
        # Identify split from prefix (added in Stage 2)
        if filename.startswith('CarDD-TR'):
            split = 'train'
        elif filename.startswith('CarDD-VAL'):
            split = 'val'
        elif filename.startswith('CarDD-TE'):
            split = 'test'
        else:
            print(f"Warning: Unknown split for {filename}, defaulting to train")
            split = 'train'
            
        target_img_dir = os.path.join(processed_root, split, 'images')
        target_mask_dir = os.path.join(processed_root, split, 'masks')
        ensure_dir(target_img_dir)
        ensure_dir(target_mask_dir)
        
        # Move Image
        shutil.copy(img_path, os.path.join(target_img_dir, filename))
        
        # Move Mask
        mask_filename = filename.replace(os.path.splitext(filename)[1], ".png")
        src_mask_path = os.path.join(mask_pool, mask_filename)
        dst_mask_path = os.path.join(target_mask_dir, mask_filename)
        shutil.copy(src_mask_path, dst_mask_path)
        
        # Calc Stats
        mask = read_mask(dst_mask_path)
        if mask is not None:
            split_stats[split]['count'] += 1
            split_stats[split]['damage_pixels'] += np.count_nonzero(mask)

    save_json(split_stats, os.path.join(vis_path, 'split_stats.json'))
    
    # Visualization: Split Distribution
    splits = ['train', 'val', 'test']
    counts = [split_stats[s]['count'] for s in splits]
    
    plot_bar(splits, counts, "Data Split Distribution", "Split", "Image Count", 
             os.path.join(vis_path, 'split_distribution.png'))
             
    # Clean up pool?
    # shutil.rmtree(pool_path) # Optional, strictly cleaner to remove
    
    print("Stage 5 Split Validation Complete.")

if __name__ == "__main__":
    process_splits()
