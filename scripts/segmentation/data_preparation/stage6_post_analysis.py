import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.io import load_params, save_json, get_files, ensure_dir, read_mask
from src.utils.visualization import plot_bar, plot_histogram

def analyze_processed_data():
    params = load_params()
    processed_root = params['data']['processed_path']
    vis_path = os.path.join(params['visualization']['output_dir'], 'stage6_post_analysis')
    ensure_dir(vis_path)

    stats = {
        'split_counts': {},
        'damage_ratios': {'train': [], 'val': [], 'test': []},
        'resolution_check': {'passed': 0, 'failed': 0},
        'mask_integrity': {'empty': 0, 'populated': 0}
    }

    target_size = tuple(params['preprocessing']['target_size'])
    splits = ['train', 'val', 'test']

    print("Running Post-Split Analysis...")
    
    for split in splits:
        split_img_dir = os.path.join(processed_root, split, 'images')
        split_mask_dir = os.path.join(processed_root, split, 'masks')
        
        if not os.path.exists(split_img_dir):
            print(f"Warning: Split {split} not found at {split_img_dir}")
            continue

        images = get_files(split_img_dir)
        stats['split_counts'][split] = len(images)
        
        print(f"Analyzing {split} split...")
        for img_path in tqdm(images):
            filename = os.path.basename(img_path)
            # Find corresponding mask
            mask_filename = filename.replace(os.path.splitext(filename)[1], ".png")
            mask_path = os.path.join(split_mask_dir, mask_filename)
            
            # 1. Resolution Check
            # Read image only to check metadata if possible, but cv2 needs read
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                if (w, h) == target_size:
                    stats['resolution_check']['passed'] += 1
                else:
                    stats['resolution_check']['failed'] += 1
            
            # 2. Mask Stats
            mask = read_mask(mask_path)
            if mask is not None:
                damage_pixels = np.count_nonzero(mask)
                total_pixels = mask.shape[0] * mask.shape[1]
                ratio = damage_pixels / total_pixels
                
                stats['damage_ratios'][split].append(float(ratio))
                
                if damage_pixels == 0:
                    stats['mask_integrity']['empty'] += 1
                else:
                    stats['mask_integrity']['populated'] += 1

    save_json(stats, os.path.join(vis_path, 'post_analysis_stats.json'))

    # Visualizations
    # 1. Split Distribution
    plot_bar(
        list(stats['split_counts'].keys()), 
        list(stats['split_counts'].values()), 
        "Image Count per Split", "Split", "Count", 
        os.path.join(vis_path, 'final_split_counts.png')
    )
    
    # 2. Damage Ratio Distribution per Split
    for split in splits:
        if stats['damage_ratios'][split]:
            plot_histogram(
                stats['damage_ratios'][split], 
                f"Damage Pixel Ratio - {split}", "Ratio", "Frequency", 
                os.path.join(vis_path, f'damage_ratio_{split}.png')
            )

    print("Stage 6 Post-Analysis Complete.")

if __name__ == "__main__":
    analyze_processed_data()
