import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add src to python path to allow importing utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.io import load_params, save_json, read_mask, get_files, ensure_dir
from src.utils.visualization import plot_histogram, plot_bar, plot_pie
from src.utils.paths import ProjectPaths

def analyze_raw_data():
    params = load_params()
    raw_path = ProjectPaths.RAW_DATA_DIR
    vis_path = ProjectPaths.SEGMENTATION_VIS_DIR / 'stage1_analysis'
    ensure_dir(vis_path)

    stats = {
        'total_images': 0,
        'split_counts': {},
        'resolutions': [],
        'mask_stats': {
            'empty': 0,
            'populated': 0,
            'damage_ratios': []
        }
    }

    splits = ['CarDD-TR', 'CarDD-VAL', 'CarDD-TE']
    
    # Check if raw path exists
    if not os.path.exists(raw_path):
        print(f"Error: Raw path {raw_path} does not exist.")
        return

    for split in splits:
        split_path = os.path.join(raw_path, split)
        image_dir = os.path.join(split_path, f"{split}-Image")
        mask_dir = os.path.join(split_path, f"{split}-Mask") # Assuming mask folder name follows this pattern
        
        # Adjust if mask dir is named differently (e.g. just 'Mask' or 'CarDD-TR-Mask')
        # Based on user description: "Each contains: *-Image (RGB images), *-Mask (binary segmentation masks)"
        # So "CarDD-TR/CarDD-TR-Image" and "CarDD-TR/CarDD-TR-Mask" seems likely.
        
        if not os.path.exists(image_dir):
            print(f"Warning: {image_dir} not found. Trying alternative...")
            # Fallback logic could go here if needed
            continue

        images = get_files(image_dir)
        masks = get_files(mask_dir)
        
        stats['split_counts'][split] = len(images)
        stats['total_images'] += len(images)
        
        print(f"Analyzing {split}...")
        for mask_path in tqdm(masks):
            mask = read_mask(mask_path)
            if mask is None:
                continue
            
            h, w = mask.shape
            stats['resolutions'].append((w, h))
            
            # Pixels > 0 are damage
            damage_pixels = np.count_nonzero(mask)
            total_pixels = h * w
            ratio = damage_pixels / total_pixels
            
            stats['mask_stats']['damage_ratios'].append(float(ratio))
            
            if damage_pixels == 0:
                stats['mask_stats']['empty'] += 1
            else:
                stats['mask_stats']['populated'] += 1

    # Save Stats
    save_json(stats, os.path.join(vis_path, 'stats.json'))
    
    # Visualization
    # 1. Image Counts per Split
    plot_bar(
        list(stats['split_counts'].keys()), 
        list(stats['split_counts'].values()), 
        "Image Count per Split", "Split", "Count", 
        os.path.join(vis_path, 'split_counts.png')
    )
    
    # 2. Resolution Distribution (Widths and Heights)
    widths = [r[0] for r in stats['resolutions']]
    heights = [r[1] for r in stats['resolutions']]
    plot_histogram(widths, "Image Width Distribution", "Width", "Frequency", os.path.join(vis_path, 'width_dist.png'))
    plot_histogram(heights, "Image Height Distribution", "Height", "Frequency", os.path.join(vis_path, 'height_dist.png'))
    
    # 3. Damage Ratio Distribution
    plot_histogram(
        stats['mask_stats']['damage_ratios'], 
        "Damage Pixel Ratio per Image", "Ratio", "Frequency", 
        os.path.join(vis_path, 'damage_ratio_dist.png')
    )
    
    # 4. Empty vs Populated Masks
    plot_pie(
        [stats['mask_stats']['empty'], stats['mask_stats']['populated']], 
        ['Empty', 'Populated'], 
        "Mask Content Distribution", 
        os.path.join(vis_path, 'mask_content_pie.png')
    )

    print("Stage 1 Analysis Complete.")

if __name__ == "__main__":
    analyze_raw_data()
