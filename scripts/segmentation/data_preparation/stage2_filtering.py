import cv2
import os
import sys
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.io import load_params, save_json, read_image, read_mask, get_files, ensure_dir
from src.utils.visualization import plot_bar
from src.utils.paths import ProjectPaths

def filter_data():
    params = load_params()
    raw_path = ProjectPaths.RAW_DATA_DIR
    # Use segmentation subfolder
    filtered_path = ProjectPaths.INTERMEDIATE_DATA_DIR / 'filtered' / 'segmentation'
    vis_path = ProjectPaths.SEGMENTATION_VIS_DIR / 'stage2_filtering'
    ensure_dir(filtered_path)
    ensure_dir(vis_path)

    log = {
        'processed': 0,
        'kept': 0,
        'removed_corrupted': 0,
        'removed_missing_mask': 0,
        'removed_empty_mask': 0,
        'removed_files': []
    }

    splits = ['CarDD-TR', 'CarDD-VAL', 'CarDD-TE']

    for split in splits:
        split_path = os.path.join(raw_path, split)
        image_dir = os.path.join(split_path, f"{split}-Image")
        mask_dir = os.path.join(split_path, f"{split}-Mask")

        if not os.path.exists(image_dir):
            continue

        images = get_files(image_dir)
        
        print(f"Filtering {split}...")
        for img_path in tqdm(images):
            log['processed'] += 1
            filename = os.path.basename(img_path)
            # Assuming mask has same filename but png extension? Or same name?
            # User sample: "CarDD-TR/CarDD-TR-Image/000001.jpg" -> "CarDD-TR/CarDD-TR-Mask/000001.png" usually
            # Or extension might be .png in mask.
            # Strategy: look for mask with same basename, maybe different extension.
            
            img_name_no_ext = os.path.splitext(filename)[0]
            
            # Check for mask file in mask_dir with common extensions
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(mask_dir, img_name_no_ext + ext)
                if os.path.exists(potential_path):
                    mask_path = potential_path
                    break
            
            if mask_path is None:
                log['removed_missing_mask'] += 1
                log['removed_files'].append({'file': filename, 'reason': 'missing_mask'})
                continue

            # Check for corruption
            img = read_image(img_path)
            mask = read_mask(mask_path)

            if img is None:
                log['removed_corrupted'] += 1
                log['removed_files'].append({'file': filename, 'reason': 'corrupted_image'})
                continue
            if mask is None:
                log['removed_corrupted'] += 1
                log['removed_files'].append({'file': filename, 'reason': 'corrupted_mask'})
                continue

            # Check for empty mask
            if np.count_nonzero(mask) == 0:
                log['removed_empty_mask'] += 1
                log['removed_files'].append({'file': filename, 'reason': 'empty_mask'})
                continue

            # Save to intermediate (copy) - Flatten structure or keep splits?
            # Keeping splits is safer for future steps referencing original logic, 
            # BUT usually intermediate joins them to do global cleaning/splitting later.
            # Implementation Plan says Stage 5 does split. So here we might want to FLATTEN 
            # or Keep structure if Stage 5 logic depends on original splits.
            # User req: "Save valid data to data/intermediate/filtered/segmentation"
            # Let's simple flatten into images/ and masks/ subfolders to make Cleaning (Dedup) easier.
            
            out_img_dir = os.path.join(filtered_path, 'images')
            out_mask_dir = os.path.join(filtered_path, 'masks')
            ensure_dir(out_img_dir)
            ensure_dir(out_mask_dir)
            
            # We need to ensure unique filenames if flattening.
            # Prefix with split name to avoid collisions.
            new_filename = f"{split}_{filename}"
            # Ensure mask uses .png for consistency if we want
            new_mask_filename = f"{split}_{img_name_no_ext}.png"

            shutil.copy(img_path, os.path.join(out_img_dir, new_filename))
            cv2.imwrite(os.path.join(out_mask_dir, new_mask_filename), mask) # Resave mask to ensure format
            
            log['kept'] += 1

    save_json(log, os.path.join(vis_path, 'filtering_log.json'))
    
    # Visualize Reasons
    reasons = ['Corrupted', 'Missing Mask', 'Empty Mask', 'Kept']
    counts = [log['removed_corrupted'], log['removed_missing_mask'], log['removed_empty_mask'], log['kept']]
    
    plot_bar(reasons, counts, "Filtering Results", "Category", "Count", os.path.join(vis_path, 'filtering_stats.png'))
    
    print("Stage 2 Filtering Complete.")

if __name__ == "__main__":
    filter_data()
