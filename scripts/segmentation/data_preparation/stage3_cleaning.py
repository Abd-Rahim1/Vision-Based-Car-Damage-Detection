import os
import sys
import numpy as np
import shutil
import cv2
import imagehash
from PIL import Image
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.utils.io import load_params, save_json, read_image, read_mask, get_files, ensure_dir
from src.utils.visualization import plot_bar
from src.utils.paths import ProjectPaths

def clean_data():
    params = load_params()
    filtered_path = ProjectPaths.INTERMEDIATE_DATA_DIR / 'filtered' / 'segmentation'
    cleaned_path = ProjectPaths.INTERMEDIATE_DATA_DIR / 'cleaned' / 'segmentation'
    vis_path = ProjectPaths.SEGMENTATION_VIS_DIR / 'stage3_cleaning'
    ensure_dir(cleaned_path)
    ensure_dir(vis_path)

    log = {
        'total_input': 0,
        'removed_duplicates': 0,
        'removed_non_binary': 0,
        'final_count': 0
    }

    img_dir = os.path.join(filtered_path, 'images')
    mask_dir = os.path.join(filtered_path, 'masks')
    
    out_img_dir = os.path.join(cleaned_path, 'images')
    out_mask_dir = os.path.join(cleaned_path, 'masks')
    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)

    images = get_files(img_dir)
    log['total_input'] = len(images)
    
    hashes = {}
    duplicates = []
    
    print("Cleaning data...")
    for img_path in tqdm(images):
        filename = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(filename)[0]
        # Potential mask path (assuming png from Stage 2)
        mask_path = os.path.join(mask_dir, img_name_no_ext + ".png")
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {filename} during Stage 3. Skipping.")
            continue
            
        # 1. Deduplication using Perceptual Hash
        # Use PIL for imagehash
        try:
            pil_img = Image.open(img_path)
            h = str(imagehash.phash(pil_img))
        except Exception as e:
            print(f"Error hashing {filename}: {e}")
            continue
            
        if h in hashes:
            duplicates.append(filename)
            log['removed_duplicates'] += 1
            continue
        else:
            hashes[h] = filename
            
        # 2. Binary Mask Validation
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unique_values = np.unique(mask)
        
        # We expect 0 and 255 (or 0 and 1). If we find others, we threshold.
        # But if it's too noisy, maybe we discard? 
        # For professional pipeline, thresholding is usually safe for binary masks unless values are semantic classes.
        # Here we enforce 0 and 1 for Detectron2? Or 0 and 255?
        # Detectron2 standard dataset dicts mask can be RLE or polygon. 
        # If we save as file, 0 and 255 is standard for visualization/readability.
        
        if len(unique_values) > 2:
            # Check if values are close to 0/255 -> JPEG artifacts?
            # Or if it's just resize interpolation artifacts from raw data?
            # Enforce binary by thresholding at 127
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
        # 3. Save
        shutil.copy(img_path, os.path.join(out_img_dir, filename))
        cv2.imwrite(os.path.join(out_mask_dir, filename.replace(os.path.splitext(filename)[1], ".png")), mask)
        log['final_count'] += 1

    save_json(log, os.path.join(vis_path, 'cleaning_log.json'))
    
    # Visualization
    plot_bar(
        ['Input', 'Duplicates Removed', 'Final'], 
        [log['total_input'], log['removed_duplicates'], log['final_count']], 
        "Cleaning Stats", "Metric", "Count", 
        os.path.join(vis_path, 'cleaning_stats.png')
    )
    
    print("Stage 3 Cleaning Complete.")

if __name__ == "__main__":
    clean_data()
