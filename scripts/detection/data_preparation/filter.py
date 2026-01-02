import os
import json
import shutil
from tqdm import tqdm
from typing import Dict
from PIL import Image
from utils import load_config, create_directory, get_image_dimensions, load_coco_annotations, save_coco_annotations

class DataFilter:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_config(config_path)
        self.filter_config = self.config['data_preparation']['filtering']
        
    def filter_dataset(self, input_path: str, output_path: str) -> Dict:
        """Filter dataset and copy valid data to output directory"""
        print(" Filtering dataset...")

        create_directory(output_path)
        stats = {
            'removed_images': 0,
            'removed_annotations': 0,
            'splits': {}
        }

        # COCO dataset
        coco_input_path = os.path.join(input_path, "CarDD_COCO")
        coco_output_path = os.path.join(output_path, "CarDD_COCO")
        if os.path.exists(coco_input_path):
            coco_stats = self._filter_coco_dataset(coco_input_path, coco_output_path)
            stats['removed_images'] += coco_stats['removed_images']
            stats['removed_annotations'] += coco_stats['removed_annotations']
            stats['splits']['coco'] = coco_stats['splits']

        # SOD dataset
        sod_input_path = os.path.join(input_path, "CarDD_SOD")
        sod_output_path = os.path.join(output_path, "CarDD_SOD")
        if os.path.exists(sod_input_path):
            sod_stats = self._filter_sod_dataset(sod_input_path, sod_output_path)
            stats['removed_images'] += sod_stats['removed_images']
            stats['splits']['sod'] = sod_stats['splits']

        return stats
    
    def _filter_coco_dataset(self, input_path: str, output_path: str) -> Dict:
        stats = {'removed_images': 0, 'removed_annotations': 0, 'splits': {}}
        splits = ['train2017', 'val2017', 'test2017']

        for split in splits:
            try:
                coco_data = load_coco_annotations(input_path, split)
                input_images_dir = os.path.join(input_path, split)
                output_images_dir = os.path.join(output_path, split)
                create_directory(output_images_dir)
                create_directory(os.path.join(output_path, "annotations"))

                valid_images = []
                removed_images = []

                for image_info in tqdm(coco_data['images'], desc=f"Filtering {split}"):
                    input_image_path = os.path.join(input_images_dir, image_info['file_name'])
                    if self._is_valid_image(input_image_path):
                        output_image_path = os.path.join(output_images_dir, image_info['file_name'])
                        shutil.copy2(input_image_path, output_image_path)
                        valid_images.append(image_info)
                    else:
                        removed_images.append(image_info)

                valid_image_ids = {img['id'] for img in valid_images}
                valid_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in valid_image_ids]
                removed_annotations = len(coco_data['annotations']) - len(valid_annotations)

                coco_data['images'] = valid_images
                coco_data['annotations'] = valid_annotations
                save_coco_annotations(coco_data, output_path, split)

                stats['splits'][split] = {
                    'removed_images': len(removed_images),
                    'removed_annotations': removed_annotations,
                    'remaining_images': len(valid_images),
                    'remaining_annotations': len(valid_annotations)
                }

                stats['removed_images'] += len(removed_images)
                stats['removed_annotations'] += removed_annotations
                print(f" {split}: Removed {len(removed_images)} images, {removed_annotations} annotations")

            except FileNotFoundError:
                print(f" Skipping {split} - annotations not found")
                continue

        return stats
    
    def _filter_sod_dataset(self, input_path: str, output_path: str) -> Dict:
        stats = {'removed_images': 0, 'splits': {}}
        splits = ['CarDD-TR', 'CarDD-VAL', 'CarDD-TE']

        for split in splits:
            input_image_dir = os.path.join(input_path, split, f"{split}-Image")
            input_mask_dir = os.path.join(input_path, split, f"{split}-Mask")
            if not os.path.exists(input_image_dir):
                continue

            output_image_dir = os.path.join(output_path, split, f"{split}-Image")
            output_mask_dir = os.path.join(output_path, split, f"{split}-Mask")
            create_directory(output_image_dir)
            create_directory(output_mask_dir)

            images = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            removed_count = 0

            for image_name in tqdm(images, desc=f"Filtering {split}"):
                input_image_path = os.path.join(input_image_dir, image_name)
                input_mask_path = os.path.join(input_mask_dir, image_name)
                if self._is_valid_image(input_image_path):
                    output_image_path = os.path.join(output_image_dir, image_name)
                    shutil.copy2(input_image_path, output_image_path)
                    if os.path.exists(input_mask_path):
                        output_mask_path = os.path.join(output_mask_dir, image_name)
                        shutil.copy2(input_mask_path, output_mask_path)
                else:
                    removed_count += 1

            stats['splits'][split] = {'removed_images': removed_count, 'remaining_images': len(images) - removed_count}
            stats['removed_images'] += removed_count
            print(f" {split}: Removed {removed_count} images")

        return stats
    
    def _is_valid_image(self, image_path: str) -> bool:
        try:
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.filter_config['allowed_extensions']:
                return False

            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > self.filter_config['max_file_size_mb']:
                return False

            width, height = get_image_dimensions(image_path)
            min_w, min_h = self.filter_config['min_image_size']
            if width < min_w or height < min_h:
                return False

            # Open image safely
            with Image.open(image_path) as img:
                img.load()

            return True
        except Exception as e:
            print(f" Skipping invalid image: {image_path} ({e})")
            return False


def main():
    # Set paths explicitly
    input_path = r"C:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\dataset\CarDD_release"
    output_path = r"C:\Users\Document\OneDrive\Desktop\Car_Damage_Detection\prepared_datasets\detection"

    data_filter = DataFilter()
    stats = data_filter.filter_dataset(input_path, output_path)

    print(f"\nFiltering completed!")
    print(f"Removed images: {stats['removed_images']}")
    print(f"Removed annotations: {stats['removed_annotations']}")
    print(f"Filtered dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
