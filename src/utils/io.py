import os
import json
import yaml
import cv2
import glob
from pathlib import Path

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def read_image(path):
    return cv2.imread(path)

def read_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def save_image(img, path):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def get_files(directory, extensions=[".jpg", ".jpeg", ".png"]):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))
    return files
