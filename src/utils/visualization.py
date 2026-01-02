import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2
from .io import ensure_dir

def plot_histogram(data, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

def plot_bar(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

def plot_pie(data, labels, title, save_path):
    plt.figure(figsize=(8, 8))
    plt.pie(data, labels=labels, autopct='%1.1f%%')
    plt.title(title)
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

def plot_overlay(image, mask, title, save_path, alpha=0.5):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, cmap="jet", alpha=alpha)
    plt.title("Overlay")
    plt.axis("off")
    
    plt.suptitle(title)
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()
