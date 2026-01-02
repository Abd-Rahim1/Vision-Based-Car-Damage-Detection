# gpu_check.py
import torch
import subprocess
import sys

def check_gpu():
    print("=== GPU Diagnostic Check ===")
    
    # Check PyTorch CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA not available to PyTorch")
    
    # Check system CUDA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✅ nvidia-smi output:")
            print(result.stdout[:500])  # First 500 chars
        else:
            print("❌ nvidia-smi not found or failed")
    except FileNotFoundError:
        print("❌ nvidia-smi not installed")
    
    # Check if correct PyTorch version is installed
    print(f"\nPyTorch built with CUDA: {torch.cuda.is_available()}")

if __name__ == "__main__":
    check_gpu()