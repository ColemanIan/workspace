import torch

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

def dataset_info(folder_path):
    exr_files = [f for f in os.listdir(folder_path) if f.endswith('.exr')]
    print(f"Number of EXR files: {len(exr_files)}")
    
    total_size_bytes = sum(os.path.getsize(os.path.join(folder_path, f)) for f in exr_files)
    total_size_gb = total_size_bytes / (1024 ** 3)
    print(f"Approximate size of the dataset: {total_size_bytes} bytes ({total_size_gb:.2f} GB)")
    
    print("Info for first 5 files:")
    for f in exr_files[:5]:
        file_path = os.path.join(folder_path, f)
        exr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if exr is not None:
            print(f"  File: {f}, shape: {exr.shape}, dtype: {exr.dtype}")
        else:
            print(f"  File: {f}, could not be read.")

# Check if CUDA (NVIDIA's GPU driver) is available
gpu_available = torch.cuda.is_available()
print(f"Is GPU available? {gpu_available}")

if gpu_available:
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Run a quick math operation on the GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"Result of GPU calculation: {x * 2}")
else:
    print("GPU NOT FOUND. Check your NVIDIA Container Toolkit installation.")


dataset_info("./dataset_full/IndoorHDRDataset2018")