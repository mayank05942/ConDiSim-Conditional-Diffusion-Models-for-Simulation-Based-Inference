#!/usr/bin/env python
# Simple script to inspect Vilar dataset structure
import numpy as np
import os

# Path to dataset
dataset_path = '/cephyr/users/nautiyal/Alvis/diffusion/vilar/datasets/vilar_dataset_10000.npz'

# Load the data
print(f"Loading dataset: {dataset_path}")
data = np.load(dataset_path, allow_pickle=True)

# Print keys and shapes
print("\nDataset keys:")
for key in data.keys():
    print(f"- {key}: {data[key].shape}")



# Check specifically for scalers
print("\nChecking for scalers:")
scaler_keys = [key for key in data.keys() if 'scaler' in key.lower()]
if scaler_keys:
    print(f"Found scaler keys: {scaler_keys}")
    for key in scaler_keys:
        print(f"\n{key}:")
        scaler = data[key]
        if hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
            print(f"  Mean: {scaler.mean_}")
            print(f"  Scale: {scaler.scale_}")
        else:
            print(f"  Type: {type(scaler)}")
            print(f"  Attributes: {dir(scaler)}")
else:
    print("No explicit scaler keys found.")
    
    # Look for objects that might be scalers
    for key in data.keys():
        obj = data[key]
        if hasattr(obj, '__class__') and 'scaler' in str(obj.__class__).lower():
            print(f"\nPossible scaler found in {key}:")
            print(f"  Type: {type(obj)}")
            if hasattr(obj, 'scale_') and hasattr(obj, 'mean_'):
                print(f"  Mean: {obj.mean_}")
                print(f"  Scale: {obj.scale_}")

print("\nInspection complete.")
