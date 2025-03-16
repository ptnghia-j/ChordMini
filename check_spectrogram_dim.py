#!/usr/bin/env python
# Simple script to check the dimensions of generated spectrograms

import os
import numpy as np
import glob
from pathlib import Path
import argparse

def check_spectrogram_dimensions():
    print("Checking spectrogram dimensions...")
    spec_dir = "./data/synth/spectrograms"
    
    # Find all spectrogram files
    spec_files = []
    for root, dirs, files in os.walk(spec_dir):
        for file in files:
            if file.endswith("_spec.npy"):
                spec_files.append(os.path.join(root, file))
    
    if not spec_files:
        print(f"No spectrogram files found in {spec_dir}")
        return
        
    print(f"Found {len(spec_files)} spectrogram files")
    
    # Check dimensions of first few files
    dims = {}
    nan_files = []
    extreme_files = []
    
    for i, spec_file in enumerate(spec_files[:10]):
        try:
            spec = np.load(spec_file)
            shape_str = "x".join(str(dim) for dim in spec.shape)
            
            if shape_str not in dims:
                dims[shape_str] = 0
            dims[shape_str] += 1
            
            # Determine if CQT or STFT based on frequency dimension
            freq_dim = spec.shape[1] if len(spec.shape) > 1 else 0
            spec_type = "CQT (Constant-Q Transform)" if freq_dim <= 256 else "STFT"
            
            print(f"File {i+1}: {os.path.basename(spec_file)}, Shape: {spec.shape} ({spec_type})")
            
            # Check for NaN or extreme values
            has_nan = np.isnan(spec).any()
            max_abs = np.abs(spec).max()
            
            if has_nan:
                nan_files.append(spec_file)
                
            if max_abs > 1000:
                extreme_files.append((spec_file, max_abs))
                
            print(f"  Contains NaN: {has_nan}, Max absolute value: {max_abs:.2f}")
            
        except Exception as e:
            print(f"Error loading {spec_file}: {e}")
    
    # Print summary
    print("\nSpectrogram dimension summary:")
    for shape, count in dims.items():
        shape_parts = shape.split("x")
        if len(shape_parts) >= 2:
            freq_dim = int(shape_parts[1])
            spec_type = "CQT" if freq_dim <= 256 else "STFT"
            print(f"  Shape {shape}: {count} files ({spec_type})")
        else:
            print(f"  Shape {shape}: {count} files (unknown format)")
    
    if nan_files:
        print("\nWARNING: Found NaN values in some spectrograms:")
        for file in nan_files[:5]:  # Show max 5
            print(f"  - {os.path.basename(file)}")
        if len(nan_files) > 5:
            print(f"  ... and {len(nan_files) - 5} more files")
            
    if extreme_files:
        print("\nWARNING: Found extreme values in some spectrograms:")
        for file, val in extreme_files[:5]:  # Show max 5
            print(f"  - {os.path.basename(file)}: {val:.2f}")
        if len(extreme_files) > 5:
            print(f"  ... and {len(extreme_files) - 5} more files")
    
    print("\nDimension check complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check spectrogram dimensions")
    parser.add_argument("--dir", type=str, default="./data/synth/spectrograms",
                       help="Directory containing spectrogram files")
    
    args = parser.parse_args()
    check_spectrogram_dimensions()