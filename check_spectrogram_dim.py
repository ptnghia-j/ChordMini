import os
import numpy as np
import glob

# Path to the synthesized spectrograms
spec_dir = os.path.join("data", "synth", "spectrograms")

# Find all spectrogram files
spec_files = []
for root, dirs, files in os.walk(spec_dir):
    for file in files:
        if file.endswith("_spec.npy") or file.endswith(".npy"):
            spec_files.append(os.path.join(root, file))

print(f"Found {len(spec_files)} spectrogram files")

# Check dimensions of the first few files
for i, spec_file in enumerate(spec_files[:5]):
    try:
        spec = np.load(spec_file)
        print(f"File: {os.path.basename(spec_file)}")
        print(f"Shape: {spec.shape}")
        print(f"Type: {spec.dtype}")
        print(f"Min: {np.min(spec)}, Max: {np.max(spec)}")
        print("-" * 40)
    except Exception as e:
        print(f"Error loading {spec_file}: {e}")
        print("-" * 40)
    
    if i >= 4:  # Only check 5 files
        break

# If there's at least one file, analyze more details of the first file
if spec_files:
    try:
        spec = np.load(spec_files[0])
        print("\nDetailed analysis of first file:")
        
        if len(spec.shape) >= 2:
            # Print some sample values from the first file
            print(f"First time frame values: {spec[0][:5]}...")
            if len(spec.shape) > 2:
                print(f"First channel, first time frame values: {spec[0][0][:5]}...")
    except Exception as e:
        print(f"Error analyzing first file: {e}")