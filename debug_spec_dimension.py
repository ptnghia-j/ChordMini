import os
import sys
import torch
import numpy as np
from pathlib import Path
from modules.data.SynthDataset import SynthDataset

# Set paths
project_root = os.path.abspath(os.path.dirname(__file__))
spec_dir = os.path.join(project_root, "data", "synth", "spectrograms")
label_dir = os.path.join(project_root, "data", "synth", "labels")

print(f"Analyzing data in:")
print(f"  Spectrograms: {spec_dir}")
print(f"  Labels: {label_dir}")

# Find a few spectrogram files for direct inspection
spec_files = []
for root, dirs, files in os.walk(spec_dir):
    for file in files:
        if file.endswith("_spec.npy"):
            spec_files.append(os.path.join(root, file))
            if len(spec_files) >= 5:
                break

# Directly load spectrogram files
print("\nDIRECT FILE INSPECTION:")
print("-" * 50)
for i, file_path in enumerate(spec_files):
    try:
        spec = np.load(file_path)
        print(f"File {i+1}: {os.path.basename(file_path)}")
        print(f"  Shape: {spec.shape}")
        print(f"  Data type: {spec.dtype}")
        print(f"  First frame shape: {spec[0].shape}")
        print(f"  Min/Max values: {spec.min():.4f}/{spec.max():.4f}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    print("-" * 30)

# Create a temporary dataset to check sample dimensions
print("\nSYNTHDATASET INSPECTION:")
print("-" * 50)
# First create a chord mapping
temp_dataset = SynthDataset(spec_dir, label_dir, chord_mapping=None, seq_len=1, stride=1)
unique_chords = set(sample['chord_label'] for sample in temp_dataset.samples)
chord_mapping = {chord: idx for idx, chord in enumerate(sorted(unique_chords))}

# Create the actual dataset with seq_len=108 to match the error message
dataset = SynthDataset(spec_dir, label_dir, chord_mapping=chord_mapping, seq_len=108, stride=108)

# Check raw sample dimensions
print(f"Dataset has {len(dataset.samples)} total frames")
for i in range(min(5, len(dataset.samples))):
    sample = dataset.samples[i]
    print(f"Sample {i+1} spectro shape: {sample['spectro'].shape}")
    print(f"Sample {i+1} chord: {sample['chord_label']}")

# Create data loader and check batch dimensions
loader = dataset.get_train_iterator(batch_size=1)
batch = next(iter(loader))

print("\nBATCH INSPECTION:")
print("-" * 50)
print(f"Batch 'spectro' tensor shape: {batch['spectro'].shape}")
print(f"Batch 'chord_idx' shape: {batch['chord_idx'].shape}")

# Check what's happening during forward pass of BaseTransformer
print("\nSIMULATED FORWARD PASS:")
print("-" * 50)
from modules.models.Transformer.ChordNet import ChordNet
from modules.models.Transformer.BaseTransformer import BaseTransformer

# Create a sample batch with the same dimensions as in the error message
x = torch.randn(1, 108, 12)  # Attempt to recreate the problematic shape
print(f"Input tensor shape: {x.shape}")

# Try reshaping like in the forward method of ChordNet
try:
    x_unsqueezed = x.unsqueeze(1)  # Add channel dimension
    print(f"After unsqueeze: {x_unsqueezed.shape}")
    
    # Extract shape information 
    if len(x_unsqueezed.shape) == 4:
        b, c, t, f = x_unsqueezed.shape
        print(f"Dimensions: batch={b}, channels={c}, time={t}, freq={f}")
    
    # Simulate the potential problem by extracting a slice
    sliced = x_unsqueezed[:, 0, :, :]
    print(f"After extracting first channel: {sliced.shape}")
except Exception as e:
    print(f"Error in forward simulation: {e}")

print("\nCONCLUSION:")
print("-" * 50)
print("If the spectrograms in the files have 144 frequency bins but the model")
print("sees 12, there might be a dimension mismatch. Check if there's any")
print("reshaping, transposition, or feature extraction happening before the data")
print("reaches the model's forward method.")