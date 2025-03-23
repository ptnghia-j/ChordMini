"""
Generate synthetic test data for testing the ChordMini model locally.
This script creates a small dataset of synthetic spectrograms and chord labels
that can be used to verify the processing pipeline without requiring real data.
"""

import os
import numpy as np
import random
import argparse
from pathlib import Path
import shutil
import time

def create_directory_structure(base_dir):
    """
    Create the directory structure for synthetic data
    """
    # Main directories
    spec_dir = Path(base_dir) / "spectrograms"
    label_dir = Path(base_dir) / "labels"
    logits_dir = Path(base_dir) / "logits"
    
    # Create directories with 3-digit prefixes (for a small subset)
    prefix_dirs = ["001", "002", "003"]
    
    for prefix in prefix_dirs:
        (spec_dir / prefix).mkdir(parents=True, exist_ok=True)
        (label_dir / prefix).mkdir(parents=True, exist_ok=True)
        (logits_dir / prefix).mkdir(parents=True, exist_ok=True)
    
    return spec_dir, label_dir, logits_dir, prefix_dirs

def generate_random_spectrogram(frames, freq_bins):
    """
    Generate a random spectrogram with given dimensions
    """
    # Create log-normal distributed values to mimic real spectrograms
    spec = np.random.lognormal(mean=0, sigma=1, size=(frames, freq_bins))
    # Add some structure - emphasize lower frequencies
    freq_decay = np.linspace(1.0, 0.1, freq_bins)
    spec *= freq_decay[np.newaxis, :]
    # Add time structure - smoothly changing
    for f in range(1, frames):
        random_factor = 0.8 + 0.4 * np.random.random()
        spec[f] = spec[f-1] * random_factor + spec[f] * (1 - random_factor)
    
    return spec

def generate_chord_labels(duration, chord_set):
    """
    Generate synthetic chord labels with realistic timing
    """
    labels = []
    current_time = 0.0
    
    while current_time < duration:
        # Randomly select a chord from the chord set
        chord = random.choice(chord_set)
        # Chord segment length between 1-4 seconds
        segment_length = 1.0 + 3.0 * random.random()
        # Make sure we don't exceed total duration
        end_time = min(current_time + segment_length, duration)
        
        labels.append((current_time, end_time, chord))
        current_time = end_time
    
    return labels

def create_label_file(labels, output_path):
    """
    Write chord labels to a .lab file
    """
    with open(output_path, 'w') as f:
        for start, end, chord in labels:
            f.write(f"{start:.6f} {end:.6f} {chord}\n")

def generate_logits(frames, num_classes):
    """
    Generate synthetic teacher logits
    """
    # Generate mostly small values with occasional spikes
    base_logits = np.random.normal(loc=-5.0, scale=1.0, size=(frames, num_classes))
    
    # For each frame, pick 1-3 classes to be more probable
    for f in range(frames):
        num_peaks = random.randint(1, 3)
        peak_indices = random.sample(range(num_classes), num_peaks)
        for idx in peak_indices:
            # Create a peak value
            base_logits[f, idx] = random.uniform(2.0, 8.0)
    
    return base_logits

def generate_dataset(base_dir, num_samples=20, frame_rate=10, freq_bins=144, duration=10.0, 
                    num_classes=170, verbose=True):
    """
    Generate a complete synthetic dataset with spectrograms, labels, and logits
    """
    # Common chord symbols for realistic label generation
    common_chords = ["N", "C", "C:min", "D", "D:min", "E", "E:min", "F", "F:min", 
                    "G", "G:min", "A", "A:min", "B", "B:min", "G:7", "C:maj7"]
    
    spec_dir, label_dir, logits_dir, prefix_dirs = create_directory_structure(base_dir)
    
    # Generate unique IDs for samples
    start_id = 100000  # 6-digit IDs
    ids = [start_id + i for i in range(num_samples)]
    
    if verbose:
        print(f"Generating {num_samples} synthetic samples:")
        print(f"  - {int(duration)} seconds each")
        print(f"  - {freq_bins} frequency bins")
        print(f"  - Frame rate: {frame_rate} Hz")
        print(f"  - {num_classes} chord classes for logits")
    
    start_time = time.time()
    for i, sample_id in enumerate(ids):
        # Assign to a prefix directory
        prefix = prefix_dirs[i % len(prefix_dirs)]
        id_str = f"{sample_id:06d}"
        
        # Calculate frames based on duration and frame rate
        frames = int(duration * frame_rate)
        
        # Create spectrogram and save
        if verbose and i % 5 == 0:
            print(f"Generating sample {i+1}/{num_samples} (ID: {id_str})...")
        
        # Generate the spectrogram
        spec = generate_random_spectrogram(frames, freq_bins)
        spec_path = spec_dir / prefix / f"{id_str}_spec.npy"
        np.save(spec_path, spec)
        
        # Generate chord labels
        chord_labels = generate_chord_labels(duration, common_chords)
        label_path = label_dir / f"{id_str}.lab"
        create_label_file(chord_labels, label_path)
        
        # Generate teacher logits
        logits = generate_logits(frames, num_classes)
        logits_path = logits_dir / prefix / f"{id_str}_logits.npy"
        np.save(logits_path, logits)
    
    end_time = time.time()
    if verbose:
        print(f"Dataset generation complete in {end_time - start_time:.2f} seconds")
        print(f"Data saved to {base_dir}")
        
        # Count files created
        spec_count = sum(1 for _ in Path(spec_dir).glob("**/*.npy"))
        label_count = sum(1 for _ in Path(label_dir).glob("**/*.lab"))
        logits_count = sum(1 for _ in Path(logits_dir).glob("**/*.npy"))
        
        print(f"Created:")
        print(f"  - {spec_count} spectrogram files")
        print(f"  - {label_count} label files")
        print(f"  - {logits_count} logits files")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for ChordMini model testing")
    parser.add_argument("--output", type=str, default="./data/synth_test",
                        help="Output directory for synthetic data")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of samples to generate")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration of each sample in seconds")
    parser.add_argument("--freq-bins", type=int, default=144,
                        help="Number of frequency bins (144 for CQT, 1024/2048 for STFT)")
    parser.add_argument("--frame-rate", type=int, default=10,
                        help="Frame rate in Hz")
    parser.add_argument("--classes", type=int, default=170,
                        help="Number of chord classes for logits")
    parser.add_argument("--clean", action="store_true",
                        help="Clean output directory before generating data")
    
    args = parser.parse_args()
    
    # Clean output directory if requested
    if args.clean and os.path.exists(args.output):
        print(f"Cleaning output directory: {args.output}")
        shutil.rmtree(args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Generating synthetic data to {args.output}")
    generate_dataset(
        base_dir=args.output,
        num_samples=args.samples,
        frame_rate=args.frame_rate,
        freq_bins=args.freq_bins,
        duration=args.duration,
        num_classes=args.classes,
        verbose=True
    )
    
    print("\nTo use this dataset, run train_student.py with these arguments:")
    print(f"python train_student.py --spec_dir {args.output}/spectrograms --label_dir {args.output}/labels --logits_dir {args.output}/logits --use_kd_loss --batch_size 8 --max_epochs 2")

if __name__ == "__main__":
    # Use spawn method for multiprocessing to avoid CUDA issues
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
