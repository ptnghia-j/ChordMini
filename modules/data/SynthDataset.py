import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import time
import multiprocessing
from functools import partial
import pickle
import warnings
from tqdm import tqdm
import hashlib

class SynthDataset(Dataset):
    """
    Dataset for loading preprocessed spectrograms and chord labels.
    Optimized implementation with multiprocessing and caching.
    """
    def __init__(self, spec_dir, label_dir, chord_mapping=None, seq_len=10, stride=None, 
                 frame_duration=0.1, num_workers=None, cache_file=None, verbose=True):
        self.spec_dir = Path(spec_dir)
        self.label_dir = Path(label_dir)
        self.chord_mapping = chord_mapping
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.frame_duration = frame_duration
        self.samples = []
        self.segment_indices = []
        self.verbose = verbose
        
        # Safely determine number of workers based on environment
        if num_workers is None:
            try:
                # Start with CPU count but cap at reasonable values
                self.num_workers = min(4, os.cpu_count() or 1)
                
                # Check if we're running in a container with limited resources
                if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
                    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                        mem_limit = int(f.read().strip()) / (1024 * 1024 * 1024)  # Convert to GB
                        # If less than 4GB memory, reduce workers to avoid OOM
                        if mem_limit < 4:
                            self.num_workers = 1
                            if verbose:
                                print(f"Limited memory detected ({mem_limit:.1f}GB), using single worker")
            except Exception as e:
                # Fallback to safe default
                self.num_workers = 1
                if verbose:
                    print(f"Error determining CPU count: {e}, using single worker")
        else:
            self.num_workers = num_workers
        
        # Generate a safer cache file name using hashing if none provided
        if cache_file is None:
            # Create a stable cache file name from paths using hashing
            cache_key = f"{spec_dir}_{label_dir}_{seq_len}_{stride}_{frame_duration}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            self.cache_file = f"dataset_cache_{cache_hash}.pkl"
            if verbose:
                print(f"Using cache file: {self.cache_file}")
        else:
            self.cache_file = cache_file
        
        # Map from chord name to index
        if self.chord_mapping is not None:
            self.chord_to_idx = self.chord_mapping
        else:
            self.chord_to_idx = {}
            
        # Load all data
        self._load_data()
        
        # Generate sequence segments 
        self._generate_segments()
        
        # Split data for train/eval/test
        total_segs = len(self.segment_indices)
        self.train_indices = list(range(0, int(total_segs * 0.8)))
        self.eval_indices = list(range(int(total_segs * 0.8), int(total_segs * 0.9)))
        self.test_indices = list(range(int(total_segs * 0.9), total_segs))
        
    def _load_data(self):
        """Optimized data loading with caching and multiprocessing"""
        start_time = time.time()
        
        # Try to load from cache first
        if os.path.exists(self.cache_file):
            if self.verbose:
                print(f"Loading dataset from cache: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Validate cache data has expected format
                    if ('samples' in cache_data and 'chord_to_idx' in cache_data and 
                        isinstance(cache_data['samples'], list) and 
                        isinstance(cache_data['chord_to_idx'], dict)):
                        self.samples = cache_data['samples']
                        self.chord_to_idx = cache_data['chord_to_idx']
                        
                        if self.verbose:
                            print(f"Loaded {len(self.samples)} samples from cache in {time.time() - start_time:.2f}s")
                        return
                    else:
                        print("Cache format invalid, rebuilding dataset")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache, rebuilding dataset: {e}")
        
        # Check if directories exist
        if not self.spec_dir.exists():
            warnings.warn(f"Spectrogram directory does not exist: {self.spec_dir}")
        if not self.label_dir.exists():
            warnings.warn(f"Label directory does not exist: {self.label_dir}")

        # Find spectrogram files faster with specific pattern
        if self.verbose:
            print("Finding spectrogram files (this may take a moment)...")
            
        # Create a fast mapping of label files for lookup
        label_files = {}
        for label_path in self.label_dir.glob("**/*.lab"):
            key = label_path.stem
            if key.endswith("_lab"):
                key = key[:-4]  # Remove '_lab' suffix
            label_files[key] = label_path
        
        if self.verbose:
            print(f"Found {len(label_files)} label files")
            
        # Find all spectrogram files once
        spec_files = list(self.spec_dir.glob("**/*.npy"))
        
        if not spec_files:
            warnings.warn("No spectrogram files found. Check your data paths.")
            return
            
        if self.verbose:
            print(f"Found {len(spec_files)} spectrogram files")
            
        # Process files in parallel if appropriate
        try:
            if self.num_workers > 1 and len(spec_files) > 10:
                # Split files into chunks for parallel processing
                chunk_size = max(1, len(spec_files) // self.num_workers)
                chunks = [spec_files[i:i + chunk_size] for i in range(0, len(spec_files), chunk_size)]
                
                if self.verbose:
                    print(f"Processing files with {self.num_workers} workers ({len(chunks)} chunks)")
                
                # Use context manager to ensure pool is properly closed
                with multiprocessing.Pool(processes=self.num_workers) as pool:
                    worker_func = partial(self._process_file_chunk, label_files=label_files)
                    
                    # Process chunks and collect results with a timeout to prevent hangs
                    all_samples = []
                    for chunk_samples in tqdm(pool.imap(worker_func, chunks), 
                                            total=len(chunks), 
                                            desc="Loading data",
                                            disable=not self.verbose):
                        all_samples.extend(chunk_samples)
                        
                self.samples = all_samples
            else:
                # Fall back to sequential processing if needed
                self.samples = []
                for spec_file in tqdm(spec_files, desc="Loading data", disable=not self.verbose):
                    processed = self._process_file(spec_file, label_files)
                    if processed:
                        self.samples.extend(processed)
        except Exception as e:
            # If parallel processing fails, fall back to sequential
            self.samples = []
            if self.verbose:
                print(f"Parallel processing failed: {e}")
                print("Falling back to sequential processing...")
            for spec_file in tqdm(spec_files, desc="Loading data", disable=not self.verbose):
                processed = self._process_file(spec_file, label_files)
                if processed:
                    self.samples.extend(processed)
        
        # Cache the dataset for future use with proper error handling
        if self.samples:
            try:
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
                    
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({
                        'samples': self.samples,
                        'chord_to_idx': self.chord_to_idx
                    }, f)
                if self.verbose:
                    print(f"Saved dataset cache to {self.cache_file}")
            except Exception as e:
                if self.verbose:
                    print(f"Error saving cache (will continue without caching): {e}")
                
        # Report on spectrogram dimensions
        if self.samples:
            # Analyze the first sample
            first_spec = self.samples[0]['spectro']
            freq_dim = first_spec.shape[-1] if len(first_spec.shape) > 0 else 0
            spec_type = "CQT (Constant-Q Transform)" if freq_dim <= 256 else "STFT"
            if self.verbose:
                print(f"Loaded {len(self.samples)} valid samples")
                print(f"Spectrogram frequency dimension: {freq_dim} (likely {spec_type})")
                
                # Report on class distribution
                chord_counter = Counter(sample['chord_label'] for sample in self.samples)
                print(f"Found {len(chord_counter)} unique chord classes")
                
                end_time = time.time()
                print(f"Dataset loading completed in {end_time - start_time:.2f} seconds")
        else:
            warnings.warn("No samples loaded. Check your data paths and structure.")
    
    def _process_file_chunk(self, spec_files, label_files):
        """Process a chunk of files for parallel processing"""
        chunk_samples = []
        for spec_file in spec_files:
            processed = self._process_file(spec_file, label_files)
            if processed:
                chunk_samples.extend(processed)
        return chunk_samples
        
    def _process_file(self, spec_file, label_files):
        """Process a single spectrogram file and its matching label file"""
        samples = []
        try:
            # Extract file name for matching
            base_name = spec_file.stem
            if base_name.endswith("_spec"):
                base_name = base_name[:-5]  # Remove '_spec' suffix
                
            # Find matching label file directly from dictionary
            label_file = label_files.get(base_name)
            
            if not label_file or not label_file.exists():
                # Try some alternative naming patterns
                found = False
                for suffix in ["", "_lab"]:
                    alt_name = f"{base_name}{suffix}"
                    if alt_name in label_files:
                        label_file = label_files[alt_name]
                        found = True
                        break
                        
                if not found:
                    return samples  # Skip this file if no label found
            
            # Load spectrogram data
            spec = np.load(spec_file)
            
            # Check for NaN values
            if np.isnan(spec).any():
                warnings.warn(f"NaN values found in {spec_file}, replacing with zeros")
                spec = np.nan_to_num(spec, nan=0.0)
            
            # Check for extreme values
            if np.abs(spec).max() > 1000:
                warnings.warn(f"Extreme values found in {spec_file}, max: {np.abs(spec).max()}")
            
            # Load label file
            chord_labels = self._parse_label_file(label_file)
            
            # Create a sample for each frame
            if len(spec.shape) <= 1:  # Single frame
                chord_label = self._find_chord_at_time(chord_labels, 0.0)
                
                # Make sure the chord label exists in the mapping
                if self.chord_mapping is None:
                    if chord_label not in self.chord_to_idx:
                        self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                elif chord_label not in self.chord_mapping:
                    warnings.warn(f"Unknown chord label {chord_label} found in {label_file}")
                    return samples
                    
                samples.append({
                    'spectro': spec,
                    'chord_label': chord_label,
                    'song_id': base_name
                })
            else:  # Multiple frames
                for t in range(spec.shape[0]):
                    frame_time = t * self.frame_duration
                    chord_label = self._find_chord_at_time(chord_labels, frame_time)
                    
                    # Make sure the chord label exists in the mapping
                    if self.chord_mapping is None:
                        if chord_label not in self.chord_to_idx:
                            self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                    elif chord_label not in self.chord_mapping:
                        warnings.warn(f"Unknown chord label {chord_label}, using 'N'")
                        chord_label = "N"
                        
                    samples.append({
                        'spectro': spec[t],
                        'chord_label': chord_label,
                        'song_id': base_name
                    })
                    
        except Exception as e:
            warnings.warn(f"Error processing file {spec_file}: {e}")
            
        return samples
    
    def _parse_label_file(self, label_file):
        """Parse a label file into a list of (start_time, end_time, chord) tuples"""
        result = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            chord = parts[2]
                            result.append((start_time, end_time, chord))
        except Exception as e:
            print(f"Error parsing label file {label_file}: {e}")
            
        return result
    
    def _find_chord_at_time(self, chord_labels, time):
        """Find the chord label at a specific time point"""
        if not chord_labels:
            return "N"  # Return no-chord for empty label files
            
        for start, end, chord in chord_labels:
            if start <= time < end:
                return chord
                
        # If we have chord labels but time is out of bounds
        if chord_labels and time >= chord_labels[-1][1]:
            # Use the last chord if the time exceeds the end
            return chord_labels[-1][2]
            
        return "N"  # No chord found
    
    def _generate_segments(self):
        """Generate segments more efficiently using song boundaries"""
        if not self.samples:
            warnings.warn("No samples to generate segments from")
            return
        
        # Group samples by song_id
        song_samples = {}
        for i, sample in enumerate(self.samples):
            song_id = sample['song_id']
            if song_id not in song_samples:
                song_samples[song_id] = []
            song_samples[song_id].append(i)
        
        if self.verbose:
            print(f"Found {len(song_samples)} unique songs")
        
        # Generate segments for each song
        start_time = time.time()
        total_segments = 0
        
        for song_id, indices in song_samples.items():
            if len(indices) < self.seq_len:
                # For very short songs, create a single segment with padding
                if len(indices) > 0:
                    self.segment_indices.append((indices[0], indices[0] + self.seq_len))
                    total_segments += 1
                continue
                
            # Create segments with stride, respecting song boundaries
            for start_idx in range(0, len(indices) - self.seq_len + 1, self.stride):
                segment_start = indices[start_idx]
                segment_end = indices[start_idx + self.seq_len - 1] + 1
                self.segment_indices.append((segment_start, segment_end))
                total_segments += 1
        
        if self.verbose:
            end_time = time.time()
            print(f"Generated {total_segments} segments in {end_time - start_time:.2f} seconds")
    
    def __len__(self):
        return len(self.segment_indices)
    
    def __getitem__(self, idx):
        """Get a segment by index, with proper padding for song boundaries"""
        if not self.segment_indices:
            raise IndexError("Dataset is empty - no segments available")
            
        seg_start, seg_end = self.segment_indices[idx]
        sequence = []
        label_seq = []
        
        # Get sample shape for padding
        first_spec = self.samples[0]['spectro']
        
        # Get song_id of the starting sample to check for song boundary
        start_song_id = self.samples[seg_start]['song_id']
        
        for i in range(seg_start, seg_end):
            if i < len(self.samples):
                sample_i = self.samples[i]
                
                # Check if we've crossed a song boundary
                if sample_i['song_id'] != start_song_id:
                    # We've crossed a song boundary, pad the rest of the sequence
                    padding_needed = seg_end - i
                    
                    # Use last valid sample shape for padding
                    if sequence:
                        padding_shape = sequence[-1].shape
                    else:
                        padding_shape = first_spec.shape
                        
                    # Add padding
                    for _ in range(padding_needed):
                        sequence.append(torch.zeros(padding_shape, dtype=torch.float))
                        label_seq.append(self.chord_to_idx.get("N", 0))
                        
                    break
                    
                # Regular case: add the sample to the sequence
                spec_vec = torch.tensor(sample_i['spectro'], dtype=torch.float)
                chord_label = sample_i['chord_label']
                chord_idx = self.chord_to_idx.get(chord_label, self.chord_to_idx.get("N", 0))
                
                sequence.append(spec_vec)
                label_seq.append(chord_idx)
            else:
                # We've reached the end of the dataset, pad with zeros
                if sequence:
                    padding_shape = sequence[-1].shape
                else:
                    padding_shape = first_spec.shape
                    
                sequence.append(torch.zeros(padding_shape, dtype=torch.float))
                label_seq.append(self.chord_to_idx.get("N", 0))
        
        # Ensure we have exactly seq_len frames
        if len(sequence) < self.seq_len:
            padding_needed = self.seq_len - len(sequence)
            
            if sequence:
                padding_shape = sequence[-1].shape
            else:
                padding_shape = first_spec.shape
                
            for _ in range(padding_needed):
                sequence.append(torch.zeros(padding_shape, dtype=torch.float))
                label_seq.append(self.chord_to_idx.get("N", 0))
        
        # Return full frame-level label sequence
        sample_out = {
            'spectro': torch.stack(sequence, dim=0),       # [seq_len, feature_dim]
            'chord_idx': torch.tensor(label_seq, dtype=torch.long)  # [seq_len]
        }
        
        return sample_out
    
    def get_train_iterator(self, batch_size=128, shuffle=True, num_workers=4, pin_memory=True):
        """Get an optimized DataLoader for the training set"""
        if not self.train_indices:
            warnings.warn("No training segments available")
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory
            )
            
        return DataLoader(
            SynthSegmentSubset(self, self.train_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,  # Use multiple workers for data loading
            pin_memory=pin_memory,   # Pin memory for faster GPU transfer
            persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
        )
    
    def get_eval_iterator(self, batch_size=128, shuffle=False, num_workers=4, pin_memory=True):
        """Get an optimized DataLoader for the evaluation set"""
        if not self.eval_indices:
            warnings.warn("No evaluation segments available")
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory
            )
            
        return DataLoader(
            SynthSegmentSubset(self, self.eval_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
    
    def get_test_iterator(self, batch_size=128, shuffle=False, num_workers=4, pin_memory=True):
        """Get an optimized DataLoader for the test set"""
        if not self.test_indices:
            warnings.warn("No test segments available")
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory
            )
            
        return DataLoader(
            SynthSegmentSubset(self, self.test_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )


class SynthSegmentSubset(Dataset):
    """Subset of the SynthDataset based on specified indices"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.indices)} indices")
        return self.dataset[self.indices[idx]]


def visualize_spectrogram(spec):
    """Visualize a spectrogram sample"""
    plt.figure(figsize=(10, 6))
    plt.imshow(spec.T, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()


def analyze_chord_distribution(dataset):
    """Analyze and print chord distribution from dataset"""
    chord_counter = Counter()
    
    for sample in dataset.samples:
        chord_counter[sample['chord_label']] += 1
    
    total = sum(chord_counter.values())
    print(f"\nChord distribution (total: {total}):")
    for chord, count in chord_counter.most_common(20):
        print(f"{chord}: {count} ({count/total*100:.2f}%)")


if __name__ == "__main__":
    # Path to test data
    import sys
    from pathlib import Path
    
    # Get the project root directory
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = str(Path(__file__).resolve().parents[2])  # Go up 2 levels from this file
    
    spec_dir = os.path.join(project_root, "data", "synth", "spectrograms")
    label_dir = os.path.join(project_root, "data", "synth", "labels")
    
    print(f"Testing SynthDataset with data from:")
    print(f"  Spectrograms: {spec_dir}")
    print(f"  Labels: {label_dir}")
    
    # Build a chord mapping
    temp_dataset = SynthDataset(spec_dir, label_dir, chord_mapping=None, seq_len=1, stride=1)
    unique_chords = set(sample['chord_label'] for sample in temp_dataset.samples)
    chord_mapping = {chord: idx for idx, chord in enumerate(sorted(unique_chords))}
    
    # Make sure 'N' is included for no-chord label
    if "N" not in chord_mapping:
        chord_mapping["N"] = len(chord_mapping)
    
    print(f"\nGenerated chord mapping with {len(chord_mapping)} unique chords")
    print(f"First 5 mappings: {dict(list(chord_mapping.items())[:5])}")
    
    # Create dataset with reasonable sequence length
    dataset = SynthDataset(spec_dir, label_dir, chord_mapping=chord_mapping, seq_len=10, stride=5)
    
    # Print dataset stats
    print(f"\nDataset Statistics:")
    print(f"  Total frames: {len(dataset.samples)}")
    print(f"  Total segments: {len(dataset)}")
    print(f"  Train segments: {len(dataset.train_indices)}")
    print(f"  Eval segments: {len(dataset.eval_indices)}")
    print(f"  Test segments: {len(dataset.test_indices)}")
    
    # Analyze chord distribution
    analyze_chord_distribution(dataset)
    
    # Test loaders
    train_loader = dataset.get_train_iterator(batch_size=16, shuffle=True)
    val_loader = dataset.get_eval_iterator(batch_size=16)
    test_loader = dataset.get_test_iterator(batch_size=16)
    
    # Display sample batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Spectrogram shape: {sample_batch['spectro'].shape}")
    print(f"  Target chord indices: {sample_batch['chord_idx']}")
    
    # Visualize a spectrogram if possible
    try:
        print("\nVisualizing first spectrogram in batch...")
        visualize_spectrogram(sample_batch['spectro'][0])
    except Exception as e:
        print(f"Could not visualize spectrogram: {e}")
        
    print("\nTest complete!")