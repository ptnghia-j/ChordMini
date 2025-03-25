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
import re
from modules.utils.device import get_device, to_device, clear_gpu_cache

# We can simplify the multiprocessing setup since we're using a single worker
# This code is still useful for the __main__ case to ensure proper testing behavior
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn' for testing")
    except RuntimeError:
        warnings.warn("Could not set multiprocessing start method to 'spawn'.")

class SynthDataset(Dataset):
    """
    Dataset for loading preprocessed spectrograms and chord labels.
    Optimized implementation for GPU acceleration with single worker.
    """
    def __init__(self, spec_dir, label_dir, chord_mapping=None, seq_len=10, stride=None, 
                 frame_duration=0.1, num_workers=0, cache_file=None, verbose=True,
                 use_cache=True, metadata_only=True, cache_fraction=0.1, logits_dir=None,
                 lazy_init=False, require_teacher_logits=False, device=None,
                 pin_memory=False, prefetch_factor=2, batch_gpu_cache=False,
                 small_dataset_percentage=None):
        """
        Initialize the dataset with optimized settings for GPU acceleration.
        
        Args:
            spec_dir: Directory containing spectrograms
            label_dir: Directory containing labels
            chord_mapping: Mapping of chord names to indices
            seq_len: Sequence length for segmentation
            stride: Stride for segmentation (default: same as seq_len)
            frame_duration: Duration of each frame in seconds
            num_workers: Number of workers for data loading (forced to 0 for GPU compatibility)
            cache_file: Path to cache file
            verbose: Whether to print verbose output
            use_cache: Whether to use caching
            metadata_only: Whether to cache only metadata
            cache_fraction: Fraction of samples to cache
            logits_dir: Directory containing teacher logits
            lazy_init: Whether to use lazy initialization
            require_teacher_logits: Whether to require teacher logits
            device: Device to use (default: auto-detect)
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch (for DataLoader)
            batch_gpu_cache: Whether to cache batches on GPU for repeated access patterns
            small_dataset_percentage: Optional percentage of the dataset to use (0-1.0)
        """
        # First, log initialization time start to track potential timeout issues
        import time
        init_start_time = time.time()
        if verbose:
            print(f"SynthDataset initialization started at {time.strftime('%H:%M:%S')}")
            print(f"Using spec_dir: {spec_dir}")
            print(f"Using label_dir: {label_dir}")
        
        # Check for CUDA availability
        self.cuda_available = torch.cuda.is_available()
        
        # Force num_workers to 0 for GPU compatibility
        self.num_workers = 0
        if num_workers is not None and num_workers > 0 and verbose:
            print(f"Forcing num_workers to 0 (was {num_workers}) for single-worker GPU optimization")
        
        # Initialize basic parameters
        self.spec_dir = Path(spec_dir)
        self.label_dir = Path(label_dir)
        self.logits_dir = Path(logits_dir) if logits_dir is not None else None
        self.chord_mapping = chord_mapping
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.frame_duration = frame_duration
        self.samples = []
        self.segment_indices = []
        self.verbose = verbose
        self.use_cache = use_cache and cache_file is not None
        self.metadata_only = metadata_only  # Only cache metadata, not full spectrograms
        self.cache_fraction = cache_fraction  # Fraction of samples to cache (default: 10%)
        self.lazy_init = lazy_init
        self.require_teacher_logits = require_teacher_logits
        
        # Disable pin_memory since we're using a single worker
        self.pin_memory = False
        if pin_memory and verbose:
            print("Disabling pin_memory since we're using a single worker")
            
        self.prefetch_factor = prefetch_factor
        self.batch_gpu_cache = batch_gpu_cache
        self.small_dataset_percentage = small_dataset_percentage
        
        # Map from chord name to index
        if self.chord_mapping is not None:
            self.chord_to_idx = self.chord_mapping
        else:
            self.chord_to_idx = {}
            
        # Calculate and store the numeric ID regex pattern (6 digits)
        self.numeric_id_pattern = re.compile(r'(\d{6})')
        
        # Auto-detect device if not provided - use device module with safer initialization
        if device is None:
            try:
                self.device = get_device()
            except Exception as e:
                if verbose:
                    print(f"Error initializing GPU device: {e}")
                    print("Falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        if self.verbose:
            print(f"Using device: {self.device}")
            
        # Initialize GPU batch cache cautiously
        try:
            self.gpu_batch_cache = {} if self.batch_gpu_cache and self.device.type == 'cuda' else None
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not initialize GPU batch cache: {e}")
            self.gpu_batch_cache = None
        
        # Safety check: if require_teacher_logits is True, logits_dir must be provided
        if self.require_teacher_logits and self.logits_dir is None:
            raise ValueError("require_teacher_logits=True requires a valid logits_dir")
        
        # Initialize zero tensor caches
        self._zero_spec_cache = {}
        self._zero_logit_cache = {}
        
        # Cache the N chord index
        self._n_chord_idx = 0  # Default value, will be updated later if needed
            
        # Generate a safer cache file name using hashing if none provided
        if cache_file is None:
            cache_key = f"{spec_dir}_{label_dir}_{seq_len}_{stride}_{frame_duration}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            self.cache_file = f"dataset_cache_{cache_hash}.pkl"
            if verbose:
                print(f"Using cache file: {self.cache_file}")
        else:
            self.cache_file = cache_file
            
        # Only load data if not using lazy initialization
        if not self.lazy_init:
            if verbose:
                print(f"Starting full data loading at {time.time() - init_start_time:.1f}s from init start")
                print("This may take a while - consider using lazy_init=True for faster startup")
            self._load_data()
            self._generate_segments()
        else:
            if verbose:
                print(f"Using lazy initialization (faster startup) at {time.time() - init_start_time:.1f}s from init start")
            # In lazy mode, scan paths and build lightweight metadata, but don't load files
            self.samples = []
            self.segment_indices = []
            
            # First, find all valid spectrogram files with numeric ID pattern
            if self.verbose:
                print("Scanning for files with 6-digit numeric IDs...")
            
            # Create a mapping of label files for quick lookup
            label_files_dict = {}
            for label_path in Path(label_dir).glob("**/*.lab"):
                # Extract the 6-digit ID from the filename
                numeric_match = self.numeric_id_pattern.search(str(label_path.stem))
                if numeric_match:
                    numeric_id = numeric_match.group(1)
                    label_files_dict[numeric_id] = label_path
            
            if self.verbose:
                print(f"Found {len(label_files_dict)} label files with valid numeric IDs")
            
            # Store file paths for lazy loading, focusing on files with numeric IDs
            self.spec_files = []
            for spec_path in Path(spec_dir).glob("**/*.npy"):
                # Extract the 6-digit ID from the filename
                numeric_match = self.numeric_id_pattern.search(str(spec_path.stem))
                if numeric_match:
                    numeric_id = numeric_match.group(1)
                    # Only include files that have matching label files
                    if numeric_id in label_files_dict:
                        self.spec_files.append((spec_path, numeric_id))
            
            if self.verbose:
                print(f"Found {len(self.spec_files)} spectrogram files with valid numeric IDs (lazy mode)")
            
            # Track which song IDs we've processed to avoid duplicates
            processed_song_ids = set()
            # Build minimal metadata for each file without loading content
            song_samples = {}  # Group indices by song_id
            
            for spec_file, numeric_id in self.spec_files:
                # Skip if we've already processed this numeric ID
                if numeric_id in processed_song_ids:
                    continue
                
                # Get the 3-digit directory prefix
                dir_prefix = numeric_id[:3]
                
                # Construct the exact paths to the label file
                label_file = label_files_dict.get(numeric_id)
                if not label_file or not label_file.exists():
                    continue  # Skip if no matching label file
                
                # Parse the label file
                chord_labels = self._parse_label_file(label_file)
                if not chord_labels:
                    continue  # Skip if label file is empty or invalid
                
                # Get shape info without loading the full data
                try:
                    # Use memory-mapped mode to get shape without loading
                    spec_info = np.load(spec_file, mmap_mode='r')
                    spec_shape = spec_info.shape
                    
                    # Create metadata for each frame
                    for t in range(spec_shape[0] if len(spec_shape) > 1 else 1):
                        frame_time = t * self.frame_duration
                        chord_label = self._find_chord_at_time(chord_labels, frame_time)
                        
                        # Make sure the chord label exists in the mapping
                        if self.chord_mapping is None:
                            if chord_label not in self.chord_to_idx:
                                self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                        elif chord_label not in self.chord_mapping:
                            chord_label = "N"  # Use no-chord for unknown chords
                        
                        # Store metadata (not the actual spectrogram)
                        sample_idx = len(self.samples)
                        sample_data = {
                            'spec_path': str(spec_file),
                            'chord_label': chord_label,
                            'song_id': numeric_id,
                            'frame_idx': t,
                            'dir_prefix': dir_prefix  # Store the directory prefix for faster lookup
                        }
                        
                        # Add logits path if logits directory is provided
                        if self.logits_dir is not None:
                            logits_path = self.logits_dir / dir_prefix / f"{numeric_id}_logits.npy"
                            if os.path.exists(logits_path):
                                sample_data['logit_path'] = str(logits_path)
                        
                        self.samples.append(sample_data)
                        
                        # Track this sample in the song group for segmenting
                        if numeric_id not in song_samples:
                            song_samples[numeric_id] = []
                        song_samples[numeric_id].append(sample_idx)
                    
                    # Mark this song ID as processed
                    processed_song_ids.add(numeric_id)
                        
                except Exception as e:
                    if verbose:
                        print(f"Error scanning file {spec_file}: {e}")
            
            # Now generate segments from the metadata (similar to _generate_segments)
            if song_samples:
                if verbose:
                    print(f"Found {len(song_samples)} unique songs with metadata")
                
                for song_id, indices in song_samples.items():
                    if len(indices) < self.seq_len:
                        # For very short songs, create a single segment with padding
                        if len(indices) > 0:
                            self.segment_indices.append((indices[0], indices[0] + self.seq_len))
                        continue
                    
                    # Create segments with stride, respecting song boundaries
                    for start_idx in range(0, len(indices) - self.seq_len + 1, self.stride):
                        segment_start = indices[start_idx]
                        segment_end = indices[start_idx + self.seq_len - 1] + 1
                        self.segment_indices.append((segment_start, segment_end))
                
                if verbose:
                    print(f"Generated {len(self.segment_indices)} segments in lazy mode")
            else:
                warnings.warn("No valid samples found in lazy mode initialization")

            
        # Split data for train/eval/test
        total_segs = len(self.segment_indices)
        self.train_indices = list(range(0, int(total_segs * 0.8)))
        self.eval_indices = list(range(int(total_segs * 0.8), int(total_segs * 0.9)))
        self.test_indices = list(range(int(total_segs * 0.9), total_segs))
        
        # Pre-allocate tensors for common shapes to reduce allocations
        self._zero_spec_cache = {}
        self._zero_logit_cache = {}
        
        # Create a thread-local tensor cache to store commonly accessed tensors on GPU
        # This minimizes CPU-GPU transfers for frequently used tensors
        if self.device.type == 'cuda':
            try:
                self._init_gpu_cache()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not initialize GPU cache: {e}")
                    print("GPU caching will be disabled")
                # Reset GPU cache to avoid potential errors
                self._zero_spec_cache = {}
                self._zero_logit_cache = {}
                self.batch_gpu_cache = None
        
        # Report total initialization time
        init_time = time.time() - init_start_time
        if verbose:
            print(f"Dataset initialization completed in {init_time:.2f} seconds")
            # Add a warning if initialization was slow
            if init_time > 60:  # More than a minute
                print(f"NOTE: Slow initialization detected ({init_time:.1f}s). For large datasets, consider:")
                print("1. Using lazy_init=True to speed up startup")
                print("2. Using metadata_only=True to reduce memory usage")
                print("3. Using a smaller dataset with small_dataset_percentage=0.01")

    def _init_gpu_cache(self):
        """Initialize GPU cache for common tensors to minimize transfers with enhanced error handling"""
        if self.device.type == 'cuda':
            try:
                # Common zero tensors for different dimensions
                freq_dims = [144, 128, 256, 512]  # Common frequency dimensions
                for dim in freq_dims:
                    # Allocate once and reuse - more efficient than creating each time
                    self._zero_spec_cache[dim] = torch.zeros(dim, device=self.device)
                
                # Common zero tensors for logits
                logit_dims = [25, 72, 170]  # Common chord class counts
                for dim in logit_dims:
                    self._zero_logit_cache[dim] = torch.zeros(dim, device=self.device)
                
                # Cache chord indices for quick lookup
                if self.chord_mapping:
                    self._n_chord_idx = torch.tensor(self.chord_to_idx.get("N", 0), 
                                                   device=self.device, dtype=torch.long)
                else:
                    self._n_chord_idx = torch.tensor(0, device=self.device, dtype=torch.long)
                    
                if self.verbose:
                    cache_mb = sum(tensor.element_size() * tensor.nelement() 
                                 for tensor in list(self._zero_spec_cache.values()) + 
                                 list(self._zero_logit_cache.values())) / (1024 * 1024)
                    print(f"Allocated {cache_mb:.2f}MB for GPU tensor cache")
            except Exception as e:
                if self.verbose:
                    print(f"Error initializing GPU cache: {e}")
                self._zero_spec_cache = {}
                self._zero_logit_cache = {}

    def _load_data(self):
        """Load data from files or cache with optimized memory usage and error handling"""
        start_time = time.time()
        
        # Try to load data from cache first
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Verify cache format
                if isinstance(cache_data, dict) and 'samples' in cache_data and 'chord_to_idx' in cache_data:
                    self.samples = cache_data['samples']
                    self.chord_to_idx = cache_data['chord_to_idx']
                    
                    # IMPORTANT: Check if we're using small_dataset_percentage and if cache is not already filtered
                    if self.small_dataset_percentage is not None and self.small_dataset_percentage < 1.0:
                        original_count = len(self.samples)
                        if 'small_dataset_percentage' not in cache_data or cache_data.get('small_dataset_percentage') != self.small_dataset_percentage:
                            if self.verbose:
                                print(f"NOTE: Cache was created with full dataset but now using small_dataset_percentage={self.small_dataset_percentage}")
                                print(f"Will filter samples from {original_count} to {int(original_count * self.small_dataset_percentage)} (approx)")
                            
                            # Group samples by song_id to maintain song integrity in the subset
                            song_groups = {}
                            for sample in self.samples:
                                song_id = sample['song_id']
                                if song_id not in song_groups:
                                    song_groups[song_id] = []
                                song_groups[song_id].append(sample)
                            
                            # Select only the percentage of songs needed
                            songs_to_keep = max(1, int(len(song_groups) * self.small_dataset_percentage))
                            selected_song_ids = list(song_groups.keys())[:songs_to_keep]
                            
                            # Filter samples to only those from selected songs
                            filtered_samples = []
                            for song_id in selected_song_ids:
                                filtered_samples.extend(song_groups[song_id])
                            
                            # Update samples with filtered set
                            self.samples = filtered_samples
                            if self.verbose:
                                print(f"Filtered from {original_count} to {len(self.samples)} samples ({len(self.samples)/original_count*100:.1f}%)")
                        else:
                            if self.verbose:
                                print(f"Cache already filtered to small_dataset_percentage={cache_data.get('small_dataset_percentage')}")
                    
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
        
        # First, create a mapping of label files by numeric ID
        label_files_dict = {}
        for label_path in self.label_dir.glob("**/*.lab"):
            # Extract the 6-digit ID from the filename
            numeric_match = self.numeric_id_pattern.search(str(label_path.stem))
            if numeric_match:
                numeric_id = numeric_match.group(1)
                label_files_dict[numeric_id] = label_path
        
        if self.verbose:
            print(f"Found {len(label_files_dict)} label files with valid numeric IDs")
        
        # Find all spectrogram files with numeric IDs
        valid_spec_files = []
        
        # Look for the specific format first: {dir_prefix}/{numeric_id}_spec.npy
        for prefix_dir in self.spec_dir.glob("**/"):
            if prefix_dir.is_dir() and len(prefix_dir.name) == 3 and prefix_dir.name.isdigit():
                dir_prefix = prefix_dir.name
                # Look for files with pattern {numeric_id}_spec.npy where numeric_id starts with dir_prefix
                for spec_path in prefix_dir.glob(f"{dir_prefix}???_spec.npy"):
                    # Extract the 6-digit ID from the filename
                    filename = spec_path.stem
                    if filename.endswith("_spec"):
                        filename = filename[:-5]  # Remove '_spec' suffix
                    
                    numeric_match = self.numeric_id_pattern.search(filename)
                    if numeric_match:
                        numeric_id = numeric_match.group(1)
                        # Only include files that have matching label files
                        if numeric_id in label_files_dict:
                            valid_spec_files.append((spec_path, numeric_id))
        
        # If we didn't find any files with the specific pattern, fall back to the general search
        if not valid_spec_files and self.verbose:
            print("No spectrogram files found with pattern {dir_prefix}/{numeric_id}_spec.npy, trying general search...")
            
            for spec_path in self.spec_dir.glob("**/*.npy"):
                # Extract the 6-digit ID from the filename
                numeric_match = self.numeric_id_pattern.search(str(spec_path.stem))
                if numeric_match:
                    numeric_id = numeric_match.group(1)
                    # Only include files that have matching label files
                    if numeric_id in label_files_dict:
                        valid_spec_files.append((spec_path, numeric_id))
        
        if not valid_spec_files:
            warnings.warn("No spectrogram files found with valid numeric IDs. Check your data paths.")
            return
            
        if self.verbose:
            print(f"Found {len(valid_spec_files)} spectrogram files with valid numeric IDs")
            # Print sample paths to help diagnose directory structure
            if valid_spec_files:
                print("Sample spectrogram paths:")
                for i, (path, _) in enumerate(valid_spec_files[:3]):
                    print(f"  {i+1}. {path}")
        
        # Handle small dataset percentage option
        if self.small_dataset_percentage is not None:
            # Ensure consistent sampling by using a fixed seed
            np.random.seed(42)
            
            # Get file count based on percentage
            sample_size = max(1, int(len(valid_spec_files) * self.small_dataset_percentage))
            
            # Sample files - prefer deterministic subset
            if sample_size < len(valid_spec_files):
                # Sort files for deterministic behavior
                valid_spec_files.sort(key=lambda x: str(x[0]))
                
                # Take the first portion based on percentage
                valid_spec_files = valid_spec_files[:sample_size]
                
                if self.verbose:
                    print(f"Using {sample_size} files ({self.small_dataset_percentage*100:.2f}% of dataset) for quick testing")
                    print(f"First file: {valid_spec_files[0][0]}")
                    if len(valid_spec_files) > 1:
                        print(f"Last file: {valid_spec_files[-1][0]}")
        
        # Sequential processing 
        self.samples = []
        self.total_processed = 0
        self.total_skipped = 0
        self.skipped_reasons = {
            'missing_label': 0,
            'missing_logits': 0,
            'load_error': 0,
            'format_error': 0
        }
        
        for spec_file, numeric_id in tqdm(valid_spec_files, desc="Loading data", disable=not self.verbose):
            self.total_processed += 1
            processed = self._process_file(spec_file, numeric_id, label_files_dict)
            if processed:
                self.samples.extend(processed)
            else:
                self.total_skipped += 1
        
        # Log statistics about skipped files
        if hasattr(self, 'total_processed') and self.total_processed > 0:
            skip_percentage = (self.total_skipped / self.total_processed) * 100
            if self.verbose:
                print(f"\nFile processing statistics:")
                print(f"  Total processed: {self.total_processed}")
                print(f"  Skipped: {self.total_skipped} ({skip_percentage:.1f}%)")
                if hasattr(self, 'skipped_reasons'):
                    for reason, count in self.skipped_reasons.items():
                        if count > 0:
                            reason_pct = (count / self.total_skipped) * 100 if self.total_skipped > 0 else 0
                            print(f"    - {reason}: {count} ({reason_pct:.1f}%)")
        
        # Cache the dataset for future use with proper error handling
        if self.samples and self.use_cache:
            try:
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
                
                # If using partial caching, select a contiguous portion instead of random songs
                if self.cache_fraction < 1.0:
                    # Group samples by song_id
                    song_groups = {}
                    for sample in self.samples:
                        song_id = sample['song_id']
                        if song_id not in song_groups:
                            song_groups[song_id] = []
                        song_groups[song_id].append(sample)
                    
                    # Get a sorted list of song IDs for deterministic results
                    song_ids = sorted(song_groups.keys())
                    
                    # Select a contiguous portion of songs up to the target fraction
                    total_samples = len(self.samples)
                    target_samples = max(1, int(total_samples * self.cache_fraction))
                    
                    samples_to_cache = []
                    samples_selected = 0
                    
                    # Take the first n songs that fit within our target sample count
                    for song_id in song_ids:
                        if samples_selected >= target_samples:
                            break
                        
                        song_samples = song_groups[song_id]
                        samples_to_cache.extend(song_samples)
                        samples_selected += len(song_samples)
                    
                    if self.verbose:
                        song_count = len(samples_to_cache) // 100  # Approximate song count for display
                        print(f"Caching first {len(samples_to_cache)} samples from {song_count} songs "
                              f"({len(samples_to_cache)/total_samples*100:.1f}% of total)")
                else:
                    samples_to_cache = self.samples
                
                # If metadata-only, create and store metadata
                if self.metadata_only:
                    samples_meta = []
                    for sample in samples_to_cache:
                        # Store metadata and file path instead of actual array
                        meta = {k: sample[k] for k in sample if k != 'spectro'}
                        spec_path = os.path.join(self.spec_dir, f"{sample['song_id']}.npy")
                        if os.path.exists(spec_path):
                            meta['spec_path'] = spec_path
                        samples_meta.append(meta)
                    
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump({
                            'samples': samples_meta,
                            'chord_to_idx': self.chord_to_idx,
                            'metadata_only': True,
                            'is_partial_cache': self.cache_fraction < 1.0,
                            'small_dataset_percentage': self.small_dataset_percentage
                        }, f)
                else:
                    # Full cache including spectrograms (original approach)
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump({
                            'samples': samples_to_cache,
                            'chord_to_idx': self.chord_to_idx,
                            'metadata_only': False,
                            'is_partial_cache': self.cache_fraction < 1.0,
                            'small_dataset_percentage': self.small_dataset_percentage
                        }, f)
                        
                if self.verbose:
                    print(f"Saved dataset cache to {self.cache_file}")
                    if self.small_dataset_percentage is not None:
                        print(f"Cache includes small_dataset_percentage={self.small_dataset_percentage}")
            except Exception as e:
                if self.verbose:
                    print(f"Error saving cache (will continue without caching): {e}")
                
        # Report on spectrogram dimensions
        if self.samples:
            # Safely analyze the first sample to determine dimensions
            first_sample = self.samples[0]
            
            # Check if 'spectro' key exists in the sample, and if not, try to load it from path
            if 'spectro' in first_sample:
                first_spec = first_sample['spectro']
            elif 'spec_path' in first_sample and os.path.exists(first_sample['spec_path']):
                try:
                    # Load the spectrogram from file
                    first_spec = np.load(first_sample['spec_path'])
                    # If it's a multi-frame spectrogram and we have a frame index, get that frame
                    if 'frame_idx' in first_sample and len(first_spec.shape) > 1:
                        frame_idx = first_sample['frame_idx']
                        if frame_idx < first_spec.shape[0]:
                            first_spec = first_spec[frame_idx]
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading first spectrogram for dimension check: {e}")
                    # Use default expected dimensions
                    first_spec = np.zeros((144,))  # Default expected CQT shape
            else:
                # If no spectrogram data is available, use default shape
                first_spec = np.zeros((144,))  # Default expected CQT shape
                if self.verbose:
                    print("WARNING: Could not determine spectrogram shape from first sample")
                    print("Using default frequency dimension of 144")
            
            # Now safely determine frequency dimension from the loaded/created spectrogram
            freq_dim = first_spec.shape[-1] if hasattr(first_spec, 'shape') and len(first_spec.shape) > 0 else 144
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
    
    def _process_file(self, spec_file, numeric_id, label_files_dict, return_skip_reason=False):
        """Process a single spectrogram file with 6-digit ID pattern"""
        samples = []
        skip_reason = None
        
        try:
            # We already have the numeric ID extracted from the calling function
            # Get the 3-digit directory prefix
            dir_prefix = numeric_id[:3]
            
            # Get the matching label file directly from the dictionary
            label_file = label_files_dict.get(numeric_id)
            if not label_file or not os.path.exists(str(label_file)):
                if hasattr(self, 'skipped_reasons'):
                    self.skipped_reasons['missing_label'] += 1
                skip_reason = 'missing_label'
                if return_skip_reason:
                    return [], skip_reason
                return []
            
            # Find matching logit file if logits_dir is provided
            logit_file = None
            if self.logits_dir is not None:
                # Construct the expected logits path using the fixed pattern
                logit_file = self.logits_dir / dir_prefix / f"{numeric_id}_logits.npy"
                
                # Check if the logits file exists
                if not os.path.exists(logit_file):
                    if self.verbose and not hasattr(self, '_missing_logits_warning'):
                        print(f"WARNING: No matching logits file found at {logit_file}")
                        self._missing_logits_warning = True
                    
                    if hasattr(self, 'skipped_reasons'):
                        self.skipped_reasons['missing_logits'] += 1
                    skip_reason = 'missing_logits'
                    if return_skip_reason:
                        return [], skip_reason
                    return []
            
            # At this point, we have all required components (spec, label, and logits if enabled)
            
            # Load spectrogram data - if metadata_only, we'll store the path instead
            if self.metadata_only:
                # Just check if file exists and record metadata
                if os.path.exists(spec_file):
                    # Load minimal information needed for song identification and structure
                    spec_info = np.load(spec_file, mmap_mode='r')
                    spec_shape = spec_info.shape
                    # Parse the label file to obtain chord_labels
                    chord_labels = self._parse_label_file(label_file)
                    # Create sample with metadata only
                    for t in range(spec_shape[0] if len(spec_shape) > 1 else 1):
                        frame_time = t * self.frame_duration
                        chord_label = self._find_chord_at_time(chord_labels, frame_time)
                        
                        # Make sure the chord label exists in the mapping
                        if self.chord_mapping is None:
                            if chord_label not in self.chord_to_idx:
                                self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                        elif chord_label not in self.chord_mapping:
                            warnings.warn(f"Unknown chord label {chord_label}, using 'N'")
                            chord_label = "N"
                            
                        # Store the correct path format including _spec suffix
                        expected_spec_path = str(self.spec_dir / dir_prefix / f"{numeric_id}_spec.npy")
                        
                        samples.append({
                            'spec_path': str(spec_file),  # Keep original path for loading
                            'expected_spec_path': expected_spec_path,  # Store the expected format path
                            'chord_label': chord_label,
                            'song_id': numeric_id,
                            'frame_idx': t,
                            'dir_prefix': dir_prefix  # Store the directory prefix for faster lookup
                        })
                        
                        # Record logit path if available
                        if logit_file is not None:
                            samples[-1]['logit_path'] = str(logit_file)
            else:
                # Original behavior - load full spectrogram
                spec = np.load(spec_file)
                
                # Check for NaN values
                if np.isnan(spec).any():
                    warnings.warn(f"NaN values found in {spec_file}, replacing with zeros")
                    spec = np.nan_to_num(spec, nan=0.0)
                
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
                        return []  # Skip this file if label not in mapping
                        
                    sample_dict = {
                        'spectro': spec,
                        'chord_label': chord_label,
                        'song_id': numeric_id,
                        'dir_prefix': dir_prefix
                    }
                    
                    # Add logits if available
                    if logit_file is not None:
                        try:
                            teacher_logits = self._load_logits_file(logit_file)
                            if teacher_logits is not None:
                                sample_dict['teacher_logits'] = teacher_logits
                        except Exception as e:
                            warnings.warn(f"Error loading logits file {logit_file}: {e}")
                            return []  # Skip this file if logits can't be loaded
                    
                    samples.append(sample_dict)
                else:  # Multiple frames
                    # Pre-load logits for frame-wise access
                    teacher_logits = None
                    if logit_file is not None:
                        try:
                            teacher_logits = self._load_logits_file(logit_file)
                        except Exception as e:
                            warnings.warn(f"Error loading logits file {logit_file}: {e}")
                            return []  # Skip this file if logits can't be loaded
                    
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
                            
                        sample_dict = {
                            'spectro': spec[t],
                            'chord_label': chord_label,
                            'song_id': numeric_id,
                            'dir_prefix': dir_prefix,
                            'frame_idx': t
                        }
                        
                        # Add frame-specific logits if available
                        if teacher_logits is not None:
                            if isinstance(teacher_logits, np.ndarray):
                                if len(teacher_logits.shape) > 1 and t < teacher_logits.shape[0]:
                                    # Multi-frame logits - extract the specific frame
                                    sample_dict['teacher_logits'] = teacher_logits[t]
                                else:
                                    # Single-frame logits - use as is
                                    sample_dict['teacher_logits'] = teacher_logits
                            
                        samples.append(sample_dict)
            
        except Exception as e:
            if hasattr(self, 'skipped_reasons'):
                if "format" in str(e).lower() or "corrupt" in str(e).lower():
                    self.skipped_reasons['format_error'] += 1
                    skip_reason = 'format_error'
                else:
                    self.skipped_reasons['load_error'] += 1
                    skip_reason = 'load_error'
            
            warnings.warn(f"Error processing file {spec_file}: {str(e)}")
            
            # Log the first few errors in detail to help debugging
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            
            if self._error_count < 5 and self.verbose:
                import traceback
                print(f"\nDetailed error processing {spec_file}:")
                print(traceback.format_exc())
                self._error_count += 1
            
            if return_skip_reason:
                return [], skip_reason
            return []
            
        # Return processed samples (or empty list if we skipped this file)
        if return_skip_reason:
            return samples, skip_reason
        return samples
    
    def _load_logits_file(self, logit_file):
        """Load logits from file with format detection and error handling"""
        try:
            # For npy files, load directly with additional error handling
            try:
                teacher_logits = np.load(logit_file)
            except (ValueError, OSError) as e:
                if "corrupt" in str(e).lower():
                    # Special handling for corrupt files
                    warnings.warn(f"Corrupt numpy file detected at {logit_file}: {str(e)}")
                    return None
                raise  # Re-raise other errors
            
            # Sanity check shape and values
            if isinstance(teacher_logits, np.ndarray):
                if self.verbose and not hasattr(self, '_logit_shape_reported'):
                    print(f"Teacher logits shape: {teacher_logits.shape}, min: {teacher_logits.min()}, max: {teacher_logits.max()}")
                    self._logit_shape_reported = True
                
                # Handle NaN or inf values
                if np.isnan(teacher_logits).any() or np.isinf(teacher_logits).any():
                    teacher_logits = np.nan_to_num(teacher_logits, nan=0.0, posinf=100.0, neginf=-100.0)
                    if self.verbose and not hasattr(self, '_logit_nan_reported'):
                        print(f"WARNING: NaN or inf values found in teacher logits and replaced")
                        self._logit_nan_reported = True
                
                return teacher_logits
            else:
                if self.verbose and not hasattr(self, '_logit_type_warning'):
                    print(f"WARNING: Loaded teacher logits has unexpected type: {type(teacher_logits)}")
                    self._logit_type_warning = True
                return None
                
        except Exception as e:
            warnings.warn(f"Error loading logits file {logit_file}: {str(e)}")
            
            # Add more context for the first few errors
            if not hasattr(self, '_logits_error_count'):
                self._logits_error_count = 0
                
            if self._logits_error_count < 5 and self.verbose:
                import traceback
                print(f"\nDetailed error loading logits file {logit_file}:")
                print(traceback.format_exc())
                self._logits_error_count += 1
                
            return None
    
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
        
        # Optional filtering of samples that don't have teacher logits when required
        if self.require_teacher_logits:
            original_count = len(self.samples)
            filtered_samples = []
            for sample in self.samples:
                # Include sample only if it has teacher logits or a valid logit path
                if 'teacher_logits' in sample or ('logit_path' in sample and os.path.exists(sample['logit_path'])):
                    filtered_samples.append(sample)
            
            # Update samples with the filtered list
            self.samples = filtered_samples
            
            if self.verbose:
                new_count = len(self.samples)
                removed = original_count - new_count
                removed_percent = (removed / original_count * 100) if original_count > 0 else 0
                print(f"Filtered out {removed} samples without teacher logits ({removed_percent:.1f}%)")
                print(f"Remaining samples with teacher logits: {new_count}")
                
                # Check if we filtered too aggressively
                if new_count < original_count * 0.1:  # Less than 10% samples remain
                    warnings.warn(f"WARNING: Only {new_count} samples ({new_count/original_count*100:.1f}%) have teacher logits.")
                    warnings.warn(f"This may indicate an issue with logits availability or paths.")
        
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
    
    def _get_zero_tensor(self, shape, tensor_type='spec'):
        """
        Get a zero tensor of the given shape from cache for efficient reuse.
        Enhanced with safer error handling for CUDA issues.
        
        Args:
            shape: Shape or dimension (int) for the zero tensor
            tensor_type: 'spec' for spectrogram, 'logit' for logits
            
        Returns:
            torch.Tensor: Zero tensor of the given shape on the appropriate device
        """
        try:
            # Handle scalar dimension or tuple shapes
            if isinstance(shape, (list, tuple)):
                if len(shape) == 1:
                    dim = shape[0]
                else:
                    # For multi-dimensional shapes, we create a new tensor
                    return torch.zeros(shape, device=self.device, dtype=torch.float)
            else:
                dim = shape
                
            # Use cached tensor if available and CUDA is working
            if self.device.type == 'cuda':
                try:
                    if tensor_type == 'spec' and dim in self._zero_spec_cache:
                        return self._zero_spec_cache[dim].clone()
                    elif tensor_type == 'logit' and dim in self._zero_logit_cache:
                        return self._zero_logit_cache[dim].clone()
                except RuntimeError:
                    # If CUDA error occurs with cached tensors, clear cache and continue with fresh tensors
                    if self.verbose and not hasattr(self, '_cuda_cache_warning'):
                        print("CUDA error with cached tensors. Clearing cache and creating fresh tensors.")
                        self._cuda_cache_warning = True
                    self._zero_spec_cache = {}
                    self._zero_logit_cache = {}
            
            # Create a new tensor (either as fallback or if no cached tensor)
            return torch.zeros(dim, device=self.device, dtype=torch.float)
            
        except Exception as e:
            # If any CUDA error happens, fall back to CPU tensors
            if self.verbose and not hasattr(self, '_device_fallback_warning'):
                print(f"Error creating tensor on {self.device}: {e}")
                print("Falling back to CPU tensors")
                self._device_fallback_warning = True
            
            # Use CPU as fallback device
            return torch.zeros(shape if isinstance(shape, (list, tuple)) else (shape,), 
                              dtype=torch.float, device='cpu')

    def __len__(self):
        return len(self.segment_indices)
    
    def __getitem__(self, idx):
        """Get a segment by index, with proper padding and direct GPU loading with improved performance"""
        if not self.segment_indices:
            raise IndexError("Dataset is empty - no segments available")
        
        # Try to use GPU batch cache with error handling
        if self.batch_gpu_cache and idx in self.gpu_batch_cache:
            try:
                return self.gpu_batch_cache[idx]
            except RuntimeError:
                # Clear cache if CUDA error occurs
                self.gpu_batch_cache = {}
                if self.verbose and not hasattr(self, '_cache_error_warned'):
                    print("CUDA error with cached batch. Clearing GPU batch cache.")
                    self._cache_error_warned = True
        
        # Initialize target device and get segment indices
        target_device = self.device
        seg_start, seg_end = self.segment_indices[idx]
        
        try:
            # Performance optimization: Pre-allocate lists with known size to avoid resizing
            sequence = []
            label_seq = []
            teacher_logits_seq = []
            has_teacher_logits = False
            
            # Get first sample to determine shape for padding
            first_sample = self.samples[seg_start]
            start_song_id = first_sample['song_id']
            
            # Skip unnecessary processing for determining first_spec shape
            # Just use a standard expected dimension for initial allocation
            expected_dim = 144
            
            # Check for teacher logits availability in the dataset
            check_for_logits = self.logits_dir is not None
            
            # Process each sample in the segment, using consistent device
            for i in range(seg_start, seg_end):
                if i >= len(self.samples):
                    # We've reached the end of the dataset, add padding
                    padding_shape = sequence[-1].shape if sequence else (expected_dim,)
                    sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=target_device))
                    label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=target_device))
                    
                    if has_teacher_logits or check_for_logits:
                        logit_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,)
                        teacher_logits_seq.append(torch.zeros(logit_shape, dtype=torch.float, device=target_device))
                    continue
                    
                sample_i = self.samples[i]
                
                # Check if we've crossed a song boundary (skip padding and break)
                if sample_i['song_id'] != start_song_id:
                    break
                    
                # Lazy load spectrogram directly to target device
                if 'spectro' not in sample_i and 'spec_path' in sample_i:
                    try:
                        spec_path = sample_i['spec_path']
                        spec = np.load(spec_path)
                        
                        # For multi-frame spectrograms, extract the specific frame
                        if sample_i.get('frame_idx') is not None and len(spec.shape) > 1:
                            frame_idx = sample_i['frame_idx']
                            if frame_idx < spec.shape[0]:
                                spec_vec = torch.from_numpy(spec[frame_idx]).to(dtype=torch.float, device=target_device)
                            else:
                                padding_shape = sequence[-1].shape if sequence else (expected_dim,)
                                spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=target_device)
                        else:
                            spec_vec = torch.from_numpy(spec).to(dtype=torch.float, device=target_device)
                    except Exception:
                        # Use zero tensor on error
                        padding_shape = sequence[-1].shape if sequence else (expected_dim,)
                        spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=target_device)
                else:
                    # Use stored spectrogram if available
                    if 'spectro' in sample_i:
                        if isinstance(sample_i['spectro'], np.ndarray):
                            spec_vec = torch.from_numpy(sample_i['spectro']).to(dtype=torch.float, device=target_device)
                        else:
                            spec_vec = sample_i['spectro'].clone().detach().to(target_device)
                    else:
                        # Create a zero tensor
                        padding_shape = sequence[-1].shape if sequence else (expected_dim,)
                        spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=target_device)
                
                # Get chord label
                chord_label = sample_i['chord_label']
                chord_idx = self.chord_to_idx.get(chord_label, self.chord_to_idx.get("N", 0))
                chord_idx_tensor = torch.tensor(chord_idx, dtype=torch.long, device=target_device)
                
                sequence.append(spec_vec)
                label_seq.append(chord_idx_tensor)
                
                # Handle teacher logits efficiently
                if 'logit_path' in sample_i and 'teacher_logits' not in sample_i:
                    try:
                        logit_path = sample_i['logit_path']
                        teacher_logits = np.load(logit_path)
                        
                        # Process multi-frame logits - ensure consistent dimensionality
                        if len(teacher_logits.shape) > 1:
                            if 'frame_idx' in sample_i:
                                frame_idx = sample_i['frame_idx']
                                if frame_idx < teacher_logits.shape[0]:
                                    # Extract specific frame
                                    teacher_logits = teacher_logits[frame_idx]
                                else:
                                    # Create zeros tensor for out-of-bounds frame
                                    teacher_logits = np.zeros(teacher_logits.shape[-1] if len(teacher_logits.shape) > 1 else 170)
                            elif teacher_logits.shape[0] == 1:
                                # Single-frame logits stored as [1, num_classes]
                                teacher_logits = teacher_logits[0]
                            else:
                                # Multi-frame logits without frame index - take first frame
                                teacher_logits = teacher_logits[0]
                        
                        # Ensure teacher_logits is 1D
                        if len(teacher_logits.shape) > 1:
                            # If still multi-dimensional, flatten to 1D
                            if self.verbose and not hasattr(self, '_teacher_logits_dim_warning'):
                                print(f"WARNING: Teacher logits have unexpected shape: {teacher_logits.shape}. Flattening to 1D.")
                                self._teacher_logits_dim_warning = True
                            teacher_logits = teacher_logits.reshape(-1)[:170]  # Limit to 170 classes
                        
                        # Create tensor and add to sequence
                        logits_tensor = torch.from_numpy(teacher_logits).to(dtype=torch.float, device=target_device)
                        teacher_logits_seq.append(logits_tensor)
                        has_teacher_logits = True
                        
                    except Exception as e:
                        if self.verbose and not hasattr(self, '_logit_error_warned'):
                            print(f"Error loading teacher logits: {e}")
                            self._logit_error_warned = True
                        
                        # Skip processing error but continue with other data
                        if has_teacher_logits or check_for_logits:
                            logit_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,)
                            teacher_logits_seq.append(torch.zeros(logit_shape, dtype=torch.float, device=target_device))
                elif 'teacher_logits' in sample_i:
                    try:
                        # Extract teacher logits and ensure consistent dimensionality
                        temp_logits = sample_i['teacher_logits']
                        
                        # Convert numpy array to tensor if needed
                        if isinstance(temp_logits, np.ndarray):
                            # Ensure 1D tensor
                            if len(temp_logits.shape) > 1:
                                temp_logits = temp_logits.reshape(-1)[:170]  # Limit to 170 classes
                            logits_tensor = torch.from_numpy(temp_logits).to(dtype=torch.float, device=target_device)
                        else:
                            # Already a tensor
                            if temp_logits.dim() > 1:
                                # Flatten to 1D if needed
                                logits_tensor = temp_logits.reshape(-1)[:170]
                            else:
                                logits_tensor = temp_logits
                            
                            # Move to target device
                            logits_tensor = logits_tensor.clone().detach().to(target_device)
                        
                        teacher_logits_seq.append(logits_tensor)
                        has_teacher_logits = True
                    except Exception as e:
                        if self.verbose and not hasattr(self, '_stored_logit_error_warned'):
                            print(f"Error processing stored teacher logits: {e}")
                            self._stored_logit_error_warned = True
                        
                        # Create zero tensor as fallback
                        if has_teacher_logits or check_for_logits:
                            logit_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,)
                            teacher_logits_seq.append(torch.zeros(logit_shape, dtype=torch.float, device=target_device))
                elif has_teacher_logits or check_for_logits:
                    # If we need teacher logits but none are available for this sample
                    logit_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,)
                    teacher_logits_seq.append(torch.zeros(logit_shape, dtype=torch.float, device=target_device))
            
            # Pad to ensure we have exactly seq_len frames
            current_len = len(sequence)
            if current_len < self.seq_len:
                padding_needed = self.seq_len - current_len
                padding_shape = sequence[-1].shape if sequence else (expected_dim,)
                for _ in range(padding_needed):
                    sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=target_device))
                    label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=target_device))
                    
                    if has_teacher_logits or check_for_logits:
                        logit_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,)
                        teacher_logits_seq.append(torch.zeros(logit_shape, dtype=torch.float, device=target_device))
            
            # Stack tensors, ensuring consistent dimensions
            sample_out = {
                'spectro': torch.stack(sequence, dim=0),
                'chord_idx': torch.stack(label_seq, dim=0)
            }
            
            # Include teacher logits if we have them - CRITICAL FIX FOR DIMENSIONALITY ISSUE
            if has_teacher_logits or check_for_logits:
                if teacher_logits_seq:
                    try:
                        # First ensure all tensors have the same shape
                        first_shape = teacher_logits_seq[0].shape
                        for i in range(len(teacher_logits_seq)):
                            if teacher_logits_seq[i].shape != first_shape:
                                # Resize tensor to match first tensor shape
                                if self.verbose and not hasattr(self, '_teacher_logits_shape_mismatch'):
                                    print(f"Tensor shape mismatch: {teacher_logits_seq[i].shape} vs {first_shape}. Resizing.")
                                    self._teacher_logits_shape_mismatch = True
                                
                                # Create a new tensor with the right shape
                                new_tensor = torch.zeros(first_shape, dtype=torch.float, device=target_device)
                                # Copy data where possible
                                min_size = min(new_tensor.numel(), teacher_logits_seq[i].numel())
                                new_tensor.view(-1)[:min_size].copy_(teacher_logits_seq[i].view(-1)[:min_size])
                                teacher_logits_seq[i] = new_tensor
                        
                        # Stack teacher logits with consistent shape
                        stacked_logits = torch.stack(teacher_logits_seq)
                        
                        # Add batch dimension if needed and ensure 3D format [batch, seq_len, classes]
                        if stacked_logits.dim() == 2:  # Shape is [seq_len, classes]
                            sample_out['teacher_logits'] = stacked_logits.unsqueeze(0)  # Add batch dimension to make [1, seq_len, classes]
                        elif stacked_logits.dim() == 1:  # Shape is [classes]
                            # Reshape to [1, 1, classes]
                            sample_out['teacher_logits'] = stacked_logits.unsqueeze(0).unsqueeze(0)
                        else:
                            # Already 3D or higher - ensure it's [batch, seq_len, classes]
                            sample_out['teacher_logits'] = stacked_logits.unsqueeze(0) if stacked_logits.dim() == 2 else stacked_logits
                        
                        # Final verification - ensure we have [batch, seq_len, classes] format
                        if sample_out['teacher_logits'].dim() != 3:
                            # Log shape and reshape to correct dimensions
                            if self.verbose and not hasattr(self, '_teacher_logits_dim_fix'):
                                print(f"Fixing teacher logits shape from {sample_out['teacher_logits'].shape} to [1, {self.seq_len}, classes]")
                                self._teacher_logits_dim_fix = True
                            
                            # Create a properly formatted tensor - consistent format is [1, seq_len, classes]
                            correct_shape = (1, self.seq_len, stacked_logits.shape[-1] if stacked_logits.dim() > 0 else 170)
                            proper_tensor = torch.zeros(correct_shape, dtype=torch.float, device=target_device)
                            
                            # Try to copy existing data where possible
                            if stacked_logits.dim() > 0:
                                try:
                                    if stacked_logits.dim() == 1:
                                        # Single class vector
                                        proper_tensor[0, 0, :min(stacked_logits.shape[0], proper_tensor.shape[2])] = stacked_logits[:proper_tensor.shape[2]]
                                    elif stacked_logits.dim() == 2:
                                        # Multiple class vectors
                                        max_seq = min(stacked_logits.shape[0], proper_tensor.shape[1])
                                        max_class = min(stacked_logits.shape[1], proper_tensor.shape[2])
                                        proper_tensor[0, :max_seq, :max_class] = stacked_logits[:max_seq, :max_class]
                                    elif stacked_logits.dim() == 3:
                                        # Already have batch dimension
                                        max_batch = min(stacked_logits.shape[0], proper_tensor.shape[0])
                                        max_seq = min(stacked_logits.shape[1], proper_tensor.shape[1])
                                        max_class = min(stacked_logits.shape[2], proper_tensor.shape[2])
                                        proper_tensor[:max_batch, :max_seq, :max_class] = stacked_logits[:max_batch, :max_seq, :max_class]
                                    else:
                                        # Higher dimensions - flatten and use what we can
                                        flat_stacked = stacked_logits.reshape(-1)
                                        flat_proper = proper_tensor.reshape(-1)
                                        max_copy = min(flat_stacked.shape[0], flat_proper.shape[0])
                                        flat_proper[:max_copy] = flat_stacked[:max_copy]
                                except Exception as e:
                                    if self.verbose and not hasattr(self, '_tensor_copy_error'):
                                        print(f"Error copying tensor data: {e}")
                                        self._tensor_copy_error = True
                            
                            sample_out['teacher_logits'] = proper_tensor
                    
                    except Exception as e:
                        if self.verbose and not hasattr(self, '_stacking_error'):
                            print(f"Error stacking teacher logits: {e}")
                            print(f"Teacher logits shapes: {[t.shape for t in teacher_logits_seq[:3]]}")
                            self._stacking_error = True
                        
                        # Create a default tensor as fallback - consistent format is [1, seq_len, classes]
                        proper_shape = (1, self.seq_len, 170)
                        sample_out['teacher_logits'] = torch.zeros(proper_shape, dtype=torch.float, device=target_device)
                else:
                    # No teacher logits, but we should include them for consistent batch structure
                    proper_shape = (1, self.seq_len, 170)
                    sample_out['teacher_logits'] = torch.zeros(proper_shape, dtype=torch.float, device=target_device)
            
            # Cache the result for future use if GPU caching is enabled
            if self.batch_gpu_cache and target_device.type == 'cuda':
                try:
                    self.gpu_batch_cache[idx] = sample_out
                    # Limit cache size to prevent memory issues
                    if len(self.gpu_batch_cache) > 256:
                        oldest_key = next(iter(self.gpu_batch_cache))
                        del self.gpu_batch_cache[oldest_key]
                except Exception:
                    # If caching fails, just ignore and return the sample
                    self.gpu_batch_cache = {}  # Clear cache to prevent future errors
            
            return sample_out
        
        except Exception as e:
            # For all errors, log but don't generate dummy data
            if self.verbose:
                print(f"Error in dataset.__getitem__: {e}")
                
            # Return what we have so far, or raise the exception if we don't have anything
            if sequence and label_seq:
                try:
                    return {
                        'spectro': torch.stack(sequence, dim=0),
                        'chord_idx': torch.stack(label_seq, dim=0)
                    }
                except:
                    # If stacking fails, then raise the original exception
                    raise
            else:
                # If we have nothing to return, raise the original exception
                raise

    def get_train_iterator(self, batch_size=128, shuffle=True, num_workers=None, pin_memory=None):
        """Get an optimized DataLoader for the training set"""
        # Always use a single worker (0) for GPU compatibility
        num_workers_val = 0
        
        # Always disable pin_memory for single worker
        pin_memory_val = False
        
        if not self.train_indices:
            warnings.warn("No training segments available")
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory_val,
                num_workers=num_workers_val
            )
        
        return DataLoader(
            SynthSegmentSubset(self, self.train_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers_val,
            pin_memory=pin_memory_val
        )
    
    def get_eval_iterator(self, batch_size=128, shuffle=False, num_workers=None, pin_memory=None):
        """Get an optimized DataLoader for the evaluation set"""
        # Always use a single worker (0) for GPU compatibility
        num_workers_val = 0
        
        # Always disable pin_memory for single worker
        pin_memory_val = False
        
        if not self.eval_indices:
            warnings.warn("No evaluation segments available")
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory_val,
                num_workers=num_workers_val
            )
            
        return DataLoader(
            SynthSegmentSubset(self, self.eval_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers_val,
            pin_memory=pin_memory_val
        )
    
    def get_test_iterator(self, batch_size=128, shuffle=False, num_workers=None, pin_memory=None):
        """Get an optimized DataLoader for the test set"""
        # Always use a single worker (0) for GPU compatibility
        num_workers_val = 0
        
        # Always disable pin_memory for single worker
        pin_memory_val = False
        
        if not self.test_indices:
            warnings.warn("No test segments available")
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory_val,
                num_workers=num_workers_val
            )
            
        return DataLoader(
            SynthSegmentSubset(self, self.test_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers_val,
            pin_memory=pin_memory_val
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
    
    # Test loaders
    train_loader = dataset.get_train_iterator(batch_size=16, shuffle=True)
    val_loader = dataset.get_eval_iterator(batch_size=16)
    test_loader = dataset.get_test_iterator(batch_size=16)
    
    # Display sample batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Spectrogram shape: {sample_batch['spectro'].shape}")
    print(f"  Target chord indices: {sample_batch['chord_idx'].shape}")
    
    print("\nTest complete!")