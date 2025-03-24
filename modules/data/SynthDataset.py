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
            self._load_data()
            self._generate_segments()
        else:
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
                warnings.warn(f"Failed to initialize GPU cache: {str(e)}")
                # Reset cache structures to avoid issues
                self._zero_spec_cache = {}
                self._zero_logit_cache = {}
                # Store scalar for N chord index as fallback
                self._n_chord_idx = 0

    def _load_data(self):
        """Optimized data loading with caching for single worker"""
        start_time = time.time()
        
        # Try to load from cache first
        if self.use_cache and os.path.exists(self.cache_file):
            if self.verbose:
                print(f"Loading dataset from cache: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Validate and use cache data
                    if ('samples' in cache_data and 'chord_to_idx' in cache_data and 
                        isinstance(cache_data['samples'], list) and 
                        isinstance(cache_data['chord_to_idx'], dict)):
                        
                        # Cache validation successful, load the data
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
                            'is_partial_cache': self.cache_fraction < 1.0
                        }, f)
                else:
                    # Full cache including spectrograms (original approach)
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump({
                            'samples': samples_to_cache,
                            'chord_to_idx': self.chord_to_idx,
                            'metadata_only': False,
                            'is_partial_cache': self.cache_fraction < 1.0
                        }, f)
                        
                if self.verbose:
                    print(f"Saved dataset cache to {self.cache_file}")
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
        """Get a segment by index, with proper padding and direct GPU loading (with improved error handling)"""
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
            
        seg_start, seg_end = self.segment_indices[idx]
        sequence = []
        label_seq = []
        teacher_logits_seq = []  # Always track teacher logits sequence
        has_teacher_logits = False  # Track if any samples have teacher logits
        
        # Get first sample to determine shape for padding
        first_sample = self.samples[seg_start]
        first_spec = None
        expected_dim = 144  # Default expected frequency dimension
        
        # Check for teacher logits availability in the dataset
        # FIX: Explicitly initialize check_for_logits variable
        check_for_logits = self.logits_dir is not None or any('teacher_logits' in s or 'logit_path' in s 
                                                           for s in self.samples[:min(100, len(self.samples))])
        
        # Lazy loading for first sample to determine shape - Fix the context manager issue
        if 'spectro' not in first_sample and 'spec_path' in first_sample:
            try:
                # Try the original path first
                spec_path = first_sample['spec_path']
                
                # If loading fails and we have an expected path, try that instead
                if 'expected_spec_path' in first_sample and not os.path.exists(spec_path):
                    expected_path = first_sample['expected_spec_path']
                    if os.path.exists(expected_path):
                        spec_path = expected_path
                        if self.verbose and not hasattr(self, '_using_expected_path'):
                            print(f"Using expected path format: {expected_path}")
                            self._using_expected_path = True
                
                # FIX: Don't use with statement for numpy mmap mode
                try:
                    # Use memory-mapped mode to get shape without loading
                    mmap_spec = np.load(spec_path, mmap_mode='r')
                    if first_sample.get('frame_idx') is not None and len(mmap_spec.shape) > 1:
                        frame_idx = first_sample['frame_idx']
                        # Now load just the frame we need
                        full_spec = np.load(spec_path)
                        first_spec = full_spec[frame_idx] if frame_idx < full_spec.shape[0] else np.zeros((expected_dim,))
                    else:
                        # Single-frame spectrogram
                        first_spec = np.array(mmap_spec)
                    
                    # Clean up the memory-mapped array explicitly
                    del mmap_spec
                    
                except Exception as e:
                    if self.verbose and not hasattr(self, '_mmap_error_reported'):
                        print(f"Error using memory-mapped mode: {e}")
                        print("Falling back to direct loading")
                        self._mmap_error_reported = True
                    
                    # Fall back to direct loading
                    try:
                        full_spec = np.load(spec_path)
                        if first_sample.get('frame_idx') is not None and len(full_spec.shape) > 1:
                            frame_idx = first_sample['frame_idx']
                            first_spec = full_spec[frame_idx] if frame_idx < full_spec.shape[0] else np.zeros((expected_dim,))
                        else:
                            first_spec = full_spec
                    except Exception as e2:
                        if self.verbose:
                            print(f"Failed to load spectrogram: {e2}")
                            print("Using zero tensor instead")
                        first_spec = np.zeros((expected_dim,))
                    
                # Convert numpy to PyTorch tensor and move to GPU
                try:
                    first_spec = torch.tensor(first_spec, dtype=torch.float, device=self.device)
                except RuntimeError as e:
                    # If CUDA error, fall back to CPU
                    if self.verbose and not hasattr(self, '_cuda_tensor_error'):
                        print(f"CUDA error creating tensor: {e}")
                        print("Falling back to CPU tensor")
                        self._cuda_tensor_error = True
                    first_spec = torch.tensor(first_spec, dtype=torch.float, device='cpu')
                
                # Validate the shape after loading
                if first_spec is not None:
                    if first_spec.dim() != 1 or first_spec.shape[0] < 10:
                        if self.verbose and not hasattr(self, '_shape_warning_logged'):
                            print(f"Warning: Unusual first_spec shape: {first_spec.shape}, expected 1D tensor with {expected_dim} elements")
                            self._shape_warning_logged = True
                        
                        # If shape is unreasonable, fall back to expected shape
                        if first_spec.dim() == 0 or first_spec.shape[0] < 10:
                            first_spec = self._get_zero_tensor(expected_dim, 'spec')
                else:
                    first_spec = self._get_zero_tensor(expected_dim, 'spec')
                
            except Exception as e:
                # Fallback to zeros with a reasonable shape on error
                warnings.warn(f"Error loading first sample: {e}, using zero padding")
                # Use cached zero tensor of common shape
                first_spec = self._get_zero_tensor(expected_dim, 'spec')
        elif 'spectro' in first_sample:
            # Use stored spectrogram if available - move to GPU with error handling
            try:
                if isinstance(first_sample['spectro'], np.ndarray):
                    # Convert numpy array to tensor on GPU
                    first_spec = torch.tensor(first_sample['spectro'], 
                                           dtype=torch.float, device=self.device)
                else:
                    # Already a tensor - just move to the right device
                    first_spec = first_sample['spectro'].to(self.device)
            except RuntimeError:
                # Fall back to CPU if CUDA error
                if isinstance(first_sample['spectro'], np.ndarray):
                    first_spec = torch.tensor(first_sample['spectro'], dtype=torch.float, device='cpu')
                else:
                    first_spec = first_sample['spectro'].to('cpu')
        else:
            # Default fallback - use cached zero tensor
            first_spec = self._get_zero_tensor(expected_dim, 'spec')

        # Check for teacher logits availability in the dataset
        check_for_logits = self.logits_dir is not None or any('teacher_logits' in s or 'logit_path' in s 
                                                           for s in self.samples[:min(100, len(self.samples))])
        
        # Process remaining samples in the segment
        start_song_id = self.samples[seg_start]['song_id']
        
        # Process sequence with enhanced error handling
        try:
            # Process each sample in the segment, using error handling for each step
            for i in range(seg_start, seg_end):
                if i < len(self.samples):
                    sample_i = self.samples[i]
                    
                    # Check if we've crossed a song boundary
                    if sample_i['song_id'] != start_song_id:
                        # We've crossed a song boundary, pad the rest of the sequence
                        padding_needed = seg_end - i
                        
                        # Get the padding shape, with better validation
                        if sequence and sequence[-1].dim() > 0:
                            padding_shape = sequence[-1].shape
                        elif first_spec is not None and first_spec.dim() > 0:
                            padding_shape = first_spec.shape
                        else:
                            # If all else fails, use a default expected shape
                            padding_shape = (expected_dim,)
                        
                        # Use cached zero tensors for padding (avoids repeated GPU transfers)
                        for _ in range(padding_needed):
                            padding_tensor = self._get_zero_tensor(padding_shape, 'spec')
                            sequence.append(padding_tensor)
                            # Use cached N chord index
                            label_seq.append(self._n_chord_idx)
                            
                            # Also add padding to teacher_logits_seq
                            if has_teacher_logits or check_for_logits:
                                if teacher_logits_seq and teacher_logits_seq[0].shape:
                                    logit_shape = teacher_logits_seq[0].shape
                                    teacher_logits_seq.append(self._get_zero_tensor(logit_shape, 'logit'))
                                else:
                                    # Default to common logit dimension
                                    teacher_logits_seq.append(self._get_zero_tensor(170, 'logit'))
                        break
                        
                    # Lazy load spectrogram directly to GPU
                    if 'spectro' not in sample_i and 'spec_path' in sample_i:
                        try:
                            spec_path = sample_i['spec_path']
                            spec = np.load(spec_path)
                            
                            # For multi-frame spectrograms, extract the specific frame
                            if sample_i.get('frame_idx') is not None and len(spec.shape) > 1:
                                frame_idx = sample_i['frame_idx']
                                if frame_idx < spec.shape[0]:
                                    # Create tensor directly on GPU to avoid CPU->GPU transfer
                                    spec_vec = torch.tensor(spec[frame_idx], 
                                                          dtype=torch.float, device=self.device)
                                else:
                                    # Handle out of range index using cached zero tensor
                                    padding_shape = sequence[-1].shape if sequence else first_spec.shape
                                    spec_vec = self._get_zero_tensor(padding_shape, 'spec')
                            else:
                                # Create tensor directly on GPU
                                spec_vec = torch.tensor(spec, dtype=torch.float, device=self.device)
                        except Exception as e:
                            # Use cached zero tensor on error
                            warnings.warn(f"Error loading {sample_i['spec_path']}: {e}")
                            padding_shape = sequence[-1].shape if sequence else first_spec.shape
                            spec_vec = self._get_zero_tensor(padding_shape, 'spec')
                    else:
                        # Use stored spectrogram if available
                        if 'spectro' in sample_i:
                            if isinstance(sample_i['spectro'], np.ndarray):
                                # Convert numpy array to tensor on GPU
                                spec_vec = torch.tensor(sample_i['spectro'], dtype=torch.float, device=self.device)
                            else:
                                # Already a tensor - just move to the right device
                                spec_vec = sample_i['spectro'].to(self.device)
                        else:
                            # Create a zero tensor on GPU using cached tensors
                            padding_shape = sequence[-1].shape if sequence else first_spec.shape
                            spec_vec = self._get_zero_tensor(padding_shape, 'spec')
                    
                    # Get chord label and convert to index
                    chord_label = sample_i['chord_label']
                    chord_idx = self.chord_to_idx.get(chord_label, self.chord_to_idx.get("N", 0))
                    # Create the tensor directly on GPU
                    chord_idx_tensor = torch.tensor(chord_idx, dtype=torch.long, device=self.device)
                    
                    sequence.append(spec_vec)
                    label_seq.append(chord_idx_tensor)
                    
                    # Also handle lazy loading for teacher logits directly to GPU
                    has_logits_for_this_sample = False
                    if 'logit_path' in sample_i and 'teacher_logits' not in sample_i:
                        try:
                            logit_path = sample_i['logit_path']
                            # Load the logits
                            teacher_logits = np.load(logit_path)
                            
                            # Process multi-frame logits
                            if len(teacher_logits.shape) > 1 and 'frame_idx' in sample_i:
                                frame_idx = sample_i['frame_idx']
                                if frame_idx < teacher_logits.shape[0]:
                                    teacher_logits = teacher_logits[frame_idx]
                            
                            # Convert directly to GPU tensor
                            logits_tensor = torch.tensor(teacher_logits, dtype=torch.float, device=self.device)
                            teacher_logits_seq.append(logits_tensor)
                            has_logits_for_this_sample = True
                            has_teacher_logits = True
                        except Exception as e:
                            if not hasattr(self, '_logit_load_errors'):
                                self._logit_load_errors = 0
                            if self._logit_load_errors < 5:
                                warnings.warn(f"Error loading teacher logits from {logit_path}: {e}")
                                self._logit_load_errors += 1
                    elif 'teacher_logits' in sample_i:
                        # Use stored teacher logits and move to GPU
                        if isinstance(sample_i['teacher_logits'], np.ndarray):
                            logits_tensor = torch.tensor(sample_i['teacher_logits'], 
                                                       dtype=torch.float, device=self.device)
                        else:
                            # Already a tensor - just move to device
                            logits_tensor = sample_i['teacher_logits'].to(self.device)
                            
                        teacher_logits_seq.append(logits_tensor)
                        has_logits_for_this_sample = True
                        has_teacher_logits = True
                    
                    # Add zero tensor on GPU if we're collecting logits but this sample doesn't have any
                    if (has_teacher_logits or check_for_logits) and not has_logits_for_this_sample:
                        # Use shape from previous logit if available, otherwise use default
                        if teacher_logits_seq and len(teacher_logits_seq) > 0:
                            logit_shape = teacher_logits_seq[0].shape
                            teacher_logits_seq.append(self._get_zero_tensor(logit_shape, 'logit'))
                        else:
                            # Default class count if we have no prior information
                            teacher_logits_seq.append(self._get_zero_tensor(170, 'logit'))
                else:
                    # We've reached the end of the dataset, pad with zeros
                    padding_shape = sequence[-1].shape if sequence else first_spec.shape
                    sequence.append(self._get_zero_tensor(padding_shape, 'spec'))
                    label_seq.append(self._n_chord_idx)
                    
                    # Also add zero padding to teacher logits if needed
                    if has_teacher_logits or check_for_logits:
                        if teacher_logits_seq and len(teacher_logits_seq) > 0:
                            logit_shape = teacher_logits_seq[0].shape
                            teacher_logits_seq.append(self._get_zero_tensor(logit_shape, 'logit'))
                        else:
                            # Default logit shape
                            teacher_logits_seq.append(self._get_zero_tensor(170, 'logit'))
            
            # Ensure we have exactly seq_len frames
            if len(sequence) < self.seq_len:
                padding_needed = self.seq_len - len(sequence)
                padding_shape = sequence[-1].shape if sequence else first_spec.shape
                for _ in range(padding_needed):
                    sequence.append(self._get_zero_tensor(padding_shape, 'spec'))
                    label_seq.append(self._n_chord_idx)
                    
                    # Add padding to teacher logits if needed
                    if has_teacher_logits or check_for_logits:
                        if teacher_logits_seq and len(teacher_logits_seq) > 0:
                            logit_shape = teacher_logits_seq[0].shape
                            teacher_logits_seq.append(self._get_zero_tensor(logit_shape, 'logit'))
                        else:
                            teacher_logits_seq.append(self._get_zero_tensor(170, 'logit'))
            
            # Create sample output with the collected data - directly stacked on GPU with error handling
            try:
                sample_out = {
                    'spectro': torch.stack(sequence, dim=0),
                    'chord_idx': torch.stack(label_seq, dim=0)
                }
                
                # Include teacher logits if we have them
                if has_teacher_logits or check_for_logits:
                    if teacher_logits_seq:
                        try:
                            # FIX: Ensure all tensors have consistent dimensions before stacking
                            # Check the shapes and standardize them
                            fixed_logits_seq = []
                            if len(teacher_logits_seq) > 0:
                                # Get the first shape to use as reference
                                first_logit = teacher_logits_seq[0]
                                first_shape = first_logit.shape
                                
                                # Process all logits to ensure consistent dimensionality
                                for logit in teacher_logits_seq:
                                    logit_shape = logit.shape
                                    
                                    # Handle dimension mismatch
                                    if len(logit_shape) != len(first_shape):
                                        # If tensor has extra dimension (e.g., [1, 432, 170] vs [432, 170])
                                        if len(logit_shape) > len(first_shape) and logit_shape[0] == 1:
                                            # Remove the extra dimension
                                            logit = logit.squeeze(0)
                                        elif len(logit_shape) < len(first_shape) and first_shape[0] == 1:
                                            # Add missing dimension
                                            logit = logit.unsqueeze(0)
                                    
                                    # Ensure exact shape match (for length/width)
                                    if logit.shape != first_shape:
                                        if self.verbose and not hasattr(self, '_shape_mismatch_warning'):
                                            print(f"Teacher logits shape mismatch: found {logit.shape} and {first_shape}")
                                            self._shape_mismatch_warning = True
                                        
                                        # If shape is unreasonable, fall back to expected shape
                                        if logit.shape != first_shape:
                                            logit = self._get_zero_tensor(first_shape, 'logit')
                                    
                                    fixed_logits_seq.append(logit)
                                
                                # Now stack the standardized tensors
                                sample_out['teacher_logits'] = torch.stack(fixed_logits_seq, dim=0)
                            else:
                                # Create a fallback tensor if we have an empty sequence
                                logits_shape = (len(sequence), 170)
                                sample_out['teacher_logits'] = torch.zeros(logits_shape, 
                                                                     dtype=torch.float, device=self.device)
                        except Exception as e:
                            # Create a fallback tensor if stacking fails
                            if self.verbose:
                                print(f"Error stacking teacher logits: {e}")
                            logits_shape = (len(sequence), 170)
                            sample_out['teacher_logits'] = torch.zeros(logits_shape, 
                                                                     dtype=torch.float, device=self.device)
                
                # FIX: Critical fix for CUDA OOM - when running with workers,
                # move everything to CPU before returning to avoid shared memory issues
                if self.num_workers > 0 and self.cuda_available:
                    # Move data to CPU for safe multiprocessing (only if using workers)
                    for key in sample_out:
                        if isinstance(sample_out[key], torch.Tensor) and sample_out[key].device.type == 'cuda':
                            # Explicitly clone to ensure memory is not shared before moving to CPU
                            sample_out[key] = sample_out[key].clone().cpu()
                else:
                    # Store in GPU batch cache if enabled and not using workers
                    if self.batch_gpu_cache and torch.cuda.is_available():
                        try:
                            self.gpu_batch_cache[idx] = sample_out
                            
                            # Limit cache size to avoid OOM
                            max_cached_items = 100
                            if len(self.gpu_batch_cache) > max_cached_items:
                                oldest_key = next(iter(self.gpu_batch_cache))
                                del self.gpu_batch_cache[oldest_key]
                        except Exception as e:
                            # If caching fails, clear the cache and continue
                            if self.verbose and not hasattr(self, '_cache_write_error'):
                                print(f"Error writing to GPU cache: {e}")
                                print("Clearing GPU cache")
                                self._cache_write_error = True
                            self.gpu_batch_cache = {}
                
                return sample_out
                
            except RuntimeError as e:
                # If CUDA error occurs when stacking, fall back to CPU tensors
                if self.verbose:
                    print(f"CUDA error stacking tensors: {e}")
                    print("Falling back to CPU tensors")
                
                # Convert all tensors to CPU before stacking
                sequence_cpu = [t.cpu() if t.device.type == 'cuda' else t for t in sequence]
                label_seq_cpu = [t.cpu() if t.device.type == 'cuda' else t for t in label_seq]
                
                sample_out = {
                    'spectro': torch.stack(sequence_cpu, dim=0),
                    'chord_idx': torch.stack(label_seq_cpu, dim=0)
                }
                
                if has_teacher_logits or check_for_logits:
                    if teacher_logits_seq:
                        try:
                            # Move logits to CPU and standardize shapes before stacking
                            fixed_logits_seq = []
                            if len(teacher_logits_seq) > 0:
                                # Get the first shape to use as reference
                                first_logit = teacher_logits_seq[0].cpu()
                                first_shape = first_logit.shape
                                
                                # Process all logits to ensure consistent dimensionality
                                for logit in teacher_logits_seq:
                                    logit = logit.cpu() if logit.device.type == 'cuda' else logit
                                    logit_shape = logit.shape
                                    
                                    # Handle dimension mismatch (same logic as GPU path)
                                    if len(logit_shape) != len(first_shape):
                                        if len(logit_shape) > len(first_shape) and logit_shape[0] == 1:
                                            logit = logit.squeeze(0)
                                        elif len(logit_shape) < len(first_shape) and first_shape[0] == 1:
                                            logit = logit.unsqueeze(0)
                                    
                                    # Ensure exact shape match
                                    if logit.shape != first_shape:
                                        logit = self._reshape_tensor_to_match(logit, first_shape)
                                    
                                    fixed_logits_seq.append(logit)
                                
                                # Now stack the standardized tensors
                                sample_out['teacher_logits'] = torch.stack(fixed_logits_seq, dim=0)
                            else:
                                # Default placeholder for teacher logits
                                logits_shape = (len(sequence), 170)
                                sample_out['teacher_logits'] = torch.zeros(logits_shape, dtype=torch.float)
                        except Exception:
                            # Default placeholder for teacher logits
                            logits_shape = (len(sequence), 170)
                            sample_out['teacher_logits'] = torch.zeros(logits_shape, dtype=torch.float)
                
                return sample_out
                
        except Exception as e:
            # Last resort fallback - completely regenerate output from scratch
            if self.verbose:
                print(f"Critical error in dataset.__getitem__: {e}")
                print("Generating fallback output")
            
            # Create a simple fallback output with appropriate dimensions
            spectro = torch.zeros((self.seq_len, expected_dim), dtype=torch.float)
            chord_idx = torch.zeros(self.seq_len, dtype=torch.long)
            
            sample_out = {
                'spectro': spectro,
                'chord_idx': chord_idx
            }
            
            # FIX: Make sure check_for_logits is defined in this exception handler
            if 'check_for_logits' not in locals():
                check_for_logits = self.logits_dir is not None
            
            if has_teacher_logits or check_for_logits:
                sample_out['teacher_logits'] = torch.zeros((self.seq_len, 170), dtype=torch.float)
            
            # FIX: Ensure we return CPU tensors when using workers
            if self.num_workers > 0 and self.cuda_available:
                for key in sample_out:
                    if isinstance(sample_out[key], torch.Tensor) and sample_out[key].device.type == 'cuda':
                        sample_out[key] = sample_out[key].cpu()
            
            return sample_out

    def _reshape_tensor_to_match(self, tensor, target_shape):
        """Helper method to reshape a tensor to match a target shape by padding/trimming as needed."""
        try:
            result_tensor = tensor
            
            # Handle different number of dimensions
            if len(result_tensor.shape) != len(target_shape):
                # If tensor has more dimensions than target, try to squeeze
                if len(result_tensor.shape) > len(target_shape):
                    # Squeeze dimensions of size 1
                    result_tensor = result_tensor.squeeze()
                    # If still not matching, add error handling here
                    if len(result_tensor.shape) != len(target_shape):
                        # Create new tensor with target shape
                        return torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
                # If tensor has fewer dimensions than target, try to unsqueeze
                else:
                    # Add dimensions until we match target
                    while len(result_tensor.shape) < len(target_shape):
                        result_tensor = result_tensor.unsqueeze(0)
            
            # Handle dimension sizes
            if result_tensor.shape != target_shape:
                new_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
                
                # Copy data with size limiting to avoid out-of-bounds
                # Handle each dimension separately
                if len(target_shape) == 1:
                    # 1D tensor
                    copy_size = min(result_tensor.shape[0], target_shape[0])
                    new_tensor[:copy_size] = result_tensor[:copy_size]
                elif len(target_shape) == 2:
                    # 2D tensor
                    copy_size0 = min(result_tensor.shape[0], target_shape[0])
                    copy_size1 = min(result_tensor.shape[1], target_shape[1])
                    new_tensor[:copy_size0, :copy_size1] = result_tensor[:copy_size0, :copy_size1]
                elif len(target_shape) == 3:
                    # 3D tensor
                    copy_size0 = min(result_tensor.shape[0], target_shape[0])
                    copy_size1 = min(result_tensor.shape[1], target_shape[1])
                    copy_size2 = min(result_tensor.shape[2], target_shape[2])
                    new_tensor[:copy_size0, :copy_size1, :copy_size2] = result_tensor[:copy_size0, :copy_size1, :copy_size2]
                
                result_tensor = new_tensor
            
            return result_tensor
        except Exception as e:
            # If reshaping fails, return a zero tensor with the target shape
            if self.verbose and not hasattr(self, '_reshape_error_logged'):
                print(f"Error reshaping tensor: {e}")
                self._reshape_error_logged = True
            return torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)

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