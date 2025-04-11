import multiprocessing
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from functools import partial
import pickle
import warnings
from tqdm import tqdm
import hashlib
import re
from modules.utils.device import get_device, to_device, clear_gpu_cache

# Define a wrapper function for multiprocessing
def process_file_wrapper(args):
    """Wrapper function for multiprocessing file processing"""
    dataset_instance, spec_file, file_id, label_files_dict, return_skip_reason = args
    return dataset_instance._process_file(spec_file, file_id, label_files_dict, return_skip_reason)

class SynthDataset(Dataset):
    """
    Dataset for loading preprocessed spectrograms and chord labels.
    Optimized implementation for GPU acceleration with single worker.
    Supports three dataset formats:
    - 'fma': Uses numeric 6-digit IDs with format ddd/dddbbb_spec.npy 
    - 'maestro': Uses arbitrary filenames with format maestro-v3.0.0/file-name_spec.npy
    - 'combined': Loads both 'fma' and 'maestro' datasets simultaneously
    """
    def __init__(self, spec_dir, label_dir, chord_mapping=None, seq_len=10, stride=None, 
                 frame_duration=0.1, num_workers=0, cache_file=None, verbose=True,
                 use_cache=True, metadata_only=True, cache_fraction=0.1, logits_dir=None,
                 lazy_init=False, require_teacher_logits=False, device=None,
                 pin_memory=False, prefetch_factor=2, batch_gpu_cache=False,
                 small_dataset_percentage=None, dataset_type='fma'):
        """
        Initialize the dataset with optimized settings for GPU acceleration.
        
        Args:
            spec_dir: Directory containing spectrograms (or list of directories for 'combined' type)
            label_dir: Directory containing labels (or list of directories for 'combined' type)
            chord_mapping: Mapping of chord names to indices
            seq_len: Sequence length for segmentation
            stride: Stride for segmentation (default: same as seq_len)
            frame_duration: Duration of each frame in seconds
            num_workers: Number of workers for data loading
            cache_file: Path to cache file
            verbose: Whether to print verbose output
            use_cache: Whether to use caching
            metadata_only: Whether to cache only metadata
            cache_fraction: Fraction of samples to cache
            logits_dir: Directory containing teacher logits (or list of directories for 'combined' type)
            lazy_init: Whether to use lazy initialization
            require_teacher_logits: Whether to require teacher logits
            device: Device to use (default: auto-detect)
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch (for DataLoader)
            batch_gpu_cache: Whether to cache batches on GPU for repeated access patterns
            small_dataset_percentage: Optional percentage of the dataset to use (0-1.0)
            dataset_type: Type of dataset format ('fma', 'maestro', or 'combined')
        """
        # First, log initialization time start to track potential timeout issues
        import time
        init_start_time = time.time()
        if verbose:
            print(f"SynthDataset initialization started at {time.strftime('%H:%M:%S')}")
            print(f"Using spec_dir: {spec_dir}")
            print(f"Using label_dir: {label_dir}")
        
        # Support for both single path and list of paths for combined dataset mode
        self.is_combined_mode = isinstance(spec_dir, list) and isinstance(label_dir, list)
        
        # Convert to list format for consistency internally
        self.spec_dirs = [Path(d) for d in spec_dir] if isinstance(spec_dir, list) else [Path(spec_dir)]
        self.label_dirs = [Path(d) for d in label_dir] if isinstance(label_dir, list) else [Path(label_dir)]
        
        # For compatibility with existing methods that expect self.spec_dir and self.label_dir
        # Use the first entry as default for these attributes
        self.spec_dir = self.spec_dirs[0] if self.spec_dirs else None
        self.label_dir = self.label_dirs[0] if self.label_dirs else None
        
        # Handle logits directories similarly
        if logits_dir is not None:
            self.logits_dirs = [Path(d) for d in logits_dir] if isinstance(logits_dir, list) else [Path(logits_dir)]
            self.logits_dir = self.logits_dirs[0] if self.logits_dirs else None
        else:
            self.logits_dirs = None
            self.logits_dir = None
            
        # Check for CUDA availability
        self.cuda_available = torch.cuda.is_available()
        
        # Force num_workers to 0 for GPU compatibility
        self.num_workers = num_workers
        
        # Initialize basic parameters
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
        self.dataset_type = dataset_type  # Dataset format type ('fma', 'maestro', or 'combined')
        
        if self.dataset_type not in ['fma', 'maestro', 'combined']:
            warnings.warn(f"Unknown dataset_type '{dataset_type}', defaulting to 'fma'")
            self.dataset_type = 'fma'
        
        # Disable pin_memory since we're using a single worker
        self.pin_memory = True
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
            
        # Set up regex patterns - always define both patterns regardless of dataset type
        # For Maestro dataset, match any filename pattern ending with _spec, _logits, or .lab
        self.file_pattern = re.compile(r'(.+?)(?:_spec|_logits)?\.(?:npy|lab)$')
        # For FMA dataset - the 6-digit numeric ID pattern
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
            cache_key = f"{spec_dir}_{label_dir}_{seq_len}_{stride}_{frame_duration}_{dataset_type}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            self.cache_file = f"dataset_cache_{dataset_type}_{cache_hash}.pkl"
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
            self.samples = []
            self.segment_indices = []
            
        # Split data for train/eval/test
        total_segs = len(self.segment_indices)
        self.train_indices = list(range(0, int(total_segs * 0.8)))
        self.eval_indices = list(range(int(total_segs * 0.8), int(total_segs * 0.9)))
        self.test_indices = list(range(int(total_segs * 0.9), total_segs))
        
        # Pre-allocate tensors for common shapes to reduce allocations
        self._zero_spec_cache = {}
        self._zero_logit_cache = {}
        
        # Create a thread-local tensor cache to store commonly accessed tensors on GPU
        if self.device.type == 'cuda':
            try:
                self._init_gpu_cache()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not initialize GPU cache: {e}")
                    print("GPU caching will be disabled")
                self._zero_spec_cache = {}
                self._zero_logit_cache = {}
                self.batch_gpu_cache = None
        
        # Report total initialization time
        init_time = time.time() - init_start_time
        if verbose:
            print(f"Dataset initialization completed in {init_time:.2f} seconds")
            if init_time > 60:
                print(f"NOTE: Slow initialization detected ({init_time:.1f}s). For large datasets, consider:")
                print("1. Using lazy_init=True to speed up startup")
                print("2. Using metadata_only=True to reduce memory usage")
                print("3. Using a smaller dataset with small_dataset_percentage=0.01")

    def _init_gpu_cache(self):
        """Initialize GPU cache with commonly used zero tensors for better memory efficiency"""
        if not hasattr(self, 'device') or self.device.type != 'cuda':
            return
        
        # Pre-allocate common tensor shapes to avoid repeated allocation
        common_shapes = [
            (self.seq_len, 144),  # Standard spectrogram sequence
            (1, 144),            # Single frame
            (self.seq_len, 25)    # Common logits/predictions size
        ]
        
        # Create zero tensors for common shapes and cache them
        for shape in common_shapes:
            # Cache for spectrograms
            if shape not in self._zero_spec_cache:
                self._zero_spec_cache[shape] = torch.zeros(shape, dtype=torch.float32, device=self.device)
            
            # Cache for logits
            if shape not in self._zero_logit_cache:
                self._zero_logit_cache[shape] = torch.zeros(shape, dtype=torch.float32, device=self.device)
                
        # Store the N chord index for quick lookup
        self._n_chord_idx = self.chord_to_idx.get("N", 0)

    def _load_logits_file(self, logit_file):
        """Load teacher logits file with error handling"""
        try:
            teacher_logits = np.load(logit_file)
            if np.isnan(teacher_logits).any():
                # Handle corrupted logits with NaN values
                if self.verbose:
                    print(f"Warning: NaN values in logits file {logit_file}, fixing...")
                teacher_logits = np.nan_to_num(teacher_logits, nan=0.0)
            return teacher_logits
        except Exception as e:
            if self.require_teacher_logits:
                raise RuntimeError(f"Error loading required logits file {logit_file}: {e}")
            if self.verbose:
                print(f"Warning: Error loading logits file {logit_file}: {e}")
            return None
    
    def get_tensor_cache(self, shape, is_logits=False):
        """Get a cached zero tensor of the appropriate shape, or create a new one"""
        if is_logits:
            cache_dict = self._zero_logit_cache
        else:
            cache_dict = self._zero_spec_cache
            
        if shape in cache_dict:
            # Return a clone to avoid modifying the cached tensor
            return cache_dict[shape].clone()
        else:
            # Create a new tensor if not in cache
            return torch.zeros(shape, dtype=torch.float32, device=self.device)
            
    def normalize_spectrogram(self, spec, mean=None, std=None):
        """Normalize a spectrogram using mean and std"""
        if mean is None and std is None:
            # Default normalization if no parameters provided
            spec_mean = torch.mean(spec)
            spec_std = torch.std(spec)
            if spec_std == 0:
                spec_std = 1.0
            return (spec - spec_mean) / spec_std
        else:
            # Use provided normalization parameters
            return (spec - mean) / (std if std != 0 else 1.0)
    
    # def apply_gpu_cache(self, batch):
    #     """Apply GPU batch caching for repeated access patterns"""
    #     if not self.gpu_batch_cache:
    #         return batch
            
    #     # Generate a cache key from the batch
    #     key_tensor = batch['spectro'][:, 0, 0] if isinstance(batch['spectro'], torch.Tensor) and len(batch['spectro'].shape) > 2 else None
        
    #     if key_tensor is not None:
    #         # Create a tuple key from the first values
    #         key = tuple(key_tensor.cpu().numpy().tolist())
            
    #         # Check if batch is already cached
    #         if key in self.gpu_batch_cache:
    #             return self.gpu_batch_cache[key]
            
    #         # Cache the new batch (only if it's small enough)
    #         if len(self.gpu_batch_cache) < 1000:  # Limit cache size
    #             self.gpu_batch_cache[key] = batch
                
    #     return batch

    def _load_data(self):
        """Load data from files or cache with optimized memory usage and error handling"""
        start_time = time.time()
        
        # Try to load data from cache first
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if isinstance(cache_data, dict) and 'samples' in cache_data and 'chord_to_idx' in cache_data:
                    self.samples = cache_data['samples']
                    self.chord_to_idx = cache_data['chord_to_idx']
                    
                    if self.verbose:
                        print(f"Loaded {len(self.samples)} samples from cache file {self.cache_file}")
                    
                    return
                else:
                    print("Cache format invalid, rebuilding dataset")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache, rebuilding dataset: {e}")
        
        # Check if directories exist
        for spec_path in self.spec_dirs:
            if not spec_path.exists():
                warnings.warn(f"Spectrogram directory does not exist: {spec_path}")
        
        for label_path in self.label_dirs:
            if not label_path.exists():
                warnings.warn(f"Label directory does not exist: {label_path}")
        
        # First, create a mapping of label files for quick lookup from all label directories
        label_files_dict = {}
        maestro_label_count = 0
        fma_label_count = 0
        
        for label_dir in self.label_dirs:
            # Add debug info about which directories we're scanning
            if self.verbose:
                is_maestro = "maestro" in str(label_dir).lower()
                print(f"Scanning {'Maestro' if is_maestro else 'FMA'} labels from: {label_dir}")
                
            # Use recursive glob to scan all subdirectories
            for label_path in label_dir.glob("**/*.lab"):
                is_maestro = "maestro" in str(label_dir).lower()
                
                if is_maestro:  # Maestro label file handling
                    # Get the full path relative to label_dir for consistent addressing
                    rel_path = label_path.relative_to(label_dir)
                    
                    # Create a combined ID that includes all parent directories
                    file_id = str(rel_path.with_suffix(''))  # Remove .lab extension
                    
                    if file_id.endswith('_spec'):
                        file_id = file_id[:-5]
                    
                    # Debug first few files
                    if maestro_label_count < 3 and self.verbose:
                        print(f"  Maestro label: {label_path}, ID: {file_id}")
                        
                    label_files_dict[file_id] = label_path
                    maestro_label_count += 1
                else:  # FMA label file handling - unchanged
                    numeric_match = self.numeric_id_pattern.search(str(label_path.stem))
                    if numeric_match:
                        numeric_id = numeric_match.group(1)
                        label_files_dict[numeric_id] = label_path
                        fma_label_count += 1
                    
        if self.verbose:
            print(f"Found {len(label_files_dict)} label files with valid IDs across all directories")
            print(f"  FMA: {fma_label_count} label files")
            print(f"  Maestro: {maestro_label_count} label files")
        
        # Find all spectrogram files from all spectrogram directories
        valid_spec_files = []
        maestro_spec_count = 0
        fma_spec_count = 0
        
        for spec_dir in self.spec_dirs:
            # Determine search logic based on directory path
            use_maestro_logic = "maestro" in str(spec_dir).lower()
            
            if use_maestro_logic:
                # Maestro-specific search logic
                if self.verbose:
                    print(f"Using Maestro logic for {spec_dir}")
                    
                # Look recursively through all subdirectories for Maestro files
                for spec_path in spec_dir.glob("**/*.npy"):
                    if "_spec.npy" in str(spec_path):
                        # Get the full path relative to spec_dir for consistent addressing
                        rel_path = spec_path.relative_to(spec_dir)
                        
                        # Create file ID from the relative path (without extension) to match label file pattern
                        file_id = str(rel_path.with_suffix(''))
                        if file_id.endswith('_spec'):
                            file_id = file_id[:-5]
                        
                        # Debug first few files
                        if maestro_spec_count < 10 and self.verbose:
                            print(f"  Maestro spec: {spec_path}")
                            print(f"  ID: {file_id}")
                            print(f"  In dict: {file_id in label_files_dict}")
                            
                        if file_id in label_files_dict:
                            valid_spec_files.append((spec_path, file_id))
                            maestro_spec_count += 1
            else:
                # FMA-specific search logic - unchanged
                for prefix_dir in spec_dir.glob("**/"):
                    if prefix_dir.is_dir() and len(prefix_dir.name) == 3 and prefix_dir.name.isdigit():
                        dir_prefix = prefix_dir.name
                        for spec_path in prefix_dir.glob(f"{dir_prefix}???_spec.npy"):
                            filename = spec_path.stem
                            if filename.endswith("_spec"):
                                filename = filename[:-5]
                            
                            numeric_match = self.numeric_id_pattern.search(filename)
                            if numeric_match:
                                numeric_id = numeric_match.group(1)
                                if numeric_id in label_files_dict:
                                    valid_spec_files.append((spec_path, numeric_id))
                                    fma_spec_count += 1
        
        if not valid_spec_files:
            warnings.warn(f"No valid spectrogram files found for dataset type '{self.dataset_type}'. Check your data paths.")
            return
            
        if self.verbose:
            print(f"Found {len(valid_spec_files)} valid spectrogram files")
            print(f"  FMA: {fma_spec_count} spectrogram files")
            print(f"  Maestro: {maestro_spec_count} spectrogram files")
            
            if valid_spec_files:
                print("Sample spectrogram paths:")
                for i, (path, _) in enumerate(valid_spec_files[:3]):
                    print(f"  {i+1}. {path}")
        
        # Handle small dataset percentage option with special handling for combined mode
        if self.small_dataset_percentage is not None:
            # Ensure consistent sampling by using a fixed seed
            np.random.seed(42)
            
            # Special handling for combined dataset to ensure both datasets are represented
            if self.is_combined_mode and self.dataset_type == 'combined' and len(self.spec_dirs) > 1:
                # Group files by which dataset they belong to
                dataset_files = {}
                for spec_path, file_id in valid_spec_files:
                    # Determine which dataset this file belongs to based on path
                    dataset_key = None
                    for i, dir_path in enumerate(self.spec_dirs):
                        if str(spec_path).startswith(str(dir_path)):
                            dataset_key = i
                            break
                    
                    if dataset_key is not None:
                        if dataset_key not in dataset_files:
                            dataset_files[dataset_key] = []
                        dataset_files[dataset_key].append((spec_path, file_id))
                
                # Add debugging to verify dataset detection
                if self.verbose:
                    print(f"Dataset distribution before sampling:")
                    for i, dir_path in enumerate(self.spec_dirs):
                        dataset_name = "Maestro" if "maestro" in str(dir_path) else "FMA"
                        file_count = len(dataset_files.get(i, []))
                        print(f"  {dataset_name}: {file_count} files detected from {dir_path}")
                
                # Apply percentage to each dataset individually
                sampled_files = []
                for dataset_key, files in dataset_files.items():
                    # Sort files for deterministic behavior within each dataset
                    files.sort(key=lambda x: str(x[0]))
                    
                    # Calculate sample size for this dataset
                    dataset_sample_size = max(1, int(len(files) * self.small_dataset_percentage))
                    
                    # Take the first portion based on percentage
                    sampled_dataset_files = files[:dataset_sample_size]
                    sampled_files.extend(sampled_dataset_files)
                    
                    # Always show this information since it's critical for understanding dataset mixture
                    dataset_name = "Maestro" if "maestro" in str(self.spec_dirs[dataset_key]) else "FMA"
                    print(f"Using {len(sampled_dataset_files)} {dataset_name} files ({self.small_dataset_percentage*100:.2f}% of {len(files)})")
                    if sampled_dataset_files:
                        print(f"First {dataset_name} file: {sampled_dataset_files[0][0]}")
                        if len(sampled_dataset_files) > 1:
                            print(f"Last {dataset_name} file: {sampled_dataset_files[-1][0]}")
                
                # Update valid_spec_files with the combined sampled files
                valid_spec_files = sampled_files
            else:
                # Original behavior for single datasets
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
        
        self.samples = []
        self.total_processed = 0
        self.total_skipped = 0
        self.skipped_reasons = {
            'missing_label': 0,
            'missing_logits': 0,
            'load_error': 0,
            'format_error': 0
        }
        
        num_cpus = max(1, self.num_workers)
        
        if len(valid_spec_files) < num_cpus * 4:
            num_cpus = max(1, len(valid_spec_files) // 2)
            if self.verbose:
                print(f"Small dataset detected, reducing worker count to {num_cpus}")

        args_list = [(self, spec_file, file_id, label_files_dict, True) 
                    for spec_file, file_id in valid_spec_files]
        
        if self.verbose:
            print(f"Processing {len(args_list)} files with {num_cpus} parallel workers")
            
        try:
            with Pool(processes=num_cpus) as pool:
                process_results = list(tqdm(
                    pool.imap(process_file_wrapper, args_list),
                    total=len(args_list),
                    desc=f"Loading data (parallel {'lazy' if self.lazy_init else 'full'})",
                    disable=not self.verbose
                ))
            
            for samples, skip_reason in process_results:
                self.total_processed += 1
                if samples:
                    self.samples.extend(samples)
                else:
                    self.total_skipped += 1
                    if skip_reason in self.skipped_reasons:
                        self.skipped_reasons[skip_reason] += 1
        
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            if self.verbose:
                print(f"ERROR in multiprocessing: {e}")
                print(f"Traceback:\n{error_msg}")
                print(f"Attempting fallback to sequential processing...")
            
            process_results = []
            for args in tqdm(args_list, desc="Loading data (sequential fallback)"):
                process_results.append(process_file_wrapper(args))
            
            for samples, skip_reason in process_results:
                self.total_processed += 1
                if samples:
                    self.samples.extend(samples)
                else:
                    self.total_skipped += 1
                    if skip_reason in self.skipped_reasons:
                        self.skipped_reasons[skip_reason] += 1
        
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
        
        if self.samples and self.use_cache:
            try:
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
                
                samples_to_cache = self.samples
                
                if self.metadata_only:
                    samples_meta = []
                    for sample in samples_to_cache:
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
                
        if self.samples:
            first_sample = self.samples[0]
            
            if 'spectro' in first_sample:
                first_spec = first_sample['spectro']
            elif 'spec_path' in first_sample and os.path.exists(first_sample['spec_path']):
                try:
                    first_spec = np.load(first_sample['spec_path'])
                    if 'frame_idx' in first_sample and len(first_spec.shape) > 1:
                        first_spec = first_spec[first_sample['frame_idx']]
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading first spectrogram: {e}")
                    first_spec = np.zeros((144,))
            else:
                first_spec = np.zeros((144,))
                if self.verbose:
                    print("WARNING: Could not determine spectrogram shape from first sample")
                    print("Using default frequency dimension of 144")
            
            freq_dim = first_spec.shape[-1] if hasattr(first_spec, 'shape') and len(first_spec.shape) > 0 else 144
            spec_type = "CQT (Constant-Q Transform)" if freq_dim <= 256 else "STFT"
            
            if self.verbose:
                print(f"Loaded {len(self.samples)} valid samples")
                print(f"Spectrogram frequency dimension: {freq_dim} (likely {spec_type})")
                
                chord_counter = Counter(sample['chord_label'] for sample in self.samples)
                print(f"Found {len(chord_counter)} unique chord classes")
                
                # Add detailed chord distribution analysis
                from modules.utils.chords import get_chord_quality
                
                # Count samples by chord quality
                quality_counter = Counter()
                for sample in self.samples:
                    chord_label = sample['chord_label']
                    quality = get_chord_quality(chord_label)
                    quality_counter[quality] += 1
                
                # Sort qualities by count for better reporting
                total_samples = len(self.samples)
                print(f"Dataset loading completed in {time.time() - start_time:.2f} seconds")
                print(f"Chord quality distribution:")
                for quality, count in quality_counter.most_common():
                    percentage = (count / total_samples) * 100
                    print(f"  {quality}: {count} samples ({percentage:.2f}%)")
                
                # Print the most common chord types to see what we have
                print("\nMost common chord types:")
                for chord, count in chord_counter.most_common(20):
                    percentage = (count / total_samples) * 100
                    print(f"  {chord}: {count} samples ({percentage:.2f}%)")
                    
                # List some less common chord types to see what unusual chords exist
                print("\nSome less common chord types:")
                less_common = [item for item in chord_counter.most_common()[100:120]]
                for chord, count in less_common:
                    percentage = (count / total_samples) * 100
                    print(f"  {chord}: {count} samples ({percentage:.2f}%)")
                
                end_time = time.time()
                print(f"Dataset loading completed in {end_time - start_time:.2f} seconds")
                
                # Add additional metrics at the end of loading
                if self.samples and hasattr(self, 'is_combined_mode') and self.is_combined_mode:
                    # Create a breakdown of samples by dataset (based on file path)
                    dataset_sample_counts = {}
                    for sample in self.samples:
                        if 'spec_path' in sample:
                            path = sample['spec_path']
                            dataset_key = "unknown"
                            if "maestro" in str(path).lower():
                                dataset_key = "maestro"
                            else:
                                dataset_key = "fma"
                            
                            if dataset_key not in dataset_sample_counts:
                                dataset_sample_counts[dataset_key] = 0
                            dataset_sample_counts[dataset_key] += 1
                    
                    if self.verbose:
                        print("\nSample distribution by dataset source:")
                        for dataset_key, count in dataset_sample_counts.items():
                            percentage = (count / total_samples) * 100
                            print(f"  {dataset_key}: {count} samples ({percentage:.2f}%)")
        else:
            warnings.warn("No samples loaded. Check your data paths and structure.")
    def _process_file(self, spec_file, file_id, label_files_dict, return_skip_reason=False):
        """Process a single spectrogram file based on dataset type"""
        samples = []
        skip_reason = None
        
        try:
            if self.dataset_type == 'maestro':
                dir_prefix = spec_file.parent.name
            else:
                dir_prefix = file_id[:3] if len(file_id) >= 3 else file_id
            
            label_file = label_files_dict.get(file_id)
            if not label_file or not os.path.exists(str(label_file)):
                if hasattr(self, 'skipped_reasons'):
                    self.skipped_reasons['missing_label'] += 1
                skip_reason = 'missing_label'
                if return_skip_reason:
                    return [], skip_reason
                return []
            
            logit_file = None
            if self.logits_dirs is not None:
                # Detect if this is a Maestro file by checking the file_id pattern
                is_maestro_file = 'maestro' in str(spec_file).lower() or (
                    '/' in file_id and 'maestro' in file_id.lower())
                
                for logits_dir in self.logits_dirs:
                    if is_maestro_file:
                        # Use Maestro-specific logic for Maestro files
                        if '/' in file_id:
                            parent_dir, base_name = file_id.split('/', 1)
                            temp_logit_file = logits_dir / parent_dir / f"{base_name}_logits.npy"
                            if os.path.exists(temp_logit_file):
                                logit_file = temp_logit_file
                                break
                        else:
                            temp_logit_file = logits_dir / f"{file_id}_logits.npy"
                            if os.path.exists(temp_logit_file):
                                logit_file = temp_logit_file
                                break
                    else:
                        # Use FMA logic for FMA files
                        temp_logit_file = logits_dir / dir_prefix / f"{file_id}_logits.npy"
                        if os.path.exists(temp_logit_file):
                            logit_file = temp_logit_file
                            break
            
            if self.metadata_only:
                if os.path.exists(spec_file):
                    spec_info = np.load(spec_file, mmap_mode='r')
                    spec_shape = spec_info.shape
                    chord_labels = self._parse_label_file(label_file)
                    for t in range(spec_shape[0] if len(spec_shape) > 1 else 1):
                        frame_time = t * self.frame_duration
                        chord_label = self._find_chord_at_time(chord_labels, frame_time)
                        
                        if self.chord_mapping is None:
                            if chord_label not in self.chord_to_idx:
                                self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                        elif chord_label not in self.chord_mapping:
                            warnings.warn(f"Unknown chord label {chord_label}, using 'N'")
                            chord_label = "N"
                        
                        if self.dataset_type == 'maestro':
                            expected_spec_path = str(spec_file)
                        else:
                            expected_spec_path = str(self.spec_dir / dir_prefix / f"{file_id}_spec.npy")
                        
                        samples.append({
                            'spec_path': str(spec_file),
                            'expected_spec_path': expected_spec_path,
                            'chord_label': chord_label,
                            'song_id': file_id,
                            'frame_idx': t,
                            'dir_prefix': dir_prefix
                        })
                        
                        if logit_file is not None:
                            samples[-1]['logit_path'] = str(logit_file)
            else:
                spec = np.load(spec_file)
                
                if np.isnan(spec).any():
                    warnings.warn(f"NaN values found in {spec_file}, replacing with zeros")
                    spec = np.nan_to_num(spec, nan=0.0)
                
                chord_labels = self._parse_label_file(label_file)
                
                if len(spec.shape) <= 1:
                    chord_label = self._find_chord_at_time(chord_labels, 0.0)
                    
                    if self.chord_mapping is None:
                        if chord_label not in self.chord_to_idx:
                            self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                    elif chord_label not in self.chord_mapping:
                        warnings.warn(f"Unknown chord label {chord_label} found in {label_file}")
                        return []
                        
                    sample_dict = {
                        'spectro': spec,
                        'chord_label': chord_label,
                        'song_id': file_id,
                        'dir_prefix': dir_prefix
                    }
                    
                    if logit_file is not None:
                        try:
                            teacher_logits = self._load_logits_file(logit_file)
                            if teacher_logits is not None:
                                sample_dict['teacher_logits'] = teacher_logits
                        except Exception as e:
                            warnings.warn(f"Error loading logits file {logit_file}: {e}")
                            return []
                    
                    samples.append(sample_dict)
                else:
                    teacher_logits = None
                    if logit_file is not None:
                        try:
                            teacher_logits = self._load_logits_file(logit_file)
                        except Exception as e:
                            warnings.warn(f"Error loading logits file {logit_file}: {e}")
                            return []
                    
                    for t in range(spec.shape[0]):
                        frame_time = t * self.frame_duration
                        chord_label = self._find_chord_at_time(chord_labels, frame_time)
                        
                        if self.chord_mapping is None:
                            if chord_label not in self.chord_to_idx:
                                self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                        elif chord_label not in self.chord_mapping:
                            warnings.warn(f"Unknown chord label {chord_label}, using 'N'")
                            chord_label = "N"
                            
                        sample_dict = {
                            'spectro': spec[t],
                            'chord_label': chord_label,
                            'song_id': file_id,
                            'dir_prefix': dir_prefix,
                            'frame_idx': t
                        }
                        
                        if teacher_logits is not None:
                            if isinstance(teacher_logits, np.ndarray):
                                if len(teacher_logits.shape) > 1 and t < teacher_logits.shape[0]:
                                    sample_dict['teacher_logits'] = teacher_logits[t]
                                else:
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
            
            if return_skip_reason:
                return [], skip_reason
            return []
            
        if return_skip_reason:
            return samples, skip_reason
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
            return "N"
            
        for start, end, chord in chord_labels:
            if start <= time < end:
                return chord
                
        if chord_labels and time >= chord_labels[-1][1]:
            return chord_labels[-1][2]
            
        return "N"
    
    def _generate_segments(self):
        """Generate segments more efficiently using song boundaries"""
        if not self.samples:
            warnings.warn("No samples to generate segments from")
            return
        
        if self.require_teacher_logits:
            original_count = len(self.samples)
            filtered_samples = []
            for sample in self.samples:
                if 'teacher_logits' in sample or ('logit_path' in sample and os.path.exists(sample['logit_path'])):
                    filtered_samples.append(sample)
            
            self.samples = filtered_samples
            
            if self.verbose:
                new_count = len(self.samples)
                removed = original_count - new_count
                removed_percent = (removed / original_count * 100) if original_count > 0 else 0
                print(f"Filtered out {removed} samples without teacher logits ({removed_percent:.1f}%)")
                print(f"Remaining samples with teacher logits: {new_count}")
                
                if new_count < original_count * 0.1:
                    warnings.warn(f"WARNING: Only {new_count} samples ({new_count/original_count*100:.1f}%) have teacher logits.")
                    warnings.warn(f"This may indicate an issue with logits availability or paths.")
        
        song_samples = {}
        for i, sample in enumerate(self.samples):
            song_id = sample['song_id']
            if song_id not in song_samples:
                song_samples[song_id] = []
            song_samples[song_id].append(i)
        
        if self.verbose:
            print(f"Found {len(song_samples)} unique songs")
        
        start_time = time.time()
        total_segments = 0
        
        for song_id, indices in song_samples.items():
            if len(indices) < self.seq_len:
                if len(indices) > 0:
                    self.segment_indices.append((indices[0], indices[0] + self.seq_len))
                    total_segments += 1
                continue
                
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
        """Get a segment by index, with proper padding and direct GPU loading with improved performance"""
        if not self.segment_indices:
            raise IndexError("Dataset is empty - no segments available")
            
        # Get segment indices
        seg_start, seg_end = self.segment_indices[idx]
        
        # Initialize lists for data with expected size
        sequence = []
        label_seq = []
        teacher_logits_seq = []
        has_teacher_logits = False
        
        # Get first sample to determine consistent song ID
        first_sample = self.samples[seg_start]
        start_song_id = first_sample['song_id']
        
        # Process each sample in the segment
        for i in range(seg_start, seg_end):
            if i >= len(self.samples):
                # We've reached the end of the dataset, add padding
                padding_shape = sequence[-1].shape if sequence else (144,)
                sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=self.device))
                label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=self.device))
                
                if has_teacher_logits:
                    logits_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (25,)
                    teacher_logits_seq.append(torch.zeros(logits_shape, dtype=torch.float, device=self.device))
                continue
                
            sample_i = self.samples[i]
            
            # Check if we've crossed a song boundary
            if sample_i['song_id'] != start_song_id:
                break
            
            # Load spectrogram - either from memory or from disk
            if 'spectro' not in sample_i and 'spec_path' in sample_i:
                try:
                    spec_path = sample_i['spec_path']
                    spec = np.load(spec_path)
                    
                    # Handle frame index for multi-frame spectrograms
                    if sample_i.get('frame_idx') is not None and len(spec.shape) > 1:
                        frame_idx = sample_i['frame_idx']
                        if frame_idx < spec.shape[0]:
                            spec_vec = torch.from_numpy(spec[frame_idx]).to(dtype=torch.float, device=self.device)
                        else:
                            # Use zero tensor for out-of-bounds frame
                            padding_shape = sequence[-1].shape if sequence else (144,)
                            spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
                    else:
                        # Single-frame spectrogram
                        spec_vec = torch.from_numpy(spec).to(dtype=torch.float, device=self.device)
                except Exception:
                    # Handle loading errors with zero tensor
                    padding_shape = sequence[-1].shape if sequence else (144,)
                    spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
            else:
                # Use stored spectrogram if available
                if 'spectro' in sample_i:
                    if isinstance(sample_i['spectro'], np.ndarray):
                        spec_vec = torch.from_numpy(sample_i['spectro']).to(dtype=torch.float, device=self.device)
                    else:
                        spec_vec = sample_i['spectro'].clone().detach().to(self.device)
                else:
                    # Use zero tensor if no spectrogram data
                    padding_shape = sequence[-1].shape if sequence else (144,)
                    spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
            
            # Get chord label index
            chord_label = sample_i['chord_label']
            chord_idx = self.chord_to_idx.get(chord_label, self.chord_to_idx.get("N", 0))
            chord_idx_tensor = torch.tensor(chord_idx, dtype=torch.long, device=self.device)
            
            # Add spectrogram and chord label to sequences
            sequence.append(spec_vec)
            label_seq.append(chord_idx_tensor)
            
            # Handle teacher logits - either from memory or from disk
            if 'teacher_logits' in sample_i:
                # Use stored teacher logits
                if isinstance(sample_i['teacher_logits'], np.ndarray):
                    logits_tensor = torch.from_numpy(sample_i['teacher_logits']).to(dtype=torch.float, device=self.device)
                else:
                    logits_tensor = sample_i['teacher_logits'].clone().detach().to(self.device)
                
                # Just store the original logits - we'll process them all at once later
                teacher_logits_seq.append(logits_tensor)
                has_teacher_logits = True
                
            elif 'logit_path' in sample_i:
                # Load teacher logits from disk
                try:
                    logit_path = sample_i['logit_path']
                    logits = np.load(logit_path)
                    
                    # Handle frame index for multi-frame logits
                    if 'frame_idx' in sample_i and len(logits.shape) > 1:
                        frame_idx = sample_i['frame_idx']
                        if frame_idx < logits.shape[0]:
                            logits_vec = logits[frame_idx]
                        else:
                            logits_vec = np.zeros(logits.shape[1] if len(logits.shape) > 1 else logits.shape[0])
                    else:
                        # Use the logits directly, handling different shapes later
                        logits_vec = logits
                    
                    # Convert to tensor without reshaping yet
                    logits_tensor = torch.from_numpy(logits_vec).to(dtype=torch.float, device=self.device)
                    teacher_logits_seq.append(logits_tensor)
                    has_teacher_logits = True
                    
                except Exception as e:
                    if teacher_logits_seq:
                        # Use zero tensor with matching shape if loading fails
                        logits_vec = torch.zeros_like(teacher_logits_seq[0], device=self.device)
                        teacher_logits_seq.append(logits_vec)
                    elif self.verbose and not hasattr(self, '_logits_load_error'):
                        print(f"Warning: Error loading logits, will use zeros: {e}")
                        self._logits_load_error = True
                        # Create dummy tensor
                        teacher_logits_seq.append(torch.zeros(170, dtype=torch.float, device=self.device))
                        has_teacher_logits = True
                        
            elif has_teacher_logits:
                # No teacher logits for this sample but we have them for others
                # Create zero tensor with matching shape for consistency
                if teacher_logits_seq:
                    logits_vec = torch.zeros_like(teacher_logits_seq[0], device=self.device)
                    teacher_logits_seq.append(logits_vec)
        
        # Pad to ensure consistent sequence length
        current_len = len(sequence)
        if current_len < self.seq_len:
            padding_needed = self.seq_len - current_len
            padding_shape = sequence[-1].shape if sequence else (144,)
            for _ in range(padding_needed):
                sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=self.device))
                label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=self.device))
                
                if has_teacher_logits and teacher_logits_seq:
                    logits_shape = teacher_logits_seq[0].shape
                    teacher_logits_seq.append(torch.zeros(logits_shape, dtype=torch.float, device=self.device))
        
        # Create the output dictionary
        sample_out = {
            'spectro': torch.stack(sequence, dim=0),
            'chord_idx': torch.stack(label_seq, dim=0)
        }
        
        # Add teacher logits if available - ensures consistent shape
        if has_teacher_logits and teacher_logits_seq:
            try:
                # Use a fixed number of classes to ensure consistent shape
                fixed_num_classes = 170  # Common for chord recognition: ~170 chord classes
                
                # Convert and resize all tensors to this fixed size
                normalized_logits = []
                for logits_tensor in teacher_logits_seq:
                    # Handle different input dimensions
                    if logits_tensor.dim() == 0:  # Scalar tensor
                        normalized = torch.zeros(fixed_num_classes, dtype=torch.float, device=self.device)
                        normalized[0] = logits_tensor.item() if logits_tensor.numel() > 0 else 0.0
                        
                    elif logits_tensor.dim() == 1:  # Already a vector
                        normalized = torch.zeros(fixed_num_classes, dtype=torch.float, device=self.device)
                        # Copy as much data as possible
                        copy_len = min(logits_tensor.shape[0], fixed_num_classes)
                        normalized[:copy_len] = logits_tensor[:copy_len]
                        
                    else:  # Multi-dimensional tensor
                        # Flatten and then normalize
                        flattened = logits_tensor.reshape(-1)
                        normalized = torch.zeros(fixed_num_classes, dtype=torch.float, device=self.device)
                        copy_len = min(flattened.shape[0], fixed_num_classes)
                        normalized[:copy_len] = flattened[:copy_len]
                    
                    normalized_logits.append(normalized)
                
                # Stack the uniformly shaped logits
                teacher_logits_stacked = torch.stack(normalized_logits, dim=0)
                
                # Add to the output dictionary
                sample_out['teacher_logits'] = teacher_logits_stacked
                
                if self.verbose and not hasattr(self, '_logit_shape_info'):
                    print(f"Teacher logits shape: {teacher_logits_stacked.shape}")
                    self._logit_shape_info = True
                    
            except Exception as e:
                if self.verbose and not hasattr(self, '_logits_error'):
                    print(f"Error processing teacher logits: {e}")
                    print(f"Will use dummy tensor instead")
                    self._logits_error = True
                
                # Create dummy tensor with fixed shape
                dummy_logits = torch.zeros((self.seq_len, fixed_num_classes), dtype=torch.float, device=self.device)
                sample_out['teacher_logits'] = dummy_logits
        
        # Apply GPU batch caching if enabled
        if self.batch_gpu_cache and self.device.type == 'cuda':
            try:
                key = idx  # Use the index as the key
                self.gpu_batch_cache[key] = sample_out
                
                # Limit cache size to avoid memory issues
                if len(self.gpu_batch_cache) > 256:
                    oldest_key = next(iter(self.gpu_batch_cache))
                    del self.gpu_batch_cache[oldest_key]
            except Exception as e:
                if self.verbose and not hasattr(self, '_cache_error_warning'):
                    print(f"Warning: Error in GPU batch caching: {e}")
                    self._cache_error_warning = True
                
                # Clear cache if an error occurs
                self.gpu_batch_cache = {}
        
        return sample_out

    def _get_data_iterator(self, indices, name, batch_size=128, shuffle=False, num_workers=None, pin_memory=None):
        """Helper method to get a data iterator for a specific subset of indices
        
        Args:
            indices: List of indices to use
            name: Name of the subset for warning message
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for DataLoader
            pin_memory: Whether to use pinned memory for DataLoader
            
        Returns:
            DataLoader object
        """
        if not indices:
            warnings.warn(f"No {name} segments available")
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                num_workers=num_workers
            )
        
        return DataLoader(
            SynthSegmentSubset(self, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers
        )

    def get_train_iterator(self, batch_size=128, shuffle=True, num_workers=None, pin_memory=None):
        """Get data iterator for training set"""
        return self._get_data_iterator(
            self.train_indices, 
            "training", 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def get_eval_iterator(self, batch_size=128, shuffle=False, num_workers=None, pin_memory=None):
        """Get data iterator for evaluation set"""
        return self._get_data_iterator(
            self.eval_indices, 
            "evaluation", 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def get_test_iterator(self, batch_size=128, shuffle=False, num_workers=None, pin_memory=None):
        """Get data iterator for test set"""
        return self._get_data_iterator(
            self.test_indices, 
            "test", 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

class SynthSegmentSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.indices)} indices")
        return self.dataset[self.indices[idx]]