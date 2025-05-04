import os # Ensure os is imported
import warnings # Ensure warnings is imported
import torch # Ensure torch is imported
import numpy as np # Ensure numpy is imported
import pickle # Ensure pickle is imported
import hashlib # Ensure hashlib is imported
import re # Ensure re is imported
import time # Ensure time is imported
import traceback # Ensure traceback is imported
import glob # Ensure glob is imported
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.utils.device import get_device, to_device, clear_gpu_cache
from modules.utils.chords import Chords # Ensure Chords is imported

# --- ADDITIONAL IMPORTS ---
from typing import List, Tuple, Dict, Optional
# --- END ADDITIONAL IMPORTS ---

# Define a wrapper function for multiprocessing
def process_file_wrapper(args):
    """Wrapper function for multiprocessing file processing"""
    # Unpack arguments, including the dataset_type for the specific file
    dataset_instance, spec_file, file_id, label_files_dict, logit_files_dict, current_dataset_type, return_skip_reason = args
    # Pass dataset type to _process_file
    return dataset_instance._process_file(spec_file, file_id, label_files_dict, logit_files_dict, current_dataset_type, return_skip_reason)

# --- Add ID Normalization Helper ---
def _normalize_id(id_string):
    """Replace spaces and hyphens with underscores for consistent IDs."""
    if not isinstance(id_string, str):
        return id_string # Return as is if not a string
    # Replace multiple spaces/hyphens with single underscore, then strip leading/trailing
    # Also handle potential multiple underscores resulting from replacements
    normalized = re.sub(r'[\s-]+', '_', id_string)
    normalized = re.sub(r'_+', '_', normalized).strip('_') # Consolidate multiple underscores
    # --- REMOVED LOWERCASE ---
    return normalized
# --- End Helper ---

def get_rel_id(file_path: Path, root_dir: Path, suffix_to_remove: str = '', ext_to_remove: str = '') -> str:
    """Get the relative path from root_dir, remove suffix and extension, and convert to posix string."""
    rel = file_path.relative_to(root_dir)
    rel = rel.with_suffix('')  # remove extension
    rel_str = str(rel)
    if suffix_to_remove and rel_str.endswith(suffix_to_remove):
        rel_str = rel_str[:-len(suffix_to_remove)]
    return rel_str.replace("\\", "/")

class SynthDataset(Dataset):
    """
    Dataset for loading preprocessed spectrograms and chord labels.
    Optimized implementation for GPU acceleration with single worker.
    Supports dataset formats:
    - 'fma': Uses numeric 6-digit IDs with format ddd/dddbbb_spec.npy
    - 'maestro': Uses arbitrary filenames with format maestro-v3.0.0/file-name_spec.npy
    - 'dali_synth': Uses hex IDs with format xxx/hexid_spec.npy (xxx is alphanumeric)
    - 'labeled': Uses real ground truth labels from LabeledDataset structure (e.g., LabeledDataset/Labels/Artist/Album/file.lab) with corresponding features (e.g., LabeledDataset_synth/spectrograms/Artist/Album/file_spec.npy)
    - 'combined': Loads 'fma', 'maestro', 'dali_synth', and 'labeled' datasets simultaneously
    """
    def __init__(self, spec_dir, label_dir, logits_dir, chord_mapping, seq_len, stride, frame_duration,
                 verbose=False, device=None, pin_memory=False, prefetch_factor=1, num_workers=0,
                 require_teacher_logits=False, use_cache=True, metadata_only=False, cache_fraction=0.1,
                 lazy_init=False, batch_gpu_cache=False, small_dataset_percentage=None, dataset_type='fma',
                 cache_dir=None, return_skip_reason=False): # Added return_skip_reason
        """
        Initialize the dataset with optimized settings for GPU acceleration.

        Args:
            spec_dir: Directory containing spectrograms (or list of directories for 'combined' type)
            label_dir: Directory containing labels (or list of directories for 'combined' type)
            logits_dir: Directory containing teacher logits (or list of directories for 'combined' type)
            chord_mapping: Mapping of chord names to indices
            seq_len: Sequence length for segmentation
            stride: Stride for segmentation (default: same as seq_len)
            frame_duration: Duration of each frame in seconds
            verbose: Whether to print verbose output
            device: Device to use (default: auto-detect)
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch (for DataLoader)
            num_workers: Number of workers for data loading
            require_teacher_logits: Whether to require teacher logits
            use_cache: Whether to use caching
            metadata_only: Whether to cache only metadata
            cache_fraction: Fraction of samples to cache
            lazy_init: Whether to use lazy initialization
            batch_gpu_cache: Whether to cache batches on GPU for repeated access patterns
            small_dataset_percentage: Optional percentage of the dataset to use (0-1.0)
            dataset_type: Type of dataset format ('fma', 'maestro', or 'combined')
            cache_dir: Directory to save cache files
            return_skip_reason: Whether to return the reason for skipping files
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

        # --- Moved Cache File Logic Earlier ---
        # Generate a safer cache file name using hashing if cache_dir is provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_key_parts = [str(p) for p in self.spec_dirs + self.label_dirs]
            if self.logits_dirs: cache_key_parts.extend([str(p) for p in self.logits_dirs])
            cache_key_parts.extend([str(seq_len), str(stride), str(frame_duration), dataset_type])
            cache_key = "_".join(cache_key_parts)
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            self.cache_file = os.path.join(cache_dir, f"dataset_cache_{dataset_type}_{cache_hash}.pkl")
            if verbose:
                print(f"Using cache file: {self.cache_file}")
        else:
            self.cache_file = None # No cache file if cache_dir is not provided
            if verbose:
                print("Cache directory not specified, caching disabled.")
        # --- End Moved Cache File Logic ---

        # Initialize basic parameters
        self.chord_mapping = chord_mapping
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.frame_duration = frame_duration
        self.samples = []
        self.segment_indices = []
        self.verbose = verbose
        # Use self.cache_file which is now defined
        self.use_cache = use_cache and self.cache_file is not None
        self.metadata_only = metadata_only  # Only cache metadata, not full spectrograms
        self.cache_fraction = cache_fraction  # Fraction of samples to cache (default: 10%)
        self.lazy_init = lazy_init
        self.require_teacher_logits = require_teacher_logits
        self.dataset_type = dataset_type  # Dataset format type

        # Add 'dali_synth' and 'labeled' to valid types
        # Define known types
        self.known_types = ['fma', 'maestro', 'dali_synth', 'labeled']
        valid_types = self.known_types + ['combined'] # Add 'combined' as a valid meta-type

        # Determine types to load
        self.types_to_load = []
        raw_types = []
        if self.dataset_type == 'combined':
            raw_types = self.known_types # Load all known types
        elif '+' in self.dataset_type:
            raw_types = self.dataset_type.split('+')
        else:
            raw_types = [self.dataset_type]

        # Validate and store types to load
        for t in raw_types:
            if t in self.known_types:
                self.types_to_load.append(t)
            else:
                warnings.warn(f"Unknown dataset type '{t}' specified, ignoring.")

        if not self.types_to_load:
             warnings.warn(f"No valid dataset types specified ('{dataset_type}'). Defaulting to 'fma'.")
             self.types_to_load = ['fma']
             self.dataset_type = 'fma' # Update main type if defaulted

        if verbose:
            print(f"Attempting to load dataset types: {self.types_to_load}")


        # Disable pin_memory since we're using a single worker
        self.pin_memory = True
        if pin_memory and verbose:
            print("Disabling pin_memory since we're using a single worker")

        self.prefetch_factor = prefetch_factor
        self.batch_gpu_cache = batch_gpu_cache
        self.small_dataset_percentage = small_dataset_percentage

        # Map from chord name to index
        if self.chord_mapping is not None:
            self.chord_to_idx = self.chord_mapping.copy()  # Make a copy to avoid modifying the original

            # Add plain note names (C, D, etc.) as aliases for major chords (C:maj, D:maj)
            # This ensures compatibility with both formats
            for root in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                maj_chord = f"{root}:maj"
                if maj_chord in self.chord_to_idx and root not in self.chord_to_idx:
                    self.chord_to_idx[root] = self.chord_to_idx[maj_chord]
                    if self.verbose and root == 'C':  # Only log once to avoid spam
                        print(f"Added plain note mapping: {root} -> {self.chord_to_idx[root]} (same as {maj_chord})")

                # Also add the reverse mapping if needed
                if root in self.chord_to_idx and maj_chord not in self.chord_to_idx:
                    self.chord_to_idx[maj_chord] = self.chord_to_idx[root]
                    if self.verbose and root == 'C':  # Only log once to avoid spam
                        print(f"Added explicit major mapping: {maj_chord} -> {self.chord_to_idx[maj_chord]} (same as {root})")
        else:
            self.chord_to_idx = {}

        # Set up regex patterns - always define all patterns regardless of dataset type
        # For Maestro/DALI: Match hex ID or general filename before _spec/_logits or .lab
        self.file_pattern = re.compile(r'([0-9a-fA-F]{32}|.+?)(?:_spec|_logits)?\.(?:npy|lab)$')
        # For FMA: 6-digit numeric ID pattern (match anywhere in filename for flexibility)
        self.numeric_id_pattern = re.compile(r'(\d{6})')
        # For DALI prefix: 3 alphanumeric chars
        self.dali_prefix_pattern = re.compile(r'^[0-9a-zA-Z]{3}$')
        # For LabeledDataset: General pattern to capture filename before extension
        self.labeled_file_pattern = re.compile(r'(.+)\.(lab|svl|txt)$') # Capture base name from label file
        self.labeled_spec_pattern = re.compile(r'(.+)_spec\.npy$') # Capture base name from spec file
        self.labeled_logit_pattern = re.compile(r'(.+)_logits\.npy$') # Capture base name from logit file

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
            # For lazy initialization, we still need to load minimal data to set up segments
            self._load_data()
            self._generate_segments()

        # Split data for train/eval/test
        total_segs = len(self.segment_indices)
        if total_segs == 0:
            if verbose:
                print("WARNING: No segments found. Check your data paths and make sure files exist.")
            # Create dummy indices to prevent errors
            self.train_indices = []
            self.eval_indices = []
            self.test_indices = []
        else:
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

    def _load_data(self):
        """Load data from files or cache with optimized memory usage and error handling"""
        start_time = time.time()
        logger = self.verbose # Use self.verbose for logging control

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

        # Create mappings of label and logit files for quick lookup
        label_files_dict = {}
        logit_files_dict = {}
        # Initialize counts for all potential types, even if not used in this run
        label_counts = {t: 0 for t in self.known_types}
        logit_counts = {t: 0 for t in self.known_types}
        spec_counts = {t: 0 for t in self.known_types}
        valid_spec_files: List[Tuple[Path, str, str]] = [] # (path, normalized_id, type)

        # --- Scan Files Based on Types to Load ---
        if logger: print(f"\n--- Scanning Files for Types: {self.types_to_load} ---")

        # --- Scan Label Directories ---
        if logger: print("\nScanning Label Directories...")
        for label_dir_base in self.label_dirs:
            if not label_dir_base.exists():
                warnings.warn(f"Label directory does not exist: {label_dir_base}")
                continue
            if logger: print(f"Scanning: {label_dir_base}")

            # Labeled Type Logic
            if 'labeled' in self.types_to_load:
                label_root_parent = label_dir_base.parent
                if logger: print(f"  Checking for 'labeled' structure (root: {label_root_parent})")
                files_found, ids_added = 0, 0
                try:
                    for label_path in label_dir_base.rglob("*.lab"):
                        files_found += 1
                        try:
                            rel_id_raw = get_rel_id(label_path, label_root_parent, ext_to_remove=".lab")
                            modified_id_raw = rel_id_raw
                            if '/' in rel_id_raw:
                                parts = rel_id_raw.split('/', 1)
                                if parts[0].lower().endswith('labels'):
                                    modified_id_raw = f"{parts[0][:-len('Labels')]}/{parts[1]}"

                            file_id = _normalize_id(modified_id_raw)
                            if file_id not in label_files_dict: # Avoid overwriting if found by other types
                                label_files_dict[file_id] = label_path
                                label_counts['labeled'] += 1
                                ids_added += 1
                            elif logger and ids_added < 5: # Log if already exists
                                 print(f"    [Labeled Scan] ID '{file_id}' already exists, skipping.")

                        except ValueError: continue # Skip if not relative to root_parent
                        except Exception as e_gen:
                             if logger: print(f"    Error processing labeled label {label_path}: {e_gen}")
                    if logger: print(f"    -> Found {files_found} .lab, added {ids_added} 'labeled' entries.")
                except Exception as e:
                     if logger: print(f"    Error during 'labeled' scan in {label_dir_base}: {e}")


            # FMA Type Logic
            if 'fma' in self.types_to_load:
                if logger: print("  Checking for 'fma' structure (ddd/*.lab)")
                files_found, ids_added = 0, 0
                try:
                    for prefix_dir in label_dir_base.glob("**/"): # Check subdirs
                        if re.fullmatch(r'\d{3}', prefix_dir.name):
                            for label_path in prefix_dir.glob("*.lab"):
                                files_found += 1
                                numeric_match = self.numeric_id_pattern.search(label_path.name)
                                if numeric_match:
                                    file_id_raw = numeric_match.group(1)
                                    file_id = _normalize_id(file_id_raw)
                                    if file_id not in label_files_dict:
                                        label_files_dict[file_id] = label_path
                                        label_counts['fma'] += 1
                                        ids_added += 1
                                    elif logger and ids_added < 5:
                                         print(f"    [FMA Scan] ID '{file_id}' already exists, skipping.")
                    if logger: print(f"    -> Found {files_found} .lab, added {ids_added} 'fma' entries.")
                except Exception as e:
                     if logger: print(f"    Error during 'fma' scan in {label_dir_base}: {e}")

            # Maestro & DALI Type Logic (General Pattern)
            other_types = [t for t in ['maestro', 'dali_synth'] if t in self.types_to_load]
            if other_types:
                if logger: print(f"  Checking for {'/'.join(other_types)} structure (*.lab)")
                files_found, ids_added_m, ids_added_d = 0, 0, 0
                try:
                    for label_path in label_dir_base.rglob("*.lab"):
                        files_found += 1
                        match = self.file_pattern.search(label_path.name)
                        if match:
                            file_id_raw = match.group(1)
                            file_id = _normalize_id(file_id_raw)
                            current_type = None
                            # Heuristic: Check if ID looks like DALI hex
                            if 'dali_synth' in other_types and re.fullmatch(r'[0-9a-fA-F]{32}', file_id_raw):
                                current_type = 'dali_synth'
                            elif 'maestro' in other_types: # Assume maestro otherwise
                                current_type = 'maestro'

                            if current_type and file_id not in label_files_dict:
                                label_files_dict[file_id] = label_path
                                label_counts[current_type] += 1
                                if current_type == 'maestro': ids_added_m += 1
                                else: ids_added_d += 1
                            elif logger and (ids_added_m + ids_added_d < 5):
                                 print(f"    [M/D Scan] ID '{file_id}' already exists or type mismatch, skipping.")
                    if logger: print(f"    -> Found {files_found} .lab, added {ids_added_m} 'maestro', {ids_added_d} 'dali_synth' entries.")
                except Exception as e:
                     if logger: print(f"    Error during 'maestro/dali' scan in {label_dir_base}: {e}")


        # --- Scan Logit Directories ---
        if logger: print("\nScanning Logit Directories...")
        if self.logits_dirs:
            for logit_dir_base in self.logits_dirs:
                if not logit_dir_base.exists():
                    warnings.warn(f"Logit directory does not exist: {logit_dir_base}")
                    continue
                if logger: print(f"Scanning: {logit_dir_base}")

                # Labeled Type Logic
                if 'labeled' in self.types_to_load:
                    logit_root = logit_dir_base # Assume this is the root like LabeledDataset_synth/logits
                    if logger: print(f"  Checking for 'labeled' structure (root: {logit_root})")
                    files_found, ids_added = 0, 0
                    try:
                        for logit_path in logit_dir_base.rglob("*_logits.npy"):
                            files_found += 1
                            try:
                                rel_id_raw = get_rel_id(logit_path, logit_root, suffix_to_remove="_logits", ext_to_remove=".npy")
                                file_id = _normalize_id(rel_id_raw)
                                if file_id not in logit_files_dict:
                                    logit_files_dict[file_id] = logit_path
                                    logit_counts['labeled'] += 1
                                    ids_added += 1
                                elif logger and ids_added < 5:
                                     print(f"    [Labeled Scan] Logit ID '{file_id}' already exists, skipping.")
                            except ValueError: continue
                            except Exception as e_gen:
                                 if logger: print(f"    Error processing labeled logit {logit_path}: {e_gen}")
                        if logger: print(f"    -> Found {files_found} logits, added {ids_added} 'labeled' entries.")
                    except Exception as e:
                         if logger: print(f"    Error during 'labeled' logit scan in {logit_dir_base}: {e}")

                # FMA Type Logic
                if 'fma' in self.types_to_load:
                    if logger: print("  Checking for 'fma' structure (ddd/*_logits.npy)")
                    files_found, ids_added = 0, 0
                    try:
                        for prefix_dir in logit_dir_base.glob("**/"):
                            if re.fullmatch(r'\d{3}', prefix_dir.name):
                                for logit_path in prefix_dir.glob("*_logits.npy"):
                                    files_found += 1
                                    # Match ID before _logits.npy
                                    match = re.search(r'(\d{6})_logits\.npy$', logit_path.name)
                                    if match:
                                        file_id_raw = match.group(1)
                                        file_id = _normalize_id(file_id_raw)
                                        if file_id not in logit_files_dict:
                                            logit_files_dict[file_id] = logit_path
                                            logit_counts['fma'] += 1
                                            ids_added += 1
                                        elif logger and ids_added < 5:
                                             print(f"    [FMA Scan] Logit ID '{file_id}' already exists, skipping.")
                        if logger: print(f"    -> Found {files_found} logits, added {ids_added} 'fma' entries.")
                    except Exception as e:
                        if logger: print(f"    Error during 'fma' logit scan in {logit_dir_base}: {e}")

                # Maestro & DALI Type Logic
                other_types = [t for t in ['maestro', 'dali_synth'] if t in self.types_to_load]
                if other_types:
                    if logger: print(f"  Checking for {'/'.join(other_types)} structure (*_logits.npy)")
                    files_found, ids_added_m, ids_added_d = 0, 0, 0
                    try:
                        for logit_path in logit_dir_base.rglob("*_logits.npy"):
                            files_found += 1
                            # Match ID before _logits.npy
                            match = re.search(r'([0-9a-fA-F]{32}|.+?)_logits\.npy$', logit_path.name)
                            if match:
                                file_id_raw = match.group(1)
                                file_id = _normalize_id(file_id_raw)
                                current_type = None
                                if 'dali_synth' in other_types and re.fullmatch(r'[0-9a-fA-F]{32}', file_id_raw):
                                    current_type = 'dali_synth'
                                elif 'maestro' in other_types:
                                    current_type = 'maestro'

                                if current_type and file_id not in logit_files_dict:
                                    logit_files_dict[file_id] = logit_path
                                    logit_counts[current_type] += 1
                                    if current_type == 'maestro': ids_added_m += 1
                                    else: ids_added_d += 1
                                elif logger and (ids_added_m + ids_added_d < 5):
                                     print(f"    [M/D Scan] Logit ID '{file_id}' already exists or type mismatch, skipping.")
                        if logger: print(f"    -> Found {files_found} logits, added {ids_added_m} 'maestro', {ids_added_d} 'dali_synth' entries.")
                    except Exception as e:
                        if logger: print(f"    Error during 'maestro/dali' logit scan in {logit_dir_base}: {e}")
        elif self.require_teacher_logits:
             raise ValueError("require_teacher_logits=True but no logits_dirs provided.")


        # --- Scan Spectrogram Directories ---
        if logger: print("\nScanning Spectrogram Directories...")
        temp_spec_dict: Dict[str, Tuple[Path, str]] = {} # Use dict to handle potential duplicates across scans: {norm_id: (path, type)}

        for spec_dir_base in self.spec_dirs:
            if not spec_dir_base.exists():
                warnings.warn(f"Spectrogram directory does not exist: {spec_dir_base}")
                continue
            if logger: print(f"Scanning: {spec_dir_base}")

            # Labeled Type Logic
            if 'labeled' in self.types_to_load:
                spec_root = spec_dir_base # Assume this is the root like LabeledDataset_synth/spectrograms
                if logger: print(f"  Checking for 'labeled' structure (root: {spec_root})")
                files_found, ids_added = 0, 0
                try:
                    for spec_path in spec_dir_base.rglob("*_spec.npy"):
                        files_found += 1
                        try:
                            rel_id_raw = get_rel_id(spec_path, spec_root, suffix_to_remove="_spec", ext_to_remove=".npy")
                            file_id = _normalize_id(rel_id_raw)
                            if file_id not in temp_spec_dict:
                                temp_spec_dict[file_id] = (spec_path, 'labeled')
                                spec_counts['labeled'] += 1
                                ids_added += 1
                            elif logger and ids_added < 5:
                                 print(f"    [Labeled Scan] Spec ID '{file_id}' already exists, skipping.")
                        except ValueError: continue
                        except Exception as e_gen:
                             if logger: print(f"    Error processing labeled spec {spec_path}: {e_gen}")
                    if logger: print(f"    -> Found {files_found} specs, added {ids_added} 'labeled' entries.")
                except Exception as e:
                     if logger: print(f"    Error during 'labeled' spec scan in {spec_dir_base}: {e}")

            # FMA Type Logic
            if 'fma' in self.types_to_load:
                if logger: print("  Checking for 'fma' structure (ddd/*_spec.npy)")
                files_found, ids_added = 0, 0
                try:
                    for prefix_dir in spec_dir_base.glob("**/"):
                        if re.fullmatch(r'\d{3}', prefix_dir.name):
                            for spec_path in prefix_dir.glob("*_spec.npy"):
                                files_found += 1
                                match = re.search(r'(\d{6})_spec\.npy$', spec_path.name)
                                if match:
                                    file_id_raw = match.group(1)
                                    file_id = _normalize_id(file_id_raw)
                                    if file_id not in temp_spec_dict:
                                        temp_spec_dict[file_id] = (spec_path, 'fma')
                                        spec_counts['fma'] += 1
                                        ids_added += 1
                                    elif logger and ids_added < 5:
                                         print(f"    [FMA Scan] Spec ID '{file_id}' already exists, skipping.")
                    if logger: print(f"    -> Found {files_found} specs, added {ids_added} 'fma' entries.")
                except Exception as e:
                     if logger: print(f"    Error during 'fma' spec scan in {spec_dir_base}: {e}")

            # Maestro & DALI Type Logic
            other_types = [t for t in ['maestro', 'dali_synth'] if t in self.types_to_load]
            if other_types:
                if logger: print(f"  Checking for {'/'.join(other_types)} structure (*_spec.npy)")
                files_found, ids_added_m, ids_added_d = 0, 0, 0
                try:
                    for spec_path in spec_dir_base.rglob("*_spec.npy"):
                        files_found += 1
                        match = re.search(r'([0-9a-fA-F]{32}|.+?)_spec\.npy$', spec_path.name)
                        if match:
                            file_id_raw = match.group(1)
                            file_id = _normalize_id(file_id_raw)
                            current_type = None
                            if 'dali_synth' in other_types and re.fullmatch(r'[0-9a-fA-F]{32}', file_id_raw):
                                current_type = 'dali_synth'
                            elif 'maestro' in other_types:
                                current_type = 'maestro'

                            if current_type and file_id not in temp_spec_dict:
                                temp_spec_dict[file_id] = (spec_path, current_type)
                                spec_counts[current_type] += 1
                                if current_type == 'maestro': ids_added_m += 1
                                else: ids_added_d += 1
                            elif logger and (ids_added_m + ids_added_d < 5):
                                 print(f"    [M/D Scan] Spec ID '{file_id}' already exists or type mismatch, skipping.")
                    if logger: print(f"    -> Found {files_found} specs, added {ids_added_m} 'maestro', {ids_added_d} 'dali_synth' entries.")
                except Exception as e:
                     if logger: print(f"    Error during 'maestro/dali' spec scan in {spec_dir_base}: {e}")

        # Convert temp_spec_dict to valid_spec_files list
        valid_spec_files = [(path, norm_id, type) for norm_id, (path, type) in temp_spec_dict.items()]
        # --- END REPLACEMENT ---

        # --- Logging and further processing ---
        if logger:
            print(f"\nFound {len(label_files_dict)} label files across all directories:")
            # Filter counts for active types or non-zero counts
            active_label_counts = {k: v for k, v in label_counts.items() if v > 0} # Show only non-zero counts
            for k, v in active_label_counts.items(): print(f"  {k.upper()}: {v}")
            if len(label_files_dict) > 0:
                print("Sample label file paths (ID -> Path):")
                # Now the ID should match the feature IDs, e.g., 'billboard/artist/album/song'
                for i, (file_id, label_path) in enumerate(list(label_files_dict.items())[:3]):
                    print(f"  ID: '{file_id}' -> Path: {label_path}")
            if self.logits_dirs:
                print(f"\nFound {len(logit_files_dict)} logit files across all directories:")
                active_logit_counts = {k: v for k, v in logit_counts.items() if v > 0} # Show only non-zero counts
                for k, v in active_logit_counts.items(): print(f"  {k.upper()}: {v}")
                if len(logit_files_dict) > 0:
                    print("Sample logit file paths (ID -> Path):")
                    for i, (file_id, logit_path) in enumerate(list(logit_files_dict.items())[:3]):
                         print(f"  ID: '{file_id}' -> Path: {logit_path}")

        # --- Process found spectrogram files ---
        if not valid_spec_files:
            warnings.warn(f"No valid spectrogram files found for dataset type(s) '{self.types_to_load}'. Check data paths.")
            self.samples = []
            self.segment_indices = []
            return

        if self.verbose:
            print(f"\nFound {len(valid_spec_files)} valid spectrogram files:")
            active_spec_counts = {k: v for k, v in spec_counts.items() if v > 0} # Show only non-zero counts
            for k, v in active_spec_counts.items(): print(f"  {k.upper()}: {v}")
            if valid_spec_files:
                print("Sample spectrogram paths:")
                for i, (path, file_id, type) in enumerate(valid_spec_files[:3]):
                    print(f"  ({type.upper()}) ID: '{file_id}' -> Path: {path}")

        # Handle small dataset percentage option
        if self.small_dataset_percentage is not None:
            np.random.seed(42) # Ensure consistent sampling

            # Group files by type ('fma', 'maestro', 'dali_synth', 'labeled')
            dataset_files = {t: [] for t in self.known_types}
            for spec_path, file_id, file_type in valid_spec_files:
                 if file_type in dataset_files: # Should always be true now
                     dataset_files[file_type].append((spec_path, file_id, file_type))
                 else: # Fallback just in case
                      if logger: print(f"Warning: Encountered unexpected file_type '{file_type}' during sampling.")


            sampled_files = []
            total_sampled_count = 0
            # Use the actual types found in valid_spec_files keys
            active_file_types = [ft for ft in dataset_files if dataset_files[ft]]
            if not active_file_types and self.verbose:
                print("No files found to sample from.")

            for file_type in active_file_types:
                files = dataset_files[file_type]
                # Calculate sample size based on the number of files of this specific type
                type_sample_size = max(1, int(len(files) * self.small_dataset_percentage))

                if type_sample_size < len(files):
                    indices = np.random.choice(len(files), type_sample_size, replace=False)
                    sampled_subset = [files[i] for i in indices]
                    if self.verbose:
                        print(f"Sampling {type_sample_size}/{len(files)} files for {file_type.upper()}")
                else:
                    sampled_subset = files # Use all if sample size >= total
                    if self.verbose:
                        print(f"Using all {len(files)} files for {file_type.upper()} (small_dataset_percentage)")

                sampled_files.extend(sampled_subset)
                total_sampled_count += len(sampled_subset)

            valid_spec_files = sampled_files # Update valid_spec_files with the sampled list
            if self.verbose:
                print(f"Total files after sampling: {total_sampled_count}")


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

        # Reduce workers if dataset is very small compared to worker count
        if len(valid_spec_files) > 0 and len(valid_spec_files) < num_cpus * 4:
            num_cpus = max(1, len(valid_spec_files) // 2) # Adjust num_cpus based on actual file count
            if self.verbose:
                print(f"Small dataset detected ({len(valid_spec_files)} files), reducing worker count to {num_cpus}")

        # Prepare arguments for parallel processing
        # Pass the dataset type ('labeled' in this case) to _process_file
        current_dataset_type = self.dataset_type # Or determine based on file if combined
        # Pass the specific type determined during scanning
        args_list = [(self, spec_file, file_id, label_files_dict, logit_files_dict, file_type, True)
                     for spec_file, file_id, file_type in valid_spec_files] # Use file_id and file_type from valid_spec_files

        if self.verbose:
            print(f"Processing {len(args_list)} files with {num_cpus} parallel workers")

        try:
            # Use multiprocessing Pool
            with Pool(processes=num_cpus) as pool:
                # Use imap for potentially better memory usage with large iterables
                process_results = list(tqdm(
                    pool.imap(process_file_wrapper, args_list),
                    total=len(args_list),
                    desc=f"Loading data (parallel {'lazy' if self.lazy_init else 'full'})",
                    disable=not self.verbose # Disable tqdm if not verbose
                ))

            # Process results from the pool
            for result in process_results:
                 # Check if the result is the expected tuple (samples, skip_reason)
                 if isinstance(result, tuple) and len(result) == 2:
                     samples, skip_reason = result
                     self.total_processed += 1
                     if samples: # If samples were successfully processed
                         self.samples.extend(samples)
                     else: # If processing resulted in skipping the file
                         self.total_skipped += 1
                         if skip_reason in self.skipped_reasons:
                             self.skipped_reasons[skip_reason] += 1
                         # Optional: Log skip reason if needed, even if return_skip_reason handled it
                         # elif self.verbose:
                         #     print(f"File skipped, reason: {skip_reason}")
                 else:
                     # Handle unexpected result format if necessary
                     if self.verbose:
                         print(f"Warning: Unexpected result format from process_file_wrapper: {result}")


        except Exception as e:
            # Fallback to sequential processing if multiprocessing fails
            import traceback
            error_msg = traceback.format_exc()
            if self.verbose:
                print(f"ERROR in multiprocessing: {e}")
                print(f"Traceback:\n{error_msg}")
                print(f"Attempting fallback to sequential processing...")

            # Sequential processing loop
            process_results = []
            for args in tqdm(args_list, desc="Loading data (sequential fallback)", disable=not self.verbose):
                try:
                    # Call the wrapper function directly
                    result = process_file_wrapper(args)
                    process_results.append(result)
                except Exception as seq_e:
                    if self.verbose:
                        print(f"Error during sequential processing of {args[1]}: {seq_e}") # Log error for specific file
                    # Append a skip result to maintain structure
                    process_results.append(([], 'load_error')) # Assume load error

            # Process results from sequential execution
            for result in process_results:
                 if isinstance(result, tuple) and len(result) == 2:
                     samples, skip_reason = result
                     self.total_processed += 1 # Increment even if skipped
                     if samples:
                         self.samples.extend(samples)
                     else:
                         self.total_skipped += 1
                         if skip_reason in self.skipped_reasons:
                             self.skipped_reasons[skip_reason] += 1
                         elif skip_reason: # Add reason if it's new
                             # Check if key exists before creating
                             if skip_reason not in self.skipped_reasons:
                                 self.skipped_reasons[skip_reason] = 0
                             self.skipped_reasons[skip_reason] += 1
                 else:
                     if self.verbose:
                         print(f"Warning: Unexpected result format during sequential fallback: {result}")


        # --- Post-processing statistics ---
        if hasattr(self, 'total_processed') and self.total_processed > 0:
            skip_percentage = (self.total_skipped / self.total_processed) * 100 if self.total_processed > 0 else 0
            if self.verbose:
                print(f"\nFile processing statistics:")
                print(f"  Total files attempted: {self.total_processed}")
                print(f"  Successfully processed: {self.total_processed - self.total_skipped}")
                print(f"  Skipped: {self.total_skipped} ({skip_percentage:.1f}%)")
                if hasattr(self, 'skipped_reasons') and self.total_skipped > 0:
                    print("  Reasons for skipping:")
                    # Sort reasons by count for clarity
                    sorted_reasons = sorted(self.skipped_reasons.items(), key=lambda item: item[1], reverse=True)
                    for reason, count in sorted_reasons:
                        if count > 0:
                            reason_pct = (count / self.total_skipped) * 100
                            print(f"    - {reason}: {count} ({reason_pct:.1f}%)")

        # --- Caching ---
        if self.samples and self.use_cache and self.cache_file:
            try:
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir: # Ensure cache_dir is not empty
                    os.makedirs(cache_dir, exist_ok=True) # Create cache directory if it doesn't exist

                samples_to_cache = self.samples # Use the final list of samples

                # --- Metadata Caching Logic ---
                if self.metadata_only:
                    samples_meta = []
                    for sample in samples_to_cache:
                        # Create a copy excluding 'spectro' if present
                        meta = {k: v for k, v in sample.items() if k != 'spectro'}
                        # Ensure 'spec_path' exists, potentially deriving it if needed (though it should be there from processing)
                        if 'spec_path' not in meta and 'song_id' in sample:
                             # This fallback might be needed if _process_file didn't add spec_path in some case
                             # Reconstruct potential path (this depends heavily on consistent ID structure)
                             # Example assumes spec_root is defined and file_id matches relative structure
                             # --- This reconstruction is complex with multiple types, rely on spec_path being present ---
                             # if spec_root and 'song_id' in sample:
                             #     potential_path = spec_root / f"{sample['song_id']}_spec.npy"
                             #     if potential_path.exists():
                             #         meta['spec_path'] = str(potential_path)
                             #     elif self.verbose:
                             #         print(f"Warning: Could not confirm spec_path for metadata caching: {sample.get('song_id')}")
                             if self.verbose:
                                 print(f"Warning: 'spec_path' missing in metadata for sample ID {sample.get('song_id')}, cannot cache path.")
                        elif 'spec_path' in meta and not Path(meta['spec_path']).exists():
                             if self.verbose:
                                 print(f"Warning: spec_path in metadata does not exist: {meta['spec_path']}")

                        samples_meta.append(meta)

                    cache_content = {
                        'samples': samples_meta,
                        'chord_to_idx': self.chord_to_idx,
                        'metadata_only': True,
                        'cache_fraction': self.cache_fraction, # Store cache fraction used
                        'small_dataset_percentage': self.small_dataset_percentage # Store percentage used
                    }
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(cache_content, f)
                    if self.verbose:
                        print(f"Saved METADATA cache ({len(samples_meta)} items) to {self.cache_file}")

                # --- Full Data Caching Logic ---
                else:
                    # Optionally apply cache_fraction here if needed (e.g., random sample)
                    # Currently, it caches all processed samples if metadata_only is False
                    cache_content = {
                        'samples': samples_to_cache,
                        'chord_to_idx': self.chord_to_idx,
                        'metadata_only': False,
                        'cache_fraction': self.cache_fraction,
                        'small_dataset_percentage': self.small_dataset_percentage
                    }
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(cache_content, f)
                    if self.verbose:
                        print(f"Saved FULL dataset cache ({len(samples_to_cache)} items) to {self.cache_file}")

                # Log cache details
                if self.verbose:
                    if self.small_dataset_percentage is not None:
                        print(f"Cache reflects small_dataset_percentage={self.small_dataset_percentage}")
                    if self.cache_fraction < 1.0:
                         print(f"Cache fraction applied: {self.cache_fraction}") # Add if fraction logic is implemented

            except Exception as e:
                if self.verbose:
                    print(f"Error saving cache (continuing without caching): {e}")
                    print(traceback.format_exc()) # Print full traceback for cache saving error


        # --- Final Sample Analysis ---
        if self.samples:
            first_sample = self.samples[0]
            freq_dim = 144 # Default frequency dimension

            # Determine frequency dimension from the first sample
            if 'spectro' in first_sample:
                # If spectrogram is loaded in memory
                spec_data = first_sample['spectro']
                if hasattr(spec_data, 'shape') and len(spec_data.shape) > 0:
                    freq_dim = spec_data.shape[-1] # Get last dimension size
            elif 'spec_path' in first_sample and Path(first_sample['spec_path']).exists():
                # If only path is stored (metadata or lazy loading)
                try:
                    # Load just the shape using mmap_mode if possible, or load the first frame
                    spec_info = np.load(first_sample['spec_path'], mmap_mode='r')
                    if len(spec_info.shape) > 1: # Multi-frame spec
                        freq_dim = spec_info.shape[-1]
                    elif len(spec_info.shape) == 1: # Single-frame spec
                        freq_dim = spec_info.shape[0]
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Error loading first spectrogram shape from {first_sample['spec_path']}: {e}")
                        print("Using default frequency dimension of 144")
            else:
                if self.verbose:
                    print("WARNING: Could not determine spectrogram shape from first sample (no 'spectro' or valid 'spec_path').")
                    print("Using default frequency dimension of 144")

            spec_type = "CQT (Constant-Q Transform)" if freq_dim <= 256 else "STFT"

            if self.verbose:
                print(f"\nLoaded {len(self.samples)} valid samples")
                print(f"Spectrogram frequency dimension: {freq_dim} (likely {spec_type})")

                # Analyze chord distribution
                chord_counter = Counter(sample['chord_label'] for sample in self.samples)
                print(f"Found {len(chord_counter)} unique chord classes")

                # Chord quality analysis (requires utility function)
                try:
                    from modules.utils.chords import get_chord_quality
                    quality_counter = Counter()
                    for sample in self.samples:
                        quality = get_chord_quality(sample['chord_label'])
                        quality_counter[quality] += 1

                    total_samples = len(self.samples)
                    print(f"\nChord quality distribution:")
                    for quality, count in quality_counter.most_common():
                        percentage = (count / total_samples) * 100
                        print(f"  {quality}: {count} samples ({percentage:.2f}%)")
                except ImportError:
                    if self.verbose:
                        print("\nNote: 'get_chord_quality' not found, skipping quality distribution analysis.")
                except Exception as e:
                     if self.verbose:
                         print(f"\nError during chord quality analysis: {e}")


                # Print most common chords
                print("\nMost common chord types (Top 20):")
                total_samples = len(self.samples) # Recalculate or ensure it's available
                for chord, count in chord_counter.most_common(20):
                    percentage = (count / total_samples) * 100
                    print(f"  {chord}: {count} samples ({percentage:.2f}%)")

                # Print some less common chords (e.g., 100th to 120th)
                print("\nSome less common chord types (100-120):")
                less_common = chord_counter.most_common()[100:120]
                if less_common:
                    for chord, count in less_common:
                        percentage = (count / total_samples) * 100
                        print(f"  {chord}: {count} samples ({percentage:.2f}%)")
                else:
                    print("  (Not enough unique chords to show less common ones in this range)")


                # Sample distribution by dataset source (if combined mode was used - adapt if needed)
                # This part might need adjustment based on how types are tracked with the new logic
                # if hasattr(self, 'is_combined_mode') and self.is_combined_mode:
                #     dataset_sample_counts = Counter()
                #     # Need a way to determine source type for each sample, e.g., store 'type' in sample dict
                #     for sample in self.samples:
                #         # Assuming 'type' was added during processing or can be inferred from path/ID
                #         sample_type = sample.get('dataset_type', 'unknown') # Example key
                #         dataset_sample_counts[sample_type] += 1
                #
                #     if self.verbose:
                #         print("\nSample distribution by dataset source:")
                #         for dataset_key, count in dataset_sample_counts.items():
                #             percentage = (count / total_samples) * 100
                #             print(f"  {dataset_key}: {count} samples ({percentage:.2f}%)")

        else:
            # Warning if no samples were loaded at all
            warnings.warn("No samples loaded after processing. Check data paths, file formats, and filtering criteria.")

        # Initialize the internal chord processor (if needed elsewhere)
        self.chord_processor = Chords()
        if self.chord_mapping:
            self.chord_processor.set_chord_mapping(self.chord_mapping)

        end_time = time.time()
        if self.verbose:
             print(f"\nDataset loading process completed in {end_time - start_time:.2f} seconds")

    # ... rest of the class methods (_process_file, _generate_segments, __len__, __getitem__, etc.) ...
    # Ensure _process_file uses the correct lookup_key and handles potential errors gracefully.
    # The _process_file method provided in the original code seems mostly compatible,
    # but double-check the dir_prefix logic for 'labeled' type.

    def _process_file(self, spec_file, file_id, label_files_dict, logit_files_dict, current_dataset_type, return_skip_reason=False):
        """Process a single spectrogram file based on dataset type"""
        samples = []
        skip_reason = None
        lookup_key = file_id # Use the normalized ID generated from spec scan

        # --- Minimal Debugging ---
        # Add a counter to limit verbose logging per file ID
        log_count_attr = f'_log_count_{lookup_key}'
        log_count = getattr(self, log_count_attr, 0)

        if self.verbose and log_count < 1: # Log only once per unique lookup_key attempt
             print(f"\n--- Processing spec: {spec_file.name} ---")
             print(f"  Lookup ID (from spec): '{lookup_key}'")
             if lookup_key in label_files_dict:
                 print(f"  ✅ Match found in label_files_dict.")
             else:
                 print(f"  ❌ Match NOT found in label_files_dict.")
                 # Log a few label keys for comparison
                 label_keys_sample = list(label_files_dict.keys())
                 print(f"     Sample label keys: {label_keys_sample[:5]}...") # Show more samples
             if self.logits_dirs:
                 if lookup_key in logit_files_dict:
                     print(f"  ✅ Match found in logit_files_dict.")
                 else:
                     print(f"  ❌ Match NOT found in logit_files_dict.")
                     logit_keys_sample = list(logit_files_dict.keys())
                     print(f"     Sample logit keys: {logit_keys_sample[:5]}...") # Show more samples
             setattr(self, log_count_attr, log_count + 1) # Increment log count
        # --- End Minimal Debugging ---

        try:
            # Determine dir_prefix based on type - adjust for 'labeled'
            dir_prefix = None
            # Use the directory part of the file_id (relative path) for 'labeled'
            if current_dataset_type == 'labeled':
                 # file_id for labeled is like 'billboard/artist/album/song'
                 dir_prefix = str(Path(lookup_key).parent) if '/' in lookup_key or '\\' in lookup_key else ''
            # Keep logic for other types if this function is used for them
            elif current_dataset_type == 'maestro':
                # Maestro IDs might not have inherent directory structure in the ID itself
                # Use parent dir of the spec file as a fallback, but might not be reliable if structure varies
                dir_prefix = spec_file.parent.name
            elif current_dataset_type == 'dali_synth':
                 # DALI IDs are hex, prefix is usually parent dir name (e.g., '00a') or first 3 chars of ID
                 dir_prefix = spec_file.parent.name
                 # Fallback using file_id if parent name doesn't match pattern
                 if not hasattr(self, 'dali_prefix_pattern') or not self.dali_prefix_pattern.match(dir_prefix):
                     dir_prefix = file_id[:3] if len(file_id) >= 3 else file_id # Use first 3 chars of hex ID
            elif current_dataset_type == 'fma': # FMA
                # FMA uses first 3 digits of the 6-digit ID
                dir_prefix = file_id[:3] if len(file_id) >= 3 else file_id
            else: # Fallback/Unknown
                 dir_prefix = ''


            # Find corresponding label file using the lookup_key
            label_file = label_files_dict.get(lookup_key)
            if not label_file or not os.path.exists(str(label_file)):
                # Check if the reason is already set to avoid redundant logging/counting
                if skip_reason != 'missing_label':
                    # if self.verbose: # Reduced verbosity for skips
                    #     print(f"  [SKIP] Label lookup failed for ID '{lookup_key}'. Label file path resolved to: {label_file}")
                    if hasattr(self, 'skipped_reasons'): # Ensure attribute exists
                        # Safely increment counter
                        self.skipped_reasons['missing_label'] = self.skipped_reasons.get('missing_label', 0) + 1
                    skip_reason = 'missing_label'
                if return_skip_reason:
                    return [], skip_reason
                return [] # Return empty list, not None

            # Find corresponding logit file if needed, using lookup_key
            logit_file = None
            if self.logits_dirs is not None:
                logit_file = logit_files_dict.get(lookup_key)

                if self.require_teacher_logits and (not logit_file or not os.path.exists(str(logit_file))):
                    if skip_reason != 'missing_logits': # Avoid double counting if label also missing
                        # if self.verbose: # Reduced verbosity for skips
                        #      print(f"  [SKIP] Required Logit lookup failed for ID '{lookup_key}'. Logit file path resolved to: {logit_file}")
                        if hasattr(self, 'skipped_reasons'):
                            self.skipped_reasons['missing_logits'] = self.skipped_reasons.get('missing_logits', 0) + 1
                        skip_reason = 'missing_logits'
                    if return_skip_reason:
                        return [], skip_reason
                    return [] # Return empty list

            # --- Metadata Only Logic ---
            if self.metadata_only:
                if os.path.exists(spec_file):
                    try:
                        # Use mmap_mode for potentially faster shape reading without loading full data
                        spec_info = np.load(spec_file, mmap_mode='r')
                        spec_shape = spec_info.shape
                        # Ensure spec_shape is usable
                        if not spec_shape: raise ValueError("Spectrogram shape is empty")

                        chord_labels = self._parse_label_file(label_file)
                        num_frames = spec_shape[0] if len(spec_shape) > 1 else 1

                        for t in range(num_frames):
                            frame_time = t * self.frame_duration
                            chord_label = self._find_chord_at_time(chord_labels, frame_time)
                            chord_label_mapped = chord_label # Keep original for now

                            # Map chord label using chord_mapping if available
                            if self.chord_mapping is not None:
                                if chord_label not in self.chord_mapping:
                                    # Handle plain root notes as major chords
                                    if chord_label in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                                        major_chord = f"{chord_label}:maj"
                                        if major_chord in self.chord_mapping:
                                            chord_label_mapped = major_chord
                                        else:
                                            # warnings.warn(f"Unknown chord label {chord_label} (even as major), using 'N'")
                                            chord_label_mapped = "N" # Map unknown to 'N'
                                    else:
                                        # warnings.warn(f"Unknown chord label {chord_label}, using 'N'")
                                        chord_label_mapped = "N" # Map unknown to 'N'
                                else:
                                     chord_label_mapped = chord_label # Already in mapping

                            # If no chord mapping provided, build one dynamically (less common now)
                            elif chord_label not in self.chord_to_idx:
                                self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                                chord_label_mapped = chord_label # Use original label

                            sample_meta = {
                                'spec_path': str(spec_file), # Store the actual path to the spec file
                                'chord_label': chord_label_mapped, # Store the potentially mapped label
                                'song_id': lookup_key, # Use normalized key
                                'frame_idx': t,
                                'dir_prefix': dir_prefix # Keep for potential use
                            }
                            # Add logit path if available
                            if logit_file is not None:
                                sample_meta['logit_path'] = str(logit_file)

                            samples.append(sample_meta)

                    except Exception as meta_e:
                         # Handle errors during metadata processing for this file
                         if skip_reason not in ['format_error', 'load_error']: # Avoid double counting
                             if hasattr(self, 'skipped_reasons'):
                                 self.skipped_reasons['format_error'] = self.skipped_reasons.get('format_error', 0) + 1
                             skip_reason = 'format_error'
                         # warnings.warn(f"Error processing metadata for {spec_file}: {meta_e}") # Reduce verbosity
                         if return_skip_reason: return [], skip_reason
                         return []

            # --- Full Data Logic ---
            else:
                try:
                    spec = np.load(spec_file)
                    # Check for NaN values after loading
                    if np.isnan(spec).any():
                        # warnings.warn(f"NaN values found in {spec_file}, replacing with zeros")
                        spec = np.nan_to_num(spec, nan=0.0) # Replace NaNs

                    chord_labels = self._parse_label_file(label_file)
                    num_frames = spec.shape[0] if len(spec.shape) > 1 else 1

                    # Load teacher logits once if needed and available
                    teacher_logits_full = None
                    if logit_file is not None:
                        teacher_logits_full = self._load_logits_file(logit_file)
                        # If loading failed and logits are required, _load_logits_file might raise error or return None
                        if teacher_logits_full is None and self.require_teacher_logits:
                             # This case should ideally be caught earlier, but double-check
                             if skip_reason != 'missing_logits':
                                 if hasattr(self, 'skipped_reasons'):
                                     self.skipped_reasons['missing_logits'] = self.skipped_reasons.get('missing_logits', 0) + 1
                                 skip_reason = 'missing_logits'
                             if return_skip_reason: return [], skip_reason
                             return []


                    for t in range(num_frames):
                        frame_time = t * self.frame_duration
                        chord_label = self._find_chord_at_time(chord_labels, frame_time)
                        chord_label_mapped = chord_label

                        # Map chord label using chord_mapping
                        if self.chord_mapping is not None:
                            if chord_label not in self.chord_mapping:
                                if chord_label in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                                    major_chord = f"{chord_label}:maj"
                                    if major_chord in self.chord_mapping:
                                        chord_label_mapped = major_chord
                                    else:
                                        chord_label_mapped = "N"
                                else:
                                    chord_label_mapped = "N"
                            # else: chord_label_mapped remains chord_label

                        elif chord_label not in self.chord_to_idx: # Dynamic mapping
                            self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                            # chord_label_mapped remains chord_label

                        # Prepare sample dictionary
                        sample_dict = {
                            'spectro': spec[t] if num_frames > 1 else spec, # Get frame or full spec
                            'chord_label': chord_label_mapped,
                            'song_id': lookup_key,
                            'dir_prefix': dir_prefix,
                            'frame_idx': t
                        }

                        # Add teacher logits for the current frame if available
                        if teacher_logits_full is not None:
                            if isinstance(teacher_logits_full, np.ndarray):
                                # Handle multi-frame or single-frame logits array
                                if len(teacher_logits_full.shape) > 1 and t < teacher_logits_full.shape[0]:
                                    sample_dict['teacher_logits'] = teacher_logits_full[t]
                                elif len(teacher_logits_full.shape) == 1 and num_frames == 1: # Single logit vector for single spec frame
                                     sample_dict['teacher_logits'] = teacher_logits_full
                                elif len(teacher_logits_full.shape) > 0 and num_frames > 1: # Need to decide how to handle mismatch
                                     # Option 1: Use first logit frame?
                                     # sample_dict['teacher_logits'] = teacher_logits_full[0]
                                     # Option 2: Skip or use zeros? For now, let's assume shape matches or use first frame if possible
                                     if teacher_logits_full.shape[0] > 0:
                                         sample_dict['teacher_logits'] = teacher_logits_full[0] # Fallback: use first logit frame
                                     # else: log warning?
                            # else: Logits might be in unexpected format

                        samples.append(sample_dict)

                except Exception as full_data_e:
                     # Handle errors during full data loading/processing for this file
                     if skip_reason not in ['format_error', 'load_error']:
                         if hasattr(self, 'skipped_reasons'):
                             # Prioritize format_error if applicable
                             err_str = str(full_data_e).lower()
                             if "format" in err_str or "corrupt" in err_str or "cannot load" in err_str:
                                 self.skipped_reasons['format_error'] = self.skipped_reasons.get('format_error', 0) + 1
                                 skip_reason = 'format_error'
                             else:
                                 self.skipped_reasons['load_error'] = self.skipped_reasons.get('load_error', 0) + 1
                                 skip_reason = 'load_error'
                     # warnings.warn(f"Error processing full data for {spec_file}: {full_data_e}") # Reduce verbosity
                     if return_skip_reason: return [], skip_reason
                     return []


        except Exception as e:
            # Catch-all for unexpected errors during file processing setup
            if skip_reason not in ['format_error', 'load_error']: # Avoid double counting
                if hasattr(self, 'skipped_reasons'):
                    self.skipped_reasons['load_error'] = self.skipped_reasons.get('load_error', 0) + 1
                skip_reason = 'load_error'

            # warnings.warn(f"Unexpected error processing file {spec_file}: {str(e)}") # Reduce verbosity
            # print(traceback.format_exc()) # Optionally print traceback for debugging

            if return_skip_reason:
                return [], skip_reason
            return [] # Return empty list

        # Return processed samples and the final skip reason (None if successful)
        if return_skip_reason:
            return samples, skip_reason
        # Ensure samples is always a list, even if empty
        return samples if samples else []

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

    def _get_data_iterator(self, indices, name, batch_size=128, shuffle=False, num_workers=None, pin_memory=None, sampler=None):
        """Helper method to get a data iterator for a specific subset of indices

        Args:
            indices: List of indices to use
            name: Name of the subset for warning message
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for DataLoader
            pin_memory: Whether to use pinned memory for DataLoader
            sampler: Optional sampler for distributed training

        Returns:
            DataLoader object
        """
        if not indices:
            warnings.warn(f"No {name} segments available")
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle if sampler is None else False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                sampler=sampler
            )

        return DataLoader(
            SynthSegmentSubset(self, indices),
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            sampler=sampler
        )

    def get_train_iterator(self, batch_size=128, shuffle=True, num_workers=None, pin_memory=None, sampler=None):
        """Get data iterator for training set"""
        return self._get_data_iterator(
            self.train_indices,
            "training",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler
        )

    def get_eval_iterator(self, batch_size=128, shuffle=False, num_workers=None, pin_memory=None, sampler=None):
        """Get data iterator for evaluation set"""
        return self._get_data_iterator(
            self.eval_indices,
            "evaluation",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler
        )

    def get_test_iterator(self, batch_size=128, shuffle=False, num_workers=None, pin_memory=None, sampler=None):
        """Get data iterator for test set"""
        return self._get_data_iterator(
            self.test_indices,
            "test",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler
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

        # Get the sample from the parent dataset
        sample = self.dataset[self.indices[idx]]

        # Always return the dictionary format for non-distributed training
        # This ensures compatibility with both distributed and non-distributed modes
        return sample