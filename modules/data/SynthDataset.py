import torch
import numpy as np
import os
import re
import time
import pickle
import warnings
import hashlib # Add hashlib import
import traceback
import glob # Import glob
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.utils.device import get_device, to_device, clear_gpu_cache
from modules.utils.chords import Chords # Import Chords class

# Define a wrapper function for multiprocessing
def process_file_wrapper(args):
    """Wrapper function for multiprocessing file processing"""
    # Pass dataset_type to _process_file
    dataset_instance, spec_file, file_id, label_files_dict, logit_files_dict, return_skip_reason, dataset_type = args
    return dataset_instance._process_file(spec_file, file_id, label_files_dict, logit_files_dict, return_skip_reason, dataset_type)

class SynthDataset(Dataset):
    """
    Dataset for loading preprocessed spectrograms and chord labels.
    Optimized implementation for GPU acceleration with single worker.
    Supports dataset formats:
    - 'fma': Uses numeric 6-digit IDs with format ddd/dddbbb_spec.npy
    - 'maestro': Uses arbitrary filenames with format maestro-v3.0.0/file-name_spec.npy
    - 'dali_synth': Uses hex IDs with format xxx/hexid_spec.npy (xxx is alphanumeric)
    - 'labeled_synth': Uses specific subdirs (billboard, etc.) under LabeledDataset/Labels, LabeledDataset_synth/spectrograms, LabeledDataset_synth/logits
    - 'combined': Loads 'fma', 'maestro', and 'dali_synth' datasets simultaneously
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
            dataset_type: Type of dataset format ('fma', 'maestro', 'dali_synth', 'labeled_synth', or 'combined')
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
        self.dataset_type = dataset_type  # Dataset format type

        # Add 'labeled_synth' to valid types
        # --- MODIFICATION START ---
        # Add combined types to the list of valid dataset types
        valid_dataset_types = ['fma', 'maestro', 'dali_synth', 'labeled_synth', 'combined',
                               'fma+maestro', 'fma+dali_synth', 'maestro+dali_synth']
        # --- MODIFICATION END ---
        if self.dataset_type not in valid_dataset_types:
            warnings.warn(f"Unknown dataset_type '{dataset_type}', defaulting to 'fma'")
            self.dataset_type = 'fma'

        # Define subdirectories for 'labeled_synth' type
        self.labeled_synth_subdirs = ['billboard', 'caroleKing', 'queen', 'theBeatles']

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

        # Instantiate the chord parser
        self.chord_parser = Chords()

        # Set up regex patterns - always define all patterns regardless of dataset type
        # General pattern to capture base name before _spec, _logits, or .lab/.txt
        self.file_pattern = re.compile(r'(.+?)(?:_spec|_logits)?\.(?:npy|lab|txt)$')
        # For FMA: 6-digit numeric ID pattern
        self.numeric_id_pattern = re.compile(r'(\d{6})')
        # For DALI prefix: 3 alphanumeric chars
        self.dali_prefix_pattern = re.compile(r'^[0-9a-zA-Z]{3}$')

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

        # Generate a safer cache file name using hashing if none provided
        # --- Moved this block BEFORE _load_data() call ---
        if cache_file is None:
            cache_key = f"{spec_dir}_{label_dir}_{seq_len}_{stride}_{frame_duration}_{dataset_type}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            self.cache_file = f"dataset_cache_{dataset_type}_{cache_hash}.pkl"
            if verbose:
                print(f"Using cache file: {self.cache_file}")
        else:
            self.cache_file = cache_file
        # --- End moved block ---

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
            (self.seq_len, 25),   # Common logits/predictions size
            (self.seq_len, 170)   # Common large voca logits size
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
        """Load teacher logits file with error handling and shape correction"""
        try:
            teacher_logits = np.load(logit_file)

            # --- Add shape correction ---
            if teacher_logits.ndim == 3 and teacher_logits.shape[0] == 1:
                teacher_logits = np.squeeze(teacher_logits, axis=0)
                if self.verbose and not hasattr(self, f'_warned_squeeze_{logit_file}'):
                    # Log only once per file to avoid spam
                    # print(f"Corrected logits shape from (1, N, C) to (N, C) for {logit_file}") # Commented out
                    setattr(self, f'_warned_squeeze_{logit_file}', True)
            # --- End shape correction ---

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

        # Try to load data from cache first
        # Now self.cache_file is guaranteed to exist
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
        logit_files_dict = {} # Added for logits
        label_counts = {'fma': 0, 'maestro': 0, 'dali_synth': 0, 'labeled_synth': 0} # Added labeled_synth
        logit_counts = {'fma': 0, 'maestro': 0, 'dali_synth': 0, 'labeled_synth': 0} # Added labeled_synth

        # --- Scan label directories ---
        for label_dir in self.label_dirs:
            # Initialize with None to force explicit type determination
            current_type = None

            # Skip LabeledDataset_augmented directory completely
            if "LabeledDataset_augmented" in str(label_dir):
                if self.verbose:
                    print(f"SKIPPING LabeledDataset_augmented directory: {label_dir}")
                continue

            # Determine dataset type based on path
            if "maestro" in str(label_dir).lower():
                current_type = 'maestro'
            elif "dali_synth" in str(label_dir).lower():
                current_type = 'dali_synth'
            # Check if this label_dir corresponds to the LabeledDataset structure
            elif self.dataset_type == 'labeled_synth' and "LabeledDataset/Labels" in str(label_dir):
                current_type = 'labeled_synth'
            # Only set to 'fma' if it's explicitly in the path or as a last resort
            elif "fma" in str(label_dir).lower() or "logits/synth" in str(label_dir).lower():
                current_type = 'fma'
            else:
                # If we can't determine the type, log a warning and skip this directory
                if self.verbose:
                    print(f"WARNING: Could not determine dataset type for {label_dir}, defaulting to 'fma'")
                current_type = 'fma'
                if self.verbose: print(f"Scanning LABELED_SYNTH labels base: {label_dir}")
                # Scan within specific subdirectories
                for subdir in self.labeled_synth_subdirs:
                    subdir_path = label_dir / subdir
                    if subdir_path.exists():
                        if self.verbose: print(f"  Scanning subdir: {subdir}")
                        # Scan for .lab and .txt files
                        for label_path in list(subdir_path.glob("**/*.lab")) + list(subdir_path.glob("**/*.txt")):
                            match = self.file_pattern.search(label_path.name)
                            if match:
                                base_name = match.group(1)
                                file_id = f"{subdir}/{base_name}" # Use subdir/basename as ID
                                label_files_dict[file_id] = label_path
                                label_counts[current_type] += 1
                    elif self.verbose: print(f"  Subdir not found: {subdir_path}")
                continue # Skip the generic glob below for this type

            # Generic scanning for other types (FMA, Maestro, DALI)
            if self.verbose:
                print(f"Scanning {current_type.upper()} labels from: {label_dir}")

            # Scan for .lab and .txt files
            for label_path in list(label_dir.glob("**/*.lab")) + list(label_dir.glob("**/*.txt")):
                match = self.file_pattern.search(label_path.name)
                if match:
                    file_id = match.group(1)
                    # Refine ID for Maestro if it contains path separators
                    if current_type == 'maestro' and '/' in str(label_path.relative_to(label_dir)):
                        file_id = str(label_path.relative_to(label_dir)).replace('.lab', '').replace('.txt', '')
                    label_files_dict[file_id] = label_path
                    label_counts[current_type] += 1

        # --- Scan logit directories (only if KD is potentially used) ---
        if self.logits_dirs:
            for logit_dir in self.logits_dirs:
                # Initialize with None to force explicit type determination
                current_type = None

                # Skip LabeledDataset_augmented directory completely
                if "LabeledDataset_augmented" in str(logit_dir):
                    if self.verbose:
                        print(f"SKIPPING LabeledDataset_augmented directory: {logit_dir}")
                    continue

                # Determine dataset type based on path
                if "maestro" in str(logit_dir).lower():
                    current_type = 'maestro'
                elif "dali_synth" in str(logit_dir).lower():
                    current_type = 'dali_synth'
                # Check if this logit_dir corresponds to the LabeledDataset_synth structure
                elif self.dataset_type == 'labeled_synth' and "LabeledDataset_synth/logits" in str(logit_dir):
                    current_type = 'labeled_synth'
                # Only set to 'fma' if it's explicitly in the path or as a last resort
                elif "fma" in str(logit_dir).lower() or "logits/synth" in str(logit_dir).lower():
                    current_type = 'fma'
                else:
                    # If we can't determine the type, log a warning and skip this directory
                    if self.verbose:
                        print(f"WARNING: Could not determine dataset type for {logit_dir}, defaulting to 'fma'")
                    current_type = 'fma'
                    if self.verbose: print(f"Scanning LABELED_SYNTH logits base: {logit_dir}")
                    # Scan within specific subdirectories
                    for subdir in self.labeled_synth_subdirs:
                        subdir_path = logit_dir / subdir
                        if subdir_path.exists():
                            if self.verbose: print(f"  Scanning subdir: {subdir}")
                            for logit_path in subdir_path.glob("**/*_logits.npy"):
                                # Use a regex that specifically looks for _logits.npy
                                match = re.search(r'(.+?)_logits\.npy$', logit_path.name)
                                if match:
                                    base_name = match.group(1)
                                    file_id = f"{subdir}/{base_name}" # Use subdir/basename as ID
                                    logit_files_dict[file_id] = logit_path
                                    logit_counts[current_type] += 1
                        elif self.verbose: print(f"  Subdir not found: {subdir_path}")
                    continue # Skip the generic glob below for this type

                # Generic scanning for other types (FMA, Maestro, DALI)
                if self.verbose:
                    print(f"Scanning {current_type.upper()} logits from: {logit_dir}")

                for logit_path in logit_dir.glob("**/*_logits.npy"):
                    # Use a regex that specifically looks for _logits.npy
                    match = re.search(r'(.+?)_logits\.npy$', logit_path.name)
                    if match:
                        file_id = match.group(1)
                        # Refine ID for Maestro
                        if current_type == 'maestro' and '/' in str(logit_path.relative_to(logit_dir)):
                            file_id = str(logit_path.relative_to(logit_dir)).replace('_logits.npy', '')
                        logit_files_dict[file_id] = logit_path
                        logit_counts[current_type] += 1

        if self.verbose:
            print(f"Found {len(label_files_dict)} label files across all directories:")
            for k, v in label_counts.items(): print(f"  {k.upper()}: {v}")
            if self.logits_dirs:
                print(f"Found {len(logit_files_dict)} logit files across all directories:")
                for k, v in logit_counts.items(): print(f"  {k.upper()}: {v}")

        # --- Find all spectrogram files from all spectrogram directories ---
        valid_spec_files = []
        spec_counts = {'fma': 0, 'maestro': 0, 'dali_synth': 0, 'labeled_synth': 0} # Added labeled_synth

        for spec_dir in self.spec_dirs:
            # Initialize with None to force explicit type determination
            current_type = None
            use_maestro_logic = False
            use_dali_logic = False
            use_labeled_synth_logic = False

            # Skip LabeledDataset_augmented directory completely
            if "LabeledDataset_augmented" in str(spec_dir):
                if self.verbose:
                    print(f"SKIPPING LabeledDataset_augmented directory: {spec_dir}")
                continue

            # Determine dataset type based on path
            if "maestro" in str(spec_dir).lower():
                current_type = 'maestro'
                use_maestro_logic = True
            elif "dali_synth" in str(spec_dir).lower() and "LabeledDataset_synth" not in str(spec_dir): # Avoid conflict
                current_type = 'dali_synth'
                use_dali_logic = True
            # Check if this spec_dir corresponds to the LabeledDataset_synth structure
            elif self.dataset_type == 'labeled_synth' and "LabeledDataset_synth/spectrograms" in str(spec_dir):
                current_type = 'labeled_synth'
                use_labeled_synth_logic = True
            # Only set to 'fma' if it's explicitly in the path or as a last resort
            elif "fma" in str(spec_dir).lower() or "logits/synth" in str(spec_dir).lower():
                current_type = 'fma'
            else:
                # If we can't determine the type, log a warning and skip this directory
                if self.verbose:
                    print(f"WARNING: Could not determine dataset type for {spec_dir}, defaulting to 'fma'")
                current_type = 'fma'

            if self.verbose:
                print(f"Scanning {current_type.upper()} spectrograms from: {spec_dir}")

            # --- Labeled Synth Logic ---
            if use_labeled_synth_logic:
                if self.verbose: print(f"Scanning LABELED_SYNTH spectrograms base: {spec_dir}")
                for subdir in self.labeled_synth_subdirs:
                    subdir_path = spec_dir / subdir
                    if subdir_path.exists():
                        if self.verbose: print(f"  Scanning subdir: {subdir}")
                        for spec_path in subdir_path.glob("**/*_spec.npy"):
                            match = re.search(r'(.+?)_spec\.npy$', spec_path.name)
                            if match:
                                base_name = match.group(1)
                                file_id = f"{subdir}/{base_name}" # Use subdir/basename as ID
                                valid_spec_files.append((spec_path, file_id, current_type)) # Store type
                                spec_counts[current_type] += 1
                            elif self.verbose:
                                print(f"Could not extract base name from {current_type.upper()} file: {spec_path.name}")
                    elif self.verbose: print(f"  Subdir not found: {subdir_path}")

            # --- Maestro and DALI Logic ---
            elif use_maestro_logic or use_dali_logic:
                for spec_path in spec_dir.glob("**/*_spec.npy"):
                    # Extract ID using the general pattern
                    match = self.file_pattern.search(spec_path.name)
                    if match:
                        file_id = match.group(1)
                        # Refine ID for Maestro
                        if use_maestro_logic and '/' in str(spec_path.relative_to(spec_dir)):
                            file_id = str(spec_path.relative_to(spec_dir)).replace('_spec.npy', '')

                        # Basic validation for DALI ID format (32 hex chars)
                        if use_dali_logic and not re.fullmatch(r'[0-9a-fA-F]{32}', file_id):
                             if self.verbose: print(f"Skipping potential DALI file with non-hex ID: {spec_path.name}")
                             continue
                        valid_spec_files.append((spec_path, file_id, current_type)) # Store type
                        spec_counts[current_type] += 1
                    elif self.verbose:
                        print(f"Could not extract ID from {current_type.upper()} file: {spec_path.name}")

            # --- FMA Logic ---
            else: # FMA logic (numeric ID)
                for prefix_dir in spec_dir.glob("**/"):
                    # Check if prefix_dir name matches the 3-digit FMA pattern
                    if re.fullmatch(r'\d{3}', prefix_dir.name):
                        for spec_path in prefix_dir.glob("*_spec.npy"):
                            numeric_match = self.numeric_id_pattern.search(spec_path.name)
                            if numeric_match:
                                file_id = numeric_match.group(1)
                                valid_spec_files.append((spec_path, file_id, current_type)) # Store type
                                spec_counts[current_type] += 1
                            elif self.verbose:
                                print(f"Could not extract numeric ID from FMA file: {spec_path.name}")

        if not valid_spec_files:
            warnings.warn(f"No valid spectrogram files found for dataset type '{self.dataset_type}'. Check data paths.")
            return

        if self.verbose:
            print(f"Found {len(valid_spec_files)} valid spectrogram files:")
            for k, v in spec_counts.items(): print(f"  {k.upper()}: {v}")
            if valid_spec_files:
                print("Sample spectrogram paths:")
                for i, (path, _, type) in enumerate(valid_spec_files[:3]):
                    print(f"  ({type.upper()}) {path}")

        # Handle small dataset percentage option
        if self.small_dataset_percentage is not None:
            np.random.seed(42) # Ensure consistent sampling

            # Group files by type ('fma', 'maestro', 'dali_synth', 'labeled_synth')
            dataset_files = {'fma': [], 'maestro': [], 'dali_synth': [], 'labeled_synth': []}
            for spec_path, file_id, file_type in valid_spec_files:
                dataset_files[file_type].append((spec_path, file_id, file_type))

            sampled_files = []
            total_sampled_count = 0
            for file_type, files in dataset_files.items():
                if not files: continue
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

            valid_spec_files = sampled_files
            if self.verbose:
                print(f"Total files after sampling: {total_sampled_count}")


        self.samples = []
        self.total_processed = 0
        self.total_skipped = 0
        self.skipped_reasons = {
            'missing_label': 0,
            'missing_logits': 0,
            'load_error': 0,
            'format_error': 0,
            'missing_spec': 0, # Added previously
            'logit_frame_mismatch': 0 # Add new reason
        }

        num_cpus = max(1, self.num_workers)
        if len(valid_spec_files) < num_cpus * 4:
            num_cpus = max(1, len(valid_spec_files) // 2)
            if self.verbose:
                print(f"Small dataset detected, reducing worker count to {num_cpus}")

        # Pass dataset_type to the wrapper
        args_list = [(self, spec_file, file_id, label_files_dict, logit_files_dict, True, file_type)
                     for spec_file, file_id, file_type in valid_spec_files]

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
            # ... existing error handling ...
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
            # ... existing logging ...
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
            # ... existing caching logic ...
            try:
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)

                samples_to_cache = self.samples

                if self.metadata_only:
                    samples_meta = []
                    for sample in samples_to_cache:
                        meta = {k: sample[k] for k in sample if k not in ['spectro', 'teacher_logits']} # Exclude data arrays
                        # Ensure spec_path is stored if available
                        if 'spec_path' not in meta and 'song_id' in sample:
                             # Attempt to reconstruct spec_path if missing (might be needed if cache was partial)
                             # This reconstruction is heuristic and depends on the dataset type
                             if 'dataset_type' in sample:
                                 dtype = sample['dataset_type']
                                 sid = sample['song_id']
                                 prefix = sample.get('dir_prefix', sid[:3] if len(sid) >=3 else sid)
                                 if dtype == 'labeled_synth' and '/' in sid:
                                     subdir, basename = sid.split('/', 1)
                                     # Assuming self.spec_dir points to the base spectrograms dir
                                     meta['spec_path'] = str(self.spec_dir / subdir / f"{basename}_spec.npy")
                                 elif dtype == 'maestro':
                                      meta['spec_path'] = str(self.spec_dir / f"{sid}_spec.npy") # Simplified assumption
                                 elif dtype == 'dali_synth':
                                      meta['spec_path'] = str(self.spec_dir / prefix / f"{sid}_spec.npy")
                                 else: # FMA
                                      meta['spec_path'] = str(self.spec_dir / prefix / f"{sid}_spec.npy")

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
            # ... existing logging and analysis ...
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
                    # Create a breakdown of samples by dataset (based on file path or stored type)
                    dataset_sample_counts = {}
                    for sample in self.samples:
                        dataset_key = sample.get('dataset_type', 'unknown') # Use stored type if available
                        if dataset_key == 'unknown' and 'spec_path' in sample: # Fallback to path check
                            path = sample['spec_path']
                            if "maestro" in str(path).lower():
                                dataset_key = "maestro"
                            elif "dali_synth" in str(path).lower() and "LabeledDataset_synth" not in str(path):
                                dataset_key = "dali_synth"
                            elif "LabeledDataset_synth" in str(path):
                                dataset_key = "labeled_synth"
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

    def _process_file(self, spec_file, file_id, label_files_dict, logit_files_dict, return_skip_reason=False, dataset_type=None):
        """Process a single spectrogram file based on dataset type, skipping if any required file is missing."""
        samples = []
        skip_reason = None
        current_dataset_type = dataset_type if dataset_type is not None else self.dataset_type

        # --- Strict File Existence Checks ---
        # 1. Check Spectrogram File
        if not os.path.exists(str(spec_file)):
            if hasattr(self, 'skipped_reasons'): self.skipped_reasons['missing_spec'] += 1 # Add new reason if needed
            skip_reason = 'missing_spec'
            if return_skip_reason: return [], skip_reason
            return []

        # 2. Check Label File
        label_file = label_files_dict.get(file_id)
        if not label_file or not os.path.exists(str(label_file)):
            # Attempt reconstruction for labeled_synth if not found in dict (e.g., cache load)
            if current_dataset_type == 'labeled_synth' and '/' in file_id:
                subdir_l, base_name_l = file_id.split('/', 1)
                potential_label_file = self.label_dir / subdir_l / f"{base_name_l}.lab"
                if not potential_label_file.exists():
                     potential_label_file = self.label_dir / subdir_l / f"{base_name_l}.txt" # Try .txt
                if potential_label_file.exists():
                    label_file = potential_label_file
                else:
                    label_file = None # Ensure label_file is None if reconstruction fails

            if not label_file: # Check again after potential reconstruction
                if hasattr(self, 'skipped_reasons'): self.skipped_reasons['missing_label'] += 1
                skip_reason = 'missing_label'
                if return_skip_reason: return [], skip_reason
                return []

        # 3. Check Logit File (only if required)
        logit_file = None
        if self.require_teacher_logits:
            logit_file = logit_files_dict.get(file_id)
            if not logit_file or not os.path.exists(str(logit_file)):
                 # Attempt reconstruction for labeled_synth
                 if current_dataset_type == 'labeled_synth' and '/' in file_id:
                     subdir_lg, base_name_lg = file_id.split('/', 1)
                     potential_logit_file = self.logits_dir / subdir_lg / f"{base_name_lg}_logits.npy"
                     if potential_logit_file.exists():
                         logit_file = potential_logit_file
                     else:
                         logit_file = None # Ensure logit_file is None

            if not logit_file: # Check again after potential reconstruction
                if hasattr(self, 'skipped_reasons'): self.skipped_reasons['missing_logits'] += 1
                skip_reason = 'missing_logits'
                if return_skip_reason: return [], skip_reason
                return []
        # --- End Strict File Existence Checks ---

        try:
            # Determine dir_prefix or subdir based on type (moved after checks)
            dir_prefix = None
            subdir = None
            if current_dataset_type == 'labeled_synth' and '/' in file_id:
                subdir, _ = file_id.split('/', 1)
            elif current_dataset_type == 'maestro':
                dir_prefix = spec_file.parent.name if spec_file.parent.name != self.spec_dir.name else None
            elif current_dataset_type == 'dali_synth':
                dir_prefix = spec_file.parent.name
                if not self.dali_prefix_pattern.match(dir_prefix):
                    dir_prefix = file_id[:3] if len(file_id) >= 3 else file_id
            else: # FMA
                dir_prefix = file_id[:3] if len(file_id) >= 3 else file_id

            # --- Load data (metadata or full) ---
            # Files are guaranteed to exist at this point if required
            if self.metadata_only:
                spec_info = np.load(spec_file, mmap_mode='r')
                spec_shape = spec_info.shape
                num_frames = spec_shape[0] if len(spec_shape) > 1 else 1
                expected_num_frames = spec_shape[0] if len(spec_shape) > 1 else 1 # Initialize expected frames

                # --- Check logit length for metadata ---
                logit_frames = 0
                current_logit_file = None
                if self.logits_dirs is not None:
                    current_logit_file = logit_file # Use the one found earlier if required
                    if not current_logit_file:
                        current_logit_file = logit_files_dict.get(file_id)
                        # Reconstruct if needed
                        if not current_logit_file or not os.path.exists(str(current_logit_file)):
                            if current_dataset_type == 'labeled_synth' and '/' in file_id:
                                subdir_lg, base_name_lg = file_id.split('/', 1)
                                potential_logit_file = self.logits_dir / subdir_lg / f"{base_name_lg}_logits.npy"
                                if potential_logit_file.exists():
                                    current_logit_file = potential_logit_file

                    if current_logit_file and os.path.exists(str(current_logit_file)):
                        try:
                            # Load only shape info if possible, or full logits if needed
                            # Using mmap_mode='r' loads metadata without reading full data
                            logit_info = np.load(current_logit_file, mmap_mode='r')
                            if logit_info.ndim >= 2: # Expecting (batch, time, classes) or (time, classes)
                                logit_frames = logit_info.shape[-2] # Get time dimension
                            elif logit_info.ndim == 1 and num_frames == 1: # Single frame case
                                logit_frames = 1
                            else: # Unexpected shape
                                logit_frames = 0

                            if logit_frames > num_frames:
                                expected_num_frames = logit_frames # Store expected length
                                if self.verbose and not hasattr(self, f'_warned_pad_meta_{file_id}'):
                                    # print(f"INFO: Logits ({logit_frames}) longer than spec ({num_frames}) for {file_id}. Storing expected length in metadata.") # Commented out
                                    setattr(self, f'_warned_pad_meta_{file_id}', True)
                        except Exception as e:
                            warnings.warn(f"Could not read logit shape for {current_logit_file}: {e}")
                # --- End check logit length ---

                chord_labels = self._parse_label_file(label_file)

                for t in range(expected_num_frames): # Iterate up to the potentially longer length
                    frame_time = t * self.frame_duration
                    chord_label = self._find_chord_at_time(chord_labels, frame_time)
                    chord_label = self._validate_chord_label(chord_label, label_file) # Validate/map chord

                    sample_meta = {
                        'spec_path': str(spec_file),
                        'chord_label': chord_label,
                        'song_id': file_id, # Store the potentially composite ID
                        'frame_idx': t,
                        'dataset_type': current_dataset_type, # Store the type
                        'expected_num_frames': expected_num_frames # Store expected length
                    }
                    # Store relevant prefix/subdir info
                    if subdir: sample_meta['subdir'] = subdir
                    if dir_prefix: sample_meta['dir_prefix'] = dir_prefix

                    # Add logit_path if logits are used (required or optional)
                    if self.logits_dirs is not None and current_logit_file and os.path.exists(str(current_logit_file)):
                        sample_meta['logit_path'] = str(current_logit_file)

                    samples.append(sample_meta)

            else: # Load full data
                spec = np.load(spec_file)
                if np.isnan(spec).any():
                    warnings.warn(f"NaN values found in {spec_file}, replacing with zeros")
                    spec = np.nan_to_num(spec, nan=0.0)

                num_frames = spec.shape[0] if len(spec.shape) > 1 else 1
                freq_bins = spec.shape[1] if len(spec.shape) > 1 else spec.shape[0] # Get freq bins

                teacher_logits_full = None
                logit_frames = 0
                current_logit_file = None
                # Load logits if they are used (required or optional)
                if self.logits_dirs is not None:
                    # Re-fetch logit file path if needed
                    current_logit_file = logit_file # Use the one found earlier if required
                    if not current_logit_file:
                        current_logit_file = logit_files_dict.get(file_id)
                        # Reconstruct if needed
                        if not current_logit_file or not os.path.exists(str(current_logit_file)):
                            if current_dataset_type == 'labeled_synth' and '/' in file_id:
                                subdir_lg, base_name_lg = file_id.split('/', 1)
                                potential_logit_file = self.logits_dir / subdir_lg / f"{base_name_lg}_logits.npy"
                                if potential_logit_file.exists():
                                    current_logit_file = potential_logit_file

                    if current_logit_file and os.path.exists(str(current_logit_file)): # Check existence again
                        try:
                            teacher_logits_full = self._load_logits_file(current_logit_file)

                            # --- Check Frame Count Mismatch and Pad Spectrogram ---
                            if teacher_logits_full is not None:
                                # Get time dimension correctly after potential squeeze in _load_logits_file
                                if teacher_logits_full.ndim >= 2: # Expecting (time, classes)
                                    logit_frames = teacher_logits_full.shape[0]
                                elif teacher_logits_full.ndim == 1 and num_frames == 1: # Single frame case
                                    logit_frames = 1
                                else: # Unexpected shape
                                    logit_frames = 0

                                if logit_frames > num_frames:
                                    pad_width = logit_frames - num_frames
                                    if self.verbose and not hasattr(self, f'_warned_pad_full_{file_id}'):
                                        # print(f"INFO: Padding spectrogram {spec_file} ({num_frames} -> {logit_frames} frames) to match logits {current_logit_file}") # Commented out
                                        setattr(self, f'_warned_pad_full_{file_id}', True)
                                    # Pad along the time axis (axis 0)
                                    spec = np.pad(spec, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
                                    num_frames = logit_frames # Update num_frames to padded length
                                elif logit_frames > 0 and logit_frames < num_frames:
                                     # Logits are shorter? This is unexpected based on the explanation. Warn.
                                     warnings.warn(f"Logits ({logit_frames}) shorter than spec ({num_frames}) for {file_id}. Truncating spec might lose data. Check generation process.")
                                     # Optionally truncate spec: spec = spec[:logit_frames]
                                     # num_frames = logit_frames
                            # --- End Frame Count Mismatch Check and Padding ---

                        except Exception as e:
                            warnings.warn(f"Error loading or processing logits file {current_logit_file}: {e}")
                            # If required, this should have been caught earlier, but double-check
                            if self.require_teacher_logits:
                                skip_reason = 'load_error'
                                if return_skip_reason: return [], skip_reason
                                return []
                            teacher_logits_full = None # Continue without logits if not required

                chord_labels = self._parse_label_file(label_file)

                # --- Process frames only if not skipped ---
                for t in range(num_frames): # Iterate using potentially updated num_frames
                    frame_time = t * self.frame_duration
                    chord_label = self._find_chord_at_time(chord_labels, frame_time)
                    chord_label = self._validate_chord_label(chord_label, label_file) # Validate/map chord

                    spec_frame = spec[t] if len(spec.shape) > 1 else spec

                    sample_dict = {
                        'spectro': spec_frame,
                        'chord_label': chord_label,
                        'song_id': file_id,
                        'frame_idx': t,
                        'dataset_type': current_dataset_type
                    }
                    # Store subdir/dir_prefix info
                    if subdir: sample_dict['subdir'] = subdir
                    if dir_prefix: sample_dict['dir_prefix'] = dir_prefix


                    if teacher_logits_full is not None:
                        # Check bounds carefully, especially after potential padding/truncation
                        if len(teacher_logits_full.shape) > 1 and t < teacher_logits_full.shape[0]:
                            sample_dict['teacher_logits'] = teacher_logits_full[t]
                        elif len(teacher_logits_full.shape) == 1 and num_frames == 1: # Single frame spec and logits
                             sample_dict['teacher_logits'] = teacher_logits_full
                        # Handle case where logits are 1D but spec is multi-frame (broadcast)
                        elif len(teacher_logits_full.shape) == 1 and num_frames > 1:
                             sample_dict['teacher_logits'] = teacher_logits_full
                        # Do NOT add logits if index t is out of bounds for teacher_logits_full

                    samples.append(sample_dict)

        except Exception as e:
            # ... existing error handling ...
            if hasattr(self, 'skipped_reasons'):
                # ... existing reason assignment ...
                pass # No changes needed here

            warnings.warn(f"Error processing file {spec_file}: {str(e)}")
            traceback.print_exc() # Print traceback for debugging

            if return_skip_reason:
                return [], skip_reason
            return []

        if return_skip_reason:
            return samples, skip_reason
        return samples

    def _validate_chord_label(self, chord_label, source_file=""):
        """Validate chord label against mapping using normalization, and default to 'N'."""
        if self.chord_mapping is None:
            # If no mapping provided, build one dynamically (less safe, relies on parser output)
            normalized_label = self.chord_parser.label_error_modify(chord_label)
            if normalized_label == "X": # Treat unknown as N for consistency
                normalized_label = "N"
            if normalized_label not in self.chord_to_idx:
                self.chord_to_idx[normalized_label] = len(self.chord_to_idx)
            return normalized_label

        # Normalize the input label using the robust parser
        # This handles enharmonics (A# -> Bb), quality variations, etc.
        normalized_label = self.chord_parser.label_error_modify(chord_label)

        # Handle cases where normalization results in N or X
        if normalized_label == "N" or normalized_label == "X":
            return "N" # Always return N for No Chord or Unknown

        # Check if the *normalized* label exists in the provided mapping
        if normalized_label in self.chord_to_idx:
            return normalized_label
        else:
            # If the normalized label is still not found, it means the target vocabulary
            # (self.chord_to_idx) doesn't contain this specific normalized chord.
            if self.verbose and not hasattr(self, f'_warned_norm_{normalized_label}'):
                 # Use a different warning message to distinguish from simple parsing errors
                 warnings.warn(f"Normalized chord label '{normalized_label}' (from original '{chord_label}' in {source_file}) not found in chord_mapping. Using 'N'.")
                 setattr(self, f'_warned_norm_{normalized_label}', True)
            return "N"

    def _parse_label_file(self, label_file):
        """Parse a label file (.lab or .txt) into a list of (start_time, end_time, chord) tuples"""
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
        self.segment_indices = [] # Initialize here

        for song_id, indices in song_samples.items():
            if len(indices) < self.seq_len:
                # Pad short songs only if they have at least one frame
                if len(indices) > 0:
                    # Ensure the segment covers seq_len frames, padding handled in __getitem__
                    self.segment_indices.append((indices[0], indices[0] + self.seq_len))
                    total_segments += 1
                continue # Skip songs shorter than seq_len otherwise

            # Generate segments using stride
            for start_idx_in_song in range(0, len(indices) - self.seq_len + 1, self.stride):
                # Get the actual sample indices from the full self.samples list
                segment_start_sample_idx = indices[start_idx_in_song]
                # The end index for slicing is start + seq_len
                segment_end_sample_idx = segment_start_sample_idx + self.seq_len
                self.segment_indices.append((segment_start_sample_idx, segment_end_sample_idx))
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
        # has_teacher_logits = False # REMOVED this flag

        # Get first sample to determine consistent song ID and potential expected length
        first_sample = self.samples[seg_start]
        start_song_id = first_sample['song_id']
        # Get dataset type and subdir if available from the sample (useful for metadata_only loading)
        sample_dataset_type = first_sample.get('dataset_type', self.dataset_type)
        sample_subdir = first_sample.get('subdir')
        # Get expected length if stored during metadata loading
        expected_seq_frames = first_sample.get('expected_num_frames')

        # --- Pre-load full spec if metadata_only and padding might be needed ---
        # This avoids loading the spec file repeatedly inside the loop if padding is required.
        full_spec_loaded = None
        if self.metadata_only and expected_seq_frames is not None:
            try:
                spec_path = first_sample['spec_path']
                if os.path.exists(spec_path):
                    full_spec_loaded = np.load(spec_path)
                    current_frames = full_spec_loaded.shape[0]
                    if current_frames < expected_seq_frames:
                        pad_width = expected_seq_frames - current_frames
                        if self.verbose and not hasattr(self, f'_warned_pad_getitem_{start_song_id}'):
                            print(f"INFO: Padding spectrogram {spec_path} ({current_frames} -> {expected_seq_frames} frames) during __getitem__ based on metadata.")
                            setattr(self, f'_warned_pad_getitem_{start_song_id}', True)
                        full_spec_loaded = np.pad(full_spec_loaded, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
                else:
                    if self.verbose and not hasattr(self, f'_warned_missing_spec_getitem_{spec_path}'):
                        warnings.warn(f"Spectrogram file not found during __getitem__: {spec_path}. Using zeros.")
                        setattr(self, f'_warned_missing_spec_getitem_{spec_path}', True)
            except Exception as e:
                 warnings.warn(f"Error pre-loading/padding spec {first_sample.get('spec_path')} in __getitem__: {e}")
                 full_spec_loaded = None # Fallback to loading inside loop or zeros
        # --- End Pre-load ---


        # Process each sample in the segment
        for i in range(seg_start, seg_end):
            # ... (padding logic for end of dataset) ...
            if i >= len(self.samples):
                # We've reached the end of the dataset, add padding
                padding_shape = sequence[-1].shape if sequence else (144,)
                sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=self.device))
                label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=self.device))

                # Pad logits only if needed for the sequence (check teacher_logits_seq)
                if teacher_logits_seq: # Check if logits were added for previous frames
                    logits_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,) # Use 170 as default guess
                    teacher_logits_seq.append(torch.zeros(logits_shape, dtype=torch.float, device=self.device))
                continue

            sample_i = self.samples[i]

            # Check if we've crossed a song boundary
            if sample_i['song_id'] != start_song_id:
                 # Pad the rest of the sequence if we hit a song boundary early
                 padding_needed = self.seq_len - len(sequence)
                 if padding_needed > 0:
                     padding_shape = sequence[-1].shape if sequence else (144,)
                     for _ in range(padding_needed):
                         sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=self.device))
                         label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=self.device))
                         # Also pad logits if they were being collected
                         if teacher_logits_seq:
                             logits_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,)
                             teacher_logits_seq.append(torch.zeros(logits_shape, dtype=torch.float, device=self.device))
                 break # Stop processing this segment

            # Load spectrogram - either from memory, pre-loaded full spec, or from disk
            spec_vec = None
            if 'spectro' in sample_i: # Already in memory (full load mode)
                if isinstance(sample_i['spectro'], np.ndarray):
                    spec_vec = torch.from_numpy(sample_i['spectro']).to(dtype=torch.float, device=self.device)
                else:
                    spec_vec = sample_i['spectro'].clone().detach().to(self.device)
            elif full_spec_loaded is not None: # Use pre-loaded and potentially padded spec (metadata mode)
                frame_idx = sample_i.get('frame_idx')
                if frame_idx is not None and frame_idx < full_spec_loaded.shape[0]:
                     spec_vec = torch.from_numpy(full_spec_loaded[frame_idx]).to(dtype=torch.float, device=self.device)
                else: # Frame index out of bounds (shouldn't happen with correct padding) or missing
                     padding_shape = (144,) # Default shape
                     if sequence: padding_shape = sequence[-1].shape
                     spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
            elif 'spec_path' in sample_i: # Load from disk (metadata mode, no padding needed or pre-load failed)
                try:
                    spec_path = sample_i['spec_path']
                    if not os.path.exists(spec_path):
                         if self.verbose and not hasattr(self, f'_warned_missing_spec_{spec_path}'):
                             warnings.warn(f"Spectrogram file not found: {spec_path}. Using zeros.")
                             setattr(self, f'_warned_missing_spec_{spec_path}', True)
                         raise FileNotFoundError(f"Spectrogram file not found: {spec_path}")

                    # Load the specific frame directly if possible, or load full and index
                    # Note: This path is less efficient if padding was needed but pre-load failed.
                    spec = np.load(spec_path)
                    frame_idx = sample_i.get('frame_idx')

                    if frame_idx is not None and len(spec.shape) > 1:
                        if frame_idx < spec.shape[0]:
                            spec_vec = torch.from_numpy(spec[frame_idx]).to(dtype=torch.float, device=self.device)
                        else:
                            padding_shape = (spec.shape[1],) if len(spec.shape) > 1 else (144,)
                            spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
                    elif len(spec.shape) > 1 and frame_idx is None: # Multi-frame spec but no index? Use first frame? Warn.
                         warnings.warn(f"Multi-frame spec {spec_path} but no frame_idx in metadata. Using frame 0.")
                         spec_vec = torch.from_numpy(spec[0]).to(dtype=torch.float, device=self.device)
                    else: # Single-frame spectrogram or 1D array
                        spec_vec = torch.from_numpy(spec).to(dtype=torch.float, device=self.device)

                except Exception as e:
                    warnings.warn(f"Error loading spec {sample_i.get('spec_path')} in __getitem__: {e}")
                    padding_shape = sequence[-1].shape if sequence else (144,)
                    spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
            else:
                # Use zero tensor if no spectrogram data
                padding_shape = sequence[-1].shape if sequence else (144,)
                spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)


            # Get chord label index
            # Ensure chord_label is valid before getting index
            chord_label = sample_i['chord_label']
            validated_chord_label = self._validate_chord_label(chord_label, sample_i.get('spec_path', 'unknown file'))
            chord_idx = self.chord_to_idx.get(validated_chord_label, self.chord_to_idx.get("N", 0))
            chord_idx_tensor = torch.tensor(chord_idx, dtype=torch.long, device=self.device)


            # Add spectrogram and chord label to sequences
            sequence.append(spec_vec)
            label_seq.append(chord_idx_tensor)

            # Handle teacher logits - either from memory or from disk
            # --- Modified Logits Handling ---
            logit_loaded_or_found = False
            if 'teacher_logits' in sample_i:
                # Use stored teacher logits
                if isinstance(sample_i['teacher_logits'], np.ndarray):
                    # Apply squeeze check here too, in case it came from cache without correction
                    logits_array = sample_i['teacher_logits']
                    if logits_array.ndim == 3 and logits_array.shape[0] == 1:
                         logits_array = np.squeeze(logits_array, axis=0)
                    logits_tensor = torch.from_numpy(logits_array).to(dtype=torch.float, device=self.device)
                else:
                    # Assume tensor is already correct shape if not numpy array
                    logits_tensor = sample_i['teacher_logits'].clone().detach().to(self.device)
                teacher_logits_seq.append(logits_tensor)
                logit_loaded_or_found = True
            elif 'logit_path' in sample_i:
                # Load teacher logits from disk
                try:
                    logit_path = sample_i['logit_path']
                    if not os.path.exists(logit_path):
                         if self.require_teacher_logits:
                             raise FileNotFoundError(f"Required teacher logits file not found: {logit_path}")
                         else:
                             if self.verbose and not hasattr(self, f'_warned_missing_logit_{logit_path}'):
                                 warnings.warn(f"Teacher logits file not found: {logit_path}. Skipping logits for this sample.")
                                 setattr(self, f'_warned_missing_logit_{logit_path}', True)
                             raise FileNotFoundError(f"Teacher logits file not found: {logit_path}") # Caught below

                    # Use the corrected _load_logits_file method
                    logits = self._load_logits_file(logit_path) # This now handles squeezing

                    if logits is None: # Handle case where _load_logits_file returned None due to error
                        raise RuntimeError(f"Failed to load logits from {logit_path}")

                    # Extract the correct frame's logits
                    frame_idx = sample_i.get('frame_idx') # Use frame_idx from metadata
                    if frame_idx is not None and len(logits.shape) > 1:
                        # Use correct shape index [1] for num_classes after squeeze
                        logits_vec = logits[frame_idx] if frame_idx < logits.shape[0] else np.zeros(logits.shape[1])
                    elif len(logits.shape) > 1 and frame_idx is None: # Multi-frame logits but no index? Warn.
                         warnings.warn(f"Multi-frame logits {logit_path} but no frame_idx in metadata. Using frame 0.")
                         logits_vec = logits[0]
                    else: # Single-frame logits or 1D array
                        logits_vec = logits # Assumes single frame or already correct vector

                    logits_tensor = torch.from_numpy(logits_vec).to(dtype=torch.float, device=self.device)
                    teacher_logits_seq.append(logits_tensor)
                    logit_loaded_or_found = True
                except Exception as e:
                    # Log error but don't necessarily stop if logits not required
                    if self.require_teacher_logits:
                        if self.verbose: print(f"ERROR: Failed to load required logits at getitem for {sample_i.get('logit_path', 'unknown path')}: {e}")
                    # If logits are required, we will pad later. If not, we just skip this frame's logits.
                    pass # Continue processing other frames

            # If logits are required but failed loading for this frame...
            # ... (existing logic for padding required logits) ...
            if self.require_teacher_logits and not logit_loaded_or_found and teacher_logits_seq:
                 logits_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,) # Guess shape
                 teacher_logits_seq.append(torch.zeros(logits_shape, dtype=torch.float, device=self.device))
            # --- End Modified Logits Handling ---


        # Pad to ensure consistent sequence length (if not already padded by song boundary break)
        current_len = len(sequence)
        if current_len < self.seq_len:
            padding_needed = self.seq_len - current_len
            padding_shape = sequence[-1].shape if sequence else (144,)
            for _ in range(padding_needed):
                sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=self.device))
                label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=self.device))

                # Also pad logits if they were being collected for this sequence
                if teacher_logits_seq:
                    logits_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,)
                    teacher_logits_seq.append(torch.zeros(logits_shape, dtype=torch.float, device=self.device))


        # Create the output dictionary
        sample_out = {
            'spectro': torch.stack(sequence, dim=0),
            'chord_idx': torch.stack(label_seq, dim=0)
        }


        # --- Process and add teacher logits based on require_teacher_logits ---
        fixed_num_classes = 170 # Define expected class number

        if self.require_teacher_logits:
            # If required, always add the key, even if it's all zeros
            if teacher_logits_seq:
                # Process the collected logits
                try:
                    # --- Refined Logit Normalization/Stacking ---
                    processed_logits_list = []
                    expected_logit_shape = None
                    for i, logits_tensor in enumerate(teacher_logits_seq):
                        # Determine expected shape from the first valid tensor
                        if expected_logit_shape is None and logits_tensor.numel() > 0:
                             # Assuming logits should be [num_classes]
                             # Use fixed_num_classes as the target dimension
                             expected_logit_shape = (fixed_num_classes,)

                        # Handle potentially empty/zero tensors added during padding/errors
                        if logits_tensor.numel() == 0:
                            if expected_logit_shape:
                                normalized = torch.zeros(expected_logit_shape, dtype=torch.float, device=self.device)
                            else: # Fallback if no valid tensor seen yet
                                normalized = torch.zeros(fixed_num_classes, dtype=torch.float, device=self.device)
                        elif logits_tensor.dim() == 0: # Scalar? Unlikely for logits
                            normalized = torch.zeros(expected_logit_shape or (fixed_num_classes,), dtype=torch.float, device=self.device)
                            normalized[0] = logits_tensor.item()
                        elif logits_tensor.dim() == 1:
                            current_len = logits_tensor.shape[0]
                            target_len = expected_logit_shape[0] if expected_logit_shape else fixed_num_classes
                            if current_len == target_len:
                                normalized = logits_tensor
                            else: # Pad or truncate
                                normalized = torch.zeros(target_len, dtype=torch.float, device=self.device)
                                copy_len = min(current_len, target_len)
                                normalized[:copy_len] = logits_tensor[:copy_len]
                        else: # More than 1D? Flatten and truncate/pad
                            flattened = logits_tensor.reshape(-1)
                            current_len = flattened.shape[0]
                            target_len = expected_logit_shape[0] if expected_logit_shape else fixed_num_classes
                            normalized = torch.zeros(target_len, dtype=torch.float, device=self.device)
                            copy_len = min(current_len, target_len)
                            normalized[:copy_len] = flattened[:copy_len]

                        processed_logits_list.append(normalized)
                    # --- End Refined Logit Normalization/Stacking ---

                    # Stack the processed list
                    teacher_logits_stacked = torch.stack(processed_logits_list, dim=0)
                    sample_out['teacher_logits'] = teacher_logits_stacked
                except Exception as e:
                    if self.verbose: print(f"ERROR processing logits for required sequence (idx {idx}): {e}. Using zeros.")
                    sample_out['teacher_logits'] = torch.zeros((self.seq_len, fixed_num_classes), dtype=torch.float, device=self.device)
            else:
                # If required but sequence is empty (errors/missing files for all frames)
                if self.verbose and not hasattr(self, f'_warned_empty_logits_{idx}'):
                     print(f"Warning: No teacher logits found or loaded for required sequence (idx {idx}). Using zeros.")
                     setattr(self, f'_warned_empty_logits_{idx}', True)
                sample_out['teacher_logits'] = torch.zeros((self.seq_len, fixed_num_classes), dtype=torch.float, device=self.device)


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