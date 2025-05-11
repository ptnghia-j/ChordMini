import torch
import numpy as np
import os
import re
import time
import pickle
import warnings
import hashlib
import traceback
import glob
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import Counter
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold # Add train_test_split and StratifiedKFold
import random

# Assuming these utilities are available in the specified paths
from modules.utils.device import get_device, to_device, clear_gpu_cache
from modules.utils.chords import Chords

# Define a wrapper function for multiprocessing
def _process_file_wrapper(args):
    """Wrapper function for multiprocessing file processing"""
    dataset_instance, spec_file, file_id, label_files_dict, logit_files_dict, return_skip_reason = args
    # Pass dataset_type implicitly as 'labeled_synth'
    return dataset_instance._process_file(spec_file, file_id, label_files_dict, logit_files_dict, return_skip_reason)

class CrossValidationDataset(Dataset):
    """
    Dataset for K-fold cross-validation using the 'labeled_synth' dataset structure.
    Shuffles tracks from specified subdirectories (e.g., billboard, theBeatles)
    and splits them into K folds for training and validation. Other subdirs
    (e.g., caroleKing, queen) can be included as fixed training data.
    """
    def __init__(self, spec_dir, label_dir, chord_mapping, seq_len=10, stride=None,
                 num_folds=4, current_fold=0, kfold_subdirs=('billboard', 'theBeatles'),
                 fixed_subdirs=('caroleKing', 'queen'),
                 frame_duration=0.1, num_workers=0, cache_file_prefix="cv_cache", verbose=True,
                 use_cache=True, metadata_only=True, logits_dir=None,
                 lazy_init=False, require_teacher_logits=False, device=None,
                 pin_memory=False, prefetch_factor=2, batch_gpu_cache=False,
                 small_dataset_percentage=None): # Add small_dataset_percentage
        """
        Initialize the cross-validation dataset.

        Args:
            spec_dir: Base directory for spectrograms (e.g., LabeledDataset_synth/spectrograms)
            label_dir: Base directory for labels (e.g., LabeledDataset/Labels)
            chord_mapping: Mapping of chord names to indices
            seq_len: Sequence length for segmentation
            stride: Stride for segmentation (default: same as seq_len)
            num_folds: Number of folds for cross-validation (default: 4)
            current_fold: The current fold index (0 to num_folds-1) to use for validation
            kfold_subdirs: Tuple of subdirectories whose tracks will be shuffled and split across folds.
            fixed_subdirs: Tuple of subdirectories whose tracks will always be included in the training set.
            frame_duration: Duration of each frame in seconds
            num_workers: Number of workers for data loading (recommend 0 for GPU)
            cache_file_prefix: Prefix for cache files (fold number will be appended)
            verbose: Whether to print verbose output
            use_cache: Whether to use caching
            metadata_only: Whether to cache only metadata
            logits_dir: Base directory for teacher logits (e.g., LabeledDataset_synth/logits)
            lazy_init: Whether to use lazy initialization (loads only metadata initially)
            require_teacher_logits: Whether to require teacher logits for all samples
            device: Device to use (default: auto-detect)
            pin_memory: Whether to pin memory (recommend False if num_workers=0)
            prefetch_factor: Number of batches to prefetch (for DataLoader)
            batch_gpu_cache: Whether to cache batches on GPU
            small_dataset_percentage: Float (0.0 to 1.0). If set, use only this fraction of the data.
        """
        init_start_time = time.time()
        if verbose:
            print(f"CrossValidationDataset initialization started (Fold {current_fold}/{num_folds})")

        self.spec_dir = Path(spec_dir)
        self.label_dir = Path(label_dir)
        self.logits_dir = Path(logits_dir) if logits_dir else None

        self.chord_mapping = chord_mapping
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.kfold_subdirs = set(kfold_subdirs)
        self.fixed_subdirs = set(fixed_subdirs)
        self.labeled_synth_subdirs = list(self.kfold_subdirs | self.fixed_subdirs) # All relevant subdirs

        self.frame_duration = frame_duration
        self.num_workers = num_workers # Recommend 0 for GPU processing in __getitem__
        self.verbose = verbose
        self.use_cache = use_cache
        self.metadata_only = metadata_only
        self.lazy_init = lazy_init # Note: Lazy init might be complex with CV splits, recommend False
        self.require_teacher_logits = require_teacher_logits
        self.dataset_type = 'labeled_synth' # Hardcoded

        self.pin_memory = pin_memory and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.batch_gpu_cache = batch_gpu_cache
        self.small_dataset_percentage = small_dataset_percentage # Store the percentage

        if current_fold < 0 or current_fold >= num_folds:
            raise ValueError(f"current_fold must be between 0 and {num_folds-1}")

        if self.lazy_init and self.verbose:
            warnings.warn("lazy_init=True is not fully recommended with CrossValidationDataset due to upfront splitting logic. Data loading might be slower than expected.")

        # Chord mapping setup (copied from SynthDataset)
        if self.chord_mapping is not None:
            self.chord_to_idx = self.chord_mapping.copy()
            # Add plain note names as aliases for major chords
            for root in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                maj_chord = f"{root}:maj"
                if maj_chord in self.chord_to_idx and root not in self.chord_to_idx:
                    self.chord_to_idx[root] = self.chord_to_idx[maj_chord]
                if root in self.chord_to_idx and maj_chord not in self.chord_to_idx:
                    self.chord_to_idx[maj_chord] = self.chord_to_idx[root]
        else:
            self.chord_to_idx = {}
            warnings.warn("No chord_mapping provided. Mapping will be built dynamically.")

        self.chord_parser = Chords()

        # Regex for file IDs (subdir/basename)
        self.file_pattern = re.compile(r'(.+?)(?:_spec|_logits)?\.(?:npy|lab|txt)$')
        self.spec_pattern = re.compile(r'(.+?)_spec\.npy$')
        self.logit_pattern = re.compile(r'(.+?)_logits\.npy$')
        self.label_pattern = re.compile(r'(.+?)\.(lab|txt)$')

        # Device setup
        if device is None:
            try:
                self.device = get_device()
            except Exception as e:
                if verbose: print(f"Error initializing GPU device: {e}. Falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = device
        if self.verbose: print(f"Using device: {self.device}")

        self.gpu_batch_cache = {} if self.batch_gpu_cache and self.device.type == 'cuda' else None
        self._zero_spec_cache = {}
        self._zero_logit_cache = {}
        # Initialize sampler storage
        self._current_train_sampler = None
        self._current_val_sampler = None
        # if self.device.type == 'cuda': # Line to be moved
        #     self._init_gpu_cache() # Line to be moved

        if self.require_teacher_logits and self.logits_dir is None:
            raise ValueError("require_teacher_logits=True requires a valid logits_dir")

        # Generate cache file name based on prefix, fold, and small_dataset_percentage
        cache_key = f"{spec_dir}_{label_dir}_{logits_dir}_{seq_len}_{stride}_{num_folds}_{current_fold}_{small_dataset_percentage}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        self.cache_file = f"{cache_file_prefix}_fold{current_fold}of{num_folds}_small{small_dataset_percentage}_{cache_hash}.pkl"
        if verbose: print(f"Using cache file: {self.cache_file}")

        # Initialize sample and segment lists
        self.train_samples = []
        self.val_samples = []
        self.train_segment_indices = []
        self.val_segment_indices = []
        self.train_song_ids = []
        self.val_song_ids = []

        # Load data (handles splitting based on current_fold)
        self._load_data()

        # Generate segments for both train and validation sets
        self._generate_segments()

        # Initialize GPU cache after multiprocessing pool in _load_data has finished
        if self.device.type == 'cuda':
            self._init_gpu_cache()

        init_time = time.time() - init_start_time
        if verbose:
            print(f"Dataset initialization for Fold {current_fold} completed in {init_time:.2f} seconds")
            print(f"  Training songs: {len(self.train_song_ids)}, Training segments: {len(self.train_segment_indices)}")
            print(f"  Validation songs: {len(self.val_song_ids)}, Validation segments: {len(self.val_segment_indices)}")

    # --- Core Data Loading and Splitting Logic ---

    def _load_data(self):
        """Loads data, performs train/validation split based on K-Fold CV with mixing strategy."""
        start_time = time.time()

        # --- Attempt to load from cache ---
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                # Validate cache content
                # Ensure 'small_dataset_percentage' matches if it was part of cache_data
                cached_small_percentage = cache_data.get('small_dataset_percentage')
                current_small_percentage = self.small_dataset_percentage
                # Treat None and 0.0 as different for cache validation if necessary, or align them.
                # For simplicity, exact match or both are None.
                small_percentage_matches = (cached_small_percentage == current_small_percentage)

                if all(k in cache_data for k in ['train_samples', 'val_samples', 'chord_to_idx', 'train_song_ids', 'val_song_ids']) and small_percentage_matches:
                    self.train_samples = cache_data['train_samples']
                    self.val_samples = cache_data['val_samples']
                    self.chord_to_idx = cache_data['chord_to_idx']
                    self.train_song_ids = cache_data['train_song_ids']
                    self.val_song_ids = cache_data['val_song_ids']
                    # Optional: Load segment indices if cached
                    self.train_segment_indices = cache_data.get('train_segment_indices', [])
                    self.val_segment_indices = cache_data.get('val_segment_indices', [])

                    if self.verbose:
                        print(f"Loaded Fold {self.current_fold} data from cache: {self.cache_file}")
                        print(f"  Cached Train songs: {len(self.train_song_ids)}, Val songs: {len(self.val_song_ids)}")
                    # If segments were cached, we can skip generation later
                    if self.train_segment_indices and self.val_segment_indices:
                        self._segments_generated = True # Flag to skip generation
                    return # Successfully loaded from cache
                else:
                    if self.verbose: print(f"Cache format invalid or missing keys, rebuilding dataset for Fold {self.current_fold}")
            except Exception as e:
                if self.verbose: print(f"Error loading cache for Fold {self.current_fold}, rebuilding: {e}")
        # --- End Cache Loading ---

        # --- Scan Files and Identify Songs ---
        if self.verbose: print("Scanning files for 'labeled_synth' structure...")
        all_spec_files = {}  # {file_id: path}
        label_files_dict = {}  # {file_id: path}
        logit_files_dict = {}  # {file_id: path}
        song_ids = []  # Combined list of all song IDs
        file_id_to_subdir = {}  # {file_id: subdir}

        # Scan Spectrograms
        for subdir in self.labeled_synth_subdirs:
            subdir_path = self.spec_dir / subdir
            if subdir_path.exists():
                for spec_path in subdir_path.glob("**/*_spec.npy"):
                    match = self.spec_pattern.search(spec_path.name)
                    if match:
                        base_name = match.group(1)
                        file_id = f"{subdir}/{base_name}"
                        all_spec_files[file_id] = spec_path
                        song_ids.append(file_id)
                        file_id_to_subdir[file_id] = subdir
            elif self.verbose:
                print(f"  Spectrogram subdir not found: {subdir_path}")

        # Scan Labels
        for subdir in self.labeled_synth_subdirs:
            subdir_path = self.label_dir / subdir
            if subdir_path.exists():
                for label_path in list(subdir_path.glob("**/*.lab")) + list(subdir_path.glob("**/*.txt")):
                    match = self.label_pattern.search(label_path.name)
                    if match:
                        base_name = match.group(1)
                        file_id = f"{subdir}/{base_name}"
                        label_files_dict[file_id] = label_path
            elif self.verbose:
                print(f"  Label subdir not found: {subdir_path}")

        # Scan Logits (if directory provided)
        if self.logits_dir:
            for subdir in self.labeled_synth_subdirs:
                subdir_path = self.logits_dir / subdir
                if subdir_path.exists():
                    for logit_path in subdir_path.glob("**/*_logits.npy"):
                        match = self.logit_pattern.search(logit_path.name)
                        if match:
                            base_name = match.group(1)
                            file_id = f"{subdir}/{base_name}"
                            logit_files_dict[file_id] = logit_path
                elif self.verbose:
                    print(f"  Logits subdir not found: {subdir_path}")

        if not all_spec_files:
            raise FileNotFoundError(f"No spectrogram files found in subdirs {self.labeled_synth_subdirs} under {self.spec_dir}")

        if self.verbose:
            print(f"Found {len(all_spec_files)} potential tracks across specified subdirs.")

        # --- Shuffle and Mix Songs ---
        np.random.seed(42)  # For reproducible shuffling
        np.random.shuffle(song_ids)

        # --- K-Fold Split Logic with Stratification ---
        if not song_ids:
            warnings.warn(f"No songs found in labeled_synth_subdirs {self.labeled_synth_subdirs} for splitting.")
            self.train_song_ids = []
            self.val_song_ids = []
        else:
            # Create stratification labels based on subdirectories to ensure balanced distribution
            # This ensures each fold has a similar distribution of songs from each subdirectory
            strat_labels = []
            for song_id in song_ids:
                subdir = song_id.split('/')[0]  # Extract subdirectory from song_id
                strat_labels.append(subdir)

            # Use StratifiedKFold to ensure balanced distribution of subdirectories across folds
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)
            all_splits = list(skf.split(song_ids, strat_labels))

            if self.current_fold >= len(all_splits):
                raise IndexError(f"current_fold {self.current_fold} is out of bounds for the generated splits.")

            train_indices, val_indices = all_splits[self.current_fold]

            # Get the song IDs for train and validation based on the current fold
            # Don't sort to preserve the stratified order from the split
            self.train_song_ids = [song_ids[i] for i in train_indices]
            self.val_song_ids = [song_ids[i] for i in val_indices]

            # Log the distribution of subdirectories in each split for verification
            if self.verbose:
                train_subdirs = [s.split('/')[0] for s in self.train_song_ids]
                val_subdirs = [s.split('/')[0] for s in self.val_song_ids]

                train_dist = Counter(train_subdirs)
                val_dist = Counter(val_subdirs)

                print(f"Subdirectory distribution in training set:")
                for subdir, count in train_dist.items():
                    print(f"  {subdir}: {count} songs ({count/len(train_subdirs)*100:.1f}%)")

                print(f"Subdirectory distribution in validation set:")
                for subdir, count in val_dist.items():
                    print(f"  {subdir}: {count} songs ({count/len(val_subdirs)*100:.1f}%)")

        if self.verbose:
            print(f"Fold {self.current_fold}: Splitting {len(song_ids)} songs across all subdirs")
            print(f"  Total Train Songs: {len(self.train_song_ids)}")
            print(f"  Total Validation Songs: {len(self.val_song_ids)}")

        # --- Apply small_dataset_percentage ---
        if self.small_dataset_percentage is not None and 0.0 < self.small_dataset_percentage < 1.0:
            if self.verbose:
                print(f"Applying small_dataset_percentage: {self.small_dataset_percentage * 100:.1f}%")

            # Reduce training set size
            if self.train_song_ids:
                try:
                    _, self.train_song_ids = train_test_split(
                        self.train_song_ids,
                        test_size=self.small_dataset_percentage, # Keep this fraction
                        random_state=42 # Use a fixed seed for reproducibility
                    )
                    if self.verbose: print(f"  Reduced training songs to: {len(self.train_song_ids)}")
                except ValueError as e:
                     if "test_size=" in str(e): # Handle case where percentage is too small
                         warnings.warn(f"Could not apply small_dataset_percentage ({self.small_dataset_percentage}) to training set (size {len(self.train_song_ids)}). Keeping original set. Error: {e}")
                     else: raise e # Re-raise other errors

            # Reduce validation set size
            if self.val_song_ids:
                try:
                    _, self.val_song_ids = train_test_split(
                        self.val_song_ids,
                        test_size=self.small_dataset_percentage, # Keep this fraction
                        random_state=42
                    )
                    if self.verbose: print(f"  Reduced validation songs to: {len(self.val_song_ids)}")
                except ValueError as e:
                     if "test_size=" in str(e): # Handle case where percentage is too small
                         warnings.warn(f"Could not apply small_dataset_percentage ({self.small_dataset_percentage}) to validation set (size {len(self.val_song_ids)}). Keeping original set. Error: {e}")
                     else: raise e # Re-raise other errors

        # --- Prepare file lists for processing ---
        train_files_to_process = []
        val_files_to_process = []

        all_files_set = set(all_spec_files.keys())

        for song_id in self.train_song_ids:
            if song_id in all_files_set:
                spec_file = all_spec_files[song_id]
                # Skip if required files are missing
                label_file = label_files_dict.get(song_id)
                if not label_file or not os.path.exists(label_file): continue
                logit_file = logit_files_dict.get(song_id)
                if self.require_teacher_logits and (not logit_file or not os.path.exists(logit_file)): continue

                train_files_to_process.append((spec_file, song_id, label_files_dict, logit_files_dict, True)) # Add return_skip_reason flag

        for song_id in self.val_song_ids:
             if song_id in all_files_set:
                spec_file = all_spec_files[song_id]
                # Skip if required files are missing
                label_file = label_files_dict.get(song_id)
                if not label_file or not os.path.exists(label_file): continue
                logit_file = logit_files_dict.get(song_id)
                if self.require_teacher_logits and (not logit_file or not os.path.exists(logit_file)): continue

                val_files_to_process.append((spec_file, song_id, label_files_dict, logit_files_dict, True)) # Add return_skip_reason flag


        # --- Process Files using Multiprocessing ---
        num_cpus = max(1, self.num_workers) if self.num_workers > 0 else os.cpu_count() // 2
        num_cpus = max(1, num_cpus) # Ensure at least 1 worker

        self.train_samples = []
        self.val_samples = []
        self.skipped_reasons = Counter() # Combined skip reasons

        def run_processing(file_list, target_sample_list, desc):
            if not file_list: return 0, 0 # No files to process

            # Adjust num_cpus if file list is small
            current_num_cpus = min(num_cpus, len(file_list))
            if current_num_cpus == 0: return 0, 0

            total_processed = 0
            total_skipped = 0

            args_list = [(self,) + item for item in file_list] # Add self as first arg for wrapper

            if self.verbose: print(f"Processing {len(args_list)} {desc} files with {current_num_cpus} workers...")

            try:
                with Pool(processes=current_num_cpus) as pool:
                    process_results = list(tqdm(
                        pool.imap(_process_file_wrapper, args_list),
                        total=len(args_list),
                        desc=f"Loading {desc} data (Fold {self.current_fold})",
                        disable=not self.verbose
                    ))

                for samples, skip_reason in process_results:
                    total_processed += 1
                    if samples:
                        target_sample_list.extend(samples)
                    elif skip_reason:
                        total_skipped += 1
                        self.skipped_reasons[skip_reason] += 1

            except Exception as e:
                error_msg = traceback.format_exc()
                if self.verbose:
                    print(f"ERROR during multiprocessing for {desc}: {e}\n{error_msg}")
                    print("Attempting sequential fallback...")

                # Sequential fallback
                for args in tqdm(args_list,
                                desc=f"Loading {desc} data (sequential fallback, Fold {self.current_fold})",
                                disable=not self.verbose):
                    try:
                        samples, skip_reason = _process_file_wrapper(args)
                        total_processed += 1
                        if samples:
                            target_sample_list.extend(samples)
                        elif skip_reason:
                            total_skipped += 1
                            self.skipped_reasons[skip_reason] += 1
                    except Exception as fallback_e:
                         if self.verbose: print(f"Error in sequential fallback for {args[1]}: {fallback_e}")
                         total_skipped += 1
                         self.skipped_reasons['sequential_error'] += 1

            return total_processed, total_skipped

        # Process training files
        train_processed, train_skipped = run_processing(train_files_to_process, self.train_samples, "training")
        # Process validation files
        val_processed, val_skipped = run_processing(val_files_to_process, self.val_samples, "validation")

        total_processed = train_processed + val_processed
        total_skipped = train_skipped + val_skipped

        if self.verbose and total_processed > 0:
            skip_percentage = (total_skipped / total_processed) * 100 if total_processed > 0 else 0
            print(f"\nFile processing statistics (Fold {self.current_fold}):")
            print(f"  Total attempted: {total_processed}")
            print(f"  Skipped: {total_skipped} ({skip_percentage:.1f}%)")
            if total_skipped > 0:
                for reason, count in self.skipped_reasons.items():
                    reason_pct = (count / total_skipped) * 100
                    print(f"    - {reason}: {count} ({reason_pct:.1f}%)")
            print(f"  Loaded Train Samples: {len(self.train_samples)}")
            print(f"  Loaded Validation Samples: {len(self.val_samples)}")


        # --- Dynamically build chord_to_idx if chord_mapping was None ---
        if self.chord_mapping is None:
            if self.verbose: print("Building chord_to_idx dynamically from all loaded samples...")
            all_labels_set = set()
            for sample_list in [self.train_samples, self.val_samples]:
                for sample in sample_list:
                    # 'chord_label' from _process_file via _validate_chord_label
                    # is already the normalized string label.
                    all_labels_set.add(sample['chord_label'])

            # Ensure "N" (No Chord) is handled, typically mapped to 0.
            final_chord_to_idx = {}
            if "N" in all_labels_set:
                final_chord_to_idx["N"] = 0
                all_labels_set.remove("N")
                next_idx = 1
            else:
                # If "N" wasn't found in data but is expected, add it.
                final_chord_to_idx["N"] = 0
                next_idx = 1
                if self.verbose: warnings.warn("Label 'N' not found in dataset, adding it to chord_to_idx with index 0.")

            sorted_labels = sorted(list(all_labels_set)) # Sort for deterministic mapping

            for label in sorted_labels:
                if label not in final_chord_to_idx: # Should not happen if "N" handled correctly
                    final_chord_to_idx[label] = next_idx
                    next_idx += 1

            self.chord_to_idx = final_chord_to_idx

            # Add plain note names as aliases for major chords (simplified for dynamic build)
            for root_note_name in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                maj_chord_name = f"{root_note_name}:maj"
                # If the major chord (e.g., "C:maj") exists in our map,
                # and the root note (e.g., "C") isn't already mapped (e.g. as a different chord type),
                # then map the root note as an alias to the major chord's index.
                if maj_chord_name in self.chord_to_idx and root_note_name not in self.chord_to_idx:
                    self.chord_to_idx[root_note_name] = self.chord_to_idx[maj_chord_name]

            if self.verbose: print(f"Dynamically built chord_to_idx with {len(self.chord_to_idx)} unique entries (aliases included).")


        # --- Caching Results ---
        if self.use_cache and (self.train_samples or self.val_samples):
            try:
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir: os.makedirs(cache_dir, exist_ok=True)

                # Prepare data for caching (handle metadata_only)
                def prepare_samples_for_cache(samples_list):
                    if not self.metadata_only:
                        return samples_list
                    meta_list = []
                    for sample in samples_list:
                        meta = {k: sample[k] for k in sample if k not in ['spectro', 'teacher_logits']}
                        # Ensure essential paths are present if possible
                        if 'spec_path' not in meta and 'song_id' in sample:
                             subdir, basename = sample['song_id'].split('/', 1)
                             meta['spec_path'] = str(self.spec_dir / subdir / f"{basename}_spec.npy")
                        if self.logits_dir and 'logit_path' not in meta and 'song_id' in sample:
                             subdir, basename = sample['song_id'].split('/', 1)
                             meta['logit_path'] = str(self.logits_dir / subdir / f"{basename}_logits.npy")
                        meta_list.append(meta)
                    return meta_list

                cache_data = {
                    'train_samples': prepare_samples_for_cache(self.train_samples),
                    'val_samples': prepare_samples_for_cache(self.val_samples),
                    'chord_to_idx': self.chord_to_idx,
                    'train_song_ids': self.train_song_ids,
                    'val_song_ids': self.val_song_ids,
                    'metadata_only': self.metadata_only,
                    # Optionally cache segments if generated
                    'train_segment_indices': getattr(self, 'train_segment_indices', []),
                    'val_segment_indices': getattr(self, 'val_segment_indices', []),
                    'small_dataset_percentage': self.small_dataset_percentage # Add small_dataset_percentage to cache data
                }

                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                if self.verbose: print(f"Saved Fold {self.current_fold} dataset cache to {self.cache_file}")

            except Exception as e:
                if self.verbose: print(f"Error saving cache for Fold {self.current_fold}: {e}")
        # --- End Caching ---

        if not self.train_samples and not self.val_samples:
             warnings.warn(f"No samples loaded for Fold {self.current_fold}. Check data paths and structure.")

        if self.verbose:
            print(f"Data loading for Fold {self.current_fold} completed in {time.time() - start_time:.2f} seconds")

    # --- Segment Generation ---
    def _generate_segments(self):
        """Generate segments for both training and validation sets."""
        # Check if segments were loaded from cache
        if hasattr(self, '_segments_generated') and self._segments_generated:
            if self.verbose: print("Segments loaded from cache, skipping generation.")
            return

        start_time = time.time()
        if self.verbose: print("Generating segments...")

        def generate_for_set(samples_list, set_name):
            segment_indices = []
            if not samples_list:
                if self.verbose: print(f"No samples in {set_name} set to generate segments from.")
                return segment_indices

            # Group samples by song_id
            song_samples = {}
            for i, sample in enumerate(samples_list):
                song_id = sample['song_id']
                if song_id not in song_samples:
                    song_samples[song_id] = []
                # Store the original index within the samples_list
                song_samples[song_id].append(i)

            if self.verbose: print(f"Found {len(song_samples)} unique songs in {set_name} set.")

            total_segments = 0
            # Add tqdm progress bar for song processing
            song_items = song_samples.items()
            if self.verbose and len(song_items) > 10:
                song_items = tqdm(song_items, desc=f"Processing songs for {set_name} set", total=len(song_samples))

            for song_id, indices in song_items:
                # indices are relative to the samples_list (train_samples or val_samples)
                num_frames_in_song = len(indices)

                if num_frames_in_song == 0: continue

                # Handle songs shorter than seq_len - pad if needed during __getitem__
                if num_frames_in_song < self.seq_len:
                     # Create a single segment starting at the beginning of the song
                     # Padding will happen in __getitem__ or _get_item_val
                     segment_start_sample_idx = indices[0]
                     segment_end_sample_idx = segment_start_sample_idx + self.seq_len # Indicate desired length
                     segment_indices.append((segment_start_sample_idx, segment_end_sample_idx))
                     total_segments += 1
                     continue

                # Generate segments using stride for songs >= seq_len
                # Use tqdm for progress bar if verbose and enough segments to warrant it
                start_offsets = range(0, num_frames_in_song - self.seq_len + 1, self.stride)
                if self.verbose and len(start_offsets) > 1000:
                    start_offsets = tqdm(start_offsets, desc=f"Generating segments for song {song_id}", leave=False)

                for start_offset in start_offsets:
                    segment_start_sample_idx = indices[start_offset]
                    # End index is exclusive for slicing, but here represents the index AFTER the last frame
                    segment_end_sample_idx = segment_start_sample_idx + self.seq_len
                    segment_indices.append((segment_start_sample_idx, segment_end_sample_idx))
                    total_segments += 1

            if self.verbose: print(f"Generated {total_segments} segments for {set_name} set.")
            return segment_indices

        self.train_segment_indices = generate_for_set(self.train_samples, "training")
        self.val_segment_indices = generate_for_set(self.val_samples, "validation")
        self._segments_generated = True # Mark as generated

        if self.verbose:
            print(f"Segment generation completed in {time.time() - start_time:.2f} seconds")

    # --- Core Data Fetching Logic (`__getitem__` style) ---

    def _get_item_internal(self, idx, use_val_set):
        """Internal helper to get a segment (train or val)."""
        if use_val_set:
            segment_indices_list = self.val_segment_indices
            samples_list = self.val_samples
            set_name = "validation"
        else:
            segment_indices_list = self.train_segment_indices
            samples_list = self.train_samples
            set_name = "training"

        if not segment_indices_list:
            raise IndexError(f"No segments available in the {set_name} set for fold {self.current_fold}")
        if idx >= len(segment_indices_list):
             raise IndexError(f"Index {idx} out of range for {set_name} set with {len(segment_indices_list)} segments")

        seg_start_idx, seg_end_idx_exclusive = segment_indices_list[idx]
        # seg_end_idx_exclusive indicates the desired length (seg_start + seq_len)

        sequence = []
        label_seq = []
        teacher_logits_seq = []

        # Get first sample to determine song ID and potential expected length
        # Need to handle case where seg_start_idx might be out of bounds if samples_list is empty
        if not samples_list or seg_start_idx >= len(samples_list):
             # This shouldn't happen if segments were generated correctly, but handle defensively
             warnings.warn(f"Segment start index {seg_start_idx} out of bounds for {set_name} samples list (len {len(samples_list)}). Returning zeros.")
             # Return zero tensors matching expected output structure
             spec_shape = (self.seq_len, 144) # Assume 144 freq bins if unknown
             logit_shape = (self.seq_len, 170) # Assume 170 classes if unknown
             return {
                 'spectro': torch.zeros(spec_shape, dtype=torch.float, device=self.device),
                 'chord_idx': torch.zeros(self.seq_len, dtype=torch.long, device=self.device),
                 'teacher_logits': torch.zeros(logit_shape, dtype=torch.float, device=self.device) if self.require_teacher_logits else None
             }

        first_sample = samples_list[seg_start_idx]
        start_song_id = first_sample['song_id']
        expected_seq_frames = first_sample.get('expected_num_frames') # From metadata if available

        # --- Pre-load full spec if metadata_only and padding might be needed ---
        # (Adapted from SynthDataset.__getitem__)
        full_spec_loaded = None
        if self.metadata_only and expected_seq_frames is not None:
            try:
                spec_path = first_sample.get('spec_path')
                if spec_path and os.path.exists(spec_path):
                    full_spec_loaded = np.load(spec_path)
                    current_frames = full_spec_loaded.shape[0]
                    if current_frames < expected_seq_frames:
                        pad_width = expected_seq_frames - current_frames
                        full_spec_loaded = np.pad(full_spec_loaded, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
                # else: handle missing file during frame processing loop
            except Exception as e:
                 warnings.warn(f"Error pre-loading/padding spec {first_sample.get('spec_path')} in _get_item_internal: {e}")
                 full_spec_loaded = None
        # --- End Pre-load ---

        # Iterate through the desired sequence length
        for frame_offset in range(self.seq_len):
            current_sample_idx = seg_start_idx + frame_offset

            # Check if we are beyond the actual available samples for this song/segment
            if current_sample_idx >= len(samples_list) or samples_list[current_sample_idx]['song_id'] != start_song_id:
                # Reached end of song or dataset boundary before seq_len: Pad
                padding_shape = sequence[-1].shape if sequence else (144,) # Infer shape or default
                sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=self.device))
                label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=self.device))

                if teacher_logits_seq or self.require_teacher_logits: # Pad logits if needed
                    logits_shape = teacher_logits_seq[0].shape if teacher_logits_seq else (170,) # Infer shape or default
                    teacher_logits_seq.append(torch.zeros(logits_shape, dtype=torch.float, device=self.device))
                continue # Continue padding until seq_len

            # --- Get data for the current frame ---
            sample_i = samples_list[current_sample_idx]

            # Load spectrogram (from memory, pre-loaded, or disk)
            spec_vec = None
            # ... (Spectrogram loading logic adapted from SynthDataset.__getitem__) ...
            # --- Start Spectrogram Loading ---
            if 'spectro' in sample_i: # Already in memory (full load mode)
                spec_data = sample_i['spectro']
                if isinstance(spec_data, np.ndarray):
                    spec_vec = torch.from_numpy(spec_data).to(dtype=torch.float, device=self.device)
                elif torch.is_tensor(spec_data):
                    spec_vec = spec_data.clone().detach().to(device=self.device, dtype=torch.float)
                else: # Handle unexpected type
                    padding_shape = sequence[-1].shape if sequence else (144,)
                    spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
            elif full_spec_loaded is not None: # Use pre-loaded and potentially padded spec (metadata mode)
                frame_idx = sample_i.get('frame_idx')
                if frame_idx is not None and frame_idx < full_spec_loaded.shape[0]:
                     spec_vec = torch.from_numpy(full_spec_loaded[frame_idx]).to(dtype=torch.float, device=self.device)
                else: # Frame index out of bounds or missing
                     padding_shape = (full_spec_loaded.shape[1],) if len(full_spec_loaded.shape) > 1 else (144,)
                     spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
            elif 'spec_path' in sample_i: # Load from disk (metadata mode, no padding needed or pre-load failed)
                try:
                    spec_path = sample_i['spec_path']
                    if not os.path.exists(spec_path):
                         raise FileNotFoundError(f"Spectrogram file not found: {spec_path}")

                    spec = np.load(spec_path)
                    frame_idx = sample_i.get('frame_idx')

                    if frame_idx is not None and len(spec.shape) > 1:
                        if frame_idx < spec.shape[0]:
                            spec_vec = torch.from_numpy(spec[frame_idx]).to(dtype=torch.float, device=self.device)
                        else: # Index out of bounds for this specific file
                            padding_shape = (spec.shape[1],)
                            spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
                    elif len(spec.shape) > 1 and frame_idx is None:
                         spec_vec = torch.from_numpy(spec[0]).to(dtype=torch.float, device=self.device) # Use frame 0?
                    else: # Single-frame spectrogram or 1D array
                        spec_vec = torch.from_numpy(spec).to(dtype=torch.float, device=self.device)

                except Exception as e:
                    warnings.warn(f"Error loading spec {sample_i.get('spec_path')} in _get_item_internal: {e}")
                    padding_shape = sequence[-1].shape if sequence else (144,)
                    spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
            else: # No spectrogram data available
                padding_shape = sequence[-1].shape if sequence else (144,)
                spec_vec = torch.zeros(padding_shape, dtype=torch.float, device=self.device)
            # --- End Spectrogram Loading ---


            # Get chord label index
            chord_label = sample_i['chord_label']
            # Use the robust validation/mapping function
            validated_chord_label = self._validate_chord_label(chord_label, sample_i.get('spec_path', 'unknown file'))
            chord_idx = self.chord_to_idx.get(validated_chord_label, self.chord_to_idx.get("N", 0))
            chord_idx_tensor = torch.tensor(chord_idx, dtype=torch.long, device=self.device)

            sequence.append(spec_vec)
            label_seq.append(chord_idx_tensor)

            # Handle teacher logits
            # ... (Logits loading/handling logic adapted from SynthDataset.__getitem__) ...
            # --- Start Logits Handling ---
            logit_loaded_or_found = False
            current_logit_tensor = None
            if 'teacher_logits' in sample_i: # Already in memory
                logit_data = sample_i['teacher_logits']
                if isinstance(logit_data, np.ndarray):
                    current_logit_tensor = torch.from_numpy(logit_data).to(dtype=torch.float, device=self.device)
                elif torch.is_tensor(logit_data):
                    current_logit_tensor = logit_data.clone().detach().to(device=self.device, dtype=torch.float)
                # else: handle unexpected type? Assume None for now.

                if current_logit_tensor is not None:
                    logit_loaded_or_found = True

            elif 'logit_path' in sample_i: # Load from disk
                try:
                    logit_path = sample_i['logit_path']
                    if not os.path.exists(logit_path):
                         if self.require_teacher_logits: raise FileNotFoundError(f"Required teacher logits file not found: {logit_path}")
                         else: raise FileNotFoundError # Caught below

                    # Use the corrected _load_logits_file method (handles squeeze)
                    logits_array = self._load_logits_file(logit_path)
                    if logits_array is None: raise RuntimeError(f"Failed to load logits from {logit_path}")

                    frame_idx = sample_i.get('frame_idx')
                    if frame_idx is not None and len(logits_array.shape) > 0: # Check ndim > 0
                        # Assuming logits_array is now (time, classes) or (classes,)
                        if len(logits_array.shape) > 1: # (time, classes)
                            if frame_idx < logits_array.shape[0]:
                                logits_vec = logits_array[frame_idx]
                            else: # Index out of bounds
                                logits_vec = np.zeros(logits_array.shape[1])
                        else: # (classes,) - assume single frame or broadcast
                            logits_vec = logits_array
                    elif len(logits_array.shape) > 0: # No frame index, use first frame or broadcast
                         logits_vec = logits_array[0] if len(logits_array.shape) > 1 else logits_array
                    else: # Empty array?
                         logits_vec = None # Indicate failure

                    if logits_vec is not None:
                        current_logit_tensor = torch.from_numpy(logits_vec).to(dtype=torch.float, device=self.device)
                        logit_loaded_or_found = True

                except Exception as e:
                    # Log error but don't necessarily stop if logits not required
                    if self.require_teacher_logits:
                        if self.verbose: print(f"ERROR: Failed to load required logits at getitem for {sample_i.get('logit_path', 'unknown path')}: {e}")
                    # Pass to handle padding later if required
                    pass

            # Append the loaded tensor or handle padding/zeros
            if logit_loaded_or_found and current_logit_tensor is not None:
                 teacher_logits_seq.append(current_logit_tensor)
            elif self.require_teacher_logits:
                 # Append a placeholder, shape will be determined later
                 teacher_logits_seq.append(torch.empty(0, dtype=torch.float, device=self.device)) # Placeholder
            # If not required and not found, do nothing for this frame's logits
            # --- End Logits Handling ---


        # --- Final Assembly and Padding/Stacking ---
        # Ensure sequence length is exactly self.seq_len (already handled by loop and padding)
        if len(sequence) != self.seq_len:
             warnings.warn(f"Sequence length mismatch: expected {self.seq_len}, got {len(sequence)} for idx {idx} ({set_name}). Re-padding.")
             # Add padding logic here if the loop didn't cover it (shouldn't be needed)
             while len(sequence) < self.seq_len:
                  padding_shape = sequence[-1].shape if sequence else (144,)
                  sequence.append(torch.zeros(padding_shape, dtype=torch.float, device=self.device))
                  label_seq.append(torch.tensor(self.chord_to_idx.get("N", 0), dtype=torch.long, device=self.device))
                  if teacher_logits_seq or self.require_teacher_logits:
                       logits_shape = teacher_logits_seq[0].shape if teacher_logits_seq and teacher_logits_seq[0].numel() > 0 else (170,)
                       teacher_logits_seq.append(torch.zeros(logits_shape, dtype=torch.float, device=self.device))


        # Stack sequences
        try:
            stacked_spectro = torch.stack(sequence, dim=0)
            stacked_labels = torch.stack(label_seq, dim=0)
        except Exception as stack_err:
             print(f"Error stacking sequence/labels for idx {idx} ({set_name}): {stack_err}")
             # Print shapes for debugging
             for i, t in enumerate(sequence): print(f" Spec {i} shape: {t.shape}")
             # Return zeros as fallback
             spec_shape = (self.seq_len, 144)
             logit_shape = (self.seq_len, 170)
             return {
                 'spectro': torch.zeros(spec_shape, dtype=torch.float, device=self.device),
                 'chord_idx': torch.zeros(self.seq_len, dtype=torch.long, device=self.device),
                 'teacher_logits': torch.zeros(logit_shape, dtype=torch.float, device=self.device) if self.require_teacher_logits else None
             }


        # Process teacher logits (stack and normalize shape)
        stacked_logits = None
        if self.require_teacher_logits or teacher_logits_seq: # Process if required OR if any were found
            fixed_num_classes = 170 # Define expected class number (adjust if needed)
            processed_logits_list = []
            expected_logit_shape = None

            for i, logits_tensor in enumerate(teacher_logits_seq):
                # Determine expected shape from the first valid tensor
                if expected_logit_shape is None and logits_tensor.numel() > 0 and logits_tensor.dim() > 0:
                    expected_logit_shape = (logits_tensor.shape[-1],) # Use last dim
                    # Override with fixed if necessary? For now, use detected shape.
                    # expected_logit_shape = (fixed_num_classes,)

                # Handle empty/zero/incorrectly shaped tensors
                current_shape = expected_logit_shape or (fixed_num_classes,)
                if logits_tensor.numel() == 0 or logits_tensor.dim() == 0:
                    normalized = torch.zeros(current_shape, dtype=torch.float, device=self.device)
                elif logits_tensor.dim() == 1:
                    if logits_tensor.shape[0] == current_shape[0]:
                        normalized = logits_tensor
                    else: # Pad or truncate 1D
                        normalized = torch.zeros(current_shape, dtype=torch.float, device=self.device)
                        copy_len = min(logits_tensor.shape[0], current_shape[0])
                        normalized[:copy_len] = logits_tensor[:copy_len]
                else: # More than 1D? Flatten and take first elements
                    flattened = logits_tensor.reshape(-1)
                    normalized = torch.zeros(current_shape, dtype=torch.float, device=self.device)
                    copy_len = min(flattened.shape[0], current_shape[0])
                    normalized[:copy_len] = flattened[:copy_len]

                processed_logits_list.append(normalized)

            # Ensure the list has the correct length (seq_len) for stacking
            while len(processed_logits_list) < self.seq_len:
                 current_shape = expected_logit_shape or (fixed_num_classes,)
                 processed_logits_list.append(torch.zeros(current_shape, dtype=torch.float, device=self.device))

            try:
                stacked_logits = torch.stack(processed_logits_list, dim=0)
            except Exception as logit_stack_err:
                 print(f"Error stacking logits for idx {idx} ({set_name}): {logit_stack_err}")
                 for i, t in enumerate(processed_logits_list): print(f" Logit {i} shape: {t.shape}")
                 # Fallback to zeros if required
                 if self.require_teacher_logits:
                      stacked_logits = torch.zeros((self.seq_len, fixed_num_classes), dtype=torch.float, device=self.device)


        # --- Construct Output ---
        sample_out = {
            'spectro': stacked_spectro,
            'chord_idx': stacked_labels
        }
        if stacked_logits is not None:
            sample_out['teacher_logits'] = stacked_logits
        elif self.require_teacher_logits: # Ensure key exists if required, even if stacking failed
             sample_out['teacher_logits'] = torch.zeros((self.seq_len, fixed_num_classes), dtype=torch.float, device=self.device)


        # --- GPU Batch Caching (Optional) ---
        # Cache based on original index and whether it's train/val
        if self.batch_gpu_cache and self.device.type == 'cuda':
            cache_key = (idx, use_val_set)
            try:
                self.gpu_batch_cache[cache_key] = sample_out
                # Limit cache size
                if len(self.gpu_batch_cache) > 256: # Example limit
                    oldest_key = next(iter(self.gpu_batch_cache))
                    del self.gpu_batch_cache[oldest_key]
            except Exception as e:
                # Disable caching on error?
                if self.verbose and not hasattr(self, '_cache_error_warning'):
                    print(f"Warning: Error in GPU batch caching: {e}. Disabling cache.")
                    self._cache_error_warning = True
                self.batch_gpu_cache = None # Disable further caching attempts
                self.gpu_batch_cache = {}

        return sample_out


    def __len__(self):
        """Return the number of training segments for the current fold."""
        return len(self.train_segment_indices)

    def __getitem__(self, idx):
        """Get a training segment by index."""
        return self._get_item_internal(idx, use_val_set=False)

    def _get_item_val(self, idx):
        """Get a validation segment by index (for use by validation iterator)."""
        return self._get_item_internal(idx, use_val_set=True)

    # --- Helper methods adapted from SynthDataset ---

    # _process_file: Handles loading/processing single file's frames
    def _process_file(self, spec_file, file_id, label_files_dict, logit_files_dict, return_skip_reason=False):
        """Process a single spectrogram file for 'labeled_synth' type."""
        samples = []
        skip_reason = None
        subdir, base_name = file_id.split('/', 1) # Guaranteed for labeled_synth

        # --- Strict File Existence Checks ---
        if not os.path.exists(str(spec_file)):
            skip_reason = 'missing_spec'
            if return_skip_reason: return [], skip_reason
            return []

        label_file = label_files_dict.get(file_id)
        if not label_file or not os.path.exists(str(label_file)):
            skip_reason = 'missing_label'
            if return_skip_reason: return [], skip_reason
            return []

        logit_file = None
        has_logit_file = False
        if self.logits_dir:
            logit_file = logit_files_dict.get(file_id)
            if logit_file and os.path.exists(str(logit_file)):
                has_logit_file = True

        if self.require_teacher_logits and not has_logit_file:
            skip_reason = 'missing_logits'
            if return_skip_reason: return [], skip_reason
            return []
        # --- End Checks ---

        try:
            # --- Load data (metadata or full) ---
            if self.metadata_only:
                # Load spec shape only
                try:
                    spec_info = np.load(spec_file, mmap_mode='r')
                    spec_shape = spec_info.shape
                    num_frames = spec_shape[0] if len(spec_shape) > 1 else 1
                    freq_bins = spec_shape[1] if len(spec_shape) > 1 else (spec_shape[0] if len(spec_shape) == 1 else 0)
                except Exception as e:
                     warnings.warn(f"Metadata load error for spec {spec_file}: {e}")
                     skip_reason = 'load_error'
                     if return_skip_reason: return [], skip_reason
                     return []

                expected_num_frames = num_frames
                logit_frames = 0
                # Check logit length if present
                if has_logit_file:
                    try:
                        logit_info = np.load(logit_file, mmap_mode='r')
                        # Handle potential (1, N, C) shape from older processing
                        if logit_info.ndim == 3 and logit_info.shape[0] == 1:
                             logit_frames = logit_info.shape[1] # Use N
                        elif logit_info.ndim >= 2: # Expecting (N, C)
                             logit_frames = logit_info.shape[0] # Use N
                        elif logit_info.ndim == 1 and num_frames == 1: # Single frame case
                             logit_frames = 1

                        if logit_frames > num_frames:
                            expected_num_frames = logit_frames # Store expected length
                    except Exception as e:
                        warnings.warn(f"Could not read logit shape for {logit_file}: {e}")

                chord_labels = self._parse_label_file(label_file)

                for t in range(expected_num_frames): # Iterate up to the potentially longer length
                    frame_time = t * self.frame_duration
                    chord_label = self._find_chord_at_time(chord_labels, frame_time)
                    chord_label = self._validate_chord_label(chord_label, label_file)

                    sample_meta = {
                        'spec_path': str(spec_file),
                        'chord_label': chord_label,
                        'song_id': file_id,
                        'frame_idx': t,
                        'dataset_type': self.dataset_type, # Store 'labeled_synth'
                        'subdir': subdir,
                        'expected_num_frames': expected_num_frames,
                        'freq_bins': freq_bins # Store freq bins
                    }
                    if has_logit_file:
                        sample_meta['logit_path'] = str(logit_file)
                    samples.append(sample_meta)

            else: # Load full data
                spec = np.load(spec_file)
                if np.isnan(spec).any(): spec = np.nan_to_num(spec, nan=0.0)
                num_frames = spec.shape[0] if len(spec.shape) > 1 else 1
                freq_bins = spec.shape[1] if len(spec.shape) > 1 else (spec.shape[0] if len(spec.shape) == 1 else 0)

                teacher_logits_full = None
                logit_frames = 0
                if has_logit_file:
                    try:
                        teacher_logits_full = self._load_logits_file(logit_file) # Handles squeeze
                        if teacher_logits_full is not None:
                            # Get time dimension correctly
                            if teacher_logits_full.ndim >= 1: # Expecting (N, C) or (C,)
                                logit_frames = teacher_logits_full.shape[0] if teacher_logits_full.ndim > 1 else 1

                            # Pad spectrogram if logits are longer
                            if logit_frames > num_frames:
                                pad_width = logit_frames - num_frames
                                spec = np.pad(spec, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
                                num_frames = logit_frames # Update num_frames
                            elif logit_frames > 0 and logit_frames < num_frames:
                                 warnings.warn(f"Logits ({logit_frames}) shorter than spec ({num_frames}) for {file_id}. Check generation.")
                                 # Decide whether to truncate spec or logits - let __getitem__ handle mismatch for now

                    except Exception as e:
                        warnings.warn(f"Error loading or processing logits file {logit_file}: {e}")
                        if self.require_teacher_logits: # Should have been caught earlier, but double-check
                            skip_reason = 'load_error'
                            if return_skip_reason: return [], skip_reason
                            return []
                        teacher_logits_full = None # Continue without logits

                chord_labels = self._parse_label_file(label_file)

                for t in range(num_frames): # Iterate using potentially updated num_frames
                    frame_time = t * self.frame_duration
                    chord_label = self._find_chord_at_time(chord_labels, frame_time)
                    chord_label = self._validate_chord_label(chord_label, label_file)

                    spec_frame = spec[t] if len(spec.shape) > 1 else spec

                    sample_dict = {
                        'spectro': spec_frame.astype(np.float32), # Ensure float32
                        'chord_label': chord_label,
                        'song_id': file_id,
                        'frame_idx': t,
                        'dataset_type': self.dataset_type,
                        'subdir': subdir,
                        'freq_bins': freq_bins
                    }

                    if teacher_logits_full is not None:
                        # Add logits for the current frame 't' if available
                        if len(teacher_logits_full.shape) > 1 and t < teacher_logits_full.shape[0]:
                            sample_dict['teacher_logits'] = teacher_logits_full[t].astype(np.float32)
                        elif len(teacher_logits_full.shape) == 1: # Single frame/broadcast case
                             sample_dict['teacher_logits'] = teacher_logits_full.astype(np.float32)
                        # Else: Logits might be shorter than spec, don't add for this 't'

                    samples.append(sample_dict)

        except Exception as e:
            warnings.warn(f"Error processing file {spec_file}: {str(e)}")
            traceback.print_exc()
            skip_reason = 'processing_error' # Generic processing error
            if return_skip_reason: return [], skip_reason
            return []

        if return_skip_reason:
            return samples, skip_reason
        return samples # Legacy return format if reason not requested

    def _load_logits_file(self, logit_file):
        """Load teacher logits file with error handling and shape correction"""
        # (Copied/adapted from SynthDataset)
        try:
            teacher_logits = np.load(logit_file)
            # Correct shape from (1, N, C) to (N, C) if needed
            if teacher_logits.ndim == 3 and teacher_logits.shape[0] == 1:
                teacher_logits = np.squeeze(teacher_logits, axis=0)
            # Handle NaNs
            if np.isnan(teacher_logits).any():
                teacher_logits = np.nan_to_num(teacher_logits, nan=0.0)
            return teacher_logits
        except Exception as e:
            if self.require_teacher_logits:
                raise RuntimeError(f"Error loading required logits file {logit_file}: {e}")
            if self.verbose: print(f"Warning: Error loading logits file {logit_file}: {e}")
            return None

    def _parse_label_file(self, label_file):
        """Parse a label file (.lab or .txt)"""
        # (Copied from SynthDataset)
        result = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            chord = " ".join(parts[2:]) # Handle chords with spaces
                            result.append((start_time, end_time, chord))
                        except ValueError:
                            if self.verbose: print(f"Skipping malformed line in {label_file}: {line.strip()}")
        except Exception as e:
            print(f"Error parsing label file {label_file}: {e}")
        return result

    def _find_chord_at_time(self, chord_labels, time):
        """Find the chord label at a specific time point"""
        # (Copied from SynthDataset)
        if not chord_labels: return "N"
        for start, end, chord in chord_labels:
            # Use a small epsilon for time comparison robustness?
            if start <= time < end:
                return chord
        # If time is beyond the last annotated segment, return the last chord
        if chord_labels and time >= chord_labels[-1][1]:
            return chord_labels[-1][2]
        return "N" # Default if no chord found (e.g., time before first chord)

    def _validate_chord_label(self, chord_label, source_file=""):
        """Validate chord label against mapping using normalization."""
        # (Copied/adapted from SynthDataset)
        if not chord_label: return "N"

        # Normalize using the parser (handles enharmonics, quality variations)
        normalized_label = self.chord_parser.label_error_modify(chord_label)

        # Handle explicit No Chord or Unknown from parser
        if normalized_label == "N" or normalized_label == "X":
            return "N"

        # If a chord_mapping was provided at initialization, use it for validation.
        if self.chord_mapping is not None:
            if normalized_label in self.chord_to_idx:
                return normalized_label
            else:
                # Normalized label not in the provided mapping
                warn_key = f'_warned_norm_{normalized_label}'
                if self.verbose and not hasattr(self, warn_key):
                     warnings.warn(f"Normalized chord '{normalized_label}' (from '{chord_label}' in {os.path.basename(str(source_file))}) not in provided chord_mapping. Using 'N'.")
                     setattr(self, warn_key, True)
                return "N"
        else:
            # If chord_mapping was None (dynamic building mode):
            # This method, when called by workers in _process_file,
            # should just return the normalized string.
            # The actual self.chord_to_idx map is built centrally in _load_data afterwards.
            # When called later by __getitem__, self.chord_to_idx will be populated,
            # and __getitem__ will handle the lookup.
            return normalized_label

    def _init_gpu_cache(self):
        """Initialize GPU cache with common zero tensors."""
        # (Copied from SynthDataset)
        if not hasattr(self, 'device') or self.device.type != 'cuda': return
        # Pre-allocate common shapes
        common_shapes = [
            (self.seq_len, 144), (1, 144), # Spectrograms
            (self.seq_len, 170), (1, 170), # Logits (common size)
            (self.seq_len, 25), (1, 25)    # Logits (smaller size)
        ]
        for shape in common_shapes:
            if shape not in self._zero_spec_cache:
                self._zero_spec_cache[shape] = torch.zeros(shape, dtype=torch.float32, device=self.device)
            if shape not in self._zero_logit_cache:
                self._zero_logit_cache[shape] = torch.zeros(shape, dtype=torch.float32, device=self.device)

    def get_tensor_cache(self, shape, is_logits=False):
        """Get a cached zero tensor or create one."""
        # (Copied from SynthDataset)
        cache_dict = self._zero_logit_cache if is_logits else self._zero_spec_cache
        if shape in cache_dict:
            return cache_dict[shape].clone() # Return clone
        else:
            # Create and potentially cache? For now, just create.
            return torch.zeros(shape, dtype=torch.float32, device=self.device)

    def normalize_spectrogram(self, spec, mean=None, std=None):
        """Normalize a spectrogram tensor."""
        # (Copied from SynthDataset)
        if mean is None or std is None:
            spec_mean = torch.mean(spec)
            spec_std = torch.std(spec)
            spec_std = spec_std if spec_std > 1e-6 else 1.0 # Avoid division by zero
            return (spec - spec_mean) / spec_std
        else:
            std_safe = std if std > 1e-6 else 1.0
            return (spec - mean) / std_safe

    # --- Data Iterator Methods ---

    def _get_data_iterator(self, indices, name, batch_size, shuffle, num_workers, pin_memory, sampler, use_val_getitem):
        """Helper method to get a DataLoader."""
        if not indices:
            warnings.warn(f"No {name} segments available for Fold {self.current_fold}")
            # Return an empty DataLoader? Or None? Let's return empty.
            return DataLoader(
                CrossValSubset(self, [], use_val_getitem=use_val_getitem), # Empty subset
                batch_size=batch_size,
                shuffle=False, # No shuffling needed for empty set
                num_workers=0, # No workers needed
                pin_memory=False,
                sampler=None # No sampler needed
            )

        effective_num_workers = self.num_workers if num_workers is None else num_workers
        effective_pin_memory = self.pin_memory if pin_memory is None else pin_memory

        # Ensure pin_memory is False if num_workers is 0
        if effective_num_workers == 0:
            effective_pin_memory = False

        subset = CrossValSubset(self, indices, use_val_getitem=use_val_getitem)

        # Store the sampler for later epoch updates
        if name == "training":
            self._current_train_sampler = sampler
        else:
            self._current_val_sampler = sampler

        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False, # Sampler handles shuffling
            num_workers=effective_num_workers,
            pin_memory=effective_pin_memory,
            sampler=sampler,
            # collate_fn=None, # Use default collate
            # worker_init_fn=None # Add if needed for seeding workers
        )

    def update_epoch(self, epoch):
        """
        Update the epoch number for all samplers to ensure different shuffling patterns.
        Call this method at the beginning of each epoch in your training loop.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        # Update train sampler if it exists and has set_epoch method
        if hasattr(self, '_current_train_sampler') and self._current_train_sampler is not None:
            if hasattr(self._current_train_sampler, 'set_epoch'):
                self._current_train_sampler.set_epoch(epoch)
                if self.verbose and epoch == 0:
                    print(f"Updated training sampler epoch to {epoch}")

        # Update validation sampler if it exists and has set_epoch method
        if hasattr(self, '_current_val_sampler') and self._current_val_sampler is not None:
            if hasattr(self._current_val_sampler, 'set_epoch'):
                self._current_val_sampler.set_epoch(epoch)
                if self.verbose and epoch == 0:
                    print(f"Updated validation sampler epoch to {epoch}")

    def get_train_iterator(self, batch_size=128, shuffle=True, num_workers=None, pin_memory=None, sampler=None, use_song_aware_sampler=True):
        """
        Get DataLoader for the training set of the current fold.

        Args:
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data (ignored if sampler is provided)
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory in GPU
            sampler: Custom sampler (if provided, overrides shuffle and use_song_aware_sampler)
            use_song_aware_sampler: Whether to use song-aware shuffling (default: True)

        Returns:
            DataLoader for the training set
        """
        indices = list(range(len(self.train_segment_indices)))

        # Create appropriate sampler if none provided
        if sampler is None and shuffle:
            if use_song_aware_sampler:
                # Use song-aware sampler for better shuffling
                sampler = SongAwareShuffleSampler(
                    self,
                    self.train_segment_indices,
                    self.train_song_ids,
                    seed=42
                )
                if self.verbose:
                    print("Using SongAwareShuffleSampler for training data")
            else:
                # Use epoch-dependent sampler for standard shuffling
                sampler = EpochDependentSampler(indices, seed=42)
                if self.verbose:
                    print("Using EpochDependentSampler for training data")

        return self._get_data_iterator(
            indices,
            "training",
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,  # Only shuffle if no sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            use_val_getitem=False  # Use main __getitem__
        )

    def get_val_iterator(self, batch_size=128, shuffle=False, num_workers=None, pin_memory=None, sampler=None):
        """
        Get DataLoader for the validation set of the current fold.

        Args:
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data (usually False for validation)
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory in GPU
            sampler: Custom sampler (if provided, overrides shuffle)

        Returns:
            DataLoader for the validation set
        """
        indices = list(range(len(self.val_segment_indices)))

        # Create epoch-dependent sampler if shuffle is True and no sampler provided
        if sampler is None and shuffle:
            sampler = EpochDependentSampler(indices, seed=43)  # Different seed from training
            if self.verbose:
                print("Using EpochDependentSampler for validation data")

        return self._get_data_iterator(
            indices,
            "validation",
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,  # Only shuffle if no sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            use_val_getitem=True  # Use _get_item_val
        )


# --- Custom Samplers for Improved Shuffling ---
class EpochDependentSampler(Sampler):
    """
    Sampler that shuffles indices with a different seed for each epoch.
    This ensures different shuffling patterns across epochs.
    """
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Use a different seed for each epoch by combining base seed and epoch number
        epoch_seed = self.seed + self.epoch
        g = torch.Generator()
        g.manual_seed(epoch_seed)

        # Create shuffled indices
        indices = torch.randperm(len(self.data_source), generator=g).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler. This affects the shuffling pattern.

        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch

class SongAwareShuffleSampler(Sampler):
    """
    Sampler that ensures segments from the same song are not scattered too widely.
    First shuffles songs, then shuffles segments within each song.
    """
    def __init__(self, dataset, segment_indices, song_ids, seed=42):
        self.dataset = dataset
        self.segment_indices = segment_indices
        self.song_ids = song_ids
        self.seed = seed
        self.epoch = 0

        # Group segments by song
        self.song_to_segments = self._group_segments_by_song()

    def _group_segments_by_song(self):
        """Group segment indices by song ID"""
        song_to_segments = {}

        for idx, (seg_start_idx, _) in enumerate(self.segment_indices):
            if seg_start_idx < len(self.dataset.train_samples):
                song_id = self.dataset.train_samples[seg_start_idx]['song_id']
                if song_id not in song_to_segments:
                    song_to_segments[song_id] = []
                song_to_segments[song_id].append(idx)

        return song_to_segments

    def __iter__(self):
        # Use a different seed for each epoch
        epoch_seed = self.seed + self.epoch
        rng = random.Random(epoch_seed)

        # Shuffle the song IDs
        song_ids = list(self.song_to_segments.keys())
        rng.shuffle(song_ids)

        # Create final indices list by taking segments from each song in shuffled order
        indices = []
        for song_id in song_ids:
            # Get segments for this song and shuffle them
            song_segments = self.song_to_segments[song_id].copy()
            rng.shuffle(song_segments)
            indices.extend(song_segments)

        return iter(indices)

    def __len__(self):
        return len(self.segment_indices)

    def set_epoch(self, epoch):
        """Set the epoch for this sampler"""
        self.epoch = epoch

# --- Subset Helper Class ---
class CrossValSubset(Dataset):
    """Helper Dataset class to wrap a subset of indices for CrossValidationDataset."""
    def __init__(self, dataset: CrossValidationDataset, indices: list, use_val_getitem: bool):
        self.dataset = dataset
        self.indices = indices
        self.use_val_getitem = use_val_getitem

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for subset with {len(self.indices)} indices")

        # Get the original segment index from our list
        original_segment_idx = self.indices[idx]

        # Call the appropriate getitem method on the parent dataset
        if self.use_val_getitem:
            return self.dataset._get_item_val(original_segment_idx)
        else:
            return self.dataset[original_segment_idx]

