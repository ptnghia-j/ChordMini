import os
import glob
import torch
import numpy as np
import random
import math
import librosa  # Add librosa import
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sortedcontainers import SortedList
from collections import defaultdict  # Import defaultdict

from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features
from modules.utils.chords import Chords, idx2voca_chord  # Import idx2voca_chord
from modules.preProcessing.preprocess import Preprocess, FeatureTypes

class CrossValidationDataset(Dataset):
    """
    Dataset for cross-validation training with knowledge distillation.
    Provides a K-fold split for training and validation.
    """
    def __init__(self, config, audio_dirs=None, label_dirs=None, chord_mapping=None, 
                 train=True, kfold=0, total_folds=5, cache_dir=None,
                 random_seed=42, device='cpu', teacher_model=None):
        """
        Initialize the cross-validation dataset.
        
        Args:
            config: Configuration object
            audio_dirs: List of directories containing audio files
            label_dirs: List of directories containing label files
            chord_mapping: Dictionary mapping chord names to indices
            train: Whether this is a training dataset or validation dataset
            kfold: Which fold to use for validation (0-indexed)
            total_folds: Total number of folds
            cache_dir: Directory to cache extracted features
            random_seed: Random seed for reproducibility
            device: Device to use
            teacher_model: Optional teacher model for generating predictions
        """
        super(CrossValidationDataset, self).__init__()
        self.config = config
        self.train = train
        self.kfold = kfold
        self.total_folds = total_folds
        self.device = device
        self.teacher_model = teacher_model
        self.chord_mapping = chord_mapping
        self.random_seed = random_seed
        
        # Initialize preprocessor to None here, before it's first accessed
        self._preprocessor = None

        # Determine and store use_large_voca consistently
        self.use_large_voca = self.config.feature.get('large_voca', False)
        logger.info(f"Dataset initialized with use_large_voca = {self.use_large_voca}")
        
        # Setup audio and label directories
        if audio_dirs is None:
            audio_dirs = [
                os.path.join(config.paths.get('root_path', '/data/music'), 'isophonic'),
                os.path.join(config.paths.get('root_path', '/data/music'), 'uspop'),
                os.path.join(config.paths.get('root_path', '/data/music'), 'robbiewilliams')
            ]
        self.audio_dirs = audio_dirs
        
        if label_dirs is None:
            # Default to using the same directories as audio
            self.label_dirs = audio_dirs
        else:
            self.label_dirs = label_dirs
        
        # Define audio_label_pairs list for analyze_label_files
        self.audio_label_pairs = []
        
        # Get N and X chord IDs based on the stored use_large_voca flag
        if self.chord_mapping:
            self.n_chord_id = self.chord_mapping.get("N", 169 if self.use_large_voca else 24)
            self.x_chord_id = self.chord_mapping.get("X", 168 if self.use_large_voca else 25)
        else:
            self.n_chord_id = 169 if self.use_large_voca else 24
            self.x_chord_id = 168 if self.use_large_voca else 25
        
        logger.info(f"Using N chord ID={self.n_chord_id}, X chord ID={self.x_chord_id}")
        
        # Setup cache directory
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Get MP3 and feature configuration
        mp3_config = config.mp3 if hasattr(config, 'mp3') else config.feature
        feature_config = config.feature if hasattr(config, 'feature') else {}
        
        # Create configuration strings for cache paths
        self.mp3_string = f"{mp3_config.get('song_hz', 22050)}_{mp3_config.get('inst_len', 10.0)}_{mp3_config.get('skip_interval', 5.0)}"
        
        feature_name = 'cqt'  # Default feature type
        self.feature_string = f"{feature_name}_{feature_config.get('n_bins', 144)}_{feature_config.get('bins_per_octave', 24)}_{feature_config.get('hop_length', 2048)}"
        
        # Set random seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize song data structures
        self.all_song_paths = {}
        self.song_names = []
        self.paths = []
        
        # Process all datasets and extract file paths
        self._find_audio_label_pairs()
        
        # Split data based on k-fold
        self._split_data_kfold()
        
        # Log dataset information
        if train:
            logger.info(f"Created training dataset with {len(self.paths)} samples from {len(self.song_names)} songs")
        else:
            logger.info(f"Created validation dataset with {len(self.paths)} samples from {len(self.song_names)} songs")
            
        # Flag to indicate if we've generated distillation data
        self._kd_data_generated = False

    def _get_preprocessor(self):
        """Lazy initialize the preprocessor when needed"""
        if not hasattr(self, '_preprocessor') or self._preprocessor is None:
            # Pass audio_dirs and label_dirs as dataset_names to handle custom directories
            all_dirs = self.audio_dirs + self.label_dirs
            self._preprocessor = Preprocess(
                self.config, 
                FeatureTypes.cqt,
                all_dirs,  # Pass all directories to be checked
                root_dir=self.config.paths.get('root_path', '')
            )
        return self._preprocessor
        
    def _find_audio_label_pairs(self):
        """Find all matching audio and label files across all directories"""
        used_song_names = []
        
        # For each dataset source
        for dataset_idx, (audio_dir, label_dir) in enumerate(zip(self.audio_dirs, self.label_dirs)):
            # Skip if directories don't exist
            if not os.path.exists(audio_dir):
                logger.warning(f"Audio directory not found: {audio_dir}")
                continue
                
            if not os.path.exists(label_dir):
                logger.warning(f"Label directory not found: {label_dir}")
                continue
                
            # Use the preprocessor to find all matching audio/label files
            preprocessor = self._get_preprocessor()
            file_list = preprocessor.get_all_files()
            
            # Populate audio_label_pairs list for analyze_label_files
            for song_name, lab_path, mp3_path, _ in file_list:
                self.audio_label_pairs.append((mp3_path, lab_path))
            
            # Create a dictionary of paths for each song
            for song_name, lab_path, mp3_path, _ in file_list:
                # Create cache path specific to this dataset
                dataset_name = Path(audio_dir).name
                cache_path = os.path.join(
                    self.cache_dir if self.cache_dir else os.path.join(audio_dir, 'cache'),
                    dataset_name,
                    self.mp3_string,
                    self.feature_string,
                    song_name.strip()
                )
                
                # Create cache directory if it doesn't exist
                os.makedirs(cache_path, exist_ok=True)
                
                # Store paths
                self.all_song_paths[song_name] = {
                    'audio_path': mp3_path,
                    'label_path': lab_path,
                    'cache_path': cache_path,
                    'dataset': dataset_name
                }
                used_song_names.append(song_name)
        
        # Sort song names to ensure consistent splitting
        self.song_names = SortedList(used_song_names)
        logger.info(f"Found {len(self.song_names)} unique songs across all datasets")

    def _split_data_kfold(self):
        """Split data into training and validation sets based on k-fold"""
        # Calculate splits for each fold
        total_songs = len(self.song_names)
        fold_size = total_songs // self.total_folds
        remainder = total_songs % self.total_folds
        
        # Calculate fold boundaries
        fold_boundaries = [0]
        for i in range(self.total_folds):
            fold_boundaries.append(fold_boundaries[-1] + fold_size + (1 if i < remainder else 0))
        
        # Determine which songs to use
        if self.train:
            selected_songs = []
            for k in range(self.total_folds):
                if k != self.kfold:
                    selected_songs.extend(self.song_names[fold_boundaries[k]:fold_boundaries[k+1]])
        else:
            selected_songs = self.song_names[fold_boundaries[self.kfold]:fold_boundaries[self.kfold+1]]
        
        # Select data paths based on selected songs
        for song_name in selected_songs:
            song_info = self.all_song_paths[song_name]
            
            # Check if features already exist in cache or need to be generated
            cache_files = glob.glob(os.path.join(song_info['cache_path'], "*.pt"))
            
            if cache_files:
                # Use cached files
                if not self.train:
                    # For validation, use only non-augmented data (1.00_0)
                    cache_files = [f for f in cache_files if "1.00_0" in f]
                self.paths.extend(cache_files)
            else:
                # Need to generate features
                if not self.train:
                    logger.warning(f"Cache files not found for validation song {song_name}. Will be processed on-the-fly.")
                    # Create an on-the-fly processing entry
                    self.paths.append({
                        'song_name': song_name,
                        'audio_path': song_info['audio_path'],
                        'label_path': song_info['label_path'],
                        'cache_path': song_info['cache_path'],
                        'needs_processing': True
                    })
                else:
                    logger.warning(f"Cache files not found for training song {song_name}. Will be processed on-the-fly.")
                    # Create an on-the-fly processing entry
                    self.paths.append({
                        'song_name': song_name,
                        'audio_path': song_info['audio_path'],
                        'label_path': song_info['label_path'],
                        'cache_path': song_info['cache_path'],
                        'needs_processing': True
                    })
        
        # Set song names to be the selected ones
        self.song_names = selected_songs
        
        # Log information
        logger.info(f"Fold {self.kfold}/{self.total_folds}: {'Training' if self.train else 'Validation'} set has {len(self.song_names)} songs and {len(self.paths)} instances")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index. Loads pre-cached data or handles on-the-fly processing placeholder.
        """
        path_info = self.paths[idx]

        if isinstance(path_info, str): # It's a path to a cached .pt file
            file_path = path_info
            try:
                # Load pre-cached data
                data = torch.load(file_path, map_location='cpu') # Load to CPU first

                # Ensure data has expected keys and convert to tensors
                spectro = torch.tensor(data['feature'], dtype=torch.float32)
                chord_idx = torch.tensor(data['chord'], dtype=torch.long)

                # Return in the expected dictionary format
                return {
                    'spectro': spectro,
                    'chord_idx': chord_idx,
                    'song_id': data.get('song_name', Path(file_path).stem), # Extract song_id if available
                    'start_frame': data.get('start_frame', 0) # Placeholder if not saved
                }
            except FileNotFoundError:
                logger.error(f"Cached file not found: {file_path}")
                # Return a dummy sample or raise error
                return self._get_dummy_sample()
            except Exception as e:
                logger.error(f"Error loading cached file {file_path}: {e}")
                # Return a dummy sample or raise error
                return self._get_dummy_sample()

        elif isinstance(path_info, dict) and path_info.get('needs_processing'):
            # Placeholder for on-the-fly processing
            # This part needs implementation similar to LabeledDataset._extract_samples
            # For now, log a warning and return a dummy sample
            logger.warning(f"On-the-fly processing requested for {path_info.get('song_name')} but not fully implemented. Returning dummy sample.")
            # TODO: Implement on-the-fly feature extraction and label processing here
            # feature, feature_per_second, song_length_second = audio_file_to_features(...)
            # chord_labels, timestamps = self._parse_label_file(...)
            # frame_level_chords = self._chord_labels_to_frames(...)
            # chord_indices = self._chord_names_to_indices(...)
            # Create segments...
            return self._get_dummy_sample(song_id=path_info.get('song_name', 'unknown_otf'))
        else:
            logger.error(f"Invalid path info at index {idx}: {path_info}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self, song_id="dummy"):
        """Returns a dummy sample to prevent crashes when loading fails."""
        seq_len = self.config.training.get('seq_len', 10)
        n_bins = self.config.feature.get('n_bins', 144)
        dummy_spectro = torch.zeros((seq_len, n_bins), dtype=torch.float32)
        dummy_chords = torch.full((seq_len,), self.n_chord_id, dtype=torch.long) # Use N chord index
        return {
            'spectro': dummy_spectro,
            'chord_idx': dummy_chords,
            'song_id': song_id,
            'start_frame': 0
        }

    def analyze_chord_distribution(self):
        """Analyze the distribution of chord qualities in the dataset."""
        logger.info("Analyzing chord quality distribution...")
        quality_counts = defaultdict(int)
        total_chords = 0

        # Use the stored N and X chord indices
        n_chord_idx = self.n_chord_id
        x_chord_idx = self.x_chord_id

        # Create reverse mapping for quality lookup if needed
        idx_to_chord_name = idx2voca_chord()
        expected_size = 170 if self.use_large_voca else 26
        if len(idx_to_chord_name) < expected_size - 5:
            logger.warning(f"idx_to_chord_name map size ({len(idx_to_chord_name)}) seems small for use_large_voca={self.use_large_voca}. Quality analysis might be inaccurate.")
            if self.chord_mapping:
                idx_to_chord_name = {v: k for k, v in self.chord_mapping.items()}

        # Iterate through all chord indices in all samples
        for path in self.paths:
            try:
                if isinstance(path, dict):
                    continue

                data = torch.load(path)

                for idx in data['chord']:
                    total_chords += 1
                    if idx == n_chord_idx:
                        quality_counts['No Chord'] += 1
                    elif idx == x_chord_idx:
                        quality_counts['Other'] += 1
                    else:
                        chord_name = idx_to_chord_name.get(idx)
                        if chord_name:
                            try:
                                quality = Chords().get_quality(chord_name)
                                quality_counts[quality or 'Other'] += 1
                            except Exception:
                                quality_counts['Other'] += 1
                        else:
                            quality_counts['Other'] += 1

            except Exception as e:
                logger.error(f"Error analyzing file {path}: {e}")

        if total_chords == 0:
            logger.warning("No chords found in samples for distribution analysis.")
            return

        logger.info("Chord quality distribution:")
        sorted_qualities = sorted(quality_counts.keys())
        for quality in sorted_qualities:
            count = quality_counts[quality]
            percentage = (count / total_chords) * 100
            logger.info(f"  {quality}: {count} samples ({percentage:.2f}%)")

        return {
            'total_chords': total_chords,
            'quality_counts': quality_counts
        }
