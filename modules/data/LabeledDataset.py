import os
import glob
import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
import traceback
from types import SimpleNamespace

from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features
from modules.utils.chords import Chords

class ConfigDict(dict):
    """
    A dictionary that allows attribute-style access (config.key) as well as item-style access (config['key']).
    This helps with compatibility between different access patterns in the codebase.
    """
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        # Allow attribute-style access for all items
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            # Special case for feature attribute since it's commonly used
            if name == 'feature':
                # Create an empty feature dictionary on-demand
                self[name] = ConfigDict()
                return self[name]
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
    def __setattr__(self, name, value):
        self[name] = value

class LabeledDataset(Dataset):
    """
    Dataset for real-world audio files with ground truth chord labels.
    Handles multiple audio sources and their matching labels.
    """
    def __init__(self, 
                 audio_dirs=None,
                 label_dirs=None,
                 chord_mapping=None,
                 seq_len=10,
                 stride=5,
                 cache_features=True,
                 cache_dir=None,
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 random_seed=42,
                 feature_config=None,
                 device='cpu'):
        """
        Initialize the dataset.
        
        Args:
            audio_dirs: List of directories containing audio files
            label_dirs: List of directories containing label files
            chord_mapping: Dictionary mapping chord names to indices
            seq_len: Sequence length for each sample (in frames)
            stride: Stride between consecutive sequences (in frames)
            cache_features: Whether to cache features in memory
            cache_dir: Directory to save cached features
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            random_seed: Random seed for reproducibility
            feature_config: Configuration for feature extraction
            device: Device to use for processing
        """
        super().__init__()
        
        if audio_dirs is None:
            audio_dirs = [
                "data/LabeledDataset/Audio/billboard",
                "data/LabeledDataset/Audio/caroleKing",
                "data/LabeledDataset/Audio/queen",
                "data/LabeledDataset/Audio/theBeatles"
            ]
        
        if label_dirs is None:
            label_dirs = [
                "data/LabeledDataset/Labels/billboardLabels",
                "data/LabeledDataset/Labels/caroleKingLabels",
                "data/LabeledDataset/Labels/queenLabels",
                "data/LabeledDataset/Labels/theBeatlesLabels"
            ]
        
        # Ensure audio_dirs and label_dirs have the same length
        if len(audio_dirs) != len(label_dirs):
            raise ValueError("audio_dirs and label_dirs must have the same length")
        
        self.audio_dirs = audio_dirs
        self.label_dirs = label_dirs
        self.chord_mapping = chord_mapping
        self.seq_len = seq_len
        self.stride = stride
        self.cache_features = cache_features
        self.cache_dir = cache_dir
        
        # Create a proper feature_config that works with both dict-style and attribute-style access
        raw_config = {} if feature_config is None else feature_config
        
        # Handle HParams objects or other dictionary-like objects
        if hasattr(raw_config, '__dict__'):
            raw_config = dict(raw_config)
        elif not isinstance(raw_config, dict):
            raw_config = {}
            
        # Make sure we have basic required structure
        if 'mp3' not in raw_config:
            raw_config['mp3'] = {
                'song_hz': 22050, 
                'inst_len': 10.0,  # Updated to match config
                'skip_interval': 5.0  # Added from config
            }
        elif isinstance(raw_config['mp3'], dict):
            # Make sure mp3 config has required keys
            if 'song_hz' not in raw_config['mp3']:
                raw_config['mp3']['song_hz'] = 22050
            if 'inst_len' not in raw_config['mp3']:
                raw_config['mp3']['inst_len'] = 10.0  # Updated to match config
            if 'skip_interval' not in raw_config['mp3']:
                raw_config['mp3']['skip_interval'] = 5.0  # Added from config
            
        # Make sure we have the 'feature' key with some default parameters
        if 'feature' not in raw_config:
            raw_config['feature'] = {
                'n_fft': 512, 
                'hop_length': 2048,  # Updated to match config
                'n_bins': 144,  # Updated to match config
                'bins_per_octave': 24,  # Updated to match config
                'hop_duration': 0.09288  # Added from config (2048/22050)
            }
        elif 'feature' in raw_config and isinstance(raw_config['feature'], dict):
            # Make sure feature config has required keys
            feature_defaults = {
                'n_fft': 512,
                'hop_length': 2048,  # Updated to match config
                'n_bins': 144,  # Updated to match config
                'bins_per_octave': 24,  # Updated to match config
                'hop_duration': 0.09288  # Added from config
            }
            
            # Add any missing defaults
            for key, default_value in feature_defaults.items():
                if key not in raw_config['feature']:
                    raw_config['feature'][key] = default_value
            
        # Convert to ConfigDict for dual access patterns
        self.feature_config = ConfigDict(raw_config)
        
        self.device = device
        
        # Create cache directory if needed
        if self.cache_features and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Cache directory: {self.cache_dir}")
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize Chords class for parsing and mapping operations
        self.chord_processor = Chords()
        if chord_mapping:
            self.chord_processor.set_chord_mapping(chord_mapping)
        self.chord_mapping = chord_mapping
        
        # Find all matched audio and label files
        logger.info("Finding audio and label files...")
        self.audio_label_pairs = self._find_matching_pairs()
        logger.info(f"Found {len(self.audio_label_pairs)} audio-label pairs")
        
        # Print some example pairs for debugging
        if self.audio_label_pairs:
            logger.info("Sample audio-label pairs:")
            for i, (audio, label) in enumerate(self.audio_label_pairs[:3]):
                logger.info(f"  Pair {i+1}: {os.path.basename(audio)} - {os.path.basename(label)}")
        
        # Extract all samples
        logger.info("Extracting samples from audio-label pairs...")
        self.samples = self._extract_samples()
        logger.info(f"Created {len(self.samples)} samples")
        
        # Handle empty dataset case
        if not self.samples:
            logger.warning("No samples were created. Creating dummy samples to avoid DataLoader error.")
            self._create_dummy_samples()
            logger.warning(f"Created {len(self.samples)} dummy samples for fallback.")
            
        # Split dataset into train, validation, and test sets
        self._split_dataset(train_ratio, val_ratio, test_ratio)

    def _parse_label_file(self, label_path):
        """
        Parse chord labels from a label file using direct file reading.
        
        Args:
            label_path: Path to label file
            
        Returns:
            Tuple of (chord_labels, timestamps)
            chord_labels: List of chord labels
            timestamps: List of (start_time, end_time) tuples
        """
        chord_labels = []
        timestamps = []
        
        try:
            # Read the file directly for greater control over parsing
            with open(label_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        chord_name = parts[2]
                        
                        # Store the original chord name, not a processed version
                        chord_labels.append(chord_name)
                        timestamps.append((start_time, end_time))
                        
            logger.debug(f"Parsed {len(chord_labels)} chord labels directly from {label_path}")
            return chord_labels, timestamps
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(label_path, 'r', encoding='latin-1') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            chord_name = parts[2]
                            
                            # Store the original chord name, not a processed version
                            chord_labels.append(chord_name)
                            timestamps.append((start_time, end_time))
                            
                logger.debug(f"Parsed {len(chord_labels)} chord labels with latin-1 encoding from {label_path}")
                return chord_labels, timestamps
            except Exception as e:
                logger.error(f"Error parsing label file with latin-1 encoding: {e}")
                return [], []
                
        except Exception as e:
            logger.error(f"Error parsing label file: {e}")
            return [], []

    def _chord_labels_to_frames(self, chord_labels, timestamps, num_frames, feature_per_second):
        """
        Convert chord labels to frame-level representation.
        
        Args:
            chord_labels: List of chord labels
            timestamps: List of (start_time, end_time) tuples
            num_frames: Total number of frames
            feature_per_second: Number of frames per second
            
        Returns:
            List of chord labels for each frame
        """
        # Initialize with "N" (no chord)
        frame_level_chords = ["N"] * num_frames
        
        # Calculate frame duration in seconds
        frame_duration = 1.0 / feature_per_second
        
        # Keep track of statistics for debugging
        assigned_frames = 0
        n_chord_frames = 0
        
        # FIRST PASS: Exact matching - assign chords where frame is fully within chord boundaries
        for i in range(num_frames):
            # Calculate time for this frame center
            frame_time = (i + 0.5) * frame_duration
            
            # Find chord that contains this time point
            found_chord = False
            for chord, (start, end) in zip(chord_labels, timestamps):
                # Use a small tolerance to account for floating point errors
                if start <= frame_time < end:
                    frame_level_chords[i] = chord
                    found_chord = True
                    assigned_frames += 1
                    break
            
            if not found_chord:
                n_chord_frames += 1
        
        # Log statistics if many frames are unassigned
        if n_chord_frames > 0.1 * num_frames:  # If more than 10% are still N chords
            n_percent = (n_chord_frames / num_frames) * 100
            logger.debug(f"Chord assignment: {assigned_frames}/{num_frames} frames assigned ({assigned_frames/num_frames:.1%})")
            logger.debug(f"N chord frames: {n_chord_frames}/{num_frames} ({n_percent:.1f}%)")
            
            # If almost all frames are N, this is likely a file format or parsing issue
            if n_chord_frames > 0.9 * num_frames:
                logger.warning(f"WARNING: {n_percent:.1f}% of frames labeled as N. Possible parsing issue with chord file.")
                
                # Show first few timestamps to aid debugging
                if timestamps:
                    logger.debug(f"First 3 timestamps: {timestamps[:3]}")
                    logger.debug(f"Song length in frames: {num_frames}, feature_per_second: {feature_per_second}")
        
        return frame_level_chords

    def _chord_names_to_indices(self, chord_names):
        """
        Convert chord names to indices using the Chords class's proven conversion method.
        
        Args:
            chord_names: List of chord names
            
        Returns:
            List of chord indices
        """
        if self.chord_mapping is None:
            # If no mapping is provided, return raw chord names
            return chord_names
        
        # Directly use the built-in chord mapping from chord_mapping dictionary first
        indices = []
        unknown_chords = set()
        use_large_voca = hasattr(self.feature_config, 'feature') and self.feature_config.feature.get('large_voca', False)
        
        # First try direct mapping using the chord_mapping dictionary - fastest approach
        for chord in chord_names:
            if chord in self.chord_mapping:
                indices.append(self.chord_mapping[chord])
            elif chord == "N":
                indices.append(self.chord_mapping.get("N", 169 if use_large_voca else 24))
            elif chord == "X":
                indices.append(self.chord_mapping.get("X", 168 if use_large_voca else 25))
            else:
                # For more complex chords, use the Chords processor
                try:
                    # First preprocess the chord using lab_file_error_modify if available
                    if hasattr(self.chord_processor, 'lab_file_error_modify'):
                        modified_chord = self.chord_processor.lab_file_error_modify([chord])[0]
                    else:
                        modified_chord = chord
                    
                    # Parse the chord components
                    root, bass, intervals, is_major = self.chord_processor.chord(modified_chord)
                    
                    # Convert to index based on vocabulary size
                    if use_large_voca:
                        # Extract quality from chord name if it contains a colon
                        quality = None
                        if ':' in modified_chord:
                            _, quality = modified_chord.split(':', 1)
                            if '/' in quality:
                                quality = quality.split('/', 1)[0]  # Remove bass note
                        else:
                            quality = 'maj'  # Default to major if no quality specified
                        
                        idx = self.chord_processor.convert_to_id_voca(root=root, quality=quality)
                    else:
                        # For standard vocabulary, use root and is_major
                        idx = self.chord_processor.convert_to_id(root=root, is_major=is_major)
                    
                    indices.append(idx)
                    
                except Exception as e:
                    # If conversion fails, use N as fallback
                    unknown_chords.add(chord)
                    indices.append(self.chord_mapping.get("N", 169 if use_large_voca else 24))
        
        # Log unknown chords
        if unknown_chords:
            if len(unknown_chords) < 20:
                logger.warning(f"Unknown chords (mapped to N): {unknown_chords}")
            else:
                logger.warning(f"Many unknown chords ({len(unknown_chords)}) found, showing first 10: {list(unknown_chords)[:10]}")
        
        return indices

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample by index"""
        sample = self.samples[idx]
        
        # Convert to torch tensors
        spectro = torch.tensor(sample['spectro'], dtype=torch.float32)
        chord_idx = torch.tensor(sample['chord_idx'], dtype=torch.long)
        
        return {
            'spectro': spectro,
            'chord_idx': chord_idx,
            'song_id': sample['song_id'],
            'start_frame': sample['start_frame'],
            'audio_path': sample['audio_path'],
            'label_path': sample['label_path']
        }

    def get_train_iterator(self, batch_size=32, shuffle=True, num_workers=2, pin_memory=True):
        """Get data loader for training set"""
        return DataLoader(
            Subset(self, self.train_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def get_val_iterator(self, batch_size=32, shuffle=False, num_workers=2, pin_memory=True):
        """Get data loader for validation set"""
        return DataLoader(
            Subset(self, self.val_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def get_test_iterator(self, batch_size=32, shuffle=False, num_workers=2, pin_memory=True):
        """Get data loader for test set"""
        return DataLoader(
            Subset(self, self.test_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def get_song_iterator(self, song_id, batch_size=32, shuffle=False, num_workers=2, pin_memory=True):
        """
        Get data loader for a specific song.
        
        Args:
            song_id: Song ID to filter by
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            
        Returns:
            DataLoader for the specified song
        """
        # Filter samples by song ID
        song_indices = [i for i, sample in enumerate(self.samples) if sample['song_id'] == song_id]
        
        return DataLoader(
            Subset(self, song_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
