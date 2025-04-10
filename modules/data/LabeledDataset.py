import os
import glob
import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset, sampler
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
        Parse chord labels from a label file using the Chords class.
        
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
            # First try to use the Chords class functionality
            use_large_voca = hasattr(self.feature_config, 'feature') and self.feature_config.feature.get('large_voca', False)
            
            try:
                if use_large_voca:
                    # Get converted chord with large vocabulary
                    df = self.chord_processor.get_converted_chord_voca(label_path)
                else:
                    # Get converted chord with standard vocabulary
                    df = self.chord_processor.get_converted_chord(label_path)
                
                # Extract chord labels and timestamps from dataframe
                for _, row in df.iterrows():
                    start_time = row['start']
                    end_time = row['end']
                    chord_id = row['chord_id']
                    
                    # Convert chord ID back to name using idx2voca_chord or chord mapping
                    if self.chord_mapping:
                        # Find the chord name from the mapping
                        for name, idx in self.chord_mapping.items():
                            if idx == chord_id:
                                chord_name = name
                                break
                        else:
                            # If chord ID not found in mapping, use N for no chord
                            chord_name = "N"
                    else:
                        # If no mapping provided, use the chord ID as string
                        chord_name = str(chord_id)
                    
                    chord_labels.append(chord_name)
                    timestamps.append((start_time, end_time))
                
                logger.debug(f"Parsed {len(chord_labels)} chord labels using Chords class from {label_path}")
                return chord_labels, timestamps
            
            except Exception as e:
                logger.warning(f"Failed to use Chords class for parsing {label_path}: {e}")
                logger.warning("Falling back to manual parsing")
                
            # Fall back to manual parsing if Chords class fails
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        chord = parts[2]
                        
                        chord_labels.append(chord)
                        timestamps.append((start_time, end_time))
        except UnicodeDecodeError:
            # Try with different encoding
            with open(label_path, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        chord = parts[2]
                        
                        chord_labels.append(chord)
                        timestamps.append((start_time, end_time))
        
        return chord_labels, timestamps
    
    def _chord_names_to_indices(self, chord_names):
        """
        Convert chord names to indices using the Chords class.
        
        Args:
            chord_names: List of chord names
            
        Returns:
            List of chord indices
        """
        if self.chord_mapping is None:
            # If no mapping is provided, return raw chord names
            return chord_names
        
        indices = []
        unknown_chords = set()
        
        # Check if large vocabulary is enabled
        use_large_voca = hasattr(self.feature_config, 'feature') and self.feature_config.feature.get('large_voca', False)
        
        for chord in chord_names:
            if chord in self.chord_mapping:
                indices.append(self.chord_mapping[chord])
            else:
                try:
                    # Try to use the Chords class to convert
                    # First, preprocess the chord name with lab_file_error_modify
                    modified_chord = chord
                    if hasattr(self.chord_processor, 'lab_file_error_modify'):
                        modified_chord = self.chord_processor.lab_file_error_modify([chord])[0]
                    
                    # Parse the chord and get its root and properties
                    root, bass, intervals, is_major = self.chord_processor.chord(modified_chord)
                    
                    # Convert to index based on vocabulary size
                    if use_large_voca:
                        # For large vocabulary, use convert_to_id_voca
                        quality = None
                        # Extract quality from chord name if it contains a colon
                        if ':' in modified_chord:
                            _, quality = modified_chord.split(':', 1)
                            if '/' in quality:
                                quality, _ = quality.split('/', 1)
                        
                        idx = self.chord_processor.convert_to_id_voca(root=root, quality=quality)
                    else:
                        # For standard vocabulary, use convert_to_id
                        idx = self.chord_processor.convert_to_id(root=root, is_major=is_major)
                    
                    indices.append(idx)
                    
                except Exception as e:
                    # If conversion fails, use fallback
                    if "X" in self.chord_mapping:
                        indices.append(self.chord_mapping["X"])  # Use 'X' (unknown) for unknown chords
                    elif "N" in self.chord_mapping:
                        indices.append(self.chord_mapping["N"])  # Fallback to 'N' if 'X' not available
                    else:
                        # If neither 'X' nor 'N' in mapping, use the first index
                        indices.append(0)
                    unknown_chords.add(chord)
        
        # Log unknown chords (at debug level)
        if 0 < len(unknown_chords) < 20:
            logger.debug(f"Unknown chords (mapped to X): {unknown_chords}")
        elif len(unknown_chords) >= 20:
            logger.debug(f"Many unknown chords ({len(unknown_chords)}) found, showing first 10: {list(unknown_chords)[:10]}")
        
        return indices

    def _extract_samples(self):
        """
        Extract samples from audio-label pairs.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        errors = {'feature': 0, 'label': 0, 'segment': 0, 'other': 0}
        
        for audio_path, label_path in tqdm(self.audio_label_pairs, desc="Processing audio files"):
            try:
                # Extract features from audio file
                if self.cache_features and self.cache_dir:
                    # Generate cache filename based on audio path
                    cache_filename = f"{Path(audio_path).stem}.npy"
                    cache_path = os.path.join(self.cache_dir, cache_filename)
                    
                    if os.path.exists(cache_path):
                        # Load from cache
                        try:
                            data = np.load(cache_path, allow_pickle=True).item()
                            feature = data['feature']
                            feature_per_second = data['feature_per_second']
                            song_length_second = data['song_length_second']
                        except Exception as e:
                            logger.warning(f"Error loading cached features for {audio_path}: {e}")
                            # Extract features with better error handling
                            feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, self.feature_config)
                            np.save(cache_path, {
                                'feature': feature,
                                'feature_per_second': feature_per_second,
                                'song_length_second': song_length_second
                            })
                    else:
                        # Extract features with better error handling
                        feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, self.feature_config)
                        np.save(cache_path, {
                            'feature': feature,
                            'feature_per_second': feature_per_second,
                            'song_length_second': song_length_second
                        })
                else:
                    # Extract features with better error handling
                    feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, self.feature_config)
                
                # Transpose to get (time, frequency) format
                feature = feature.T
                
                # Change to debug level for per-file details
                logger.debug(f"Feature shape for {os.path.basename(audio_path)}: {feature.shape}")
                
                # Parse chord labels from label file
                chord_labels, timestamps = self._parse_label_file(label_path)
                
                # Change to debug level for detailed label info
                logger.debug(f"Found {len(chord_labels)} chord labels in {os.path.basename(label_path)}")
                if chord_labels:
                    logger.debug(f"First few labels: {chord_labels[:5]}")
                    logger.debug(f"First few timestamps: {timestamps[:5]}")
                
                # Convert chord labels to frame-level representation
                num_frames = feature.shape[0]
                frame_level_chords = self._chord_labels_to_frames(
                    chord_labels, timestamps, num_frames, feature_per_second)
                
                # Convert chord names to indices
                chord_indices = self._chord_names_to_indices(frame_level_chords)
                
                # Verify we have enough frames for at least one segment
                if num_frames < self.seq_len:
                    logger.warning(f"Audio file {os.path.basename(audio_path)} has {num_frames} frames, which is less than seq_len={self.seq_len}. Skipping.")
                    errors['segment'] += 1
                    continue
                
                # Create segments with sequence length and stride
                segments_created = 0
                for i in range(0, num_frames - self.seq_len + 1, self.stride):
                    # Extract segment
                    feature_segment = feature[i:i+self.seq_len]
                    chord_indices_segment = chord_indices[i:i+self.seq_len]
                    
                    # Create sample dictionary
                    sample = {
                        'song_id': Path(audio_path).stem,
                        'spectro': feature_segment,
                        'chord_idx': chord_indices_segment,
                        'start_frame': i,
                        'audio_path': audio_path,
                        'label_path': label_path,
                        'feature_per_second': feature_per_second
                    }
                    
                    samples.append(sample)
                    segments_created += 1
                
                # Change to debug level for per-file segment creation info
                logger.debug(f"Created {segments_created} segments from {os.path.basename(audio_path)}")
                    
            except Exception as e:
                # Determine error type for statistics
                error_msg = str(e).lower()
                if 'feature' in error_msg:
                    errors['feature'] += 1
                elif 'label' in error_msg or 'parse' in error_msg:
                    errors['label'] += 1
                else:
                    errors['other'] += 1
                
                logger.error(f"Error processing {audio_path}: {e}")
                logger.error(traceback.format_exc())
        
        # Log error statistics
        total_errors = sum(errors.values())
        if total_errors > 0:
            logger.warning(f"Encountered errors during processing: {errors}")
            logger.warning(f"Total error rate: {total_errors/len(self.audio_label_pairs):.2%}")
        
        return samples
    
    def _create_dummy_samples(self, count=10):
        """Create dummy samples to avoid DataLoader errors when no real samples exist"""
        # Use the first chord_mapping index as a default if mapping exists
        default_idx = 0 if not self.chord_mapping else list(self.chord_mapping.values())[0]
        
        # Create dummy samples
        for i in range(count):
            # Create a dummy spectrogram of shape (seq_len, 144) - typical spectrogram shape
            dummy_spectro = np.zeros((self.seq_len, 144), dtype=np.float32)
            # Create dummy chord indices of shape (seq_len,)
            dummy_chords = np.full(self.seq_len, default_idx, dtype=np.int64)
            
            # Add to samples
            self.samples.append({
                'song_id': f"dummy_{i}",
                'spectro': dummy_spectro,
                'chord_idx': dummy_chords,
                'start_frame': 0,
                'audio_path': "dummy_path.wav",
                'label_path': "dummy_path.lab",
                'feature_per_second': 10.0
            })
    
    def _find_matching_pairs(self):
        """
        Find all matching audio and label files across all directories.
        
        Returns:
            List of tuples (audio_path, label_path)
        """
        audio_label_pairs = []
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        label_extensions = ['.lab', '.txt']
        
        for audio_dir, label_dir in zip(self.audio_dirs, self.label_dirs):
            # Check if directories exist
            if not os.path.exists(audio_dir):
                logger.warning(f"Audio directory not found: {audio_dir}")
                continue
            
            if not os.path.exists(label_dir):
                logger.warning(f"Label directory not found: {label_dir}")
                continue
            
            # Find audio files
            audio_files = {}
            for ext in audio_extensions:
                for path in glob.glob(os.path.join(audio_dir, f"*{ext}")):
                    # Use filename without extension as key
                    basename = os.path.splitext(os.path.basename(path))[0]
                    audio_files[basename] = path
                    
                # Also check subdirectories
                for path in glob.glob(os.path.join(audio_dir, f"**/*{ext}"), recursive=True):
                    basename = os.path.splitext(os.path.basename(path))[0]
                    audio_files[basename] = path
            
            # Find label files
            label_files = {}
            for ext in label_extensions:
                for path in glob.glob(os.path.join(label_dir, f"*{ext}")):
                    basename = os.path.splitext(os.path.basename(path))[0]
                    label_files[basename] = path
                    
                # Also check subdirectories
                for path in glob.glob(os.path.join(label_dir, f"**/*{ext}"), recursive=True):
                    basename = os.path.splitext(os.path.basename(path))[0]
                    label_files[basename] = path
            
            # Log counts for debugging
            logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")
            logger.info(f"Found {len(label_files)} label files in {label_dir}")
            
            # Match audio files with label files
            pairs_found = 0
            for basename, audio_path in audio_files.items():
                if basename in label_files:
                    audio_label_pairs.append((audio_path, label_files[basename]))
                    pairs_found += 1
            
            logger.info(f"Matched {pairs_found} pairs between {audio_dir} and {label_dir}")
        
        return audio_label_pairs
    
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
        
        # SECOND PASS: If there are still many N chords, use proximity matching with tolerance
        if n_chord_frames > 0.5 * num_frames:  # If more than 50% are N chords, try with tolerance
            tolerance = frame_duration * 0.5  # Half a frame tolerance
            
            for i in range(num_frames):
                if frame_level_chords[i] != "N":
                    continue  # Skip already assigned frames
                    
                # Calculate time for this frame center
                frame_time = (i + 0.5) * frame_duration
                
                # Find nearest chord within tolerance
                min_distance = float('inf')
                nearest_chord = None
                
                for chord, (start, end) in zip(chord_labels, timestamps):
                    # If frame is before chord starts
                    if frame_time < start:
                        distance = start - frame_time
                    # If frame is after chord ends
                    elif frame_time >= end:
                        distance = frame_time - end
                    # Should not get here if first pass worked correctly
                    else:
                        distance = 0
                        
                    if distance < min_distance:
                        min_distance = distance
                        nearest_chord = chord
                
                # Assign nearest chord if within tolerance
                if nearest_chord is not None and min_distance <= tolerance:
                    frame_level_chords[i] = nearest_chord
                    assigned_frames += 1
                    n_chord_frames -= 1
        
        # Log statistics - only if significant number of frames are unassigned
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
        
    def _split_dataset(self, train_ratio, val_ratio, test_ratio):
        """
        Split dataset into train, validation, and test sets.
        Splitting is done by song to prevent data leakage.
        
        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
        """
        # Get unique song IDs
        song_ids = list(set(sample['song_id'] for sample in self.samples))
        
        # Shuffle song IDs
        random.shuffle(song_ids)
        
        # Calculate split indices
        train_split = int(len(song_ids) * train_ratio)
        val_split = int(len(song_ids) * (train_ratio + val_ratio))
            
        # Split song IDs
        train_song_ids = set(song_ids[:train_split])
        val_song_ids = set(song_ids[train_split:val_split])
        test_song_ids = set(song_ids[val_split:])
        
        # Split samples based on song IDs
        self.train_indices = [i for i, sample in enumerate(self.samples) if sample['song_id'] in train_song_ids]
        self.val_indices = [i for i, sample in enumerate(self.samples) if sample['song_id'] in val_song_ids]
        self.test_indices = [i for i, sample in enumerate(self.samples) if sample['song_id'] in test_song_ids]
        
        logger.info(f"Dataset split - Train: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")

    def analyze_label_file(self, label_path):
        """Analyze a single label file to diagnose potential issues"""
        logger.info(f"Analyzing label file: {os.path.basename(label_path)}")
        
        try:
            # Attempt to parse the file - first few lines raw
            with open(label_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()[:10]  # First 10 lines
            
            logger.info(f"First few lines of file: {lines}")
            
            # Now parse with our normal method
            chord_labels, timestamps = self._parse_label_file(label_path)
            
            if len(chord_labels) == 0:
                logger.error(f"No chords parsed from file - possible format issue")
                return
                
            logger.info(f"Parsed {len(chord_labels)} chords")
            logger.info(f"First 5 chords: {chord_labels[:5]}")
            logger.info(f"First 5 timestamps: {timestamps[:5]}")
            
            # Calculate song duration
            if timestamps:
                duration = timestamps[-1][1]
                logger.info(f"Song duration from labels: {duration:.2f} seconds")
                
                # Calculate frame rate and expected frame count
                feature_per_second = 1.0 / self.feature_config.feature.get('hop_duration', 0.1)
                expected_frames = int(duration * feature_per_second)
                logger.info(f"Expected frames at {feature_per_second:.2f} frames/sec: {expected_frames}")
                
                # Check for gaps in chord coverage
                total_chord_time = 0
                for _, (start, end) in enumerate(timestamps):
                    total_chord_time += (end - start)
                
                coverage = (total_chord_time / duration) * 100
                logger.info(f"Chord time coverage: {coverage:.2f}% of song duration")
                
                if coverage < 90:
                    logger.warning(f"Low chord coverage ({coverage:.2f}%) may result in many N labels")
            
            # Check for chord mapping issues
            mapped = self._chord_names_to_indices(chord_labels)
            n_chord_id = 169 if self.feature_config.feature.get('large_voca', False) else 24
            n_count = sum(1 for idx in mapped if idx == n_chord_id)
            
            if n_count > 0:
                logger.info(f"N chords in file: {n_count}/{len(mapped)} ({n_count/len(mapped):.2%})")
                if n_count > 0.5 * len(mapped):
                    logger.warning(f"HIGH NUMBER OF N CHORDS: Check mapping and chord format")
            
        except Exception as e:
            logger.error(f"Error analyzing label file: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
