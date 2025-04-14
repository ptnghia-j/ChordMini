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
from collections import Counter

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
        
        # --- FIX: Determine use_large_voca based on chord_mapping size ---
        # Infer from the size of the provided mapping (excluding N/X if they exist)
        # A mapping size > 30 typically indicates the large vocabulary (170).
        # The standard small vocabulary has 26 entries (including N/X).
        num_unique_chords_in_mapping = len(chord_mapping) if chord_mapping else 0
        self.use_large_voca = num_unique_chords_in_mapping > 30 
        # --- End FIX ---
        logger.info(f"Dataset initialized with use_large_voca = {self.use_large_voca} (inferred from mapping size: {num_unique_chords_in_mapping})")

        # --- Determine N and X indices based *only* on use_large_voca ---
        self.n_chord_idx = 169 if self.use_large_voca else 24
        self.x_chord_idx = 168 if self.use_large_voca else 25
        logger.info(f"Using N index: {self.n_chord_idx}, X index: {self.x_chord_idx}")
        # --- End N/X index determination ---

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
            # Remove N and X from the mapping passed to the processor,
            # as we handle them separately based on use_large_voca flag.
            processed_mapping = {k: v for k, v in chord_mapping.items() if k not in ["N", "X"]}
            self.chord_processor.set_chord_mapping(processed_mapping)
            self.chord_processor.initialize_chord_mapping()  # Initialize based on the filtered mapping
        # Store the original mapping if needed elsewhere, but processor uses filtered one
        self.original_chord_mapping = chord_mapping
        
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

    def analyze_label_file(self, label_path):
        """Analyze a label file for diagnostic purposes."""
        from modules.utils import logger
        from collections import Counter # Ensure Counter is imported

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            logger.info(f"Analyzing label file: {os.path.basename(label_path)}")
            logger.info(f"First few lines of file: {lines[:10]}")
            
            # Parse the chord labels and timestamps
            timestamps = []
            chord_labels = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    chord = parts[2]
                    
                    timestamps.append((start_time, end_time))
                    chord_labels.append(chord)
            
            logger.info(f"Parsed {len(chord_labels)} chords")
            logger.info(f"First 5 chords: {chord_labels[:5]}")
            logger.info(f"First 5 timestamps: {timestamps[:5]}")
            
            # Calculate song duration
            if timestamps:
                duration = timestamps[-1][1]
                logger.info(f"Song duration from labels: {duration:.2f} seconds")
                logger.info(f"Expected frames at 10.77 frames/sec: {int(duration * 10.77)}")
                
                # Check for time coverage (gaps in the labels)
                total_time = sum(end - start for start, end in timestamps)
                coverage = total_time / duration * 100
                logger.info(f"Chord time coverage: {coverage:.2f}% of song duration")
            
            # Check chord index mapping using the processor's method
            unknown_count = 0
            unmapped_chords_counter = Counter() # Use Counter for efficiency
            # Use the stored instance attribute for consistency
            x_chord_idx = self.x_chord_idx # Use the class attribute determined in __init__

            for chord in chord_labels:
                # Handle N and X explicitly first
                if chord == "N":
                    idx = self.n_chord_idx
                elif chord == "X":
                    idx = self.x_chord_idx
                else:
                    # Use the processor for other chords
                    idx = self.chord_processor.get_chord_idx(chord, self.use_large_voca)

                if idx == x_chord_idx and chord != "X": # Check if it mapped to unknown
                    unknown_count += 1
                    unmapped_chords_counter[chord] += 1

            if unknown_count > 0:
                logger.error(f"Unknown chords (using get_chord_idx): {unknown_count}/{len(chord_labels)} ({unknown_count/len(chord_labels)*100:.1f}%)")
                logger.error(f"Most common unknown chords: {unmapped_chords_counter.most_common(10)}")

                unique_chords = set(chord_labels)
                # Re-check unique unmapped using the processor method and instance attribute
                unmapped_unique = [c for c in unique_chords if c not in ["N", "X"] and self.chord_processor.get_chord_idx(c, self.use_large_voca) == x_chord_idx]
                logger.error(f"{len(unmapped_unique)}/{len(unique_chords)} unique chords could not be mapped by get_chord_idx")
                logger.error(f"Examples of unmapped chords: {unmapped_unique[:10]}")
            else:
                 logger.info("All chords successfully mapped by get_chord_idx.")

            # Validate chord mapping on first few chords using the processor's method and instance attribute
            logger.info("Chord mapping validation (first 5 chords using get_chord_idx):")
            for i, chord in enumerate(chord_labels[:5]):
                 # Handle N and X explicitly first
                if chord == "N":
                    idx = self.n_chord_idx
                elif chord == "X":
                    idx = self.x_chord_idx
                else:
                    idx = self.chord_processor.get_chord_idx(chord, self.use_large_voca) # Use instance attribute
                logger.info(f"  '{chord}' -> {idx}")

            # Check specifically for G:maj mapping using the processor's method and instance attribute
            if "G:maj" in chord_labels:
                g_maj_idx = self.chord_processor.get_chord_idx("G:maj", self.use_large_voca) # Use instance attribute
                logger.info(f"G:maj mapping (using get_chord_idx): {g_maj_idx}")

            return True

        except Exception as e:
            logger.error(f"Error analyzing label file {label_path}: {e}")
            return False

    def _fix_time_scale(self, feature_shape, feature_per_second, song_length_second, timestamps):
        """
        Fix time scale issues between audio features and chord labels.
        
        Args:
            feature_shape: Shape of the extracted feature (frames, freq_bins)
            feature_per_second: Calculated frames per second
            song_length_second: Calculated song length in seconds
            timestamps: List of (start_time, end_time) tuples from chord labels
            
        Returns:
            Tuple of (corrected_feature_per_second, time_scale_factor)
        """
        if not timestamps:
            return feature_per_second, 1.0
            
        # Calculate expected number of frames based on label duration and feature rate
        label_duration = timestamps[-1][1]
        expected_frames = feature_shape[0]
        
        # Verify if feature_per_second is actually a frame duration (common mistake)
        if feature_per_second < 1.0:
            # It's likely the frame duration was passed instead of frame rate
            logger.warning(f"Detected frame duration ({feature_per_second:.5f}s) passed as frame rate")
            # Convert from duration to rate
            true_feature_per_second = 1.0 / feature_per_second
            logger.info(f"Corrected frame rate: {true_feature_per_second:.2f} fps (was {feature_per_second:.2f})")
            return true_feature_per_second, 1.0
        
        # Check for massive mismatch in calculated duration
        if song_length_second > 1000:  # If calculated duration is >1000 seconds
            logger.warning(f"Calculated song duration of {song_length_second:.2f}s is suspiciously long")
            
            # Recalculate based on more reasonable assumptions
            # Most songs are 2-10 minutes, so calculate fps from frames and label duration
            corrected_fps = expected_frames / label_duration
            
            if 8.0 <= corrected_fps <= 12.0:  # Reasonable frame rate range for music
                logger.info(f"Corrected frame rate: {corrected_fps:.2f} fps (was {feature_per_second:.2f})")
                return corrected_fps, 1.0
        
        # Check if timestamps might be in milliseconds
        if label_duration * 10 < song_length_second:
            logger.warning("Labels appear to be in seconds but audio duration is much longer")
            
            # Try to correct the frame rate instead of scaling timestamps
            corrected_fps = expected_frames / label_duration
            
            if 8.0 <= corrected_fps <= 12.0:  # Reasonable range
                logger.info(f"Using corrected frame rate: {corrected_fps:.2f} fps")
                return corrected_fps, 1.0
            else:
                logger.warning(f"Corrected fps of {corrected_fps:.2f} seems outside reasonable range")
        
        # Check if labels appear to be in milliseconds but audio is in seconds
        if label_duration > song_length_second * 10:
            logger.warning("Labels appear to be in milliseconds but audio is in seconds")
            return feature_per_second, 0.001  # Convert ms to seconds
            
        return feature_per_second, 1.0

    def _chord_labels_to_frames(self, chord_labels, timestamps, num_frames, feature_per_second):
        """
        Convert chord labels to frame-level representation with improved assignment.
        
        Args:
            chord_labels: List of chord labels
            timestamps: List of (start_time, end_time) tuples
            num_frames: Total number of frames
            feature_per_second: Number of frames per second
            
        Returns:
            List of chord labels for each frame
        """
        # Fix the massive time scale issues we're seeing in the logs
        corrected_fps, time_scale_factor = self._fix_time_scale(
            feature_shape=(num_frames, self.feature_config.feature.get('n_bins', 144)),
            feature_per_second=feature_per_second,
            song_length_second=num_frames / feature_per_second if feature_per_second > 0 else 0,
            timestamps=timestamps
        )
        
        # Use the corrected frame rate instead
        feature_per_second = corrected_fps
        
        # Log input data for debugging
        logger.debug(f"Converting {len(chord_labels)} chord labels to {num_frames} frames at {feature_per_second} fps")
        
        # Calculate frame duration in seconds
        frame_duration = 1.0 / feature_per_second
        
        # Calculate audio duration in seconds based on frames and frame duration
        audio_duration = num_frames * frame_duration
        logger.debug(f"Audio duration: {audio_duration:.2f}s ({num_frames} frames at {frame_duration:.5f}s per frame)")
        
        # Apply time scale factor to timestamps if needed
        if time_scale_factor != 1.0:
            timestamps = [(start * time_scale_factor, end * time_scale_factor) for start, end in timestamps]
            logger.info(f"Applied time scale factor {time_scale_factor} to timestamps")
            
            # Recalculate label duration for more accurate logs
            label_duration = timestamps[-1][1]
            logger.info(f"After scaling: Label duration = {label_duration:.2f}s")
        elif timestamps:
            label_duration = timestamps[-1][1]
            logger.debug(f"Label duration: {label_duration:.2f}s from chord file")
            
        # Initialize with "N" (no chord)
        frame_level_chords = ["N"] * num_frames
        
        # NEW: Find reasonable label bounds to detect issues with timestamps
        if timestamps:
            # Correctly extract only the start time for min() comparison
            min_time = min(start for start, _ in timestamps)
            max_time = max(end for _, end in timestamps)
            
            # Check for negative start times or excessive end times
            if min_time < 0:
                logger.warning(f"Found negative start time ({min_time:.2f}s) in chord labels")
                # Shift all timestamps to non-negative
                if min_time < -0.01:  # Only fix if it's substantially negative
                    shift = -min_time
                    timestamps = [(start + shift, end + shift) for start, end in timestamps]
                    logger.info(f"Shifted all timestamps by +{shift:.2f}s")
            
            # Check for extreme duration mismatch
            expected_duration = audio_duration
            if max_time > expected_duration * 2:
                logger.warning(f"Label end time ({max_time:.2f}s) far exceeds audio duration ({expected_duration:.2f}s)")
                # Do not auto-fix this case as it requires more careful handling
        
        # Keep track of assigned and unassigned frames
        assigned_frames = 0
        
        # Set a reasonable tolerance based on frame duration
        tolerance = frame_duration * 0.5  # Half a frame tolerance
        
        # First pass: Direct assignment with tolerance
        for i in range(num_frames):
            # Calculate time for this frame center (THIS IS THE KEY CHANGE)
            # Using the proper formula: frame_index * frame_duration + half_frame_duration
            frame_time = i * frame_duration + (frame_duration / 2)
            
            # Find chord that contains this time point
            for chord, (start, end) in zip(chord_labels, timestamps):
                if (start - tolerance) <= frame_time < (end + tolerance):
                    frame_level_chords[i] = chord
                    assigned_frames += 1
                    break

        # Log assignment statistics
        logger.info(f"Chord assignment: {assigned_frames}/{num_frames} frames assigned ({assigned_frames/num_frames:.1%})")
        
        # If terrible assignment, try with much larger tolerance
        if assigned_frames < num_frames * 0.3:  # Less than 30% assigned
            logger.warning(f"Poor chord assignment rate ({assigned_frames/num_frames:.1%}). Trying larger tolerance.")
            
            # Reset and try again with much larger tolerance
            frame_level_chords = ["N"] * num_frames
            assigned_frames = 0
            large_tolerance = frame_duration * 5  # 5x frame duration
            
            for i in range(num_frames):
                # Ensure proper frame time calculation here too
                frame_time = i * frame_duration + (frame_duration / 2)
                
                for chord, (start, end) in zip(chord_labels, timestamps):
                    # Use larger tolerance for better coverage
                    if (start - large_tolerance) <= frame_time < (end + large_tolerance):
                        frame_level_chords[i] = chord
                        assigned_frames += 1
                        break
            
            logger.info(f"After larger tolerance: {assigned_frames}/{num_frames} frames assigned ({assigned_frames/num_frames:.1%})")
        
        return frame_level_chords

    def _chord_names_to_indices(self, chord_names):
        """
        Convert chord names to indices using the chord_processor's get_chord_idx method.

        Args:
            chord_names: List of chord names

        Returns:
            List of chord indices
        """
        if self.chord_mapping is None:
            logger.warning("Chord mapping not set in LabeledDataset. Returning raw names.")
            return chord_names

        indices = []
        unknown_chords = Counter() # Use Counter for efficient counting

        # Use the N and X indices determined in __init__
        n_chord_idx = self.n_chord_idx
        x_chord_idx = self.x_chord_idx

        # Log mapping info once for diagnostics
        if not hasattr(self, '_chord_mapping_logged') and chord_names:
            logger.info(f"Chord mapping info - using N index: {n_chord_idx}, X index: {x_chord_idx} (use_large_voca={self.use_large_voca})") # Log flag value
            # Print a sample of the processor's mapping (which excludes N/X)
            sample_keys = list(self.chord_processor.chord_mapping.keys())[:10]
            logger.info(f"Sample processor chord mapping: {[(k, self.chord_processor.chord_mapping[k]) for k in sample_keys]}")
            self._chord_mapping_logged = True

        # Process each chord
        for chord in chord_names:
            # Handle N and X explicitly using the determined indices
            if chord == "N":
                idx = n_chord_idx
            elif chord == "X":
                idx = x_chord_idx
            else:
                # Use the get_chord_idx method for all other chords
                idx = self.chord_processor.get_chord_idx(chord, self.use_large_voca) # Use instance attribute

            indices.append(idx)

            # Track if the result was the unknown index (and wasn't explicitly X)
            if idx == x_chord_idx and chord != "X":
                unknown_chords[chord] += 1

        # Log unknown chord statistics if any were found
        total_unknown = sum(unknown_chords.values())
        if total_unknown > 0:
            total_chords = len(chord_names)
            unknown_percent = (total_unknown / total_chords) * 100
            # Limit logging verbosity - maybe only log if > 1% unknown or periodically
            if unknown_percent > 1.0 or random.random() < 0.01: # Log if >1% or 1% of the time
                 logger.warning(f"Unknown chords: {total_unknown}/{total_chords} ({unknown_percent:.1f}%)")
                 # Show most common unknowns for debugging
                 logger.warning(f"Most common unknown chords: {unknown_chords.most_common(10)}")

        return indices

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
                
                # Add diagnostic info for each file
                logger.info(f"Processing {os.path.basename(audio_path)} - Found {len(chord_labels)} chord labels")
                
                # Validate that we have chord labels
                if len(chord_labels) == 0:
                    logger.error(f"No chord labels found in {label_path}! Skipping file.")
                    errors['label'] += 1
                    continue
                    
                # Analyze timestamps to detect potential issues
                if timestamps:
                    total_duration = timestamps[-1][1]
                    expected_frames = int(total_duration * feature_per_second)
                    logger.info(f"Label duration: {total_duration:.2f}s, Audio frames: {feature.shape[0]}, Expected frames: {expected_frames}")
                    
                    # Detect significant mismatch
                    if abs(feature.shape[0] - expected_frames) > feature.shape[0] * 0.5:
                        logger.warning(f"Label vs audio duration mismatch: {total_duration:.2f}s vs {feature.shape[0]/feature_per_second:.2f}s")
                
                # Check for time discontinuities in chord labels
                if len(timestamps) > 1:
                    discontinuities = []
                    for i in range(1, len(timestamps)):
                        prev_end = timestamps[i-1][1]
                        curr_start = timestamps[i][0]
                        gap = curr_start - prev_end
                        if abs(gap) > 0.01:  # 10ms gap/overlap threshold
                            discontinuities.append((i-1, i, gap))
                    
                    if discontinuities:
                        logger.warning(f"Found {len(discontinuities)} time discontinuities in {label_path}")
                        if len(discontinuities) < 5:  # Show first few if not too many
                            for prev_idx, curr_idx, gap in discontinuities[:5]:
                                logger.warning(f"Gap of {gap:.3f}s between {chord_labels[prev_idx]} and {chord_labels[curr_idx]}")
                
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
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample by index"""
        sample = self.samples[idx]
        
        # Convert to torch tensors and move to the specified device
        spectro = torch.tensor(sample['spectro'], dtype=torch.float32).to(self.device)
        chord_idx = torch.tensor(sample['chord_idx'], dtype=torch.long).to(self.device)
        
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
        # Pin memory is generally not needed if data is already on GPU in __getitem__
        pin_memory = False if self.device.type == 'cuda' else pin_memory 
        return DataLoader(
            Subset(self, self.train_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def get_val_iterator(self, batch_size=32, shuffle=False, num_workers=2, pin_memory=True):
        """Get data loader for validation set"""
        # Pin memory is generally not needed if data is already on GPU in __getitem__
        pin_memory = False if self.device.type == 'cuda' else pin_memory
        return DataLoader(
            Subset(self, self.val_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def get_test_iterator(self, batch_size=32, shuffle=False, num_workers=2, pin_memory=True):
        """Get data loader for test set"""
        # Pin memory is generally not needed if data is already on GPU in __getitem__
        pin_memory = False if self.device.type == 'cuda' else pin_memory
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
        
        # Pin memory is generally not needed if data is already on GPU in __getitem__
        pin_memory = False if self.device.type == 'cuda' else pin_memory
        
        return DataLoader(
            Subset(self, song_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
