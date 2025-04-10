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

from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features
from modules.utils.chords import Chords
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

    def _process_audio_file(self, audio_path, label_path, cache_path, stretch_factor=1.0, shift_factor=0):
        """Process an audio file to extract features and labels"""
        # Generate features and chord labels
        try:
            # Get preprocessor
            preprocessor = self._get_preprocessor()
            
            # Create chord class instance for parsing
            chord_class = Chords()
            
            # Check if large vocabulary is enabled
            large_voca = self.config.feature.get('large_voca', False)
            
            # Get chord info
            if large_voca:
                chord_info = chord_class.get_converted_chord_voca(label_path)
            else:
                chord_info = chord_class.get_converted_chord(label_path)
            
            # Load audio file
            song_hz = self.config.mp3.get('song_hz', 22050)
            original_wav, sr = librosa.load(audio_path, sr=song_hz)
            
            # Apply time stretching and pitch shifting
            if stretch_factor != 1.0 or shift_factor != 0:
                import pyrubberband as pyrb
                x = pyrb.time_stretch(original_wav, sr, stretch_factor)
                x = pyrb.pitch_shift(x, sr, shift_factor)
            else:
                x = original_wav
            
            # Adjust chord timestamps for time stretching
            if stretch_factor != 1.0:
                chord_info['start'] = chord_info['start'] * (1/stretch_factor)
                chord_info['end'] = chord_info['end'] * (1/stretch_factor)
            
            # Get sequence parameters
            inst_len = self.config.mp3.get('inst_len', 10.0)
            hop_length = self.config.feature.get('hop_length', 2048)
            time_interval = hop_length / song_hz
            no_of_chord_datapoints_per_sequence = math.ceil(inst_len / time_interval)
            
            # Get audio information
            last_sec = chord_info.iloc[-1]['end']
            last_sec_hz = int(last_sec * song_hz)
            
            # Ensure audio is long enough
            if len(x) < last_sec_hz:
                logger.warning(f"Audio file {audio_path} is too short")
                return None
            
            # Trim audio if needed
            if len(x) > last_sec_hz:
                x = x[:last_sec_hz]
            
            # Process audio in segments
            origin_length_in_sec = last_sec_hz / song_hz
            skip_interval = self.config.mp3.get('skip_interval', 5.0)
            
            results = []
            current_start_second = 0
            
            while current_start_second + inst_len < origin_length_in_sec:
                # Extract chord sequence for this segment
                chord_list = []
                curSec = current_start_second
                
                while curSec < current_start_second + inst_len:
                    try:
                        # Find chords for current time frame
                        # First, look for chords that contain the current time point
                        available_chords = chord_info.loc[(chord_info['start'] <= curSec) & 
                                                         (chord_info['end'] > curSec)].copy()
                        
                        if len(available_chords) == 0:
                            # If no direct match, try to find chords with a small tolerance window
                            tolerance = time_interval / 2
                            available_chords = chord_info.loc[
                                # Chords starting just before or at the current time
                                ((chord_info['start'] >= curSec - tolerance) & (chord_info['start'] <= curSec + tolerance)) |
                                # Chords ending just after or at the current time
                                ((chord_info['end'] >= curSec - tolerance) & (chord_info['end'] <= curSec + tolerance)) |
                                # Chords that completely contain the current time point with tolerance
                                ((chord_info['start'] <= curSec - tolerance) & (chord_info['end'] >= curSec + tolerance))
                            ].copy()
                            
                            # Log when we need to use the tolerance window
                            if len(available_chords) > 0 and len(chord_list) % 100 == 0:  # Limit logging
                                logger.debug(f"Used tolerance window to find chord at {curSec}s")
                        
                        if len(available_chords) == 1:
                            # If only one chord, use it
                            chord = available_chords['chord_id'].iloc[0]
                        elif len(available_chords) > 1:
                            # If multiple chords, pick the one that covers the most of the current time point
                            tolerance_value = tolerance if 'tolerance' in locals() else 0
                            
                            # Calculate overlap with the current time point
                            max_starts = available_chords.apply(lambda row: max(row['start'], curSec - tolerance_value), axis=1)
                            available_chords['max_start'] = max_starts
                            
                            min_ends = available_chords.apply(
                                lambda row: min(row['end'], curSec + tolerance_value), axis=1)
                            available_chords['min_end'] = min_ends
                            
                            chords_lengths = available_chords['min_end'] - available_chords['max_start']
                            available_chords['chord_length'] = chords_lengths
                            
                            # Get chord with maximum coverage
                            chord = available_chords.loc[available_chords['chord_length'].idxmax()]['chord_id']
                        else:
                            # No chord found, use no-chord class
                            chord = 169 if large_voca else 24
                            
                            # Log when no chord is found (but limit logging frequency)
                            if current_start_second < 5.0 or (current_start_second + inst_len) > (origin_length_in_sec - 5.0):
                                # Don't log for beginning/end of song where no-chord is expected
                                pass
                            elif len(chord_list) % 100 == 0:  # Only log occasionally to avoid flood
                                logger.debug(f"No chord found at time {curSec}s")
                    except Exception as e:
                        # Error handling for chord extraction
                        chord = 169 if large_voca else 24
                        logger.warning(f"Error extracting chord at {curSec}s: {e}")
                    
                    # Handle pitch shifting for chord IDs
                    if shift_factor != 0:
                        if chord != 169 and chord != 168 and large_voca:
                            chord += shift_factor * 14
                            chord = chord % 168
                        elif chord != 24 and not large_voca:
                            chord += shift_factor * 2
                            chord = chord % 24
                    
                    chord_list.append(chord)
                    curSec += time_interval

                # Check if we have the right number of chords
                if len(chord_list) == no_of_chord_datapoints_per_sequence:
                    # Extract audio segment
                    sequence_start_time = current_start_second
                    sequence_end_time = current_start_second + inst_len
                    start_index = int(sequence_start_time * song_hz)
                    end_index = int(sequence_end_time * song_hz)
                    song_seq = x[start_index:end_index]
                    
                    # Extract CQT feature
                    n_bins = self.config.feature.get('n_bins', 144)
                    bins_per_octave = self.config.feature.get('bins_per_octave', 24)
                    feature = librosa.cqt(
                        song_seq, 
                        sr=song_hz,
                        n_bins=n_bins,
                        bins_per_octave=bins_per_octave,
                        hop_length=hop_length
                    )
                    
                    # Ensure feature length matches chord sequence length
                    if feature.shape[1] > no_of_chord_datapoints_per_sequence:
                        feature = feature[:, :no_of_chord_datapoints_per_sequence]
                    
                    if feature.shape[1] != no_of_chord_datapoints_per_sequence:
                        logger.warning(f"Feature length mismatch: {feature.shape[1]} != {no_of_chord_datapoints_per_sequence}")
                        break
                    
                    # Create result dictionary
                    etc = f"{current_start_second:.1f}_{current_start_second + inst_len:.1f}"
                    aug = f"{stretch_factor:.2f}_{shift_factor}"
                    result = {
                        'feature': feature,
                        'chord': chord_list,
                        'etc': etc,
                        'song_name': os.path.basename(audio_path)
                    }
                    
                    # Save to cache
                    cache_filename = f"{aug}_{len(results)}.pt"
                    cache_filepath = os.path.join(cache_path, cache_filename)
                    torch.save(result, cache_filepath)
                    results.append(cache_filepath)
                
                # Move to next segment
                current_start_second += skip_interval
            
            return results
                    
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def analyze_chord_distribution(self):
        """
        Analyze the distribution of chords in the dataset to identify potential imbalance issues.
        This helps diagnose issues like excessive 'N' chord labeling.
        """
        logger.info("Analyzing chord distribution in the dataset...")
        
        # Initialize counters
        chord_counts = {}
        total_frames = 0
        n_chord_frames = 0
        large_voca = self.config.feature.get('large_voca', False)
        n_chord_id = 169 if large_voca else 24
        
        # Sample up to 100 files for analysis
        sample_paths = self.paths[:min(100, len(self.paths))]
        
        # Process each file
        for path in sample_paths:
            try:
                # Skip if it's a dictionary (on-the-fly processing)
                if isinstance(path, dict):
                    continue
                    
                # Load data from file
                data = torch.load(path)
                
                # Count chord occurrences
                for chord_id in data['chord']:
                    chord_id = int(chord_id)  # Convert to int for consistent key type
                    if chord_id not in chord_counts:
                        chord_counts[chord_id] = 0
                    chord_counts[chord_id] += 1
                    total_frames += 1
                    
                    # Check if it's not an 'N' chord
                    if chord_id != n_chord_id:
                        n_chord_frames += 1
                        
            except Exception as e:
                logger.error(f"Error analyzing file {path}: {e}")
        
        # Calculate statistics
        n_chord_percent = (n_chord_frames / total_frames * 100) if total_frames > 0 else 0
        n_percent = ((total_frames - n_chord_frames) / total_frames * 100) if total_frames > 0 else 0
        
        logger.info(f"Analyzed {total_frames} frames from {len(sample_paths)} files")
        logger.info(f"  Chord frames (non-N): {n_chord_frames} ({n_chord_percent:.2f}%)")
        logger.info(f"  No-chord frames (N): {total_frames - n_chord_frames} ({n_percent:.2f}%)")
        
        # Sort and display chord distribution
        if chord_counts:
            sorted_counts = sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)
            logger.info("Top 10 chord IDs by frequency:")
            for chord_id, count in sorted_counts[:10]:
                percentage = (count / total_frames * 100) if total_frames > 0 else 0
                logger.info(f"  Chord ID {chord_id}: {count} frames ({percentage:.2f}%)")
        
        return {
            'total_frames': total_frames,
            'chord_frames': n_chord_frames,
            'no_chord_frames': total_frames - n_chord_frames,
            'chord_counts': chord_counts
        }
