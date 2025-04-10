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
                        available_chords = chord_info.loc[(chord_info['start'] <= curSec) & 
                                                         (chord_info['end'] > curSec + time_interval)].copy()
                        
                        if len(available_chords) == 0:
                            # If no chord spans the entire interval, look for overlapping chords
                            available_chords = chord_info.loc[
                                ((chord_info['start'] >= curSec) & (chord_info['start'] <= curSec + time_interval)) | 
                                ((chord_info['end'] >= curSec) & (chord_info['end'] <= curSec + time_interval))
                            ].copy()
                        
                        if len(available_chords) == 1:
                            # If only one chord, use it
                            chord = available_chords['chord_id'].iloc[0]
                        elif len(available_chords) > 1:
                            # If multiple chords, pick the one that covers the most of the interval
                            max_starts = available_chords.apply(lambda row: max(row['start'], curSec), axis=1)
                            available_chords['max_start'] = max_starts
                            
                            min_ends = available_chords.apply(
                                lambda row: min(row['end'], curSec + time_interval), axis=1)
                            available_chords['min_end'] = min_ends
                            
                            chords_lengths = available_chords['min_end'] - available_chords['max_start']
                            available_chords['chord_length'] = chords_lengths
                            
                            # Get chord with maximum coverage
                            chord = available_chords.loc[available_chords['chord_length'].idxmax()]['chord_id']
                        else:
                            # No chord found, use no-chord class
                            chord = 169 if large_voca else 24
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

    def generate_teacher_predictions(self):
        """Generate predictions from teacher model for knowledge distillation"""
        if self.teacher_model is None or self._kd_data_generated:
            return
            
        logger.info("Generating teacher model predictions for knowledge distillation")
        self.teacher_model.eval()
        
        # Process all paths
        for i, path in enumerate(self.paths):
            if i % 100 == 0:
                logger.info(f"Generating KD data: {i}/{len(self.paths)}")
                
            # Skip if path is not a file (dictionary for on-the-fly processing)
            if isinstance(path, dict):
                continue
                
            try:
                # Load data
                data = torch.load(path)
                feature = data['feature']
                
                # Convert to log-magnitude and ensure shape is correct
                feature = np.log(np.abs(feature) + 1e-6)
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Generate teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher_model(feature.permute(0, 2, 1))
                    if isinstance(teacher_logits, tuple):
                        teacher_logits = teacher_logits[0]
                    
                # Store teacher predictions with data
                data['teacher_logits'] = teacher_logits.cpu()
                torch.save(data, path)
            except Exception as e:
                logger.error(f"Error generating teacher predictions for {path}: {e}")
        
        self._kd_data_generated = True
        logger.info("Teacher predictions generation completed")

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get a sample by index"""
        path = self.paths[idx]
        
        # Check if path is a file or a dictionary for on-the-fly processing
        if isinstance(path, dict):
            # Process file on-the-fly
            logger.debug(f"Processing file on-the-fly: {path['song_name']}")
            
            # Process with default parameters (no augmentation for on-the-fly)
            processed_paths = self._process_audio_file(
                path['audio_path'], 
                path['label_path'], 
                path['cache_path']
            )
            
            if processed_paths and len(processed_paths) > 0:
                # Replace the dictionary entry with the first processed file
                self.paths[idx] = processed_paths[0]
                # Add other processed files to the dataset
                self.paths.extend(processed_paths[1:])
                
                # Load the newly processed file
                path = processed_paths[0]
            else:
                # Return dummy data if processing failed
                logger.warning(f"Failed to process {path['song_name']} on-the-fly")
                return self._get_dummy_data()
        
        # Load data from cache
        try:
            data = torch.load(path)
            
            # Convert feature to log-magnitude and torch tensor
            feature = np.log(np.abs(data['feature']) + 1e-6)
            feature = torch.tensor(feature, dtype=torch.float32)
            
            # Convert chord to torch tensor
            chord = torch.tensor(data['chord'], dtype=torch.long)
            
            # Set up return dictionary
            result = {
                'spectro': feature, 
                'chord_idx': chord,
                'song_id': data.get('song_name', Path(path).stem),
                'file_path': path
            }
            
            # Add teacher logits if available
            if 'teacher_logits' in data:
                result['teacher_logits'] = data['teacher_logits']
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return self._get_dummy_data()
    
    def _get_dummy_data(self):
        """Return dummy data in case of errors"""
        # Determine feature and chord sizes based on config
        inst_len = self.config.mp3.get('inst_len', 10.0)
        hop_length = self.config.feature.get('hop_length', 2048)
        song_hz = self.config.mp3.get('song_hz', 22050)
        time_interval = hop_length / song_hz
        seq_len = math.ceil(inst_len / time_interval)
        n_bins = self.config.feature.get('n_bins', 144)
        
        # Create dummy data
        dummy_feature = torch.zeros(n_bins, seq_len, dtype=torch.float32)
        dummy_chord = torch.zeros(seq_len, dtype=torch.long)
        
        return {
            'spectro': dummy_feature, 
            'chord_idx': dummy_chord,
            'song_id': 'dummy',
            'file_path': 'dummy'
        }
        
    def get_data_loader(self, batch_size=16, shuffle=True, num_workers=4, pin_memory=True):
        """Get a data loader for this dataset"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    @staticmethod
    def _collate_fn(batch):
        """Custom collate function to handle variable length sequences"""
        # Check if the batch is empty
        if len(batch) == 0:
            return {}
            
        # Extract all spectrograms and chords
        spectros = [item['spectro'] for item in batch]
        chords = [item['chord_idx'] for item in batch]
        
        # Get maximum sequence length
        max_len = max(s.shape[1] for s in spectros)
        
        # Pad sequences to max_len
        padded_spectros = []
        for spectro in spectros:
            curr_len = spectro.shape[1]
            if curr_len < max_len:
                # Pad with zeros
                padding = torch.zeros(spectro.shape[0], max_len - curr_len, dtype=spectro.dtype)
                padded_spectros.append(torch.cat([spectro, padding], dim=1))
            else:
                padded_spectros.append(spectro)
        
        # Stack spectrograms and chords
        spectros_tensor = torch.stack(padded_spectros)
        chords_tensor = torch.stack(chords)
        
        # Transpose spectrograms to (batch, freq, time)
        spectros_tensor = spectros_tensor.permute(0, 2, 1)
        
        # Create output dictionary
        result = {
            'spectro': spectros_tensor,
            'chord_idx': chords_tensor,
            'song_ids': [item['song_id'] for item in batch]
        }
        
        # Add teacher logits if available
        if 'teacher_logits' in batch[0]:
            teacher_logits = [item.get('teacher_logits') for item in batch]
            # Make sure all teacher logits are available and have the same shape
            if all(t is not None for t in teacher_logits):
                result['teacher_logits'] = torch.stack(teacher_logits)
            else:
                # Handle missing teacher logits
                has_none = [i for i, t in enumerate(teacher_logits) if t is None]
                logger.warning(f"Missing teacher logits in batch at indices: {has_none}")
        
        return result

    def generate_all_features(self):
        """Process all audio files to extract features (for pre-processing)"""
        from tqdm import tqdm
        import time
        
        # Count total operations to be performed
        total_songs = len(self.all_song_paths)
        stretch_factors = [1.0]
        shift_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        total_operations = total_songs * len(stretch_factors) * len(shift_factors)
        
        logger.info(f"Starting feature extraction for {total_songs} songs with {len(shift_factors)} pitch shifts")
        logger.info(f"Total operations: {total_operations} (this may take a while)")
        
        # Process each song with a progress bar
        operations_completed = 0
        start_time = time.time()
        
        for song_name, info in tqdm(self.all_song_paths.items(), desc="Processing songs", total=total_songs):
            # Skip if already processed
            cache_files = glob.glob(os.path.join(info['cache_path'], "*.pt"))
            if cache_files:
                logger.info(f"Song {song_name} already has cached features ({len(cache_files)} files)")
                operations_completed += len(stretch_factors) * len(shift_factors)
                continue
                
            song_start_time = time.time()
            logger.info(f"Processing {song_name}")
            
            # Track successful operations for this song
            song_operations = 0
            
            # Generate features with various augmentations
            for stretch_factor in stretch_factors:
                for shift_factor in tqdm(shift_factors, desc=f"Pitch shifts for {song_name}", leave=False):
                    try:
                        result = self._process_audio_file(
                            info['audio_path'],
                            info['label_path'],
                            info['cache_path'],
                            stretch_factor,
                            shift_factor
                        )
                        
                        if result:
                            song_operations += 1
                        
                        operations_completed += 1
                        
                        # Log progress occasionally
                        if operations_completed % 10 == 0:
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                ops_per_second = operations_completed / elapsed
                                estimated_total = total_operations / ops_per_second if ops_per_second > 0 else 0
                                remaining = max(0, estimated_total - elapsed)
                                
                                logger.info(f"Progress: {operations_completed}/{total_operations} operations " +
                                          f"({operations_completed/total_operations:.1%}), " +
                                          f"~{remaining/60:.1f} minutes remaining")
                    except Exception as e:
                        logger.error(f"Error processing {song_name} with shift={shift_factor}: {e}")
                        operations_completed += 1
            
            song_time = time.time() - song_start_time
            logger.info(f"Completed {song_name}: {song_operations} operations in {song_time:.1f}s " +
                      f"({song_operations/song_time:.1f} ops/s)")
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
