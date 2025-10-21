#!/usr/bin/env python3
"""
BTC Chord Recognition Module
Combined functionality from test.py and test_btc.py for API usage
"""

import os
import sys
import numpy as np
import torch
import warnings
from pathlib import Path
import librosa
import soundfile as sf
from scipy import interpolate

# Add the ChordMini directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Project imports - use absolute imports from current directory
try:
    from modules.utils import logger
    from modules.utils.mir_eval_modules import idx2voca_chord
    from modules.utils.hparams import HParams
    from modules.models.Transformer.btc_model import BTC_model
except ImportError as e:
    # If modules import fails, try relative imports
    try:
        import importlib.util

        # Import logger
        logger_spec = importlib.util.spec_from_file_location("logger", os.path.join(current_dir, "modules", "utils", "logger.py"))
        logger_module = importlib.util.module_from_spec(logger_spec)
        logger_spec.loader.exec_module(logger_module)
        logger = logger_module

        # Import mir_eval_modules
        mir_spec = importlib.util.spec_from_file_location("mir_eval_modules", os.path.join(current_dir, "modules", "utils", "mir_eval_modules.py"))
        mir_module = importlib.util.module_from_spec(mir_spec)
        mir_spec.loader.exec_module(mir_module)
        idx2voca_chord = mir_module.idx2voca_chord

        # Import hparams
        hparams_spec = importlib.util.spec_from_file_location("hparams", os.path.join(current_dir, "modules", "utils", "hparams.py"))
        hparams_module = importlib.util.module_from_spec(hparams_spec)
        hparams_spec.loader.exec_module(hparams_module)
        HParams = hparams_module.HParams

        # Import BTC model
        btc_spec = importlib.util.spec_from_file_location("btc_model", os.path.join(current_dir, "modules", "models", "Transformer", "btc_model.py"))
        btc_module = importlib.util.module_from_spec(btc_spec)
        btc_spec.loader.exec_module(btc_module)
        BTC_model = btc_module.BTC_model

    except Exception as import_error:
        print(f"Failed to import BTC modules: {import_error}")
        raise ImportError(f"Could not import required BTC modules: {e}") from import_error

# Explicitly disable MPS globally
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
    torch.backends.mps.enabled = False

def _process_audio_with_librosa_cqt(audio_file, config):
    """Process audio using librosa's CQT implementation."""
    n_fft = config.feature.get('n_fft', 512)
    original_wav, sr = librosa.load(audio_file, sr=config.mp3.get('song_hz', 22050), mono=True)
    
    if len(original_wav) < n_fft:
        original_wav = np.pad(original_wav, (0, n_fft - len(original_wav)), mode="constant", constant_values=0)
    
    current_sec_hz = 0
    feature = None
    
    while len(original_wav) > current_sec_hz + int(config.mp3.get('song_hz', 22050) * config.mp3.get('inst_len', 10.0)):
        wav_segment = original_wav[current_sec_hz:current_sec_hz + int(config.mp3.get('song_hz', 22050) * config.mp3.get('inst_len', 10.0))]

        cqt = librosa.cqt(wav_segment, sr=sr, hop_length=config.feature.get('hop_length', 2048),
                         n_bins=config.feature.get('n_bins', 144), bins_per_octave=config.feature.get('bins_per_octave', 24))
        cqt_magnitude = np.abs(cqt)
        
        if feature is None:
            feature = cqt_magnitude
        else:
            feature = np.concatenate((feature, cqt_magnitude), axis=1)
        
        current_sec_hz += int(config.mp3.get('song_hz', 22050) * config.mp3.get('inst_len', 10.0))
    
    # Process remaining audio
    if current_sec_hz < len(original_wav):
        remaining_wav = original_wav[current_sec_hz:]
        if len(remaining_wav) >= n_fft:
            cqt = librosa.cqt(remaining_wav, sr=sr, hop_length=config.feature.get('hop_length', 2048),
                             n_bins=config.feature.get('n_bins', 144), bins_per_octave=config.feature.get('bins_per_octave', 24))
            cqt_magnitude = np.abs(cqt)
            
            if feature is None:
                feature = cqt_magnitude
            else:
                feature = np.concatenate((feature, cqt_magnitude), axis=1)
    
    # Apply logarithmic scaling (crucial for model compatibility)
    if feature is not None:
        feature = np.log(np.abs(feature) + 1e-6)

    song_length_second = len(original_wav) / sr
    feature_per_second = config.feature.get('hop_length', 2048) / config.mp3.get('song_hz', 22050)

    return feature, feature_per_second, song_length_second

def _process_audio_with_alternative(audio_file, config):
    """Alternative feature extraction using STFT."""
    try:
        audio_data, sr = sf.read(audio_file)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        if sr != config.mp3.get('song_hz', 22050):
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=config.mp3.get('song_hz', 22050))
            sr = config.mp3.get('song_hz', 22050)

        hop_length = config.feature.get('hop_length', 2048)
        n_fft = config.feature.get('n_fft', 2048)
        
        stft = librosa.stft(audio_data, hop_length=hop_length, n_fft=n_fft)
        magnitude = np.abs(stft)

        # Resize to match expected feature dimensions (144 bins)
        target_bins = config.feature.get('n_bins', 144)
        if magnitude.shape[0] != target_bins:
            # Interpolate to target number of bins
            x = np.linspace(0, 1, magnitude.shape[0])
            x_new = np.linspace(0, 1, target_bins)
            magnitude_resized = np.zeros((target_bins, magnitude.shape[1]))

            for i in range(magnitude.shape[1]):
                f_interp = interpolate.interp1d(x, magnitude[:, i], kind='linear')
                magnitude_resized[:, i] = f_interp(x_new)

            magnitude = magnitude_resized

        # Apply logarithmic scaling (crucial for model compatibility)
        magnitude = np.log(np.abs(magnitude) + 1e-6)

        song_length_second = len(audio_data) / sr
        feature_per_second = hop_length / sr

        return magnitude, feature_per_second, song_length_second
        
    except Exception as e:
        logger.error(f"Alternative feature extraction failed: {e}")
        raise

def process_audio_with_padding(audio_file, config):
    """Process audio file with proper handling of short segments."""
    try:
        return _process_audio_with_librosa_cqt(audio_file, config)
    except Exception as e:
        logger.warning(f"CQT extraction failed: {e}. Trying alternative method.")
        try:
            return _process_audio_with_alternative(audio_file, config)
        except Exception as e2:
            logger.error(f"All feature extraction methods failed: {e2}")
            raise

def btc_chord_recognition(audio_file, output_file, model_variant='sl'):
    """
    Main BTC chord recognition function for API usage.
    
    Args:
        audio_file (str): Path to input audio file
        output_file (str): Path to output .lab file
        model_variant (str): 'sl' for supervised learning or 'pl' for pseudo-label
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set up logging
        logger.logging_verbosity(1)
        warnings.filterwarnings('ignore')
        
        # Force CPU usage
        device = torch.device("cpu")
        
        # Explicitly disable MPS
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            torch.backends.mps.enabled = False
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'btc_config.yaml')
        config = HParams.load(config_path)
        
        # BTC uses large vocabulary
        n_classes = config.model.get('num_chords', 170)
        idx_to_chord = idx2voca_chord()

        # Initialize BTC model
        model = BTC_model(config=config.model).to(device)
        
        # Load model checkpoint based on variant
        if model_variant == 'sl':
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'SL', 'btc_model_large_voca.pt')
        else:  # pl
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'btc', 'btc_combined_best.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found: {checkpoint_path}")
            return False
        
        # Load checkpoint with weights_only=False for compatibility
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            checkpoint = torch.load(checkpoint_path, map_location=device)
        # Extract model state dict and normalization stats based on checkpoint format
        if 'model_state_dict' in checkpoint:
            # PL format: {'model_state_dict': {...}, 'mean': tensor, 'std': tensor, ...}
            state_dict = checkpoint['model_state_dict']

            # Remove 'module.' prefix if present (from DataParallel training)
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

            model.load_state_dict(state_dict)

            # Extract normalization stats
            if 'mean' in checkpoint and 'std' in checkpoint:
                mean_val = checkpoint['mean'].item() if hasattr(checkpoint['mean'], 'item') else checkpoint['mean']
                std_val = checkpoint['std'].item() if hasattr(checkpoint['std'], 'item') else checkpoint['std']
                ckpt_mean = np.full(config.model.get('feature_size', 144), mean_val)
                ckpt_std = np.full(config.model.get('feature_size', 144), std_val)
            else:
                ckpt_mean = checkpoint.get('feature_mean', np.zeros(config.model.get('feature_size', 144)))
                ckpt_std = checkpoint.get('feature_std', np.ones(config.model.get('feature_size', 144)))
        elif 'model' in checkpoint:
            # SL format: {'model': {...}, 'mean': float, 'std': float}
            model.load_state_dict(checkpoint['model'])
            # Extract normalization stats
            mean_val = checkpoint.get('mean', 0.0)
            std_val = checkpoint.get('std', 1.0)
            ckpt_mean = np.full(config.model.get('feature_size', 144), mean_val)
            ckpt_std = np.full(config.model.get('feature_size', 144), std_val)
        else:
            # Direct state dict format
            model.load_state_dict(checkpoint)
            # Use default normalization if stats not available
            ckpt_mean = np.zeros(config.model.get('feature_size', 144))
            ckpt_std = np.ones(config.model.get('feature_size', 144))
        
        model.eval()
        
        # Process audio
        feature, feature_per_second, song_length_second = process_audio_with_padding(audio_file, config)
        
        if feature is None:
            logger.error("Feature extraction failed")
            return False
        
        # Transpose and normalize using checkpoint stats
        feature = feature.T  # Shape: [frames, features]
        logger.info(f"Feature stats BEFORE norm: Min={np.min(feature):.4f}, Max={np.max(feature):.4f}, Mean={np.mean(feature):.4f}, Std={np.std(feature):.4f}")
        epsilon = 1e-8
        feature = (feature - ckpt_mean) / (ckpt_std + epsilon)
        logger.info(f"Feature stats AFTER norm (using checkpoint stats): Min={np.min(feature):.4f}, Max={np.max(feature):.4f}, Mean={np.mean(feature):.4f}, Std={np.std(feature):.4f}")
        
        # Process in segments
        seq_len = config.model.get('seq_len', 108)
        original_num_frames = feature.shape[0]
        
        # Pad features
        num_pad = seq_len - (original_num_frames % seq_len)
        if num_pad == seq_len:
            num_pad = 0
        if num_pad > 0:
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        
        num_instance = feature.shape[0] // seq_len
        logger.info(f"Processing {num_instance} segments of length {seq_len}")

        # Run inference
        all_predictions_list = []
        with torch.no_grad():
            for t in range(num_instance):
                start_frame = t * seq_len
                end_frame = start_frame + seq_len
                segment_feature = feature[start_frame:end_frame, :]
                
                feature_tensor = torch.tensor(segment_feature, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(feature_tensor)
                predictions = torch.argmax(logits, dim=-1)
                segment_predictions = predictions.squeeze(0).cpu().numpy()
                all_predictions_list.append(segment_predictions)
        
        if not all_predictions_list:
            logger.warning("No predictions generated")
            return False
        
        all_predictions = np.concatenate(all_predictions_list, axis=0)
        all_predictions = all_predictions[:original_num_frames]
        
        # Function to standardize chord notation to match Chord-CNN-LSTM format
        def standardize_chord_notation(chord_name):
            # Convert 'N' to 'N/C' to match Chord-CNN-LSTM format
            if chord_name == 'N':
                return 'N/C'

            # Other standardizations can be added here if needed
            return chord_name

        # Generate .lab format
        lines = []
        if all_predictions.size > 0:
            prev_chord = all_predictions[0]
            start_time = 0.0
            min_segment_duration = 0.05

            for frame_idx, chord_idx in enumerate(all_predictions):
                current_time = frame_idx * feature_per_second

                if chord_idx != prev_chord:
                    segment_end_time = current_time
                    segment_duration = segment_end_time - start_time

                    if segment_duration >= min_segment_duration:
                        # Standardize chord notation
                        chord_name = standardize_chord_notation(idx_to_chord[prev_chord])
                        lines.append(f"{start_time:.6f}\t{segment_end_time:.6f}\t{chord_name}")

                    start_time = segment_end_time
                    prev_chord = chord_idx

            # Add final segment
            final_time = min(song_length_second, len(all_predictions) * feature_per_second)
            if start_time < final_time:
                last_segment_duration = final_time - start_time
                if last_segment_duration >= min_segment_duration:
                    # Standardize chord notation
                    chord_name = standardize_chord_notation(idx_to_chord[prev_chord])
                    lines.append(f"{start_time:.6f}\t{final_time:.6f}\t{chord_name}")

        # Save output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        
        logger.info(f"BTC chord recognition complete. Saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"BTC chord recognition failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
