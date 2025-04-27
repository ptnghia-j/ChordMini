import os
import argparse
import numpy as np
import torch
import warnings
import glob
from pathlib import Path
import librosa # Add librosa import
import soundfile as sf # Add soundfile import
from scipy import signal, interpolate # Add scipy imports

# Project imports
from modules.utils import logger
from modules.utils.mir_eval_modules import idx2voca_chord # BTC typically uses large voca
from modules.utils.hparams import HParams
from modules.models.Transformer.btc_model import BTC_model # Import BTC model

# Explicitly disable MPS globally at the beginning
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
    torch.backends.mps.enabled = False

def get_audio_paths(audio_dir):
    """Get paths to all audio files in directory and subdirectories."""
    audio_paths = []
    for ext in ['*.wav', '*.mp3']:
        audio_paths.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
    return audio_paths

# --- Start: Insert robust process_audio_with_padding and helpers (copied from test.py) ---
def _process_audio_with_librosa_cqt(audio_file, config):
    """Process audio using librosa's CQT implementation (requires Numba)."""
    # Get FFT size from config or use default
    n_fft = config.feature.get('n_fft', 512)

    # Load audio data
    original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)

    # Log audio length for debugging
    logger.info(f"Audio loaded: {len(original_wav)/sr:.2f} seconds ({len(original_wav)} samples)")

    # If the entire audio is too short, pad it immediately
    if len(original_wav) < n_fft:
        logger.warning(f"Entire audio file is too short ({len(original_wav)} samples). Padding to {n_fft} samples.")
        original_wav = np.pad(original_wav, (0, n_fft - len(original_wav)), mode="constant", constant_values=0)

    current_sec_hz = 0
    feature = None  # initialize feature

    # Process main segments - full-length ones first
    while len(original_wav) > current_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(current_sec_hz)
        end_idx = int(current_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        segment = original_wav[start_idx:end_idx]

        # Add extra check for segment length before CQT
        if len(segment) < n_fft:
            logger.warning(f"Segment is too short ({len(segment)} samples). Padding to {n_fft} samples.")
            segment = np.pad(segment, (0, n_fft - len(segment)), mode="constant", constant_values=0)

        # Process segment with CQT
        tmp = librosa.cqt(segment,
                          sr=sr,
                          n_bins=config.feature['n_bins'],
                          bins_per_octave=config.feature['bins_per_octave'],
                          hop_length=config.feature['hop_length'])

        if feature is None:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)

        current_sec_hz = end_idx

    # Process the final segment with proper padding
    if current_sec_hz < len(original_wav):
        final_segment = original_wav[current_sec_hz:]

        # Always ensure the final segment is at least n_fft samples long
        if len(final_segment) < n_fft:
            logger.info(f"Final segment is too short ({len(final_segment)} samples). Padding to {n_fft} samples.")
            final_segment = np.pad(final_segment, (0, n_fft - len(final_segment)), mode="constant", constant_values=0)

        # Process the properly sized segment
        tmp = librosa.cqt(final_segment,
                          sr=sr,
                          n_bins=config.feature['n_bins'],
                          bins_per_octave=config.feature['bins_per_octave'],
                          hop_length=config.feature['hop_length'])

        if feature is None:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)

    # Apply logarithmic scaling and return results
    feature = np.log(np.abs(feature) + 1e-6)
    song_length_second = len(original_wav) / config.mp3['song_hz']
    # Calculate feature_per_second based on hop_length and sample rate
    feature_per_second = config.feature['hop_length'] / config.mp3['song_hz']

    return feature, feature_per_second, song_length_second

def _process_audio_with_alternative(audio_file, config):
    """Alternative feature extraction that doesn't rely on Numba."""
    try:
        # Try to load with soundfile first
        audio_data, sr = sf.read(audio_file)
        if audio_data.ndim > 1:
            # Convert stereo to mono by averaging channels
            audio_data = np.mean(audio_data, axis=1)
    except Exception:
        # Fall back to scipy.io.wavfile if soundfile fails
        try:
            from scipy.io import wavfile
            sr, audio_data = wavfile.read(audio_file)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            # Normalize to [-1, 1] range if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
        except Exception as e:
            logger.error(f"Failed to load audio with scipy.io.wavfile: {e}")
            raise

    # Resample to target sample rate if needed
    target_sr = config.mp3['song_hz']
    if sr != target_sr:
        # Calculate resampling ratio
        ratio = target_sr / sr
        # Calculate new length
        new_length = int(len(audio_data) * ratio)
        # Resample using scipy.signal.resample
        audio_data = signal.resample(audio_data, new_length)
        sr = target_sr

    logger.info(f"Audio loaded: {len(audio_data)/sr:.2f} seconds ({len(audio_data)} samples)")

    # Create a simple spectrogram using STFT as an alternative to CQT
    n_fft = 2048  # Default FFT size
    hop_length = config.feature.get('hop_length', 512)

    # Compute STFT
    f, t, Zxx = signal.stft(audio_data, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)

    # Take magnitude and convert to log scale
    feature = np.log(np.abs(Zxx) + 1e-6)

    # Resize to match expected feature dimensions if needed
    target_bins = config.feature.get('n_bins', 144) # Use n_bins from config
    if feature.shape[0] != target_bins:
        # Simple linear interpolation for frequency bins
        x = np.linspace(0, 1, feature.shape[0])
        x_new = np.linspace(0, 1, target_bins)
        feature_resized = np.zeros((target_bins, feature.shape[1]))

        for i in range(feature.shape[1]):
            # Use scipy.interpolate.interp1d
            f_interp = interpolate.interp1d(x, feature[:, i])
            feature_resized[:, i] = f_interp(x_new)

        feature = feature_resized

    # Calculate timing information
    song_length_second = len(audio_data) / sr
    feature_per_second = hop_length / sr

    logger.info(f"Alternative feature extraction complete: {feature.shape}")
    return feature, feature_per_second, song_length_second

def process_audio_with_padding(audio_file, config):
    """
    Safely process audio file with proper handling of short segments.
    This wrapper ensures all segments meet the minimum FFT size requirements.
    Tries CQT first, falls back to STFT if Numba/Librosa fails.
    """
    try:
        # Try to use librosa's CQT first
        try:
            logger.info(f"Attempting CQT feature extraction for {os.path.basename(audio_file)}...")
            features, feat_per_sec, song_len = _process_audio_with_librosa_cqt(audio_file, config)
            logger.info("CQT feature extraction successful.")
            return features, feat_per_sec, song_len
        except ImportError as e:
            logger.error(f"Librosa CQT failed due to import error: {e}")
            logger.warning("********************************************************************************")
            logger.warning(" CQT feature extraction failed. This is likely due to a Numba/NumPy version conflict.")
            logger.warning(" The model was likely trained on CQT features. Using STFT fallback may yield poor results.")
            logger.warning(" Please resolve the environment conflict (e.g., 'pip install \"numpy<2.0\"') for best accuracy.")
            logger.warning("********************************************************************************")
            logger.info("Falling back to alternative STFT feature extraction method.")
            return _process_audio_with_alternative(audio_file, config)
        except Exception as e_cqt:
             logger.error(f"Librosa CQT failed due to unexpected error: {e_cqt}")
             logger.warning("Falling back to alternative STFT feature extraction method.")
             return _process_audio_with_alternative(audio_file, config)

    except Exception as e_main:
        logger.error(f"Error processing audio file {audio_file}: {e_main}")
        import traceback
        logger.error(traceback.format_exc())
        raise # Re-raise the exception after logging
# --- End: Insert robust process_audio_with_padding and helpers ---

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run BTC chord recognition on audio files")
    parser.add_argument('--audio_dir', type=str, default='./test',
                       help='Directory containing audio files')
    parser.add_argument('--save_dir', type=str, default='./test/output_btc',
                       help='Directory to save output .lab files')
    parser.add_argument('--model_file', type=str, default=None, # Default to None to use external path
                       help='Path to BTC model checkpoint file')
    parser.add_argument('--config', type=str, default='./config/btc_config.yaml', # Default to BTC config
                       help='Path to BTC configuration file')
    parser.add_argument('--min_segment_duration', type=float, default=0.05, # Add min duration arg
                       help='Minimum duration in seconds for a chord segment (to reduce fragmentation)')
    args = parser.parse_args()

    # Set up logging
    logger.logging_verbosity(1)
    warnings.filterwarnings('ignore')

    # Force CPU usage regardless of what's available
    device = torch.device("cpu")
    logger.info(f"Forcing CPU usage for consistent device handling")

    # Explicitly disable MPS again to be safe
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        torch.backends.mps.enabled = False

    logger.info(f"Using device: {device}")

    # Load configuration
    config = HParams.load(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # BTC uses large vocabulary
    n_classes = config.model.get('num_chords', 170) # Get from config
    idx_to_chord = idx2voca_chord()
    logger.info(f"Using large vocabulary chord set ({n_classes} chords)")

    # Initialize BTC model
    n_freq = config.model.get('feature_size', 144) # Get from config
    logger.info(f"Using n_freq={n_freq} for model input")

    # Create model instance using the model sub-config
    model = BTC_model(config=config.model).to(device)

    # Load model weights with explicit device mapping
    ckpt_mean, ckpt_std = 0.0, 1.0 # Default normalization

    # Use external checkpoint path if model_file is not specified
    if args.model_file:
        model_file = args.model_file
    else:
        # First try external storage path
        external_model_path = '/mnt/storage/checkpoints/btc/btc_model_best.pth'
        if os.path.exists(external_model_path):
            model_file = external_model_path
            logger.info(f"Using external checkpoint at {model_file}")
        else:
            # Fall back to local path
            model_file = './checkpoints/btc/btc_model_best.pth'
            logger.info(f"External checkpoint not found, using local path: {model_file}")

    if os.path.isfile(model_file):
        logger.info(f"Loading model from {model_file}")
        try:
            # First try loading with weights_only=False (for PyTorch 2.6+ compatibility)
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)
            logger.info("Model loaded successfully with weights_only=False")
        except TypeError:
            # Fall back to older PyTorch versions that don't have weights_only parameter
            logger.info("Falling back to legacy loading method (for older PyTorch versions)")
            checkpoint = torch.load(model_file, map_location=device)

        # Check if model state dict is directly available or nested
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint: # Handle older checkpoint format
             model.load_state_dict(checkpoint['model'])
        else:
            # Try to load the state dict directly
            model.load_state_dict(checkpoint)

        # Get normalization parameters
        ckpt_mean = checkpoint.get('mean', 0.0)
        ckpt_std = checkpoint.get('std', 1.0)
        logger.info(f"Using Checkpoint Normalization parameters: mean={ckpt_mean:.4f}, std={ckpt_std:.4f}")
    else:
        logger.error(f"Model file not found: {model_file}. Using default normalization.")

    # Create output directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Get all audio files
    audio_paths = get_audio_paths(args.audio_dir)
    logger.info(f"Found {len(audio_paths)} audio files to process")

    # Process each audio file
    for i, audio_path in enumerate(audio_paths):
        logger.info(f"Processing file {i+1} of {len(audio_paths)}: {os.path.basename(audio_path)}")

        try:
            # Use robust function with padding handling
            feature, feature_per_second, song_length_second = process_audio_with_padding(audio_path, config)
            # Check if feature extraction failed
            if feature is None:
                 logger.error(f"Feature extraction returned None for {audio_path}. Skipping file.")
                 continue
            logger.info(f"Feature extraction complete: {feature.shape}, {feature_per_second:.4f} sec/frame")

            # Transpose and normalize using checkpoint stats
            feature = feature.T  # Shape: [frames, features]
            logger.info(f"Feature stats BEFORE norm: Min={np.min(feature):.4f}, Max={np.max(feature):.4f}, Mean={np.mean(feature):.4f}, Std={np.std(feature):.4f}")
            epsilon = 1e-8
            feature = (feature - ckpt_mean) / (ckpt_std + epsilon)
            logger.info(f"Feature stats AFTER norm (using checkpoint stats): Min={np.min(feature):.4f}, Max={np.max(feature):.4f}, Mean={np.mean(feature):.4f}, Std={np.std(feature):.4f}")

            # --- Start: Process features in segments ---
            seq_len = config.model.get('seq_len', 108) # Get sequence length from config
            original_num_frames = feature.shape[0] # Store original length

            # Pad features to be a multiple of seq_len
            num_pad = seq_len - (original_num_frames % seq_len)
            if num_pad == seq_len: # No padding needed if already multiple
                num_pad = 0
            if num_pad > 0:
                # Pad the second dimension (features) with zeros
                feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)

            num_instance = feature.shape[0] // seq_len
            logger.info(f"Processing {num_instance} segments of length {seq_len}")

            # Prepare for model input (process segment by segment)
            all_predictions_list = []
            with torch.no_grad():
                model.eval()
                model = model.to(device) # Ensure model is on the correct device

                for t in range(num_instance):
                    start_frame = t * seq_len
                    end_frame = start_frame + seq_len
                    segment_feature = feature[start_frame:end_frame, :]

                    # Add batch dimension: [1, seq_len, features]
                    feature_tensor = torch.tensor(segment_feature, dtype=torch.float32).unsqueeze(0).to(device)

                    # BTC model processes the segment
                    logits = model(feature_tensor) # Shape: [1, seq_len, classes]

                    # Get predictions (argmax over classes dimension)
                    predictions = torch.argmax(logits, dim=-1) # Shape: [1, seq_len]
                    segment_predictions = predictions.squeeze(0).cpu().numpy() # Shape: [seq_len]
                    all_predictions_list.append(segment_predictions)

            # Concatenate predictions from all segments
            if not all_predictions_list:
                 logger.warning(f"No predictions generated for {audio_path}")
                 all_predictions = np.array([], dtype=int)
            else:
                 all_predictions = np.concatenate(all_predictions_list, axis=0) # Shape: [num_instance * seq_len]
                 # Trim predictions to original length
                 all_predictions = all_predictions[:original_num_frames]
            # --- End: Process features in segments ---


            # Find chord boundaries and generate .lab format
            lines = []
            if all_predictions.size == 0:
                 logger.warning("Prediction array is empty. Cannot generate .lab file.")
            else:
                prev_chord = all_predictions[0]
                start_time = 0.0
                current_time = 0.0
                segment_duration = 0.0

                # Process frame by frame, applying minimum segment duration
                for frame_idx, chord_idx in enumerate(all_predictions):
                    current_time = frame_idx * feature_per_second

                    # Detect chord changes
                    if chord_idx != prev_chord:
                        segment_end_time = current_time
                        segment_duration = segment_end_time - start_time

                        # Only add segment if it's longer than minimum duration
                        if segment_duration >= args.min_segment_duration:
                            lines.append(f"{start_time:.6f} {segment_end_time:.6f} {idx_to_chord[prev_chord]}\n")
                        # If segment is too short, merge with the previous one by *not* updating start_time
                        # However, the original logic implies we just discard short segments.
                        # Let's stick to discarding short segments for now.
                        # If merging is desired, the logic needs adjustment.
                        start_time = segment_end_time
                        prev_chord = chord_idx

                # Add the final segment
                total_processed_frames = len(all_predictions)
                feature_sequence_duration = total_processed_frames * feature_per_second
                # Ensure final time doesn't exceed actual song length (can happen due to padding)
                final_time = min(song_length_second, feature_sequence_duration)

                if start_time < final_time:
                    last_segment_duration = final_time - start_time
                    if last_segment_duration >= args.min_segment_duration:
                         lines.append(f"{start_time:.6f} {final_time:.6f} {idx_to_chord[prev_chord]}\n")
                    # Handle case where the last segment is too short (optional: merge with previous)
                    # elif lines: # If there was a previous line
                    #     # Modify the end time of the last added line
                    #     last_line_parts = lines[-1].strip().split()
                    #     last_line_start = float(last_line_parts[0])
                    #     last_line_chord = last_line_parts[2]
                    #     lines[-1] = f"{last_line_start:.6f} {final_time:.6f} {last_line_chord}\n"


            # Save output to .lab file
            output_filename = os.path.splitext(os.path.basename(audio_path))[0] + '.lab'
            output_path = os.path.join(args.save_dir, output_filename)

            with open(output_path, 'w') as f:
                for line in lines:
                    f.write(line)

            logger.info(f"Saved chord annotations to {output_path}")

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())  # Add detailed error trace
            continue

    logger.info("BTC Chord recognition complete")

if __name__ == "__main__":
    # Optional: Add dependency checks here if needed
    # Example:
    # try:
    #     import librosa
    #     import soundfile
    #     import scipy
    # except ImportError as e:
    #     logger.error(f"Missing dependency: {e}. Please install required packages.")
    #     exit(1)
    main()
