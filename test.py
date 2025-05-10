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
# Make sure idx2voca_chord is imported correctly if needed, or adjust based on ChordNet output
from modules.utils.mir_eval_modules import idx2voca_chord # Assuming ChordNet uses large voca
from modules.utils.hparams import HParams
from modules.models.Transformer.ChordNet import ChordNet

# Explicitly disable MPS globally at the beginning
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
    torch.backends.mps.enabled = False

def get_audio_paths(audio_dir):
    """Get paths to all audio files in directory and subdirectories."""
    audio_paths = []
    for ext in ['*.wav', '*.mp3']:
        audio_paths.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
    return audio_paths

# --- Start: Insert robust process_audio_with_padding and helpers ---
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
    target_bins = config.feature.get('n_bins', 144)
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


def load_hmm_model(hmm_path, base_model, device):
    """
    Load HMM model for chord sequence smoothing

    Args:
        hmm_path: Path to the HMM model checkpoint
        base_model: The loaded base chord recognition model
        device: Device to load the model on

    Returns:
        Loaded HMM model or None if loading fails
    """
    try:
        from modules.models.HMM.ChordHMM import ChordHMM

        logger.info(f"Loading HMM model from {hmm_path}")

        # Load checkpoint
        checkpoint = torch.load(hmm_path, map_location=device)

        # Get configuration
        config = checkpoint.get('config', {})
        num_states = config.get('num_states', 170)

        # Create HMM model
        hmm_model = ChordHMM(
            pretrained_model=base_model,  # Use the already loaded base model
            num_states=num_states,
            device=device
        ).to(device)

        # Load HMM state dict
        hmm_model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"HMM model loaded successfully with {num_states} states")
        return hmm_model
    except Exception as e:
        logger.error(f"Error loading HMM model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run chord recognition on audio files")
    parser.add_argument('--audio_dir', type=str, default='./test',
                       help='Directory containing audio files')
    parser.add_argument('--save_dir', type=str, default='./test/output',
                       help='Directory to save output .lab files')
    parser.add_argument('--model_file', type=str, default=None,
                       help='Path to model checkpoint file (if None, will use default path)')
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--smooth_predictions', action='store_true',
                       help='Apply smoothing to predictions to reduce noise')
    parser.add_argument('--min_segment_duration', type=float, default=0.0,
                       help='Minimum duration in seconds for a chord segment (to reduce fragmentation)')
    parser.add_argument('--model_scale', type=float, default=1.0,
                       help='Scaling factor for model capacity (0.5=half, 1.0=base, 2.0=double)')
    parser.add_argument('--hmm', type=str, default=None,
                       help='Path to HMM model for chord sequence smoothing')
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

    # Always use large vocabulary
    config.feature['large_voca'] = True  # Set to True always
    n_classes = 170  # Large vocabulary size

    # Use external checkpoint path if model_file is not specified
    if args.model_file:
        model_file = args.model_file
    else:
        # First try external storage path
        external_model_path = '/mnt/storage/checkpoints/student/student_model_final.pth'
        if os.path.exists(external_model_path):
            model_file = external_model_path
            logger.info(f"Using external checkpoint at {model_file}")
        else:
            # Fall back to local path
            model_file = './checkpoints/student_model_final.pth'
            logger.info(f"External checkpoint not found, using local path: {model_file}")

    idx_to_chord = idx2voca_chord()
    logger.info("Using large vocabulary chord set (170 chords)")

    # Initialize model with proper scaling
    n_freq = config.feature.get('n_bins', 144)
    logger.info(f"Using n_freq={n_freq} for model input")

    # Get base configuration for the model
    base_config = config.model.get('base_config', {})

    # If base_config is not specified, fall back to direct model parameters
    if not base_config:
        base_config = {
            'f_layer': config.model.get('f_layer', 3),
            'f_head': config.model.get('f_head', 6),
            't_layer': config.model.get('t_layer', 3),
            't_head': config.model.get('t_head', 6),
            'd_layer': config.model.get('d_layer', 3),
            'd_head': config.model.get('d_head', 6)
        }

    # Apply scale to model parameters
    model_scale = args.model_scale
    logger.info(f"Using model scale: {model_scale}")

    f_layer = max(1, int(round(base_config.get('f_layer', 3) * model_scale)))
    f_head = max(1, int(round(base_config.get('f_head', 6) * model_scale)))
    t_layer = max(1, int(round(base_config.get('t_layer', 3) * model_scale)))
    t_head = max(1, int(round(base_config.get('t_head', 6) * model_scale)))
    d_layer = max(1, int(round(base_config.get('d_layer', 3) * model_scale)))
    d_head = max(1, int(round(base_config.get('d_head', 6) * model_scale)))

    # Log scaled parameters
    logger.info(f"Scaled model parameters: f_layer={f_layer}, f_head={f_head}, t_layer={t_layer}, t_head={t_head}, d_layer={d_layer}, d_head={d_head}")

    model = ChordNet(
        n_freq=n_freq,
        n_classes=n_classes,
        n_group=config.model.get('n_group', 4),
        f_layer=f_layer,
        f_head=f_head,
        t_layer=t_layer,
        t_head=t_head,
        d_layer=d_layer,
        d_head=d_head,
        dropout=config.model.get('dropout', 0.3)
    ).to(device)

    # Load model weights with explicit device mapping
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

        # Try to extract model scale from checkpoint if available
        if 'model_scale' in checkpoint:
            loaded_scale = checkpoint['model_scale']
            if loaded_scale != model_scale:
                logger.warning(f"Model was trained with scale {loaded_scale} but using scale {model_scale} for inference")

        # Check if model state dict is directly available or nested
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Try to use the checkpoint directly as a state dict
            state_dict = checkpoint

        # Check if the state dict has 'module.' prefix (from DistributedDataParallel)
        # and remove it if necessary
        if any(k.startswith('module.') for k in state_dict.keys()):
            logger.info("Detected 'module.' prefix in state dict keys. Removing prefix for compatibility.")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Now load the cleaned state dict
        try:
            model.load_state_dict(state_dict)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model state dict: {e}")
            # Try with strict=False as a fallback
            logger.info("Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model loaded with strict=False (some weights may be missing)")

        # Set idx_to_chord attribute on the model if available in checkpoint
        if 'idx_to_chord' in checkpoint:
            model.idx_to_chord = checkpoint['idx_to_chord']
            logger.info("Loaded idx_to_chord mapping from checkpoint")
        else:
            # Set default mapping
            model.set_chord_mapping(None)
            logger.info("Using default idx_to_chord mapping")

        # Get normalization parameters
        ckpt_mean = checkpoint.get('mean', 0.0) # Use ckpt_mean/std consistently
        ckpt_std = checkpoint.get('std', 1.0)

        # Convert tensors to numpy arrays if needed
        if isinstance(ckpt_mean, torch.Tensor):
            ckpt_mean = ckpt_mean.cpu().numpy()
        if isinstance(ckpt_std, torch.Tensor):
            ckpt_std = ckpt_std.cpu().numpy()

        # Ensure we have scalar values if they're single-element arrays
        if hasattr(ckpt_mean, 'shape') and ckpt_mean.size == 1:
            ckpt_mean = float(ckpt_mean.item() if hasattr(ckpt_mean, 'item') else ckpt_mean)
        if hasattr(ckpt_std, 'shape') and ckpt_std.size == 1:
            ckpt_std = float(ckpt_std.item() if hasattr(ckpt_std, 'item') else ckpt_std)

        logger.info(f"Using Checkpoint Normalization parameters: mean={ckpt_mean:.4f}, std={ckpt_std:.4f}")
    else:
        logger.error(f"Model file not found: {model_file}")
        ckpt_mean, ckpt_std = 0.0, 1.0

    # Load HMM model if specified
    hmm_model = None
    if args.hmm and os.path.isfile(args.hmm):
        hmm_model = load_hmm_model(args.hmm, model, device)
        if hmm_model:
            logger.info("HMM model will be used for prediction smoothing")
    elif args.hmm:
        logger.warning(f"HMM model file not found: {args.hmm}")

    # Create output directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Get all audio files
    audio_paths = get_audio_paths(args.audio_dir)
    logger.info(f"Found {len(audio_paths)} audio files to process")

    # Process each audio file - ensure consistent device usage
    for i, audio_path in enumerate(audio_paths):
        logger.info(f"Processing file {i+1} of {len(audio_paths)}: {os.path.basename(audio_path)}")

        try:
            # Use our custom function with better padding handling
            feature, feature_per_second, song_length_second = process_audio_with_padding(audio_path, config)
            # Check if feature extraction failed
            if feature is None:
                 logger.error(f"Feature extraction returned None for {audio_path}. Skipping file.")
                 continue
            logger.info(f"Feature extraction complete: {feature.shape}, {feature_per_second:.4f} sec/frame")

            # Transpose and normalize using checkpoint stats
            feature = feature.T  # Shape: [frames, features]
            logger.info(f"Feature stats BEFORE norm: Min={np.min(feature):.4f}, Max={np.max(feature):.4f}, Mean={np.mean(feature):.4f}, Std={np.std(feature):.4f}")

            # Apply normalization - ensure types match
            epsilon = 1e-8
            # Handle different shapes of normalization parameters
            if hasattr(ckpt_mean, 'shape') and len(ckpt_mean.shape) > 0 and ckpt_mean.shape[0] > 1:
                # If mean/std are arrays with the same dimension as features, reshape for broadcasting
                if ckpt_mean.shape[0] == feature.shape[1]:
                    ckpt_mean_reshaped = ckpt_mean.reshape(1, -1)  # [1, features]
                    ckpt_std_reshaped = ckpt_std.reshape(1, -1)    # [1, features]
                    feature = (feature - ckpt_mean_reshaped) / (ckpt_std_reshaped + epsilon)
                else:
                    # If shapes don't match, use scalar mean/std
                    logger.warning(f"Normalization parameter shape mismatch: mean shape {ckpt_mean.shape}, feature shape {feature.shape}. Using scalar normalization.")
                    feature = (feature - float(np.mean(ckpt_mean))) / (float(np.mean(ckpt_std)) + epsilon)
            else:
                # Scalar normalization
                feature = (feature - ckpt_mean) / (ckpt_std + epsilon)

            logger.info(f"Feature stats AFTER norm (using checkpoint stats): Min={np.min(feature):.4f}, Max={np.max(feature):.4f}, Mean={np.mean(feature):.4f}, Std={np.std(feature):.4f}")

            # Get sequence length from config or checkpoint
            # Try to get from checkpoint first, then config, then use default
            n_timestep = None

            # Try to get from checkpoint
            if 'timestep' in checkpoint:
                n_timestep = checkpoint['timestep']
                logger.info(f"Using timestep={n_timestep} from checkpoint")
            # Try to get from model config
            elif hasattr(config, 'model') and hasattr(config.model, 'timestep'):
                n_timestep = config.model.timestep
                logger.info(f"Using timestep={n_timestep} from config")
            # Try to get from model attribute
            elif hasattr(model, 'timestep'):
                n_timestep = model.timestep
                logger.info(f"Using timestep={n_timestep} from model attribute")
            # Use default
            else:
                n_timestep = 108  # Default value from ChordMini
                logger.info(f"Using default timestep={n_timestep}")

            # Pad to match sequence length
            original_num_frames = feature.shape[0] # Store original length before padding
            num_pad = n_timestep - (original_num_frames % n_timestep)
            if num_pad == n_timestep: # No padding needed if already multiple
                num_pad = 0
            if num_pad > 0:
                feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)

            num_instance = feature.shape[0] // n_timestep
            logger.info(f"Processing {num_instance} segments of length {n_timestep}")

            # Initialize
            start_time = 0.0
            lines = []
            all_predictions = []

            # Process features and generate predictions
            with torch.no_grad():
                model.eval()
                model = model.cpu() # Move model to CPU explicitly
                feature_tensor = torch.tensor(feature, dtype=torch.float32).cpu() # Ensure feature tensor is on CPU

                # Get raw frame-level predictions in batches to avoid OOM errors
                raw_predictions_list = [] # Use list to collect batch outputs
                batch_size = 32  # Process in smaller batches

                for t in range(0, num_instance, batch_size):
                    batch_start_instance = t
                    batch_end_instance = min(t + batch_size, num_instance)
                    batch_count = batch_end_instance - batch_start_instance

                    # Create batch tensor
                    batch_segments = []
                    for b_idx in range(batch_count):
                        instance_idx = batch_start_instance + b_idx
                        start_frame = instance_idx * n_timestep
                        end_frame = start_frame + n_timestep
                        batch_segments.append(feature_tensor[start_frame:end_frame, :])

                    if not batch_segments: continue # Skip if batch is empty

                    segment_batch = torch.stack(batch_segments, dim=0).cpu() # Ensure batch is on CPU

                    # Get frame-level predictions using model.predict
                    with torch.no_grad():
                        # ChordNet's predict might return indices directly
                        prediction_indices = model.predict(segment_batch, per_frame=True)
                        # Ensure prediction is on CPU
                        if prediction_indices.device.type != 'cpu':
                            prediction_indices = prediction_indices.cpu()

                    # Collect raw predictions for this batch
                    raw_predictions_list.append(prediction_indices.numpy()) # Shape: [batch_count, n_timestep]


                # Concatenate predictions from all batches
                if not raw_predictions_list:
                    logger.warning(f"No predictions generated for {audio_path}")
                    all_predictions = np.array([], dtype=int)
                else:
                    # Concatenate along the batch dimension first, then flatten
                    raw_predictions_concat = np.concatenate(raw_predictions_list, axis=0) # Shape: [num_instance, n_timestep]
                    raw_predictions = raw_predictions_concat.flatten() # Shape: [num_instance * n_timestep]

                    # Trim padding predictions based on original frame count
                    raw_predictions = raw_predictions[:original_num_frames]

                    # Apply HMM smoothing if model is available
                    if hmm_model is not None:
                        logger.info("Applying HMM sequence modeling to predictions...")
                        hmm_model.eval()
                        hmm_model = hmm_model.cpu() # Ensure HMM is on CPU

                        # Run Viterbi decoding on the full (original length) feature tensor
                        # Need the unpadded, normalized feature tensor
                        original_feature_tensor = torch.tensor(feature[:original_num_frames], dtype=torch.float32).cpu()

                        with torch.no_grad():
                            # HMM decode expects [batch=1, time, features]
                            all_predictions = hmm_model.decode(original_feature_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

                        logger.info(f"HMM smoothed predictions generated: {all_predictions.shape}")
                        # Ensure HMM output length matches original frames
                        if len(all_predictions) != original_num_frames:
                             logger.warning(f"HMM output length ({len(all_predictions)}) differs from original frames ({original_num_frames}). Trimming/Padding HMM output.")
                             # Simple trim/pad - might need more sophisticated alignment
                             if len(all_predictions) > original_num_frames:
                                 all_predictions = all_predictions[:original_num_frames]
                             else: # Pad with last prediction if shorter
                                 pad_val = all_predictions[-1] if len(all_predictions) > 0 else 169 # Default to N if empty
                                 all_predictions = np.pad(all_predictions, (0, original_num_frames - len(all_predictions)), mode='constant', constant_values=pad_val)

                    else:
                        # Use raw predictions or apply simple smoothing
                        all_predictions = raw_predictions

                        # Apply smoothing if requested
                        if args.smooth_predictions:
                            from scipy.signal import medfilt
                            kernel_size = 3
                            if len(all_predictions) < kernel_size:
                                logger.warning(f"Prediction length ({len(all_predictions)}) too short for smoothing kernel ({kernel_size}). Skipping smoothing.")
                            else:
                                all_predictions = medfilt(all_predictions, kernel_size=kernel_size)
                                logger.info(f"Applied median filtering (k={kernel_size}) to predictions")

            # Find chord boundaries
            lines = []
            if all_predictions.size == 0:
                 logger.warning("Prediction array is empty. Cannot generate .lab file.")
            else:
                prev_chord = all_predictions[0]
                start_time = 0.0
                current_time = 0.0
                segment_duration = 0.0

                # Process frame by frame, applying minimum segment duration if specified
                for i, chord_idx in enumerate(all_predictions):
                    current_time = i * feature_per_second

                    # Detect chord changes
                    if chord_idx != prev_chord:
                        segment_end_time = current_time
                        segment_duration = segment_end_time - start_time

                        # Only add segment if it's longer than minimum duration
                        if segment_duration >= args.min_segment_duration:
                            lines.append(f"{start_time:.6f} {segment_end_time:.6f} {idx_to_chord[prev_chord]}\n")

                        start_time = segment_end_time
                        prev_chord = chord_idx

                # Add the final segment
                total_processed_frames = len(all_predictions)
                feature_sequence_duration = total_processed_frames * feature_per_second
                final_time = min(song_length_second, feature_sequence_duration)

                if start_time < final_time:
                    last_segment_duration = final_time - start_time
                    if last_segment_duration >= args.min_segment_duration:
                         lines.append(f"{start_time:.6f} {final_time:.6f} {idx_to_chord[prev_chord]}\n")


            # Save output to .lab file with HMM suffix if applicable
            output_filename = os.path.splitext(os.path.basename(audio_path))[0]
            if hmm_model is not None:
                output_filename += '_hmm'  # Add suffix to indicate HMM processing
            output_filename += '.lab'

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

    logger.info("Chord recognition complete")

if __name__ == "__main__":
    # Add dependency check similar to test_btc.py if desired
    main()