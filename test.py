import os
import argparse
import numpy as np
import torch
import warnings
import glob
from pathlib import Path

# Project imports
from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features, idx2voca_chord
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

def process_audio_with_padding(audio_file, config):
    """
    Safely process audio file with proper handling of short segments.
    This wrapper ensures all segments meet the minimum FFT size requirements.
    """
    import librosa
    import numpy as np
    
    try:
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
        
        currunt_sec_hz = 0
        feature = None  # initialize feature
        
        # Process main segments - full-length ones first
        while len(original_wav) > currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
            start_idx = int(currunt_sec_hz)
            end_idx = int(currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
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
            
            currunt_sec_hz = end_idx
        
        # Process the final segment with proper padding
        if currunt_sec_hz < len(original_wav):
            final_segment = original_wav[currunt_sec_hz:]
            
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
    
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        raise

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
    model_file = args.model_file or './checkpoints/student_model_final.pth'
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
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # Try to load the state dict directly
            model.load_state_dict(checkpoint)
        
        # Get normalization parameters
        mean = checkpoint.get('mean', 0.0)
        std = checkpoint.get('std', 1.0)
        logger.info(f"Normalization parameters: mean={mean}, std={std}")
    else:
        logger.error(f"Model file not found: {model_file}")
        mean, std = 0.0, 1.0  # Default values
        logger.warning("Using default normalization parameters")

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
            logger.info(f"Feature extraction complete: {feature.shape}, {feature_per_second:.4f} sec/frame")
            
            # Transpose and normalize
            feature = feature.T  # Shape: [frames, features]
            feature = (feature - mean) / std
            
            # Get sequence length from config
            n_timestep = config.model.get('timestep', 10)
            
            # Pad to match sequence length
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            if num_pad < n_timestep:
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
                # Move model to CPU explicitly
                model = model.cpu()
                
                # Ensure feature tensor is on CPU
                feature_tensor = torch.tensor(feature, dtype=torch.float32)
                
                # Get raw frame-level predictions in batches to avoid OOM errors
                raw_predictions = []
                batch_size = 32  # Process in smaller batches
                
                for t in range(0, num_instance, batch_size):
                    end_idx = min(t + batch_size, num_instance)
                    batch_count = end_idx - t
                    
                    # Create a batch of the appropriate size
                    if batch_count == 1:
                        # Handle single segment case
                        segment = feature_tensor[n_timestep * t:n_timestep * (t + 1), :].unsqueeze(0)
                    else:
                        # Collect multiple segments
                        segments = []
                        for b in range(batch_count):
                            if t + b < num_instance:
                                seg = feature_tensor[n_timestep * (t+b):n_timestep * (t+b+1), :]
                                segments.append(seg)
                        
                        if segments:
                            segment = torch.stack(segments, dim=0)
                        else:
                            continue
                    
                    # Get frame-level predictions - ensure everything stays on CPU
                    with torch.no_grad():
                        # Explicitly move segment to CPU to match model device
                        segment = segment.cpu()
                        prediction = model.predict(segment, per_frame=True)
                        # Explicitly move prediction to CPU if it's not already
                        if prediction.device.type != 'cpu':
                            prediction = prediction.cpu()
                    
                    # Flatten and collect predictions
                    if prediction.dim() > 1:
                        for p in prediction:
                            raw_predictions.append(p.cpu().numpy())
                    else:
                        raw_predictions.append(prediction.cpu().numpy())
                
                # Concatenate all raw predictions
                raw_predictions = np.concatenate(raw_predictions)
                
                # Apply HMM smoothing if model is available
                if hmm_model is not None:
                    logger.info("Applying HMM sequence modeling to predictions...")
                    hmm_model.eval()
                    
                    # Move everything to CPU for consistent processing
                    hmm_model = hmm_model.cpu()
                    
                    # Run Viterbi decoding on the full sequence for best results
                    feature_tensor = feature_tensor.cpu()
                    with torch.no_grad():
                        all_predictions = hmm_model.decode(feature_tensor).cpu().numpy()
                    
                    logger.info(f"HMM smoothed predictions generated: {all_predictions.shape}")
                else:
                    # Use raw predictions or apply simple smoothing
                    all_predictions = raw_predictions
            
                    # Apply smoothing if requested
                    if args.smooth_predictions:
                        # Apply median filtering with window size 3
                        from scipy.signal import medfilt
                        all_predictions = medfilt(all_predictions, kernel_size=3)
                        logger.info("Applied median filtering to predictions")
            
            # Find chord boundaries
            prev_chord = all_predictions[0]
            start_time = 0.0
            current_time = 0.0
            segment_duration = 0.0
            
            # Process frame by frame, applying minimum segment duration if specified
            for i, chord_idx in enumerate(all_predictions):
                current_time = i * feature_per_second
                
                # Detect chord changes
                if chord_idx != prev_chord:
                    segment_duration = current_time - start_time
                    
                    # Only add segment if it's longer than minimum duration
                    if segment_duration >= args.min_segment_duration:
                        lines.append(f"{start_time:.6f} {current_time:.6f} {idx_to_chord[prev_chord]}\n")
                    
                    start_time = current_time
                    prev_chord = chord_idx
            
            # Add the final segment
            if start_time < feature.shape[0] * feature_per_second:
                final_time = min(song_length_second, feature.shape[0] * feature_per_second)
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
    main()