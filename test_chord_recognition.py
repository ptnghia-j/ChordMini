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

def get_audio_paths(audio_dir):
    """Get paths to all audio files in directory and subdirectories."""
    audio_paths = []
    for ext in ['*.wav', '*.mp3']:
        audio_paths.extend(glob.glob(os.path.join(audio_dir, '**', ext), recursive=True))
    return audio_paths

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run chord recognition on audio files")
    parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower() == 'true'),
                       help='Use large vocabulary chord set (True) or just major/minor (False)')
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
    args = parser.parse_args()

    # Set up logging
    logger.logging_verbosity(1)
    warnings.filterwarnings('ignore')

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load configuration
    config = HParams.load(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Configure model based on vocabulary size
    if args.voca:
        config.feature['large_voca'] = True
        n_classes = 170  # Large vocabulary
        model_file = args.model_file or './checkpoints/student_model_best_large_voca.pth'
        idx_to_chord = idx2voca_chord()
        logger.info("Using large vocabulary chord set")
    else:
        config.feature['large_voca'] = False
        n_classes = 25  # Major and minor chords only (12 roots * 2 + no-chord)
        model_file = args.model_file or './checkpoints/student_model_best.pth'
        # Simplified mapping for major/minor only
        idx_to_chord = {i: f"{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][i//2]}{'m' if i%2 else ''}" 
                       for i in range(24)}
        idx_to_chord[24] = "N"  # No chord
        logger.info("Using major/minor chord set")

    # Initialize model - we need to detect the right input dimension from config
    n_freq = config.feature.get('n_bins', 144)
    logger.info(f"Using n_freq={n_freq} for model input")
    
    model = ChordNet(
        n_freq=n_freq,
        n_classes=n_classes,
        n_group=config.model.get('n_group', 4),
        f_layer=config.model.get('f_layer', 3),
        f_head=config.model.get('f_head', 6),
        t_layer=config.model.get('t_layer', 3),
        t_head=config.model.get('t_head', 6),
        d_layer=config.model.get('d_layer', 3),
        d_head=config.model.get('d_head', 6),
        dropout=config.model.get('dropout', 0.3)
    ).to(device)

    # Load model weights
    if os.path.isfile(model_file):
        logger.info(f"Loading model from {model_file}")
        checkpoint = torch.load(model_file, map_location=device)
        
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

    # Create output directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get all audio files
    audio_paths = get_audio_paths(args.audio_dir)
    logger.info(f"Found {len(audio_paths)} audio files to process")
    
    # Process each audio file
    for i, audio_path in enumerate(audio_paths):
        logger.info(f"Processing file {i+1} of {len(audio_paths)}: {os.path.basename(audio_path)}")
        
        try:
            # Extract features from audio
            feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
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
                feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get frame-level predictions
                for t in range(num_instance):
                    # Extract current segment
                    segment = feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :]
                    
                    # Get frame-level predictions
                    prediction = model.predict(segment, per_frame=True).squeeze()
                    
                    # Ensure prediction is a 1D tensor
                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)
                    
                    # Store predictions
                    all_predictions.append(prediction.cpu().numpy())
            
            # Concatenate all predictions
            all_predictions = np.concatenate(all_predictions)
            
            # Apply smoothing if requested
            if args.smooth_predictions:
                # Apply median filtering with window size 3
                from scipy.signal import medfilt
                all_predictions = medfilt(all_predictions, kernel_size=3)
            
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
            
            # Save output to .lab file
            output_filename = os.path.splitext(os.path.basename(audio_path))[0] + '.lab'
            output_path = os.path.join(args.save_dir, output_filename)
            
            with open(output_path, 'w') as f:
                for line in lines:
                    f.write(line)
            
            logger.info(f"Saved chord annotations to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            continue
    
    logger.info("Chord recognition complete")

if __name__ == "__main__":
    main()
