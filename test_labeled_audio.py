#!/usr/bin/env python3

"""
Test script for evaluating chord recognition on labeled audio files.
This script uses a custom MIR evaluation method that properly handles
nested chord label structures for accurate evaluation.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import mir_eval
import re
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from pathlib import Path
from collections import Counter
from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features, idx2voca_chord, lab_file_error_modify, calculate_chord_scores # Import calculate_chord_scores
from modules.models.Transformer.ChordNet import ChordNet
from modules.models.Transformer.btc_model import BTC_model # Import BTC_model
from modules.utils.hparams import HParams
from modules.utils.visualize import plot_confusion_matrix, plot_chord_quality_distribution_accuracy

def load_model(model_file, config, device, model_type='ChordNet'): # Add model_type argument
    """Load the model from a checkpoint file."""
    # Get model parameters based on model_type
    n_freq = config.feature.get('n_bins', config.model.get('feature_size', 144)) # Use feature_size for BTC
    n_classes = config.model.get('num_chords', 170) # Use num_chords for BTC

    if model_type == 'ChordNet':
        # Use the same architecture as in train_student.py
        n_group = 2  # Always use n_group=2 as in train_student.py
        feature_dim = n_freq // n_group

        # Print feature dimensions for debugging
        print(f"Using feature dimensions: n_freq={n_freq}, n_group={n_group}, feature_dim={feature_dim}, heads={config.model.get('f_head', 6)}")

        logger.info("Loading ChordNet model...")
        model = ChordNet(
            n_freq=n_freq,
            n_classes=n_classes,
            n_group=n_group,  # Always use n_group=2
            f_layer=config.model.get('f_layer', 3),
            f_head=config.model.get('f_head', 2),  # Use f_head=2 as in train_student.py
            t_layer=config.model.get('t_layer', 4),  # Use t_layer=4 as in train_student.py
            t_head=config.model.get('t_head', 4),  # Use t_head=4 as in train_student.py
            d_layer=config.model.get('d_layer', 3),
            d_head=config.model.get('d_head', 4),  # Use d_head=4 as in train_student.py
            dropout=config.model.get('dropout', 0.3)
        ).to(device)
    elif model_type == 'BTC':
        logger.info("Loading BTC model...")
        # Use BTC specific config parameters
        model = BTC_model(config=config.model).to(device) # Pass the model sub-config
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None, 0.0, 1.0, {}

    # Load weights with robust error handling
    checkpoint = None

    # Check if file exists
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return None, 0.0, 1.0, {}

    # Check file size to ensure it's not empty
    file_size = os.path.getsize(model_file)
    if file_size < 1000:  # Arbitrary small size threshold
        logger.error(f"Model file is too small ({file_size} bytes), likely corrupted: {model_file}")
        return None, 0.0, 1.0, {}

    # Handle PyTorch 2.6+ compatibility by adding numpy scalar to safe globals
    try:
        import numpy as np
        try:
            from torch.serialization import add_safe_globals
            # Add numpy scalar to safe globals list
            add_safe_globals([np.core.multiarray.scalar])
            logger.info("Added numpy scalar type to PyTorch safe globals list")
        except (ImportError, AttributeError) as e:
            logger.info(f"Could not add numpy scalar to safe globals: {e}")
    except Exception as e:
        logger.info(f"Error setting up numpy compatibility: {e}")

    # Try multiple loading approaches
    loading_exceptions = []

    # Approach 1: Standard torch.load with weights_only=False
    try:
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        logger.info("Model loaded successfully with weights_only=False")
    except Exception as e:
        loading_exceptions.append(f"Standard loading failed: {str(e)}")

    # Approach 2: torch.load with weights_only=True
    if checkpoint is None:
        try:
            checkpoint = torch.load(model_file, map_location=device, weights_only=True)
            logger.info("Model loaded successfully with weights_only=True")
        except Exception as e:
            loading_exceptions.append(f"weights_only=True loading failed: {str(e)}")

    # Approach 3: Legacy loading method
    if checkpoint is None:
        try:
            checkpoint = torch.load(model_file, map_location=device)
            logger.info("Model loaded successfully with legacy method")
        except Exception as e:
            loading_exceptions.append(f"Legacy loading failed: {str(e)}")

    # Approach 4: Try with _C._load_from_file
    if checkpoint is None:
        try:
            try:
                import torch._C as _C
                checkpoint = _C._load_from_file(model_file, map_location=device)
                logger.info("Model loaded successfully with _C._load_from_file")
            except ImportError:
                logger.info("torch._C not available")
        except Exception as e:
            loading_exceptions.append(f"_C._load_from_file failed: {str(e)}")

    # Approach 5: Try pickle loading as last resort
    if checkpoint is None:
        try:
            import pickle
            with open(model_file, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info("Model loaded successfully with pickle")
        except Exception as e:
            loading_exceptions.append(f"Pickle loading failed: {str(e)}")

    # If all approaches failed, raise an error with details
    if checkpoint is None:
        error_details = "\n".join(loading_exceptions)
        logger.error(f"All loading approaches failed:\n{error_details}")
        return None, 0.0, 1.0, {}

    # Checkpoint loading check is already done above

    try:
        # Check if model state dict is directly available or nested
        state_dict = None
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Try to load the state dict directly
            state_dict = checkpoint

        if state_dict is None:
            logger.error("Could not find model state dictionary in checkpoint.")
            return None, 0.0, 1.0, {}

        # Handle potential DDP prefix 'module.'
        if list(state_dict.keys())[0].startswith('module.'):
            logger.info("Removing 'module.' prefix from state dict keys for compatibility.")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Try to load state dict with strict=True first, then fall back to strict=False if it fails
        try:
            model.load_state_dict(state_dict)
            logger.info("Model state dict loaded successfully with strict=True")
        except Exception as e:
            logger.warning(f"Failed to load model state dict with strict=True: {e}")
            logger.info("Attempting to load with strict=False...")
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info("Model state dict loaded successfully with strict=False")
            except Exception as e2:
                logger.error(f"Failed to load model state dict with strict=False: {e2}")
                raise

        # Simplified normalization parameter loading
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

        # First check for direct 'mean' and 'std' keys (common in BTC model checkpoints)
        if 'mean' in checkpoint and 'std' in checkpoint:
            mean = checkpoint['mean']
            std = checkpoint['std']
            logger.info(f"Found direct normalization parameters in checkpoint: mean={mean}, std={std}")
        # Then check for normalization in the standard location (how StudentTrainer saves it)
        elif 'normalization' in checkpoint and isinstance(checkpoint['normalization'], dict):
            norm_dict = checkpoint['normalization']
            mean = norm_dict.get('mean', 0.0)
            std = norm_dict.get('std', 1.0)
            logger.info(f"Found normalization parameters in checkpoint dictionary: mean={mean}, std={std}")
        else:
            # Use default values if normalization not found
            logger.warning("Normalization parameters not found in checkpoint, using defaults (0.0, 1.0)")
            mean = 0.0
            std = 1.0

        # Convert tensors to scalar values if needed
        if isinstance(mean, torch.Tensor):
            mean = float(mean.item()) if hasattr(mean, 'item') else float(mean)
            logger.info(f"Converted mean tensor to scalar: {mean}")
        if isinstance(std, torch.Tensor):
            std = float(std.item()) if hasattr(std, 'item') else float(std)
            logger.info(f"Converted std tensor to scalar: {std}")

        # Ensure std is not zero
        if std == 0:
            logger.warning("Checkpoint std is zero, using 1.0 instead.")
            std = 1.0

        logger.info(f"Final normalization parameters: mean={mean:.4f}, std={std:.4f}")

        # Attach chord mapping
        idx_to_chord = idx2voca_chord()
        model.idx_to_chord = idx_to_chord

        logger.info(f"{model_type} model loaded successfully")
        return model, mean, std, idx_to_chord
    except Exception as e:
        logger.error(f"Error processing checkpoint: {e}")
        logger.error(traceback.format_exc())
        return None, 0.0, 1.0, {}

# HMM functionality has been removed as it's no longer needed

def flatten_nested_list(nested_list):
    """
    Recursively flatten a potentially nested list into a 1D list.
    This ensures chord labels are properly flattened for MIR evaluation.
    """
    flattened = []

    # Handle case where input might not actually be a list
    if not isinstance(nested_list, (list, tuple, np.ndarray)):
        return [nested_list]

    for item in nested_list:
        if isinstance(item, (list, tuple, np.ndarray)):
            # If item is a nested list, recursively flatten it and extend
            flattened.extend(flatten_nested_list(item))
        else:
            # If item is not a list, append it directly
            flattened.append(item)

    return flattened

def process_audio_file(audio_path, label_path, model, config, mean, std, device, idx_to_chord, model_type='ChordNet'):
    """Process a single audio-label pair and create a sample for MIR evaluation."""
    try:
        # Extract features
        feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
        feature = feature.T # Transpose to [time, features]

        # Simplified normalization handling
        # Convert tensors to numpy arrays or scalars
        if isinstance(mean, torch.Tensor):
            mean = mean.cpu().numpy() if mean.numel() > 1 else float(mean.item())
        if isinstance(std, torch.Tensor):
            std = std.cpu().numpy() if std.numel() > 1 else float(std.item())

        # Ensure std is not zero (replace with 1.0 if it is)
        if isinstance(std, np.ndarray):
            if (std == 0).any():
                logger.warning("Std contains zero values, using 1.0 instead")
                std = 1.0
        elif std == 0:
            logger.warning("Std is zero, using 1.0 instead")
            std = 1.0

        logger.info(f"Normalizing features with mean={mean} and std={std}")
        feature = (feature - mean) / std

        # Get predictions
        # Use seq_len for BTC, timestep for ChordNet
        n_timestep = config.model.get('seq_len', config.model.get('timestep', 10))
        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        if num_pad < n_timestep:
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep

        # Generate predictions using base model
        all_predictions = []
        with torch.no_grad():
            model.eval()
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            batch_size = 32 # Adjust batch size if needed

            for t in range(0, num_instance, batch_size):
                end_idx = min(t + batch_size, num_instance)
                batch_count = end_idx - t

                segments = []
                for b in range(batch_count):
                    if t + b < num_instance:
                        seg = feature_tensor[n_timestep * (t+b):n_timestep * (t+b+1), :]
                        segments.append(seg)

                if not segments:
                    continue

                # Stack segments: [batch, time, features]
                segment_batch = torch.stack(segments, dim=0).to(device)

                # Adjust input shape for ChordNet if necessary
                if model_type == 'ChordNet':
                    # ChordNet expects [batch, 1, time, features] for the BaseTransformer
                    # The BaseTransformer will handle the reshaping internally
                    # Just add a channel dimension at position 1
                    segment_batch = segment_batch.unsqueeze(1)  # [batch, 1, time, features]

                # Get prediction (model.predict should handle the specific model's logic)
                if model_type == 'ChordNet':
                    prediction = model.predict(segment_batch, per_frame=True)
                else: # For BTC model
                    # BTC model expects input of shape [batch, time, features]
                    # segment_batch is already in this format for BTC
                    # Make sure it's the right shape
                    if segment_batch.dim() == 4 and segment_batch.size(1) == 1:
                        # If we have [batch, 1, time, features], squeeze out the channel dimension
                        segment_batch = segment_batch.squeeze(1)
                    prediction = model.predict(segment_batch) # Call without per_frame

                prediction = prediction.cpu()

                # Flatten batch predictions and append
                if prediction.dim() > 1: # Should be [batch, time]
                    for p in prediction:
                        all_predictions.append(p.numpy())
                else: # Handle potential single prediction case
                    all_predictions.append(prediction.numpy())

        # Concatenate raw base model predictions
        all_predictions = np.concatenate(all_predictions) if all_predictions else np.array([])

        # Parse ground truth annotations
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    chord = parts[2]
                    annotations.append((start_time, end_time, chord))

        # Convert to frame-level chord labels
        num_frames = len(all_predictions)
        gt_frames = np.full(num_frames, "N", dtype=object)
        for start, end, chord in annotations:
            start_frame = int(start / feature_per_second)
            end_frame = min(int(end / feature_per_second) + 1, num_frames)
            if start_frame < num_frames:
                gt_frames[start_frame:end_frame] = str(chord)

        # Convert predictions to chord names
        pred_frames_raw = [idx_to_chord[int(idx)] for idx in all_predictions[:num_frames]]

        # Standardize ground truth and predicted labels
        standardized_gt_labels = [lab_file_error_modify(str(label)) for label in flatten_nested_list(gt_frames.tolist())]
        standardized_pred_labels = [lab_file_error_modify(str(label)) for label in flatten_nested_list(pred_frames_raw)]

        # Create sample dict with required fields
        sample = {
            'song_id': os.path.splitext(os.path.basename(audio_path))[0],
            'spectro': feature,
            'model_pred': all_predictions,
            'gt_annotations': annotations,
            'chord_label': standardized_gt_labels, # Use standardized GT labels
            'pred_label': standardized_pred_labels,   # Use standardized predicted labels
            'feature_per_second': feature_per_second,
            'feature_length': num_frames,
            'model_type': model_type # Store model type
        }

        return sample
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        logger.error(traceback.format_exc())
        return None

# The custom_calculate_chord_scores function has been removed
# We now use the calculate_chord_scores function from modules/utils/mir_eval_modules.py
# This ensures consistent evaluation between test_labeled_audio.py and train_cv_kd.py
# The imported function lets mir_eval handle standardization internally to avoid double standardization

def extract_chord_quality(chord):
    """
    Extract chord quality from a chord label, handling different formats.
    Supports both colon format (C:maj) and direct format (Cmaj).

    Args:
        chord: A chord label string

    Returns:
        The chord quality as a string
    """
    # Handle None or empty strings
    if not chord:
        return "N"  # Default to "N" for empty chords

    # Handle special cases
    if chord in ["N", "None", "NC"]:
        return "N"  # No chord
    if chord in ["X", "Unknown"]:
        return "X"  # Unknown chord

    # Handle colon format (e.g., "C:min")
    if ':' in chord:
        parts = chord.split(':')
        if len(parts) > 1:
            # Handle bass notes (e.g., "C:min/G")
            quality = parts[1].split('/')[0] if '/' in parts[1] else parts[1]
            return quality

    # Handle direct format without colon (e.g., "Cmin")
    import re
    root_pattern = r'^[A-G][#b]?'
    match = re.match(root_pattern, chord)
    if match:
        quality = chord[match.end():]
        if quality:
            # Handle bass notes (e.g., "Cmin/G")
            return quality.split('/')[0] if '/' in quality else quality

    # Default to major if we couldn't extract a quality
    return "maj"

def map_chord_to_quality(chord_name):
    """
    Map a chord name to its quality group.

    Args:
        chord_name (str): The chord name (e.g., "C:maj", "A:min", "G:7", "N")

    Returns:
        str: The chord quality group name
    """
    # Handle non-string input
    if not isinstance(chord_name, str):
        return "Other"

    # Handle special cases
    if chord_name in ["N", "X", "None", "Unknown", "NC"]:
        return "No Chord"

    # Extract quality using extract_chord_quality function
    quality = extract_chord_quality(chord_name)

    # Map extracted quality to broader categories
    quality_mapping = {
        # Major family
        "maj": "Major", "": "Major", "M": "Major", "major": "Major",
        # Minor family
        "min": "Minor", "m": "Minor", "minor": "Minor",
        # Dominant seventh family
        "7": "Dom7", "dom7": "Dom7", "dominant": "Dom7",
        # Major seventh family
        "maj7": "Maj7", "M7": "Maj7", "major7": "Maj7",
        # Minor seventh family
        "min7": "Min7", "m7": "Min7", "minor7": "Min7",
        # Diminished family
        "dim": "Dim", "°": "Dim", "o": "Dim", "diminished": "Dim",
        # Diminished seventh family
        "dim7": "Dim7", "°7": "Dim7", "o7": "Dim7", "diminished7": "Dim7",
        # Half-diminished family
        "hdim7": "Half-Dim", "m7b5": "Half-Dim", "ø": "Half-Dim", "half-diminished": "Half-Dim",
        # Augmented family
        "aug": "Aug", "+": "Aug", "augmented": "Aug",
        # Suspended family
        "sus2": "Sus", "sus4": "Sus", "sus": "Sus", "suspended": "Sus",
        # Additional common chord qualities
        "min6": "Min6", "m6": "Min6",
        "maj6": "Maj6", "6": "Maj6",
        "minmaj7": "Min-Maj7", "mmaj7": "Min-Maj7", "min-maj7": "Min-Maj7",
        # Special cases
        "N": "No Chord",
        "X": "Unknown",
    }

    # Return mapped quality or "Other" if not found
    return quality_mapping.get(quality, "Other")

def compute_chord_quality_accuracy(reference_labels, prediction_labels):
    """
    Compute accuracy for individual chord qualities.
    Returns a dictionary mapping quality (e.g. maj, min, min7, etc.) to accuracy.
    Also returns mapped quality accuracy for consistency with validation.
    """
    from collections import defaultdict

    total_processed = 0
    malformed_chords = 0

    # Use two sets of statistics - one for raw qualities and one for mapped qualities
    raw_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    mapped_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for ref, pred in zip(reference_labels, prediction_labels):
        total_processed += 1

        if not ref or not pred:
            malformed_chords += 1
            continue

        try:
            # Extract chord qualities using the robust method
            q_ref_raw = extract_chord_quality(ref)
            q_pred_raw = extract_chord_quality(pred)

            # Map to broader categories for consistent reporting with validation
            q_ref_mapped = map_chord_to_quality(ref)
            q_pred_mapped = map_chord_to_quality(pred)

            # Update raw statistics
            raw_stats[q_ref_raw]['total'] += 1
            if q_ref_raw == q_pred_raw:
                raw_stats[q_ref_raw]['correct'] += 1

            # Update mapped statistics
            mapped_stats[q_ref_mapped]['total'] += 1
            if q_ref_mapped == q_pred_mapped:
                mapped_stats[q_ref_mapped]['correct'] += 1
        except Exception as e:
            malformed_chords += 1
            continue

    logger.debug(f"Processed {total_processed} chord pairs, {malformed_chords} were malformed or caused errors")
    logger.debug(f"Found {len(raw_stats)} unique raw chord qualities and {len(mapped_stats)} mapped qualities")

    # Calculate accuracy for each quality (both raw and mapped)
    raw_acc = {}
    for quality, vals in raw_stats.items():
        if vals['total'] > 0:
            raw_acc[quality] = vals['correct'] / vals['total']
        else:
            raw_acc[quality] = 0.0

    mapped_acc = {}
    for quality, vals in mapped_stats.items():
        if vals['total'] > 0:
            mapped_acc[quality] = vals['correct'] / vals['total']
        else:
            mapped_acc[quality] = 0.0

    # Print both raw and mapped statistics for comparison
    logger.info("\nRaw Chord Quality Distribution:")
    total_raw = sum(stats['total'] for stats in raw_stats.values())
    for quality, stats in sorted(raw_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            percentage = (stats['total'] / total_raw) * 100
            logger.info(f"  {quality}: {stats['total']} samples ({percentage:.2f}%)")

    logger.info("\nMapped Chord Quality Distribution (matches validation):")
    total_mapped = sum(stats['total'] for stats in mapped_stats.values())
    for quality, stats in sorted(mapped_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            percentage = (stats['total'] / total_mapped) * 100
            logger.info(f"  {quality}: {stats['total']} samples ({percentage:.2f}%)")

    logger.info("\nRaw Accuracy by chord quality:")
    for quality, accuracy_val in sorted(raw_acc.items(), key=lambda x: x[1], reverse=True):
        if raw_stats[quality]['total'] >= 10:  # Only show meaningful stats
            logger.info(f"  {quality}: {accuracy_val:.4f}")

    logger.info("\nMapped Accuracy by chord quality (matches validation):")
    for quality, accuracy_val in sorted(mapped_acc.items(), key=lambda x: x[1], reverse=True):
        if mapped_stats[quality]['total'] >= 10:  # Only show meaningful stats
            logger.info(f"  {quality}: {accuracy_val:.4f}")

    # Return both raw and mapped statistics
    return mapped_acc, mapped_stats

def generate_chord_distribution_accuracy_plot(quality_stats, quality_accuracy, output_path, title=None):
    """
    Generate a bar chart showing chord distribution with accuracy line overlay.

    Args:
        quality_stats: Dictionary of quality statistics
        quality_accuracy: Dictionary of quality accuracies
        output_path: Path to save the plot
        title: Optional title for the plot

    Returns:
        Path to the saved plot
    """
    # Extract data for plotting
    qualities = []
    counts = []
    accuracies = []
    total_samples = sum(stats['total'] for stats in quality_stats.values())

    # Sort qualities by count (descending)
    sorted_qualities = sorted(quality_stats.keys(), key=lambda q: quality_stats[q]['total'], reverse=True)

    # Filter out qualities with very few samples (less than 10)
    for quality in sorted_qualities:
        if quality_stats[quality]['total'] >= 10:
            qualities.append(quality)
            counts.append(quality_stats[quality]['total'])
            accuracies.append(quality_accuracy.get(quality, 0.0))

    if not qualities:
        logger.warning("No chord qualities with sufficient samples for plotting")
        return None

    # Calculate percentages
    percentages = [100 * count / total_samples for count in counts]

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot distribution bars
    bars = ax1.bar(qualities, percentages, alpha=0.7, color='steelblue', label='Distribution (%)')
    ax1.set_ylabel('Distribution (%)', color='steelblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, max(percentages) * 1.2 if percentages else 10)

    # Add percentage labels above bars
    for bar, pct, count in zip(bars, percentages, counts):
        height = bar.get_height()
        ax1.annotate(f'{pct:.1f}%\n({count})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', color='steelblue', fontsize=10)

    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(qualities, accuracies, 'ro-', linewidth=2, markersize=8, label='Accuracy')
    ax2.set_ylabel('Accuracy', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.0)

    # Add accuracy values as text
    for i, acc in enumerate(accuracies):
        ax2.annotate(f'{acc:.2f}',
                    xy=(i, acc),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color='red', fontsize=10)

    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Set title and adjust layout
    plt.title(title or "Chord Quality Distribution and Accuracy", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.tight_layout()

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved chord distribution and accuracy plot to {output_path}")
    return output_path

def generate_confusion_matrix_heatmap(ref_labels, pred_labels, output_path, title=None):
    """
    Generate a heatmap confusion matrix for chord qualities.

    Args:
        ref_labels: List of reference chord labels
        pred_labels: List of prediction chord labels
        output_path: Path to save the plot
        title: Optional title for the plot

    Returns:
        Path to the saved plot
    """
    # Extract chord qualities
    ref_qualities = [map_chord_to_quality(chord) for chord in ref_labels]
    pred_qualities = [map_chord_to_quality(chord) for chord in pred_labels]

    # Get unique qualities (sorted by frequency in reference labels)
    quality_counts = Counter(ref_qualities)
    unique_qualities = sorted(quality_counts.keys(), key=lambda q: quality_counts[q], reverse=True)

    # Filter out qualities with very few samples
    filtered_qualities = [q for q in unique_qualities if quality_counts[q] >= 10]

    if not filtered_qualities:
        logger.warning("No chord qualities with sufficient samples for confusion matrix")
        return None

    # Create mapping from quality to index
    quality_to_idx = {q: i for i, q in enumerate(filtered_qualities)}

    # Filter data to only include selected qualities
    filtered_indices = []
    for i, (ref, pred) in enumerate(zip(ref_qualities, pred_qualities)):
        if ref in filtered_qualities and pred in filtered_qualities:
            filtered_indices.append(i)

    filtered_ref = [ref_qualities[i] for i in filtered_indices]
    filtered_pred = [pred_qualities[i] for i in filtered_indices]

    # Convert qualities to indices
    ref_indices = [quality_to_idx[q] for q in filtered_ref]
    pred_indices = [quality_to_idx[q] for q in filtered_pred]

    # Calculate confusion matrix
    cm = confusion_matrix(ref_indices, pred_indices, labels=range(len(filtered_qualities)))

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=filtered_qualities, yticklabels=filtered_qualities,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

    # Improve layout
    plt.tight_layout()
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Chord Quality Confusion Matrix', fontsize=14)

    # Set tick font sizes and rotation
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved chord quality confusion matrix to {output_path}")
    return output_path

def evaluate_dataset(dataset, config, model, device, mean, std):
    """
    Evaluate chord recognition on a dataset of audio samples.

    Args:
        dataset: List of samples for evaluation
        config: Configuration object
        model: Model to evaluate
        device: Device to run evaluation on
        mean: Mean value for normalization
        std: Standard deviation for normalization

    Returns:
        score_list_dict: Dictionary of score lists for each metric
        song_length_list: List of song lengths for weighting
        average_score_dict: Dictionary of average scores for each metric
        quality_accuracy: Dictionary of per-quality accuracies
        quality_stats: Detailed stats for chord qualities
        visualization_paths: Dictionary of paths to generated visualizations
    """
    logger.info(f"Evaluating {len(dataset)} audio samples")

    score_list_dict = {
        'root': [],
        'thirds': [],
        'triads': [],
        'sevenths': [],
        'tetrads': [],
        'majmin': [],
        'mirex': []
    }
    song_length_list = []

    all_reference_labels = []
    all_prediction_labels = []

    for sample in tqdm(dataset, desc="Evaluating songs"):
        try:
            frame_duration = config.feature.get('hop_duration', 0.1)
            feature_length = sample.get('feature_length', len(sample.get('chord_label', [])))
            timestamps = np.arange(feature_length) * frame_duration

            reference_labels = sample.get('chord_label', [])
            prediction_labels = sample.get('pred_label', [])

            if len(reference_labels) == 0 or len(prediction_labels) == 0:
                logger.warning(f"Skipping sample {sample.get('song_id', 'unknown')}: missing labels")
                continue

            all_reference_labels.extend([str(label) for label in reference_labels])
            all_prediction_labels.extend([str(label) for label in prediction_labels])

            # We no longer need to calculate durations separately
            # The calculate_chord_scores function uses frame_duration directly

            # Use the imported calculate_chord_scores function from modules/utils/mir_eval_modules.py
            # This function lets mir_eval handle standardization internally
            root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score = \
                calculate_chord_scores(timestamps, frame_duration, reference_labels, prediction_labels)

            score_list_dict['root'].append(root_score)
            score_list_dict['thirds'].append(thirds_score)
            score_list_dict['triads'].append(triads_score)
            score_list_dict['sevenths'].append(sevenths_score)
            score_list_dict['tetrads'].append(tetrads_score)
            score_list_dict['majmin'].append(majmin_score)
            score_list_dict['mirex'].append(mirex_score)

            song_length = feature_length * frame_duration
            song_length_list.append(song_length)

            logger.info(f"Song {sample.get('song_id', 'unknown')}: length={song_length:.1f}s, root={root_score:.4f}, mirex={mirex_score:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating sample {sample.get('song_id', 'unknown')}: {str(e)}")
            logger.debug(traceback.format_exc())

    average_score_dict = {}

    if song_length_list:
        total_length = sum(song_length_list)
        for metric in score_list_dict:
            weighted_sum = sum(score * length for score, length in zip(score_list_dict[metric], song_length_list))
            average_score_dict[metric] = weighted_sum / total_length if total_length > 0 else 0.0
    else:
        for metric in score_list_dict:
            average_score_dict[metric] = 0.0

    quality_accuracy = {}
    quality_stats = {}
    visualization_paths = {}

    if len(all_reference_labels) > 0 and len(all_prediction_labels) > 0:
        logger.info("\n=== Chord Quality Analysis ===")
        logger.info(f"Collected {len(all_reference_labels)} reference and {len(all_prediction_labels)} prediction labels")

        min_len = min(len(all_reference_labels), len(all_prediction_labels))
        if min_len > 0:
            logger.info(f"Using {min_len} chord pairs for quality analysis")
            ref_labels = all_reference_labels[:min_len]
            pred_labels = all_prediction_labels[:min_len]

            has_colon_ref = any(':' in str(label) for label in ref_labels[:100] if label)
            has_colon_pred = any(':' in str(label) for label in pred_labels[:100] if label)

            logger.debug(f"Format check: Reference labels have colons: {has_colon_ref}")
            logger.debug(f"Format check: Prediction labels have colons: {has_colon_pred}")
            logger.debug(f"Sample reference labels: {[str(l) for l in ref_labels[:5]]}")
            logger.debug(f"Sample prediction labels: {[str(l) for l in pred_labels[:5]]}")

            quality_accuracy, quality_stats = compute_chord_quality_accuracy(ref_labels, pred_labels)

            # Generate visualizations
            try:
                # Get dataset identifier from the first sample if available
                dataset_id = "unknown_dataset"
                if dataset and len(dataset) > 0 and 'song_id' in dataset[0]:
                    # Extract dataset name from the first part of the song_id (usually contains dataset name)
                    song_id = dataset[0]['song_id']
                    if '/' in song_id:
                        dataset_id = song_id.split('/')[0]
                    else:
                        dataset_id = song_id.split('_')[0]

                # Create output directory for visualizations
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_visualizations", dataset_id)
                os.makedirs(output_dir, exist_ok=True)

                # Generate chord distribution and accuracy plot
                dist_plot_path = os.path.join(output_dir, f"{dataset_id}_chord_distribution_accuracy.png")
                dist_plot_path = generate_chord_distribution_accuracy_plot(
                    quality_stats,
                    quality_accuracy,
                    dist_plot_path,
                    title=f"Chord Quality Distribution and Accuracy - {dataset_id}"
                )
                visualization_paths['distribution_plot'] = dist_plot_path

                # Generate confusion matrix heatmap
                cm_plot_path = os.path.join(output_dir, f"{dataset_id}_chord_confusion_matrix.png")
                cm_plot_path = generate_confusion_matrix_heatmap(
                    ref_labels,
                    pred_labels,
                    cm_plot_path,
                    title=f"Chord Quality Confusion Matrix - {dataset_id}"
                )
                visualization_paths['confusion_matrix'] = cm_plot_path

                logger.info(f"Generated visualizations saved to {output_dir}")
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
                logger.debug(traceback.format_exc())

    return score_list_dict, song_length_list, average_score_dict, quality_accuracy, quality_stats, visualization_paths

def find_matching_audio_label_pairs(audio_dir, label_dir):
    """Find matching audio and label files in the given directories."""
    audio_extensions = ['.mp3', '.wav', '.flac']
    label_extensions = ['.lab', '.txt']

    audio_files = {}
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                base_name = os.path.splitext(file)[0]
                audio_files[base_name] = os.path.join(root, file)

    label_files = {}
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in label_extensions):
                base_name = os.path.splitext(file)[0]
                label_files[base_name] = os.path.join(root, file)

    matched_pairs = []
    for base_name, audio_path in audio_files.items():
        if base_name in label_files:
            matched_pairs.append((audio_path, label_files[base_name]))

    return matched_pairs

def main():
    parser = argparse.ArgumentParser(description="Test chord recognition on labeled audio files")
    parser.add_argument('--audio_dir', type=str, required=True, nargs='+', help='Directory (or directories) containing audio files')
    parser.add_argument('--label_dir', type=str, required=True, nargs='+', help='Directory (or directories) containing label files')
    parser.add_argument('--config', type=str, default='./config/student_config.yaml', help='Path to ChordNet configuration file')
    parser.add_argument('--model', type=str, default=None, help='Path to ChordNet model file (if None, will try external path)')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Path to save results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    # HMM argument has been removed
    # Add arguments for BTC model
    parser.add_argument('--model_type', type=str, default='ChordNet', choices=['ChordNet', 'BTC'], help='Type of model to test')
    parser.add_argument('--btc_config', type=str, default='./config/btc_config.yaml', help='Path to BTC configuration file')
    parser.add_argument('--btc_model', type=str, default=None, help='Path to BTC model file (if None, will try external path)')
    args = parser.parse_args()

    logger.logging_verbosity(2 if args.verbose else 1)

    if len(args.audio_dir) != len(args.label_dir):
        logger.error("The number of audio directories must match the number of label directories.")
        return

    # Select config and model path based on model_type
    if args.model_type == 'BTC':
        config_path = args.btc_config

        # Use provided model path from either --btc_model or --model argument
        if args.btc_model:
            model_path = args.btc_model
            logger.info(f"Using BTC model from --btc_model argument: {model_path}")
        elif args.model:
            model_path = args.model
            logger.info(f"Using BTC model from --model argument: {model_path}")
        else:
            # Fall back to external storage path only if no model path is provided
            external_model_path = 'checkpoints/btc/btc_model_best.pth'
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(external_model_path), exist_ok=True)
            model_path = external_model_path
            logger.info(f"No model path provided. Using default BTC checkpoint at {model_path}")

        logger.info(f"Using BTC model type with config: {config_path} and model: {model_path}")
    else: # Default to ChordNet
        config_path = args.config

        # Use provided model path or try external path
        if args.model:
            model_path = args.model
            logger.info(f"Using ChordNet model from --model argument: {model_path}")
        else:
            # Fall back to local path only if no model path is provided
            external_model_path = 'checkpoints/student/student_model_final.pth'
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(external_model_path), exist_ok=True)
            model_path = external_model_path
            logger.info(f"No model path provided. Using default ChordNet checkpoint at {model_path}")

        logger.info(f"Using ChordNet model type with config: {config_path} and model: {model_path}")

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    # Model path check will be done before loading the model,
    # as it might be an external path that doesn't exist locally initially.

    config = HParams.load(config_path)

    device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # HMM functionality has been removed

    # Pass model_type to load_model
    model, mean, std, idx_to_chord = load_model(model_path, config, device, model_type=args.model_type)
    if model is None:
        logger.error("Model loading failed. Cannot continue.")
        return

    # HMM functionality has been removed
    logger.info("HMM functionality has been removed. Using raw model predictions.")

    all_datasets_results = {}

    for i in range(len(args.audio_dir)):
        current_audio_dir = args.audio_dir[i]
        current_label_dir = args.label_dir[i]
        dataset_identifier = os.path.basename(current_audio_dir.rstrip('/\\'))

        logger.info(f"\n\n===== Processing Dataset: {dataset_identifier} =====")
        logger.info(f"Audio directory: {current_audio_dir}")
        logger.info(f"Label directory: {current_label_dir}")

        if not os.path.exists(current_audio_dir):
            logger.error(f"Audio directory not found: {current_audio_dir}. Skipping dataset.")
            all_datasets_results[dataset_identifier] = {"error": f"Audio directory not found: {current_audio_dir}"}
            continue
        if not os.path.exists(current_label_dir):
            logger.error(f"Label directory not found: {current_label_dir}. Skipping dataset.")
            all_datasets_results[dataset_identifier] = {"error": f"Label directory not found: {current_label_dir}"}
            continue

        # Check model_path existence here, before processing each dataset with it
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}. Cannot process dataset {dataset_identifier}.")
            all_datasets_results[dataset_identifier] = {"error": f"Model file not found: {model_path}"}
            continue


        logger.info(f"Finding matching audio and label files for {dataset_identifier}...")
        matched_pairs = find_matching_audio_label_pairs(current_audio_dir, current_label_dir)
        logger.info(f"Found {len(matched_pairs)} matching audio-label pairs for {dataset_identifier}")

        if len(matched_pairs) == 0:
            logger.warning(f"No matching audio-label pairs found for {dataset_identifier}. Skipping evaluation for this dataset.")
            all_datasets_results[dataset_identifier] = {"status": "No matching files found", "average_scores": {}, "quality_accuracy": {}}
            continue

        dataset = []
        for audio_path, label_path in tqdm(matched_pairs, desc=f"Processing audio files for {dataset_identifier}"):
            sample = process_audio_file(
                audio_path, label_path, model, config, mean, std, device, idx_to_chord,
                model_type=args.model_type # Pass model_type
            )
            if sample is not None:
                dataset.append(sample)

        logger.info(f"Successfully processed {len(dataset)} of {len(matched_pairs)} audio files for {dataset_identifier}")

        if len(dataset) == 0:
            logger.warning(f"No samples were processed successfully for {dataset_identifier}. Skipping evaluation for this dataset.")
            all_datasets_results[dataset_identifier] = {"status": "No samples processed successfully", "average_scores": {}, "quality_accuracy": {}}
            continue

        logger.info(f"\nRunning evaluation for {dataset_identifier}...")
        try:
            score_list_dict, song_length_list, average_score_dict, quality_accuracy, quality_stats, visualization_paths = evaluate_dataset(
                dataset=dataset,
                config=config,
                model=model,
                device=device,
                mean=mean,
                std=std
            )

            logger.info(f"\nOverall MIR evaluation results for {dataset_identifier}:")
            for metric, score in average_score_dict.items():
                logger.info(f"{metric} score: {score:.4f}")

            if quality_accuracy:
                logger.info(f"\nIndividual Chord Quality Accuracy for {dataset_identifier}:")
                logger.info("---------------------------------")

                meaningful_qualities = [(q, acc) for q, acc in quality_accuracy.items()
                                      if quality_stats.get(q, {}).get('total', 0) >= 10 or acc > 0]

                for chord_quality, accuracy in sorted(meaningful_qualities, key=lambda x: x[1], reverse=True):
                    total = quality_stats.get(chord_quality, {}).get('total', 0)
                    correct = quality_stats.get(chord_quality, {}).get('correct', 0)
                    if total >= 10: # Only log if substantial samples
                        logger.info(f"{chord_quality}: {accuracy*100:.2f}% ({correct}/{total})")

                # Log visualization paths if available
                if visualization_paths:
                    logger.info("\nGenerated visualizations:")
                    for viz_type, path in visualization_paths.items():
                        if path:
                            logger.info(f"  {viz_type}: {path}")
            else:
                logger.warning(f"\nNo chord quality accuracy data available for {dataset_identifier}!")

            current_dataset_results = {
                'model_type': args.model_type,
                'audio_dir': current_audio_dir,
                'label_dir': current_label_dir,
                'average_scores': average_score_dict,
                'quality_accuracy': quality_accuracy,
                'quality_stats': {k: {'total': v['total'], 'correct': v['correct']}
                                 for k, v in quality_stats.items() if v['total'] >= 5},
                'visualization_paths': visualization_paths,
                'song_details': [
                    {
                        'song_id': sample['song_id'],
                        'duration': song_length_list[idx],
                        'scores': {
                            metric: score_list_dict[metric][idx]
                            for metric in score_list_dict
                        }
                    }
                    for idx, sample in enumerate(dataset) # Use idx for song_length_list and score_list_dict
                ]
            }
            all_datasets_results[dataset_identifier] = current_dataset_results

        except Exception as e:
            logger.error(f"Error during evaluation for dataset {dataset_identifier}: {e}")
            logger.error(traceback.format_exc())
            all_datasets_results[dataset_identifier] = {"error": f"Evaluation failed: {str(e)}"}

    with open(args.output, 'w') as f:
        json.dump(all_datasets_results, f, indent=2)

    logger.info(f"\n\nAll dataset evaluations complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()