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
from tqdm import tqdm
from pathlib import Path
from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features, idx2voca_chord
from modules.models.Transformer.ChordNet import ChordNet
from modules.models.Transformer.btc_model import BTC_model # Import BTC_model
from modules.utils.hparams import HParams

def load_model(model_file, config, device, model_type='ChordNet'): # Add model_type argument
    """Load the model from a checkpoint file."""
    # Get model parameters based on model_type
    n_freq = config.feature.get('n_bins', config.model.get('feature_size', 144)) # Use feature_size for BTC
    n_classes = config.model.get('num_chords', 170) # Use num_chords for BTC

    if model_type == 'ChordNet':
        logger.info("Loading ChordNet model...")
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
    elif model_type == 'BTC':
        logger.info("Loading BTC model...")
        # Use BTC specific config parameters
        model = BTC_model(config=config.model).to(device) # Pass the model sub-config
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return None, 0.0, 1.0, {}

    # Load weights with robust error handling
    checkpoint = None

    try:
        # Handle PyTorch 2.6+ compatibility by adding numpy scalar to safe globals
        try:
            import numpy as np
            from torch.serialization import add_safe_globals
            # Add numpy scalar to safe globals list
            add_safe_globals([np.core.multiarray.scalar])
            logger.info("Added numpy scalar type to PyTorch safe globals list")
        except (ImportError, AttributeError) as e:
            logger.info(f"Could not add numpy scalar to safe globals: {e}")

        # Try loading with weights_only parameter
        try:
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)
            logger.info("Model loaded successfully with weights_only=False")
        except TypeError:
            # Fall back to older PyTorch versions that don't have weights_only parameter
            logger.info("Falling back to legacy loading method (for older PyTorch versions)")
            checkpoint = torch.load(model_file, map_location=device)
            logger.info("Model loaded successfully with legacy method")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        logger.error("Trying one more approach with explicitly disabled security...")
        import torch._C as _C
        try:
            checkpoint = _C._load_from_file(model_file, map_location=device)
            logger.info("Model loaded successfully with _C._load_from_file")
        except Exception as fallback_e:
            logger.error(f"All loading attempts failed: {fallback_e}")
            logger.error(traceback.format_exc())
            return None, 0.0, 1.0, {}

    if checkpoint is None:
        logger.error("Failed to load checkpoint - checkpoint is None")
        return None, 0.0, 1.0, {}

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

        model.load_state_dict(state_dict)

        # Get normalization parameters
        mean = checkpoint.get('mean', 0.0)
        std = checkpoint.get('std', 1.0)

        # Attach chord mapping
        idx_to_chord = idx2voca_chord()
        model.idx_to_chord = idx_to_chord

        logger.info(f"{model_type} model loaded successfully")
        return model, mean, std, idx_to_chord
    except Exception as e:
        logger.error(f"Error processing checkpoint: {e}")
        logger.error(traceback.format_exc())
        return None, 0.0, 1.0, {}

# NEW FUNCTION: Create and load HMM model
def create_hmm_model(checkpoint, base_model, device):
    """Create and initialize HMM model from checkpoint"""
    try:
        from modules.models.HMM.ChordHMM import ChordHMM

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

        logger.info(f"HMM model initialized with {num_states} states")
        return hmm_model
    except Exception as e:
        logger.error(f"Error initializing HMM model: {e}")
        logger.error(traceback.format_exc())
        return None

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

def process_audio_file(audio_path, label_path, model, config, mean, std, device, idx_to_chord, hmm_model=None, model_type='ChordNet'): # Add model_type
    """Process a single audio-label pair and create a sample for MIR evaluation."""
    try:
        # Extract features
        feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
        feature = feature.T # Transpose to [time, features]

        # Ensure mean and std are NumPy arrays for calculation with feature (NumPy array)
        if isinstance(mean, torch.Tensor):
            mean = mean.cpu().numpy()
        if isinstance(std, torch.Tensor):
            std = std.cpu().numpy()

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
                    # ChordNet expects [batch, group, time, features]
                    # Reshape [batch, time, features] -> [batch, time, group, feat_per_group] -> [batch, group, time, feat_per_group]
                    n_group = config.model.get('n_group', 4)
                    batch_s, time_s, feat_s = segment_batch.shape
                    feat_per_group = feat_s // n_group
                    if feat_s % n_group != 0:
                         logger.error(f"Feature size {feat_s} not divisible by n_group {n_group}")
                         # Handle error or skip batch
                         continue
                    segment_batch = segment_batch.view(batch_s, time_s, n_group, feat_per_group)
                    segment_batch = segment_batch.permute(0, 2, 1, 3) # [batch, group, time, feat_per_group]

                # Get prediction (model.predict should handle the specific model's logic)
                # Both models' predict methods should return [batch, time] or similar frame-level predictions
                # prediction = model.predict(segment_batch, per_frame=True if model_type == 'ChordNet' else False) # BTC predict doesn't need per_frame
                if model_type == 'ChordNet':
                    prediction = model.predict(segment_batch, per_frame=True)
                else: # For BTC model
                    prediction = model.predict(segment_batch) # Call without per_frame

                prediction = prediction.cpu()

                # Flatten batch predictions and append
                if prediction.dim() > 1: # Should be [batch, time]
                    for p in prediction:
                        all_predictions.append(p.numpy())
                else: # Handle potential single prediction case
                    all_predictions.append(prediction.numpy())

        # Concatenate raw base model predictions
        raw_predictions = np.concatenate(all_predictions) if all_predictions else np.array([])

        # Apply HMM smoothing only if HMM model is available AND model_type is ChordNet
        if hmm_model is not None and model_type == 'ChordNet':
            logger.debug(f"Applying HMM smoothing to predictions for {os.path.basename(audio_path)}")
            with torch.no_grad():
                hmm_model.eval()
                # Convert predictions to tensor for HMM processing
                feature_tensor = torch.tensor(feature, dtype=torch.float32).to(device)

                # Get HMM smoothed predictions using Viterbi decoding
                smoothed_preds = hmm_model.decode(feature_tensor).cpu().numpy()

                # Use HMM predictions instead of raw predictions
                all_predictions = smoothed_preds
                logger.debug(f"HMM smoothing applied: raw shape={raw_predictions.shape}, smoothed shape={all_predictions.shape}")
        else:
            # Keep the raw predictions if no HMM model
            all_predictions = raw_predictions

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
        pred_frames = [idx_to_chord[int(idx)] for idx in all_predictions[:num_frames]]

        # Create sample dict with required fields
        sample = {
            'song_id': os.path.splitext(os.path.basename(audio_path))[0],
            'spectro': feature,
            'model_pred': all_predictions,
            'gt_annotations': annotations,
            'chord_label': [str(chord) for chord in flatten_nested_list(gt_frames.tolist())],
            'pred_label': [str(chord) for chord in flatten_nested_list(pred_frames)],
            'feature_per_second': feature_per_second,
            'feature_length': num_frames,
            'model_type': model_type, # Store model type
            'used_hmm': hmm_model is not None and model_type == 'ChordNet' # HMM only used for ChordNet
        }

        return sample
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        logger.error(traceback.format_exc())
        return None

def custom_calculate_chord_scores(timestamps, durations, reference_labels, prediction_labels):
    """
    Calculate chord evaluation metrics using mir_eval.evaluate.

    Args:
        timestamps: Array of frame timestamps
        durations: Array of frame durations
        reference_labels: List of reference chord labels
        prediction_labels: List of predicted chord labels

    Returns:
        Tuple of evaluation scores (root, thirds, triads, sevenths, tetrads, majmin, mirex)
    """
    # Ensure inputs are lists
    reference_labels = list(reference_labels)
    prediction_labels = list(prediction_labels)

    # Ensure all inputs have the same length
    min_len = min(len(timestamps), len(durations), len(reference_labels), len(prediction_labels))
    if min_len == 0:
        logger.warning("Zero length input to custom_calculate_chord_scores. Returning all zeros.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    timestamps = timestamps[:min_len]
    durations = durations[:min_len]
    reference_labels = reference_labels[:min_len]
    prediction_labels = prediction_labels[:min_len]

    # Create intervals for mir_eval
    ref_intervals = np.zeros((min_len, 2))
    ref_intervals[:, 0] = timestamps
    ref_intervals[:, 1] = timestamps + durations

    est_intervals = np.zeros((min_len, 2))
    est_intervals[:, 0] = timestamps
    est_intervals[:, 1] = timestamps + durations

    # Standardize labels before evaluation (using the function from mir_eval_modules)
    # try:
    #     from modules.utils.mir_eval_modules import lab_file_error_modify
    #     standardized_refs = [lab_file_error_modify(ref) for ref in reference_labels]
    #     standardized_preds = [lab_file_error_modify(pred) for pred in prediction_labels]
    # except ImportError:
    #     logger.warning("Could not import lab_file_error_modify. Using raw labels.")
    standardized_refs = reference_labels
    standardized_preds = prediction_labels
    # except Exception as e:
    #      logger.error(f"Error standardizing labels: {e}. Using raw labels.")
    #      standardized_refs = reference_labels
    #      standardized_preds = prediction_labels


    # Use mir_eval.chord.evaluate for robust calculation
    scores = {}
    try:
        # mir_eval.chord.evaluate handles merging, weighting, and calculates all metrics
        scores = mir_eval.chord.evaluate(ref_intervals, standardized_refs, est_intervals, standardized_preds)

        # Extract scores safely, defaulting to 0.0 if a metric is missing
        root_score = float(scores.get('root', 0.0))
        thirds_score = float(scores.get('thirds', 0.0))
        triads_score = float(scores.get('triads', 0.0))
        sevenths_score = float(scores.get('sevenths', 0.0))
        tetrads_score = float(scores.get('tetrads', 0.0))
        majmin_score = float(scores.get('majmin', 0.0))
        mirex_score = float(scores.get('mirex', 0.0))

    except Exception as e:
        logger.error(f"Error during mir_eval.chord.evaluate: {e}")
        # Return default zero scores on error
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Ensure all scores are within the valid range [0, 1]
    root_score = max(0.0, min(1.0, root_score))
    thirds_score = max(0.0, min(1.0, thirds_score))
    triads_score = max(0.0, min(1.0, triads_score))
    sevenths_score = max(0.0, min(1.0, sevenths_score))
    tetrads_score = max(0.0, min(1.0, tetrads_score))
    majmin_score = max(0.0, min(1.0, majmin_score))
    mirex_score = max(0.0, min(1.0, mirex_score))

    return root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score

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

            durations = np.diff(np.append(timestamps, [timestamps[-1] + frame_duration]))

            root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score = \
                custom_calculate_chord_scores(timestamps, durations, reference_labels, prediction_labels)

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

    return score_list_dict, song_length_list, average_score_dict, quality_accuracy, quality_stats

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
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing label files')
    parser.add_argument('--config', type=str, default='./config/student_config.yaml', help='Path to ChordNet configuration file')
    parser.add_argument('--model', type=str, default=None, help='Path to ChordNet model file (if None, will try external path)')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Path to save results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--hmm', type=str, default=None, help='Path to HMM model for sequence smoothing (ChordNet only)')
    # Add arguments for BTC model
    parser.add_argument('--model_type', type=str, default='ChordNet', choices=['ChordNet', 'BTC'], help='Type of model to test')
    parser.add_argument('--btc_config', type=str, default='./config/btc_config.yaml', help='Path to BTC configuration file')
    parser.add_argument('--btc_model', type=str, default=None, help='Path to BTC model file (if None, will try external path)')
    args = parser.parse_args()

    logger.logging_verbosity(2 if args.verbose else 1)

    if not os.path.exists(args.audio_dir):
        logger.error(f"Audio directory not found: {args.audio_dir}")
        return
    if not os.path.exists(args.label_dir):
        logger.error(f"Label directory not found: {args.label_dir}")
        return

    # Select config and model path based on model_type
    if args.model_type == 'BTC':
        config_path = args.btc_config

        # Use provided model path or try external path
        if args.btc_model:
            model_path = args.btc_model
        else:
            # Always use external storage path
            external_model_path = 'checkpoints/btc/btc_model_best.pth'
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(external_model_path), exist_ok=True)
            model_path = external_model_path
            logger.info(f"Using external BTC checkpoint at {model_path}")

        logger.info(f"Using BTC model type with config: {config_path} and model: {model_path}")
    else: # Default to ChordNet
        config_path = args.config

        # Use provided model path or try external path
        if args.model:
            model_path = args.model
        else:
            # Always use external storage path
            external_model_path = '/mnt/storage/checkpoints/student/student_model_final.pth'
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(external_model_path), exist_ok=True)
            model_path = external_model_path
            logger.info(f"Using external ChordNet checkpoint at {model_path}")

        logger.info(f"Using ChordNet model type with config: {config_path} and model: {model_path}")

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    config = HParams.load(config_path)

    device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    hmm_model = None
    hmm_checkpoint = None

    # Load HMM only if model_type is ChordNet
    if args.model_type == 'ChordNet':
        hmm_path = args.hmm

        # If no HMM path provided, use the external storage path
        if not hmm_path:
            external_hmm_path = '/mnt/storage/checkpoints/hmm/hmm_model_best.pth'
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(external_hmm_path), exist_ok=True)
            hmm_path = external_hmm_path
            logger.info(f"Using external HMM checkpoint at {hmm_path}")

        # Load HMM if path exists
        if hmm_path and os.path.exists(hmm_path):
            try:
                logger.info(f"Loading HMM checkpoint from {hmm_path}")
                hmm_checkpoint = torch.load(hmm_path, map_location=device)
                logger.info("HMM checkpoint loaded successfully, will initialize after base model")
            except Exception as e:
                logger.error(f"Error loading HMM checkpoint: {e}")
                hmm_checkpoint = None
        elif hmm_path:
            logger.warning(f"HMM model file not found: {hmm_path}")
    elif args.model_type == 'BTC' and args.hmm:
        logger.warning(f"HMM smoothing is not supported for BTC model type. Ignoring --hmm argument.")

    # Pass model_type to load_model
    model, mean, std, idx_to_chord = load_model(model_path, config, device, model_type=args.model_type)
    if model is None:
        logger.error("Model loading failed. Cannot continue.")
        return

    # Initialize HMM only if checkpoint loaded and model is ChordNet
    if hmm_checkpoint is not None and args.model_type == 'ChordNet':
        hmm_model = create_hmm_model(hmm_checkpoint, model, device)
        if hmm_model is not None:
            logger.info("HMM model initialized successfully and will be used for smoothing")
        else:
            logger.warning("HMM model initialization failed, will continue without HMM smoothing")

    logger.info(f"Finding matching audio and label files...")
    matched_pairs = find_matching_audio_label_pairs(args.audio_dir, args.label_dir)
    logger.info(f"Found {len(matched_pairs)} matching audio-label pairs")

    if len(matched_pairs) == 0:
        logger.error("No matching audio-label pairs found. Cannot continue.")
        return

    dataset = []
    for audio_path, label_path in tqdm(matched_pairs, desc="Processing audio files"):
        sample = process_audio_file(
            audio_path, label_path, model, config, mean, std, device, idx_to_chord,
            hmm_model=hmm_model, model_type=args.model_type # Pass model_type
        )
        if sample is not None:
            dataset.append(sample)

    logger.info(f"Successfully processed {len(dataset)} of {len(matched_pairs)} audio files")

    if len(dataset) == 0:
        logger.error("No samples were processed successfully. Cannot continue.")
        return

    logger.info(f"\nRunning evaluation with{'out' if hmm_model is None else ''} HMM smoothing...")
    try:
        score_list_dict, song_length_list, average_score_dict, quality_accuracy, quality_stats = evaluate_dataset(
            dataset=dataset,
            config=config,
            model=model,
            device=device,
            mean=mean,
            std=std
        )

        logger.info("\nOverall MIR evaluation results:")
        for metric, score in average_score_dict.items():
            logger.info(f"{metric} score: {score:.4f}")

        if quality_accuracy:
            logger.info("\nIndividual Chord Quality Accuracy:")
            logger.info("---------------------------------")

            meaningful_qualities = [(q, acc) for q, acc in quality_accuracy.items()
                                  if quality_stats.get(q, {}).get('total', 0) >= 10 or acc > 0]

            for chord_quality, accuracy in sorted(meaningful_qualities, key=lambda x: x[1], reverse=True):
                total = quality_stats.get(chord_quality, {}).get('total', 0)
                correct = quality_stats.get(chord_quality, {}).get('correct', 0)
                if total >= 10:
                    logger.info(f"{chord_quality}: {accuracy*100:.2f}% ({correct}/{total})")
        else:
            logger.warning("\nNo chord quality accuracy data available!")

        results = {
            'model_type': args.model_type, # Add model type to results
            'used_hmm': hmm_model is not None and args.model_type == 'ChordNet',
            'hmm_path': args.hmm if hmm_model is not None and args.model_type == 'ChordNet' else None,
            'average_scores': average_score_dict,
            'quality_accuracy': quality_accuracy,
            'quality_stats': {k: {'total': v['total'], 'correct': v['correct']}
                             for k, v in quality_stats.items() if v['total'] >= 5},
            'song_details': [
                {
                    'song_id': sample['song_id'],
                    'duration': song_length_list[i],
                    'scores': {
                        metric: score_list_dict[metric][i]
                        for metric in score_list_dict
                    }
                }
                for i, sample in enumerate(dataset)
            ]
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(traceback.format_exc())

    logger.info("Evaluation complete")

if __name__ == "__main__":
    main()