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
import traceback
from tqdm import tqdm
from pathlib import Path
from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features, idx2voca_chord
from modules.models.Transformer.ChordNet import ChordNet
from modules.utils.hparams import HParams

def load_model(model_file, config, device):
    """Load the model from a checkpoint file."""
    # Get model parameters
    n_freq = config.feature.get('n_bins', 144)
    n_classes = 170  # Large vocabulary size
    
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
        
        # Attach chord mapping
        idx_to_chord = idx2voca_chord()
        model.idx_to_chord = idx_to_chord
        
        logger.info("Model loaded successfully")
        return model, mean, std, idx_to_chord
    except Exception as e:
        logger.error(f"Error processing checkpoint: {e}")
        logger.error(traceback.format_exc())
        return None, 0.0, 1.0, {}

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

def process_audio_file(audio_path, label_path, model, config, mean, std, device, idx_to_chord):
    """Process a single audio-label pair and create a sample for MIR evaluation."""
    try:
        # Extract features
        feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
        feature = feature.T
        feature = (feature - mean) / std
        
        # Get predictions
        n_timestep = config.model.get('timestep', 10)
        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        if num_pad < n_timestep:
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep
        
        # Generate predictions
        all_predictions = []
        with torch.no_grad():
            model.eval()
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            batch_size = 32
            
            for t in range(0, num_instance, batch_size):
                end_idx = min(t + batch_size, num_instance)
                batch_count = end_idx - t
                
                if batch_count == 1:
                    segment = feature_tensor[n_timestep * t:n_timestep * (t + 1), :].unsqueeze(0).to(device)
                else:
                    segments = []
                    for b in range(batch_count):
                        if t + b < num_instance:
                            seg = feature_tensor[n_timestep * (t+b):n_timestep * (t+b+1), :]
                            segments.append(seg)
                    
                    if segments:
                        segment = torch.stack(segments, dim=0).to(device)
                    else:
                        continue
                
                prediction = model.predict(segment, per_frame=True)
                prediction = prediction.cpu()
                
                if prediction.dim() > 1:
                    for p in prediction:
                        all_predictions.append(p.numpy())
                else:
                    all_predictions.append(prediction.numpy())
        
        all_predictions = np.concatenate(all_predictions)
        
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
        # CRITICAL FIX: Ensure all elements are strings and flatten any nested lists
        sample = {
            'song_id': os.path.splitext(os.path.basename(audio_path))[0],
            'spectro': feature,
            'model_pred': all_predictions,
            'gt_annotations': annotations,
            # Explicitly flatten and convert to string any potential nested lists
            'chord_label': [str(chord) for chord in flatten_nested_list(gt_frames.tolist())],
            'pred_label': [str(chord) for chord in flatten_nested_list(pred_frames)],
            'feature_per_second': feature_per_second,
            'feature_length': num_frames,
            'model_type': 'ChordNet'
        }
        
        return sample
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        logger.error(traceback.format_exc())
        return None

def custom_calculate_chord_scores(timestamps, durations, reference_labels, prediction_labels):
    """
    Calculate chord evaluation metrics properly handling nested lists.
    
    Args:
        timestamps: Array of frame timestamps
        durations: Array of frame durations 
        reference_labels: List of reference chord labels
        prediction_labels: List of predicted chord labels
        
    Returns:
        Tuple of evaluation scores (root, thirds, triads, sevenths, tetrads, majmin, mirex)
    """
    import mir_eval
    import numpy as np
    
    # Create intervals for mir_eval
    intervals = np.zeros((len(timestamps), 2))
    intervals[:, 0] = timestamps
    intervals[:, 1] = timestamps + durations
    
    # Ensure all inputs have the same length
    min_len = min(len(intervals), len(reference_labels), len(prediction_labels))
    intervals = intervals[:min_len]
    
    # CRITICAL FIX: Better handling of reference_labels - extract full chord sequence
    flat_ref_labels = []
    
    # Check if reference_labels is a doubly nested list - often happens when it's [['N', 'N', ...]]
    if (len(reference_labels) > 0 and isinstance(reference_labels[0], list) and 
            len(reference_labels[0]) > 0 and isinstance(reference_labels[0][0], str)):
        # This is the case where reference_labels is [['N', 'N', 'E:min', ...]]
        # Extract the full inner list to preserve all chord labels
        logger.debug(f"Detected doubly-nested reference labels with {len(reference_labels[0])} items")
        flat_ref_labels = [str(chord) for chord in reference_labels[0][:min_len]]
    else:
        # Handle other cases - single level nesting or no nesting
        for ref in reference_labels[:min_len]:
            if isinstance(ref, (list, tuple, np.ndarray)):
                if len(ref) > 0:
                    flat_ref_labels.append(str(ref[0]))  # Take first element if it's a list
                else:
                    flat_ref_labels.append("N")  # Default for empty list
            else:
                flat_ref_labels.append(str(ref))  # Keep as is if not a list
    
    # Handle prediction labels with same careful approach
    flat_pred_labels = []
    for pred in prediction_labels[:min_len]:
        if isinstance(pred, (list, tuple, np.ndarray)):
            if len(pred) > 0:
                flat_pred_labels.append(str(pred[0]))
            else:
                flat_pred_labels.append("N")
        else:
            flat_pred_labels.append(str(pred))
    
    # Make sure both lists are the same length
    final_length = min(len(flat_ref_labels), len(flat_pred_labels))
    flat_ref_labels = flat_ref_labels[:final_length]
    flat_pred_labels = flat_pred_labels[:final_length]
    
    # Update durations to match the final length
    if len(durations) > final_length:
        durations = durations[:final_length]
    
    # Debug output (reduced verbosity for batch processing)
    logger.debug(f"Reference chords: {len(flat_ref_labels)}, prediction chords: {len(flat_pred_labels)}")
    
    # Initialize scores
    root_score = thirds_score = triads_score = sevenths_score = tetrads_score = majmin_score = mirex_score = 0.0
    
    try:
        # Root metric
        root_comparisons = mir_eval.chord.root(flat_ref_labels, flat_pred_labels)
        root_score = mir_eval.chord.weighted_accuracy(root_comparisons, durations)
        
        # Thirds metric
        thirds_comparisons = mir_eval.chord.thirds(flat_ref_labels, flat_pred_labels)
        thirds_score = mir_eval.chord.weighted_accuracy(thirds_comparisons, durations)
        
        # Triads metric
        triads_comparisons = mir_eval.chord.triads(flat_ref_labels, flat_pred_labels)
        triads_score = mir_eval.chord.weighted_accuracy(triads_comparisons, durations)
        
        # Sevenths metric
        sevenths_comparisons = mir_eval.chord.sevenths(flat_ref_labels, flat_pred_labels)
        sevenths_score = mir_eval.chord.weighted_accuracy(sevenths_comparisons, durations)
        
        # Tetrads metric
        tetrads_comparisons = mir_eval.chord.tetrads(flat_ref_labels, flat_pred_labels)
        tetrads_score = mir_eval.chord.weighted_accuracy(tetrads_comparisons, durations)
        
        # Majmin metric
        majmin_comparisons = mir_eval.chord.majmin(flat_ref_labels, flat_pred_labels)
        majmin_score = mir_eval.chord.weighted_accuracy(majmin_comparisons, durations)
        
        # Mirex metric
        mirex_comparisons = mir_eval.chord.mirex(flat_ref_labels, flat_pred_labels)
        mirex_score = mir_eval.chord.weighted_accuracy(mirex_comparisons, durations)
    except Exception as e:
        logger.error(f"Error in mir_eval scoring: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Ensure all scores are non-negative
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
    """
    # First try the colon format
    if ':' in chord:
        return chord.split(':')[1]
    
    # Handle special cases
    if chord in ["N", "X"]:
        return chord
    
    # Try to extract quality from direct format (e.g., Cmaj, Emin)
    import re
    # Root note pattern (A-G with optional # or b)
    root_pattern = r'^[A-G][#b]?'
    
    # Match and remove the root if present
    match = re.match(root_pattern, chord)
    if match:
        # Extract everything after the root
        quality = chord[match.end():]
        if quality:
            return quality
            
    # Default to maj if we can't determine
    return "maj"

def compute_chord_quality_accuracy(reference_labels, prediction_labels):
    """
    Compute accuracy for individual chord qualities.
    Returns a dictionary mapping quality (e.g. maj, min, min7, etc.) to accuracy.
    
    Now handles different chord formats and provides detailed counts.
    """
    from collections import defaultdict
    
    # Counter for debugging
    total_processed = 0
    malformed_chords = 0
    
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for ref, pred in zip(reference_labels, prediction_labels):
        total_processed += 1
        
        # Skip problematic chords
        if not ref or not pred:
            malformed_chords += 1
            continue
            
        try:
            q_ref = extract_chord_quality(ref)
            q_pred = extract_chord_quality(pred)
            
            stats[q_ref]['total'] += 1
            if q_ref == q_pred:
                stats[q_ref]['correct'] += 1
        except Exception as e:
            malformed_chords += 1
            continue
    
    # Use logger instead of direct print statements
    logger.debug(f"Processed {total_processed} chord pairs, {malformed_chords} were malformed or caused errors")
    logger.debug(f"Found {len(stats)} unique chord qualities in the reference labels")
    
    # Top qualities by frequency
    top_qualities = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    # Filter to only show qualities that appear at least 10 times or have >0% accuracy
    meaningful_qualities = [q for q, v in top_qualities if v['total'] >= 10 or (v['total'] > 0 and v['correct'] > 0)]
    logger.debug(f"Found {len(meaningful_qualities)} meaningful chord qualities (appear ≥10 times or have >0% accuracy)")
    
    # Show top qualities by frequency in debug mode only
    # Replace the logger.get_level() check with a simple try-except to avoid errors
    try:
        # Check if we should show detailed debug output - try using logging level
        is_debug = logger.is_debug() if hasattr(logger, 'is_debug') else False
        if is_debug:
            count = 0
            for quality, counts in top_qualities:
                if counts['total'] >= 10 or (counts['total'] > 0 and counts['correct'] > 0):
                    logger.debug(f"  {quality}: {counts['total']} instances ({counts['correct']} correct)")
                    count += 1
                    if count >= 10:
                        break
    except Exception:
        # If checking the debug level fails, just continue without showing detailed stats
        pass
    
    acc = {}
    for quality, vals in stats.items():
        if vals['total'] > 0:
            acc[quality] = vals['correct'] / vals['total']
        else:
            acc[quality] = 0.0
    return acc, stats

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
    
    # Evaluation metrics
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
    
    # Collect all reference and prediction labels for quality analysis
    all_reference_labels = []
    all_prediction_labels = []
    
    # Process each sample with a progress bar
    for sample in tqdm(dataset, desc="Evaluating songs"):
        try:
            # Extract timestamps
            frame_duration = config.feature.get('hop_duration', 0.1)
            feature_length = sample.get('feature_length', len(sample.get('chord_label', [])))
            timestamps = np.arange(feature_length) * frame_duration
            
            # Extract chord labels and predictions
            reference_labels = sample.get('chord_label', [])
            prediction_labels = sample.get('pred_label', [])
            
            # Ensure we have valid data
            if len(reference_labels) == 0 or len(prediction_labels) == 0:
                logger.warning(f"Skipping sample {sample.get('song_id', 'unknown')}: missing labels")
                continue
            
            # Collect labels for quality analysis - make sure they're strings
            all_reference_labels.extend([str(label) for label in reference_labels])
            all_prediction_labels.extend([str(label) for label in prediction_labels])
            
            # Calculate durations
            durations = np.diff(np.append(timestamps, [timestamps[-1] + frame_duration]))
            
            # Calculate scores with our custom function
            root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score = \
                custom_calculate_chord_scores(timestamps, durations, reference_labels, prediction_labels)
            
            # Store scores
            score_list_dict['root'].append(root_score)
            score_list_dict['thirds'].append(thirds_score)
            score_list_dict['triads'].append(triads_score)
            score_list_dict['sevenths'].append(sevenths_score)
            score_list_dict['tetrads'].append(tetrads_score)
            score_list_dict['majmin'].append(majmin_score)
            score_list_dict['mirex'].append(mirex_score)
            
            # Store song length
            song_length = feature_length * frame_duration
            song_length_list.append(song_length)
            
            # Print individual sample scores
            logger.info(f"Song {sample.get('song_id', 'unknown')}: length={song_length:.1f}s, root={root_score:.4f}, mirex={mirex_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating sample {sample.get('song_id', 'unknown')}: {str(e)}")
            logger.debug(traceback.format_exc())
    
    # Calculate weighted average scores
    average_score_dict = {}
    
    if song_length_list:
        total_length = sum(song_length_list)
        for metric in score_list_dict:
            weighted_sum = sum(score * length for score, length in zip(score_list_dict[metric], song_length_list))
            average_score_dict[metric] = weighted_sum / total_length if total_length > 0 else 0.0
    else:
        for metric in score_list_dict:
            average_score_dict[metric] = 0.0
    
    # Calculate chord quality accuracy if we have enough data
    quality_accuracy = {}
    quality_stats = {}
    
    if len(all_reference_labels) > 0 and len(all_prediction_labels) > 0:
        logger.info("\n=== Chord Quality Analysis ===")
        logger.info(f"Collected {len(all_reference_labels)} reference and {len(all_prediction_labels)} prediction labels")
        
        # Make sure lists are equal length
        min_len = min(len(all_reference_labels), len(all_prediction_labels))
        if min_len > 0:
            logger.info(f"Using {min_len} chord pairs for quality analysis")
            ref_labels = all_reference_labels[:min_len]
            pred_labels = all_prediction_labels[:min_len]
            
            # Check if we have proper chord format with colons
            has_colon_ref = any(':' in str(label) for label in ref_labels[:100] if label)
            has_colon_pred = any(':' in str(label) for label in pred_labels[:100] if label)
            
            logger.debug(f"Format check: Reference labels have colons: {has_colon_ref}")
            logger.debug(f"Format check: Prediction labels have colons: {has_colon_pred}")
            logger.debug(f"Sample reference labels: {[str(l) for l in ref_labels[:5]]}")
            logger.debug(f"Sample prediction labels: {[str(l) for l in pred_labels[:5]]}")
            
            # Always attempt quality extraction with our improved extractor
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
    
    # Match audio files to label files
    matched_pairs = []
    for base_name, audio_path in audio_files.items():
        if base_name in label_files:
            matched_pairs.append((audio_path, label_files[base_name]))
    
    return matched_pairs

def main():
    parser = argparse.ArgumentParser(description="Test chord recognition on labeled audio files")
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory containing label files')
    parser.add_argument('--config', type=str, default='./config/student_config.yaml', help='Path to configuration file')
    parser.add_argument('--model', type=str, default='./checkpoints/student_model_final.pth', help='Path to model file')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Path to save results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Set up logging
    logger.logging_verbosity(2 if args.verbose else 1)
    
    # Check if directories exist
    if not os.path.exists(args.audio_dir):
        logger.error(f"Audio directory not found: {args.audio_dir}")
        return
    if not os.path.exists(args.label_dir):
        logger.error(f"Label directory not found: {args.label_dir}")
        return
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Load configuration
    config = HParams.load(args.config)
    
    # Set up device 
    device = torch.device('cpu')  # CPU is more reliable for evaluation
    logger.info(f"Using device: {device}")
    
    # Load model
    model, mean, std, idx_to_chord = load_model(args.model, config, device)
    if model is None:
        logger.error("Model loading failed. Cannot continue.")
        return
    
    # Find matching audio and label files
    logger.info(f"Finding matching audio and label files...")
    matched_pairs = find_matching_audio_label_pairs(args.audio_dir, args.label_dir)
    logger.info(f"Found {len(matched_pairs)} matching audio-label pairs")
    
    if len(matched_pairs) == 0:
        logger.error("No matching audio-label pairs found. Cannot continue.")
        return
    
    # Process all matched pairs
    dataset = []
    for audio_path, label_path in tqdm(matched_pairs, desc="Processing audio files"):
        sample = process_audio_file(
            audio_path, label_path, model, config, mean, std, device, idx_to_chord
        )
        if sample is not None:
            dataset.append(sample)
    
    logger.info(f"Successfully processed {len(dataset)} of {len(matched_pairs)} audio files")
    
    if len(dataset) == 0:
        logger.error("No samples were processed successfully. Cannot continue.")
        return
    
    # Run evaluation
    logger.info("\nRunning custom MIR evaluation...")
    try:
        score_list_dict, song_length_list, average_score_dict, quality_accuracy, quality_stats = evaluate_dataset(
            dataset=dataset,
            config=config, 
            model=model, 
            device=device,
            mean=mean, 
            std=std
        )
        
        # Print overall results
        logger.info("\nOverall MIR evaluation results:")
        for metric, score in average_score_dict.items():
            logger.info(f"{metric} score: {score:.4f}")
        
        # Print chord quality accuracy with higher visibility - ONLY SHOW MEANINGFUL QUALITIES
        if quality_accuracy:
            logger.info("\nIndividual Chord Quality Accuracy:")
            logger.info("---------------------------------")
            
            # Only show qualities with at least 10 examples or >0% accuracy
            meaningful_qualities = [(q, acc) for q, acc in quality_accuracy.items() 
                                  if quality_stats.get(q, {}).get('total', 0) >= 10 or acc > 0]
            
            # Sort by accuracy (high to low)
            for chord_quality, accuracy in sorted(meaningful_qualities, key=lambda x: x[1], reverse=True):
                total = quality_stats.get(chord_quality, {}).get('total', 0)
                correct = quality_stats.get(chord_quality, {}).get('correct', 0)
                # Only show if it appears enough times
                if total >= 10:
                    logger.info(f"{chord_quality}: {accuracy*100:.2f}% ({correct}/{total})")
        else:
            logger.warning("\nNo chord quality accuracy data available!")
        
        # Save results to JSON - include stats for detailed analysis later
        results = {
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