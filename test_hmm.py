#!/usr/bin/env python3

"""
Test script for evaluating a trained ChordHMM model.
Compares chord recognition with and without HMM smoothing.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import glob
import mir_eval
from sklearn.metrics import confusion_matrix, accuracy_score

from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features
from modules.models.Transformer.ChordNet import ChordNet
from modules.models.HMM.ChordHMM import ChordHMM
from modules.utils.hparams import HParams
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.chords import idx2voca_chord, Chords

def load_hmm_model(model_path, device):
    """Load a trained HMM model"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get configuration
        config = checkpoint.get('config', {})
        num_states = config.get('num_states', 170)
        pretrained_path = config.get('pretrained_model_path')
        
        # Get chord mapping and normalization parameters
        chord_mapping = checkpoint.get('chord_mapping')
        idx_to_chord = checkpoint.get('idx_to_chord')
        mean = checkpoint.get('mean', 0.0)
        std = checkpoint.get('std', 1.0)
        
        if pretrained_path is None:
            logger.error("Pretrained model path not found in HMM checkpoint")
            return None, None, None, None, None
            
        # Load pretrained model config
        pretrained_config = HParams.load('./config/student_config.yaml')  # Default config
        
        # Load pretrained model
        pretrained_model, _, _, _ = load_pretrained_model(pretrained_path, pretrained_config, device)
        
        # Create HMM model
        hmm_model = ChordHMM(
            pretrained_model=pretrained_model,
            num_states=num_states,
            device=device
        ).to(device)
        
        # Load HMM state dict
        hmm_model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("HMM model loaded successfully")
        return hmm_model, pretrained_model, chord_mapping, idx_to_chord, mean, std
        
    except Exception as e:
        logger.error(f"Error loading HMM model: {e}")
        return None, None, None, None, None

def load_pretrained_model(model_path, config, device):
    """Load pretrained ChordNet model"""
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
    try:
        # First try safe loading with weights_only=True
        try:
            # For PyTorch 2.6+, whitelist numpy scalar type
            import numpy as np
            from numpy.core.multiarray import scalar
            import torch.serialization
            torch.serialization.add_safe_globals([scalar])
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e1:
            logger.info(f"Safe loading failed: {e1}")
            logger.info("Trying to load with weights_only=False (only do this for trusted checkpoints)")
            # Fallback to weights_only=False for older checkpoints
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
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
        
        logger.info("Pretrained model loaded successfully")
        return model, mean, std, idx_to_chord
    
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        raise

def find_matching_files(audio_dir, label_dir):
    """
    Find matching audio and label files based on filenames
    
    Returns:
        List of tuples (audio_path, label_path)
    """
    audio_label_pairs = []
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    label_extensions = ['.lab', '.txt']
    
    # Find audio files
    audio_files = {}
    for ext in audio_extensions:
        for path in glob.glob(os.path.join(audio_dir, f"*{ext}")):
            # Use filename without extension as key
            basename = os.path.splitext(os.path.basename(path))[0]
            audio_files[basename] = path
            
        # Also check subdirectories
        for path in glob.glob(os.path.join(audio_dir, f"**/*{ext}"), recursive=True):
            basename = os.path.splitext(os.path.basename(path))[0]
            audio_files[basename] = path
    
    # Find label files
    label_files = {}
    for ext in label_extensions:
        for path in glob.glob(os.path.join(label_dir, f"*{ext}")):
            basename = os.path.splitext(os.path.basename(path))[0]
            label_files[basename] = path
            
        # Also check subdirectories
        for path in glob.glob(os.path.join(label_dir, f"**/*{ext}"), recursive=True):
            basename = os.path.splitext(os.path.basename(path))[0]
            label_files[basename] = path
    
    # Match audio files with label files
    pairs_found = 0
    for basename, audio_path in audio_files.items():
        if basename in label_files:
            audio_label_pairs.append((audio_path, label_files[basename]))
            pairs_found += 1
    
    logger.info(f"Found {pairs_found} matching audio-label pairs")
    return audio_label_pairs

def evaluate_audio_file(audio_path, label_path, hmm_model, raw_model, config, mean, std, chord_mapping, idx_to_chord, device):
    """
    Evaluate a single audio file with both raw model and HMM model
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Extract features from audio file
        feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
        feature = feature.T  # Transpose to get (time, frequency) format
        
        # Apply normalization
        if mean is not None and std is not None:
            feature = (feature - mean) / (max(std, 1e-10))
        
        # Convert to tensor
        feature_tensor = torch.tensor(feature, dtype=torch.float32).to(device)
        
        # Get raw model predictions
        raw_model.eval()
        with torch.no_grad():
            batch_size = 32
            raw_preds = []
            for i in range(0, feature_tensor.shape[0], batch_size):
                batch = feature_tensor[i:i+batch_size]
                output = raw_model(batch)
                raw_preds.append(output)
            
            # Concatenate predictions
            raw_preds = torch.cat(raw_preds, 0).cpu().numpy()
        
        # Get HMM model predictions
        hmm_model.eval()
        with torch.no_grad():
            # HMM uses Viterbi decoding which expects the full sequence
            hmm_preds = hmm_model.decode(feature_tensor).cpu().numpy()
        
        # Read ground truth chord labels
        chord_helper = Chords()
        chord_helper.set_chord_mapping(chord_mapping)
        
        # Convert lab file to chord indices
        try:
            df = chord_helper.get_converted_chord_voca(label_path)
            ground_truth = df['chord_id'].values
            timestamps = list(zip(df['start'].values, df['end'].values))
            
            # Convert raw model frame-level predictions to timestamps for evaluation
            raw_timestamps = []
            raw_chord_labels = []
            for i, pred_idx in enumerate(raw_preds):
                # Convert frame index to time
                if i > 0:  # Skip first frame for start time
                    start_time = (i - 1) / feature_per_second
                else:
                    start_time = 0
                end_time = i / feature_per_second
                
                # Convert chord index to label
                if pred_idx in idx_to_chord:
                    chord = idx_to_chord[pred_idx]
                else:
                    chord = "N"  # No chord
                
                raw_timestamps.append((start_time, end_time))
                raw_chord_labels.append(chord)
            
            # Convert HMM frame-level predictions to timestamps for evaluation
            hmm_timestamps = []
            hmm_chord_labels = []
            for i, pred_idx in enumerate(hmm_preds):
                # Convert frame index to time
                if i > 0:  # Skip first frame for start time
                    start_time = (i - 1) / feature_per_second
                else:
                    start_time = 0
                end_time = i / feature_per_second
                
                # Convert chord index to label
                if pred_idx in idx_to_chord:
                    chord = idx_to_chord[pred_idx]
                else:
                    chord = "N"  # No chord
                
                hmm_timestamps.append((start_time, end_time))
                hmm_chord_labels.append(chord)
            
            # Evaluate using mir_eval
            # First, format ground truth data for mir_eval
            gt_intervals = np.array(timestamps)
            gt_labels = df['chord_id'].apply(lambda x: idx_to_chord.get(x, "N") if x in idx_to_chord else "N").values
            
            # Evaluate raw model
            raw_intervals = np.array(raw_timestamps)
            raw_scores = mir_eval.chord.evaluate(gt_intervals, gt_labels, raw_intervals, raw_chord_labels)
            
            # Evaluate HMM model
            hmm_intervals = np.array(hmm_timestamps)
            hmm_scores = mir_eval.chord.evaluate(gt_intervals, gt_labels, hmm_intervals, hmm_chord_labels)
            
            # Calculate frame-level accuracy
            raw_frame_acc = accuracy_score(ground_truth, raw_preds)
            hmm_frame_acc = accuracy_score(ground_truth, hmm_preds)
            
            # Return results
            return {
                "audio_file": os.path.basename(audio_path),
                "label_file": os.path.basename(label_path),
                "raw_model": {
                    "frame_accuracy": float(raw_frame_acc),
                    "mir_eval_scores": {k: float(v) for k, v in raw_scores.items()}
                },
                "hmm_model": {
                    "frame_accuracy": float(hmm_frame_acc),
                    "mir_eval_scores": {k: float(v) for k, v in hmm_scores.items()}
                }
            }
        
        except Exception as e:
            logger.error(f"Error evaluating file {audio_path}: {str(e)}")
            return {
                "audio_file": os.path.basename(audio_path),
                "label_file": os.path.basename(label_path),
                "error": str(e)
            }
            
    except Exception as e:
        logger.error(f"Error processing file {audio_path}: {str(e)}")
        return {
            "audio_file": os.path.basename(audio_path),
            "error": str(e)
        }

def visualize_results(results, output_path):
    """
    Visualize evaluation results comparing raw model vs HMM model
    
    Args:
        results: List of evaluation result dictionaries
        output_path: Path to save visualization
    """
    # Extract metrics for successful evaluations
    raw_root = []
    hmm_root = []
    raw_thirds = []
    hmm_thirds = []
    raw_triads = []
    hmm_triads = []
    
    # Labels for x-axis
    labels = []
    
    for result in results:
        if "error" not in result:
            raw_scores = result["raw_model"]["mir_eval_scores"]
            hmm_scores = result["hmm_model"]["mir_eval_scores"]
            
            raw_root.append(raw_scores.get("root", 0.0))
            hmm_root.append(hmm_scores.get("root", 0.0))
            
            raw_thirds.append(raw_scores.get("thirds", 0.0))
            hmm_thirds.append(hmm_scores.get("thirds", 0.0))
            
            raw_triads.append(raw_scores.get("triads", 0.0))
            hmm_triads.append(hmm_scores.get("triads", 0.0))
            
            labels.append(result["audio_file"])
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot root recognition accuracy
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, raw_root, width, label='Raw Model')
    axes[0].bar(x + width/2, hmm_root, width, label='HMM Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Root Recognition Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot thirds recognition accuracy
    axes[1].bar(x - width/2, raw_thirds, width, label='Raw Model')
    axes[1].bar(x + width/2, hmm_thirds, width, label='HMM Model')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Thirds Recognition Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot triad recognition accuracy
    axes[2].bar(x - width/2, raw_triads, width, label='Raw Model')
    axes[2].bar(x + width/2, hmm_triads, width, label='HMM Model')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Triad Recognition Accuracy')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate and display average improvements
    avg_root_improvement = np.mean(np.array(hmm_root) - np.array(raw_root))
    avg_thirds_improvement = np.mean(np.array(hmm_thirds) - np.array(raw_thirds))
    avg_triads_improvement = np.mean(np.array(hmm_triads) - np.array(raw_triads))
    
    plt.figtext(0.5, 0.01, 
                f"Average Improvements with HMM:\n"
                f"Root: {avg_root_improvement:.4f}, "
                f"Thirds: {avg_thirds_improvement:.4f}, "
                f"Triads: {avg_triads_improvement:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the text
    
    plt.savefig(output_path)
    logger.info(f"Visualization saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test HMM chord recognition model")
    parser.add_argument('--hmm', type=str, required=True,
                        help='Path to trained HMM model')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory containing label files')
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='hmm_evaluation_results.json',
                        help='Path to save results')
    parser.add_argument('--visualization', type=str, default='hmm_evaluation_results.png',
                        help='Path to save results visualization')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of files to process (for testing)')
    parser.add_argument('--temperature', type=float, default=0.5, 
                        help='Temperature for HMM transitions (lower = less smoothing, default: 0.5)')
    parser.add_argument('--emission_weight', type=float, default=0.8,
                        help='Weight for emission probabilities (higher = trust emissions more, default: 0.8)')
    parser.add_argument('--smoothing_level', type=float, default=1.5,
                        help='Dynamic smoothing level (>1 = more transitions, <1 = fewer, default: 1.5)')
    parser.add_argument('--max_segment_length', type=float, default=10.0,
                        help='Max segment length in seconds (0 to disable splitting, default: 10.0)')
    parser.add_argument('--segment_confidence', type=float, default=0.7,
                        help='Confidence threshold for segment splitting (default: 0.7)')
    parser.add_argument('--compare_raw', action='store_true',
                        help='Also generate raw model predictions for comparison')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.hmm):
        logger.error(f"HMM model not found: {args.hmm}")
        return
    
    if not os.path.exists(args.audio_dir):
        logger.error(f"Audio directory not found: {args.audio_dir}")
        return
        
    if not os.path.exists(args.label_dir):
        logger.error(f"Label directory not found: {args.label_dir}")
        return
    
    # Load configuration
    config = HParams.load(args.config)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load HMM model
    hmm_model, raw_model, chord_mapping, idx_to_chord, mean, std = load_hmm_model(args.hmm, device)
    
    if hmm_model is None:
        logger.error("Failed to load HMM model")
        return
    
    # Set HMM parameters
    if hasattr(hmm_model, 'set_temperature'):
        hmm_model.set_temperature(args.temperature)
        logger.info(f"Set HMM temperature to {args.temperature}")
    
    if hasattr(hmm_model, 'set_emission_weight'):
        hmm_model.set_emission_weight(args.emission_weight)
        logger.info(f"Set emission weight to {args.emission_weight}")
    
    if hasattr(hmm_model, 'set_smoothing_level'):
        hmm_model.set_smoothing_level(args.smoothing_level)
        logger.info(f"Set smoothing level to {args.smoothing_level}")
    
    if hasattr(hmm_model, 'set_max_segment_length'):
        hmm_model.set_max_segment_length(args.max_segment_length)
        logger.info(f"Set max segment length to {args.max_segment_length} seconds")
    
    if hasattr(hmm_model, 'set_segment_confidence_threshold'):
        hmm_model.set_segment_confidence_threshold(args.segment_confidence)
        logger.info(f"Set segment confidence threshold to {args.segment_confidence}")
    
    # Find matching audio and label files
    audio_label_pairs = find_matching_files(args.audio_dir, args.label_dir)
    
    if not audio_label_pairs:
        logger.error("No matching audio-label pairs found")
        return
    
    # Limit the number of files if specified
    if args.limit is not None and args.limit > 0:
        audio_label_pairs = audio_label_pairs[:args.limit]
        logger.info(f"Processing limited to {args.limit} files")
    
    # Process audio files and evaluate with both raw model and HMM
    results = []
    for audio_path, label_path in tqdm(audio_label_pairs, desc="Evaluating"):
        result = evaluate_audio_file(
            audio_path, label_path, hmm_model, raw_model, 
            config, mean, std, chord_mapping, idx_to_chord, device
        )
        results.append(result)
        
        # Log summary for this file
        if "error" not in result:
            raw_acc = result["raw_model"]["frame_accuracy"]
            hmm_acc = result["hmm_model"]["frame_accuracy"]
            diff = hmm_acc - raw_acc
            logger.info(f"{os.path.basename(audio_path)}: Raw acc: {raw_acc:.4f}, HMM acc: {hmm_acc:.4f}, Diff: {diff:+.4f}")
    
    # Calculate overall metrics
    successful_evals = [r for r in results if "error" not in r]
    if successful_evals:
        # Frame accuracy
        raw_frame_acc = np.mean([r["raw_model"]["frame_accuracy"] for r in successful_evals])
        hmm_frame_acc = np.mean([r["hmm_model"]["frame_accuracy"] for r in successful_evals])
        
        # MIR_EVAL metrics (averaged)
        raw_mir_metrics = {}
        hmm_mir_metrics = {}
        for metric in successful_evals[0]["raw_model"]["mir_eval_scores"]:
            raw_mir_metrics[metric] = np.mean([r["raw_model"]["mir_eval_scores"].get(metric, 0.0) for r in successful_evals])
            hmm_mir_metrics[metric] = np.mean([r["hmm_model"]["mir_eval_scores"].get(metric, 0.0) for r in successful_evals])
        
        # Overall results
        overall_results = {
            "num_files_processed": len(audio_label_pairs),
            "num_successful_evaluations": len(successful_evals),
            "raw_model": {
                "average_frame_accuracy": float(raw_frame_acc),
                "average_mir_eval_scores": raw_mir_metrics
            },
            "hmm_model": {
                "average_frame_accuracy": float(hmm_frame_acc),
                "average_mir_eval_scores": hmm_mir_metrics
            },
            "per_file_results": results,
            "hmm_config": {
                "temperature": args.temperature,
                "emission_weight": args.emission_weight,
                "smoothing_level": args.smoothing_level,
                "max_segment_length": args.max_segment_length,
                "segment_confidence": args.segment_confidence
            }
        }
        
        # Save results to JSON
        with open(args.output, 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        logger.info("===== EVALUATION SUMMARY =====")
        logger.info(f"Number of files processed: {len(audio_label_pairs)}")
        logger.info(f"Number of successful evaluations: {len(successful_evals)}")
        logger.info(f"Raw model - Average frame accuracy: {raw_frame_acc:.4f}")
        logger.info(f"HMM model - Average frame accuracy: {hmm_frame_acc:.4f}")
        logger.info(f"Accuracy improvement: {hmm_frame_acc - raw_frame_acc:+.4f}")
        
        for metric in raw_mir_metrics:
            logger.info(f"Raw model - Average {metric}: {raw_mir_metrics[metric]:.4f}")
            logger.info(f"HMM model - Average {metric}: {hmm_mir_metrics[metric]:.4f}")
            logger.info(f"Improvement in {metric}: {hmm_mir_metrics[metric] - raw_mir_metrics[metric]:+.4f}")
        
        # Create visualization
        try:
            visualize_results(results, args.visualization)
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    else:
        logger.error("No successful evaluations")

if __name__ == "__main__":
    main()
