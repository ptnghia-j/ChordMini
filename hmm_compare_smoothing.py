#!/usr/bin/env python3

"""
Script to compare different HMM smoothing levels on the same audio file.
Produces multiple lab files with different levels of detail.
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from modules.utils import logger
from modules.utils.mir_eval_modules import audio_file_to_features
from modules.models.HMM.ChordHMM import ChordHMM
from modules.utils.hparams import HParams

def load_hmm_model(model_path, device):
    """Load HMM model from checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract required information from checkpoint
        config = checkpoint.get('config', {})
        num_states = config.get('num_states', 170)
        pretrained_path = config.get('pretrained_model_path')
        
        chord_mapping = checkpoint.get('chord_mapping')
        idx_to_chord = checkpoint.get('idx_to_chord')
        mean = checkpoint.get('mean', 0.0)
        std = checkpoint.get('std', 1.0)
        
        # Load pretrained model
        from test_hmm import load_pretrained_model
        pretrained_config = HParams.load('./config/student_config.yaml')
        pretrained_model, _, _, _ = load_pretrained_model(pretrained_path, pretrained_config, device)
        
        # Create HMM model
        hmm_model = ChordHMM(
            pretrained_model=pretrained_model,
            num_states=num_states,
            device=device
        ).to(device)
        
        hmm_model.load_state_dict(checkpoint['model_state_dict'])
        
        return hmm_model, pretrained_model, chord_mapping, idx_to_chord, mean, std
    
    except Exception as e:
        logger.error(f"Error loading HMM model: {e}")
        return None, None, None, None, None, None

def process_audio(audio_file, hmm_model, raw_model, config, mean, std, idx_to_chord, device):
    """Process audio file with HMM model at different smoothing levels"""
    # Extract features
    feature, feature_per_second, song_length_second = audio_file_to_features(audio_file, config)
    feature = feature.T  # Transpose to (time, frequency)
    
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
    
    # Process with different smoothing levels
    smoothing_configs = [
        # name, smoothing_level, max_segment_length, segment_confidence
        ("very_smooth", 0.5, 20.0, 0.6),
        ("smooth", 1.0, 15.0, 0.6),
        ("balanced", 1.5, 10.0, 0.7),
        ("detailed", 2.0, 5.0, 0.7),
        ("very_detailed", 3.0, 3.0, 0.8)
    ]
    
    results = {}
    
    for name, smoothing_level, max_segment, confidence in smoothing_configs:
        # Configure HMM
        hmm_model.set_smoothing_level(smoothing_level)
        hmm_model.set_max_segment_length(max_segment)
        hmm_model.set_segment_confidence_threshold(confidence)
        
        # Process audio
        hmm_model.eval()
        with torch.no_grad():
            hmm_preds = hmm_model.decode(
                feature_tensor, 
                feature_rate=feature_per_second
            ).cpu().numpy()
        
        # Convert predictions to lab format
        lab_data = []
        current_chord = None
        segment_start = 0
        
        for i, pred_idx in enumerate(hmm_preds):
            # Convert chord index to label
            if pred_idx in idx_to_chord:
                chord = idx_to_chord[pred_idx]
            else:
                chord = "N"  # No chord
            
            # Track segments
            if chord != current_chord:
                if current_chord is not None:
                    end_time = i / feature_per_second
                    lab_data.append((segment_start, end_time, current_chord))
                
                current_chord = chord
                segment_start = i / feature_per_second
        
        # Add the final segment
        if current_chord is not None:
            end_time = len(hmm_preds) / feature_per_second
            lab_data.append((segment_start, end_time, current_chord))
        
        results[name] = {
            "predictions": hmm_preds,
            "lab_data": lab_data,
            "config": {
                "smoothing_level": smoothing_level,
                "max_segment": max_segment,
                "confidence": confidence
            }
        }
    
    # Also include raw predictions
    raw_lab_data = []
    current_chord = None
    segment_start = 0
    
    for i, pred_idx in enumerate(raw_preds):
        # Convert chord index to label
        if pred_idx in idx_to_chord:
            chord = idx_to_chord[pred_idx]
        else:
            chord = "N"  # No chord
        
        # Track segments
        if chord != current_chord:
            if current_chord is not None:
                end_time = i / feature_per_second
                raw_lab_data.append((segment_start, end_time, current_chord))
            
            current_chord = chord
            segment_start = i / feature_per_second
    
    # Add the final segment
    if current_chord is not None:
        end_time = len(raw_preds) / feature_per_second
        raw_lab_data.append((segment_start, end_time, current_chord))
    
    results["raw"] = {
        "predictions": raw_preds,
        "lab_data": raw_lab_data,
        "config": {}
    }
    
    return results, feature_per_second

def write_lab_files(results, output_dir, audio_filename):
    """Write lab files for each smoothing level"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_filename))[0]
    
    for name, result in results.items():
        # Format lab data
        lab_lines = []
        for start, end, chord in result["lab_data"]:
            lab_lines.append(f"{start:.6f} {end:.6f} {chord}")
        
        # Write lab file
        output_path = os.path.join(output_dir, f"{base_name}_{name}.lab")
        with open(output_path, "w") as f:
            f.write("\n".join(lab_lines))
        
        logger.info(f"Written {len(lab_lines)} chord segments to {output_path}")
        
        # Write config summary
        if name != "raw":
            output_summary = os.path.join(output_dir, f"{base_name}_{name}_config.txt")
            with open(output_summary, "w") as f:
                f.write(f"Smoothing level: {result['config']['smoothing_level']}\n")
                f.write(f"Max segment length: {result['config']['max_segment']} seconds\n")
                f.write(f"Confidence threshold: {result['config']['confidence']}\n")
                f.write(f"Number of segments: {len(result['lab_data'])}\n")

def visualize_results(results, feature_per_second, output_dir, audio_filename):
    """Create visualization comparing different smoothing levels"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_filename))[0]
        
        # Create a timeline plot showing segments
        plt.figure(figsize=(15, 10))
        
        # Calculate y positions and heights for each smoothing level
        names = list(results.keys())
        heights = 0.8 / len(names)
        positions = {}
        
        for i, name in enumerate(names):
            positions[name] = 1.0 - (i + 0.5) * heights
        
        # Define a color map for chord roots
        root_colors = {
            'C': '#FF0000', 'C#': '#FF4500', 'D': '#FFA500', 'D#': '#FFD700',
            'E': '#FFFF00', 'F': '#32CD32', 'F#': '#008000', 'G': '#00FFFF',
            'G#': '#1E90FF', 'A': '#0000FF', 'A#': '#800080', 'B': '#FF00FF',
            'N': '#808080'  # No chord is gray
        }
        
        # Plot segments for each smoothing level
        max_time = 0
        
        for name, result in results.items():
            y_pos = positions[name]
            
            # Plot segments
            for start, end, chord in result["lab_data"]:
                # Get root of chord
                if chord == 'N':
                    root = 'N'
                else:
                    root = chord.split(':')[0]
                
                color = root_colors.get(root, '#000000')
                max_time = max(max_time, end)
                
                # Plot segment
                plt.fill_between(
                    [start, end],
                    [y_pos - heights/2, y_pos - heights/2],
                    [y_pos + heights/2, y_pos + heights/2],
                    color=color, alpha=0.8
                )
        
        # Add labels for each smoothing level
        for name, y_pos in positions.items():
            plt.text(max_time * 1.01, y_pos, name, va='center', ha='left', fontsize=10)
        
        # Add a legend for chord roots
        legend_patches = []
        for root, color in root_colors.items():
            from matplotlib.patches import Patch
            legend_patches.append(Patch(color=color, label=root))
        
        plt.legend(handles=legend_patches, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.15), ncol=7, fontsize=10)
        
        # Set plot limits and labels
        plt.xlim(0, max_time * 1.1)
        plt.ylim(0, 1)
        plt.xlabel('Time (seconds)')
        plt.yticks([])
        plt.title(f'Chord Recognition Results with Different Smoothing Levels - {base_name}')
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Save figure
        output_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        
        # Create a segment count comparison chart
        segment_counts = [len(result["lab_data"]) for name, result in results.items()]
        
        plt.figure(figsize=(10, 5))
        plt.bar(names, segment_counts)
        plt.xlabel('Smoothing Level')
        plt.ylabel('Number of Chord Segments')
        plt.title(f'Number of Chord Segments by Smoothing Level - {base_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Save figure
        output_path = os.path.join(output_dir, f"{base_name}_segment_counts.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Segment count visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare HMM chord recognition with different smoothing levels")
    parser.add_argument('--hmm', type=str, required=True,
                        help='Path to trained HMM model')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./hmm_smoothing_comparison',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.hmm):
        logger.error(f"HMM model not found: {args.hmm}")
        return
        
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
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
    
    # Process audio with different smoothing levels
    results, feature_per_second = process_audio(args.audio, hmm_model, raw_model, config, mean, std, idx_to_chord, device)
    
    # Write lab files
    write_lab_files(results, args.output_dir, args.audio)
    
    # Create visualizations
    visualize_results(results, feature_per_second, args.output_dir, args.audio)
    
    logger.info("Comparison complete!")
    logger.info(f"Output files stored in {args.output_dir}")

if __name__ == "__main__":
    main()
