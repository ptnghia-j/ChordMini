#!/usr/bin/env python3

"""
Script to tune HMM parameters, especially temperature, to find
the optimal balance between smoothing and preserving chord variety.
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
import subprocess
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

from modules.utils import logger
from modules.models.HMM.ChordHMM import ChordHMM
from modules.utils.chords import idx2voca_chord, Chords

def parse_lab_file(file_path):
    """Parse a lab file into a list of (start, end, chord) tuples"""
    chords = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                chord = parts[2]
                chords.append((start, end, chord))
    return chords

def count_unique_chords(lab_file):
    """Count unique chords in a lab file"""
    chords = parse_lab_file(lab_file)
    unique = set(chord for _, _, chord in chords)
    return len(unique)

def calculate_diversity(lab_file):
    """Calculate chord diversity metrics for a lab file"""
    chords = parse_lab_file(lab_file)
    
    # Count unique chords
    unique_chords = set(chord for _, _, chord in chords)
    num_unique = len(unique_chords)
    
    # Count total chord segments
    total_segments = len(chords)
    
    # Calculate transitions (changes between different chords)
    transitions = 0
    for i in range(1, len(chords)):
        if chords[i][2] != chords[i-1][2]:
            transitions += 1
    
    # Calculate average segment duration
    durations = [end - start for start, end, _ in chords]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    # Count chord quality distribution
    qualities = []
    for _, _, chord in chords:
        if chord != 'N' and ':' in chord:
            quality = chord.split(':')[1]
            qualities.append(quality)
    quality_counter = Counter(qualities)
    
    return {
        "unique_chords": num_unique,
        "total_segments": total_segments,
        "transitions": transitions,
        "transition_rate": transitions / (len(chords) - 1) if len(chords) > 1 else 0,
        "avg_duration": avg_duration,
        "quality_distribution": dict(quality_counter)
    }

def run_hmm_with_temperature(hmm_model_path, audio_file, temp, output_dir):
    """Run HMM chord recognition with a specific temperature"""
    output_lab = os.path.join(output_dir, f"temp_{temp:.2f}.lab")
    
    # Run test_hmm.py with the specified temperature
    cmd = [
        "python", "test_hmm.py",
        "--hmm", hmm_model_path,
        "--audio_dir", os.path.dirname(audio_file),
        "--label_dir", "dummy",  # Not needed for this purpose
        "--temperature", str(temp),
        "--output", os.path.join(output_dir, f"temp_{temp:.2f}_results.json")
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Check if lab file was generated
        if os.path.exists(output_lab):
            return output_lab
        else:
            logger.error(f"Output lab file not found: {output_lab}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running HMM with temperature {temp}: {e}")
        logger.error(f"STDOUT: {e.stdout.decode()}")
        logger.error(f"STDERR: {e.stderr.decode()}")
        return None

def tune_temperature(hmm_model_path, audio_file, reference_lab, output_dir, 
                     temps=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]):
    """
    Tune the temperature parameter and analyze the results
    
    Args:
        hmm_model_path: Path to trained HMM model
        audio_file: Audio file to process
        reference_lab: Reference chord lab file (ground truth or student output)
        output_dir: Directory to save results
        temps: List of temperature values to try
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse reference lab file
    ref_diversity = calculate_diversity(reference_lab)
    ref_unique = count_unique_chords(reference_lab)
    
    logger.info(f"Reference lab file has {ref_unique} unique chords")
    logger.info(f"Reference diversity metrics: {ref_diversity}")
    
    # Run HMM with different temperature values
    results = []
    for temp in temps:
        logger.info(f"Testing temperature: {temp}")
        
        # Run HMM with this temperature
        output_lab = run_hmm_with_temperature(hmm_model_path, audio_file, temp, output_dir)
        
        if output_lab:
            # Calculate metrics
            diversity = calculate_diversity(output_lab)
            unique = count_unique_chords(output_lab)
            
            result = {
                "temperature": temp,
                "unique_chords": unique,
                "diversity": diversity,
                "lab_file": output_lab,
                "ref_similarity": {
                    "unique_chord_ratio": unique / max(1, ref_unique),
                    "segment_ratio": diversity['total_segments'] / max(1, ref_diversity['total_segments']),
                    "transition_ratio": diversity['transitions'] / max(1, ref_diversity['transitions'])
                }
            }
            
            results.append(result)
            
            logger.info(f"Temperature {temp}: {unique} unique chords, "
                        f"{diversity['transitions']} transitions, "
                        f"{diversity['total_segments']} segments")
                        
            # Print quality distribution
            if "quality_distribution" in diversity:
                qual_str = ", ".join([f"{q}:{c}" for q, c in diversity["quality_distribution"].items()])
                logger.info(f"Quality distribution: {qual_str}")
    
    # Save full results
    with open(os.path.join(output_dir, "temperature_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_temperature_results(results, ref_diversity, output_dir)
    
    return results

def visualize_temperature_results(results, ref_diversity, output_dir):
    """Create visualizations of temperature tuning results"""
    if not results:
        logger.error("No results to visualize")
        return
    
    # Sort results by temperature
    results.sort(key=lambda x: x["temperature"])
    
    # Extract data for plotting
    temps = [r["temperature"] for r in results]
    unique_chords = [r["unique_chords"] for r in results]
    transitions = [r["diversity"]["transitions"] for r in results]
    segments = [r["diversity"]["total_segments"] for r in results]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # Plot unique chords
    axes[0].plot(temps, unique_chords, 'o-', color='blue')
    axes[0].axhline(y=ref_diversity["unique_chords"], color='r', linestyle='--', 
                   label=f'Reference ({ref_diversity["unique_chords"]} chords)')
    axes[0].set_ylabel('Unique Chords')
    axes[0].set_title('Effect of Temperature on Chord Diversity')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot transitions
    axes[1].plot(temps, transitions, 'o-', color='green')
    axes[1].axhline(y=ref_diversity["transitions"], color='r', linestyle='--', 
                   label=f'Reference ({ref_diversity["transitions"]} transitions)')
    axes[1].set_ylabel('Chord Transitions')
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot segments
    axes[2].plot(temps, segments, 'o-', color='purple')
    axes[2].axhline(y=ref_diversity["total_segments"], color='r', linestyle='--', 
                   label=f'Reference ({ref_diversity["total_segments"]} segments)')
    axes[2].set_ylabel('Total Segments')
    axes[2].grid(True)
    axes[2].legend()
    
    # Plot chord quality diversity
    # Extract all qualities that appear in any temperature setting
    all_qualities = set()
    for result in results:
        if "quality_distribution" in result.get("diversity", {}):
            all_qualities.update(result["diversity"]["quality_distribution"].keys())
    
    # Reference qualities for comparison
    ref_qualities = ref_diversity.get("quality_distribution", {})
    
    # Plot stacked bar chart of quality distributions at different temperatures
    if all_qualities:
        quality_data = []
        for temp_idx, result in enumerate(results):
            if "quality_distribution" in result.get("diversity", {}):
                dist = result["diversity"]["quality_distribution"]
                quality_data.append({quality: dist.get(quality, 0) for quality in all_qualities})
            else:
                quality_data.append({quality: 0 for quality in all_qualities})
        
        # Convert to arrays for plotting
        qualities = sorted(all_qualities)
        quality_counts = np.zeros((len(temps), len(qualities)))
        for i, data in enumerate(quality_data):
            for j, qual in enumerate(qualities):
                quality_counts[i, j] = data.get(qual, 0)
        
        # Plot stacked bars
        bottom = np.zeros(len(temps))
        for i, qual in enumerate(qualities):
            axes[3].bar(temps, quality_counts[:, i], bottom=bottom, label=qual)
            bottom += quality_counts[:, i]
        
        axes[3].set_xlabel('Temperature')
        axes[3].set_ylabel('Chord Quality Count')
        axes[3].set_title('Chord Quality Distribution')
        axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[3].text(0.5, 0.5, "No chord quality data available", 
                    ha='center', va='center', transform=axes[3].transAxes)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temperature_analysis.png"))
    plt.close()
    
    logger.info(f"Visualization saved to {os.path.join(output_dir, 'temperature_analysis.png')}")
    
    # Create a separate figure for quality distribution comparison with reference
    if ref_qualities and all_qualities:
        plt.figure(figsize=(12, 8))
        
        # Choose the "best" temperature result based on chord diversity similarity to reference
        best_idx = 0
        min_diff = float('inf')
        for i, result in enumerate(results):
            # Simple difference metric - how close is chord count to reference
            diff = abs(result["unique_chords"] - ref_diversity["unique_chords"])
            if diff < min_diff:
                min_diff = diff
                best_idx = i
        
        best_temp = results[best_idx]["temperature"]
        best_dist = results[best_idx]["diversity"].get("quality_distribution", {})
        
        # Get all qualities from both reference and best temp
        all_compared_quals = set(ref_qualities.keys()) | set(best_dist.keys())
        qual_list = sorted(all_compared_quals)
        
        # Prepare data
        ref_counts = [ref_qualities.get(q, 0) for q in qual_list]
        best_counts = [best_dist.get(q, 0) for q in qual_list]
        
        # Set up bar positions
        bar_width = 0.35
        x = np.arange(len(qual_list))
        
        # Plot bars
        plt.bar(x - bar_width/2, ref_counts, bar_width, label='Reference')
        plt.bar(x + bar_width/2, best_counts, bar_width, label=f'Best Temp ({best_temp})')
        
        # Add labels and title
        plt.xlabel('Chord Quality')
        plt.ylabel('Count')
        plt.title('Chord Quality Distribution Comparison')
        plt.xticks(x, qual_list, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        quality_comp_path = os.path.join(output_dir, "quality_comparison.png")
        plt.savefig(quality_comp_path)
        plt.close()
        logger.info(f"Quality comparison saved to {quality_comp_path}")

def main():
    parser = argparse.ArgumentParser(description="Tune HMM temperature parameter")
    parser.add_argument('--hmm', type=str, required=True,
                        help='Path to trained HMM model')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference lab file (ground truth or student output)')
    parser.add_argument('--output_dir', type=str, default='./hmm_tuning_results',
                        help='Directory to save results')
    parser.add_argument('--temps', type=float, nargs='+', 
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0],
                        help='Temperature values to try (default: 0.1 to 2.0)')
                        
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.hmm):
        logger.error(f"HMM model not found: {args.hmm}")
        return
    
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return
        
    if not os.path.exists(args.reference):
        logger.error(f"Reference lab file not found: {args.reference}")
        return
    
    # Tune temperature
    tune_temperature(args.hmm, args.audio, args.reference, args.output_dir, args.temps)

if __name__ == "__main__":
    main()
