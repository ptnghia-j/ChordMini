#!/usr/bin/env python3

"""
Script to find optimal HMM parameters by grid search.
Evaluates different combinations of temperature and emission weight.
"""

import os
import argparse
import numpy as np
import torch
import json
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from modules.utils import logger
from modules.utils.chords import idx2voca_chord, Chords
import mir_eval

def run_hmm_with_params(hmm_model_path, audio_file, ref_file, output_dir, temp, emission_weight):
    """Run HMM with specific parameters and return evaluation metrics"""
    output_json = os.path.join(output_dir, f"temp_{temp:.2f}_ew_{emission_weight:.2f}_results.json")
    output_lab = os.path.join(output_dir, f"temp_{temp:.2f}_ew_{emission_weight:.2f}.lab")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run test_hmm.py with specified parameters
    cmd = [
        "python", "test_hmm.py",
        "--hmm", hmm_model_path,
        "--audio_dir", os.path.dirname(audio_file),
        "--label_dir", os.path.dirname(ref_file),
        "--temperature", str(temp),
        "--emission_weight", str(emission_weight),
        "--output", output_json
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Check if results file was generated
        if os.path.exists(output_json):
            with open(output_json, 'r') as f:
                results = json.load(f)
            return results
        else:
            logger.error(f"Output results not found: {output_json}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running HMM with temp={temp}, ew={emission_weight}: {e}")
        logger.error(f"STDOUT: {e.stdout.decode()}")
        logger.error(f"STDERR: {e.stderr.decode()}")
        return None

def evaluate_against_teacher(teacher_lab, hmm_lab):
    """Compare HMM output against teacher output using MIR_EVAL"""
    try:
        # Read teacher labels
        teacher_intervals, teacher_labels = mir_eval.io.load_labeled_intervals(teacher_lab)
        
        # Read HMM labels
        hmm_intervals, hmm_labels = mir_eval.io.load_labeled_intervals(hmm_lab)
        
        # Evaluate chord recognition
        scores = mir_eval.chord.evaluate(teacher_intervals, teacher_labels, hmm_intervals, hmm_labels)
        
        return scores
    except Exception as e:
        logger.error(f"Error evaluating against teacher: {e}")
        return None

def grid_search(hmm_model_path, audio_file, ref_file, teacher_file, output_dir,
               temps=[0.2, 0.3, 0.5, 0.7, 0.9], 
               emission_weights=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Perform grid search over temperature and emission weight parameters
    
    Args:
        hmm_model_path: Path to trained HMM model
        audio_file: Audio file to process
        ref_file: Reference chord lab file (ground truth)
        teacher_file: Teacher model output file for comparison
        output_dir: Directory to save results
        temps: List of temperature values to try
        emission_weights: List of emission weight values to try
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results array
    results = []
    
    # Run grid search
    for temp in temps:
        for ew in emission_weights:
            logger.info(f"Testing temperature={temp}, emission_weight={ew}")
            
            # Run HMM with these parameters
            result = run_hmm_with_params(hmm_model_path, audio_file, ref_file, output_dir, temp, ew)
            
            if result:
                # Extract metrics
                hmm_metrics = result.get("hmm_model", {}).get("average_mir_eval_scores", {})
                
                # Compare against teacher if teacher file exists
                teacher_comparison = None
                if teacher_file and os.path.exists(teacher_file):
                    hmm_output_lab = os.path.join(output_dir, f"temp_{temp:.2f}_ew_{ew:.2f}.lab")
                    if os.path.exists(hmm_output_lab):
                        teacher_comparison = evaluate_against_teacher(teacher_file, hmm_output_lab)
                
                # Record results
                param_result = {
                    "temperature": temp,
                    "emission_weight": ew,
                    "metrics": hmm_metrics,
                    "teacher_comparison": teacher_comparison
                }
                
                results.append(param_result)
                
                # Log results
                logger.info(f"Params temp={temp}, ew={ew}: "
                           f"Root accuracy: {hmm_metrics.get('root', 0):.4f}, "
                           f"Thirds: {hmm_metrics.get('thirds', 0):.4f}, "
                           f"Triads: {hmm_metrics.get('triads', 0):.4f}")
    
    # Save full results
    with open(os.path.join(output_dir, "grid_search_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_grid_search(results, output_dir)
    
    # Determine best parameters
    find_best_parameters(results, output_dir)
    
    return results

def visualize_grid_search(results, output_dir):
    """Create visualization of grid search results"""
    if not results:
        logger.error("No results to visualize")
        return
    
    # Extract parameters and metrics
    temps = sorted(list(set([r["temperature"] for r in results])))
    eweights = sorted(list(set([r["emission_weight"] for r in results])))
    
    # Create heatmaps for different metrics
    metrics = ["root", "thirds", "triads", "mirex", "majmin", "sevenths"]
    
    for metric in metrics:
        if metric not in results[0]["metrics"]:
            continue
            
        # Create heatmap data
        heatmap_data = np.zeros((len(temps), len(eweights)))
        
        for r in results:
            temp_idx = temps.index(r["temperature"])
            ew_idx = eweights.index(r["emission_weight"])
            heatmap_data[temp_idx, ew_idx] = r["metrics"].get(metric, 0)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
        plt.colorbar(label=f'{metric} accuracy')
        
        # Add labels and ticks
        plt.xlabel('Emission Weight')
        plt.ylabel('Temperature')
        plt.title(f'HMM Parameter Grid Search: {metric} accuracy')
        plt.xticks(range(len(eweights)), [f"{ew:.1f}" for ew in eweights])
        plt.yticks(range(len(temps)), [f"{t:.1f}" for t in temps])
        
        # Add values to cells
        for i in range(len(temps)):
            for j in range(len(eweights)):
                plt.text(j, i, f"{heatmap_data[i, j]:.3f}", 
                        ha="center", va="center", color="white" if heatmap_data[i, j] < 0.7 else "black")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"grid_search_{metric}.png"))
        plt.close()

def find_best_parameters(results, output_dir):
    """Find and report the best parameters for different metrics"""
    if not results:
        logger.error("No results to analyze")
        return
    
    # Metrics to consider
    metrics = ["root", "thirds", "triads", "mirex", "majmin", "sevenths"]
    
    best_params = {}
    for metric in metrics:
        if metric not in results[0]["metrics"]:
            continue
            
        # Find best parameters for this metric
        best_score = 0
        best_result = None
        
        for r in results:
            score = r["metrics"].get(metric, 0)
            if score > best_score:
                best_score = score
                best_result = r
        
        if best_result:
            best_params[metric] = {
                "temperature": best_result["temperature"],
                "emission_weight": best_result["emission_weight"],
                "score": best_score
            }
    
    # Find best overall parameters (average of all metrics)
    if best_params:
        avg_scores = {}
        for r in results:
            # Calculate average score across all metrics
            scores = [r["metrics"].get(m, 0) for m in metrics if m in r["metrics"]]
            if scores:
                avg_score = sum(scores) / len(scores)
                key = (r["temperature"], r["emission_weight"])
                avg_scores[key] = avg_score
        
        if avg_scores:
            best_overall = max(avg_scores.items(), key=lambda x: x[1])
            best_params["overall"] = {
                "temperature": best_overall[0][0],
                "emission_weight": best_overall[0][1],
                "score": best_overall[1]
            }
    
    # Save and print results
    with open(os.path.join(output_dir, "best_parameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    
    # Print summary
    logger.info("===== BEST PARAMETERS =====")
    for metric, params in best_params.items():
        logger.info(f"{metric}: temp={params['temperature']}, ew={params['emission_weight']}, score={params['score']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Optimize HMM parameters by grid search")
    parser.add_argument('--hmm', type=str, required=True,
                        help='Path to trained HMM model')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--ref', type=str, required=True,
                        help='Path to reference lab file (ground truth)')
    parser.add_argument('--teacher', type=str, default=None,
                        help='Path to teacher output lab file for comparison')
    parser.add_argument('--output_dir', type=str, default='./hmm_optimization',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.hmm):
        logger.error(f"HMM model not found: {args.hmm}")
        return
        
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return
        
    if not os.path.exists(args.ref):
        logger.error(f"Reference lab file not found: {args.ref}")
        return
    
    if args.teacher and not os.path.exists(args.teacher):
        logger.warning(f"Teacher lab file not found: {args.teacher}")
    
    # Run grid search
    grid_search(args.hmm, args.audio, args.ref, args.teacher, args.output_dir)

if __name__ == "__main__":
    main()
