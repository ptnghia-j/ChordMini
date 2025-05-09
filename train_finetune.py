import os
import sys
import argparse
import glob
import gc
import random
import time
import json
import traceback
import multiprocessing
from pathlib import Path
from collections import Counter, defaultdict # Add defaultdict
import hashlib # Add hashlib import

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F # Add F for padding
import matplotlib.pyplot as plt # Import matplotlib

# Project imports
from modules.utils.device import get_device, is_cuda_available, is_gpu_available, clear_gpu_cache
from modules.data.SynthDataset import SynthDataset, SynthSegmentSubset # Use SynthDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.models.Transformer.btc_model import BTC_model
from modules.training.StudentTrainer import StudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord, Chords
from modules.training.Tester import Tester
# Import the enhanced standardization function
from modules.utils.mir_eval_modules import large_voca_score_calculation, audio_file_to_features, calculate_chord_scores, lab_file_error_modify as standardize_chord_label_mir, compute_individual_chord_accuracy # Import compute_individual_chord_accuracy

# Define the direct label parsing function (copied from original train_finetune.py)
def parse_lab_file_direct(label_path, chord_processor, frame_duration, total_frames):
    """
    Parses a .lab file directly, processes chords using the enhanced standardization,
    and aligns labels to frames.
    Returns a list of chord labels corresponding to each frame.
    """
    if not os.path.exists(label_path):
        logger.error(f"Label file not found during direct parse: {label_path}")
        return ['N'] * total_frames # Return 'N' for all frames if file not found

    try:
        intervals = []
        raw_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=2) # Split max 2 times for robust chord names
                if len(parts) == 3:
                    try:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        chord_name = parts[2]
                        # Basic validation
                        if start_time < 0 or end_time < start_time:
                             logger.warning(f"Skipping invalid interval in {label_path}: {line.strip()}")
                             continue
                        intervals.append((start_time, end_time))
                        raw_labels.append(chord_name)
                    except ValueError:
                        logger.warning(f"Skipping malformed line (non-numeric times?) in {label_path}: {line.strip()}")
                elif line.strip(): # Log other non-empty, non-conforming lines
                     logger.warning(f"Skipping malformed line in {label_path}: {line.strip()}")

        if not intervals:
            logger.warning(f"No valid intervals found in {label_path}")
            return ['N'] * total_frames # Return 'N' for all frames if file is empty/invalid

        # Process chord names using the enhanced standardization function from mir_eval_modules
        # Note: chord_processor is no longer used here, standardization handles it.
        processed_labels = [standardize_chord_label_mir(name) for name in raw_labels]

        # Align labels to frames
        frame_labels = ['N'] * total_frames # Default to No Chord
        last_assigned_idx = -1
        for i in range(total_frames):
            frame_mid_time = i * frame_duration + (frame_duration / 2.0) # Midpoint of the frame
            assigned = False
            # Find the interval containing the frame midpoint
            for j, (start, end) in enumerate(intervals):
                # Use a small tolerance for end time comparison
                # Ensure frame_mid_time is strictly less than end unless it's the very last interval
                is_last_interval = (j == len(intervals) - 1)
                if start <= frame_mid_time < end or (is_last_interval and abs(frame_mid_time - end) < 1e-6):
                    frame_labels[i] = processed_labels[j]
                    last_assigned_idx = j
                    assigned = True
                    break

            # Handle frames potentially after the last annotated interval
            # If a frame wasn't assigned and comes after the start of the last known interval,
            # assign the last known chord label. Check bounds carefully.
            if not assigned and last_assigned_idx != -1 and intervals and frame_mid_time >= intervals[last_assigned_idx][0]:
                 # Ensure last_assigned_idx is valid
                 if 0 <= last_assigned_idx < len(processed_labels):
                     frame_labels[i] = processed_labels[last_assigned_idx]
                 else:
                      # This case should ideally not happen if logic is correct
                      logger.warning(f"Invalid last_assigned_idx {last_assigned_idx} encountered for frame {i} in {label_path}. Assigning 'N'.")
                      frame_labels[i] = 'N'


        # Log assignment statistics
        assigned_count = sum(1 for lbl in frame_labels if lbl != 'N')
        # Reduce verbosity of this log, maybe use DEBUG level
        # logger.debug(f"Direct parse assignment for {os.path.basename(label_path)}: {assigned_count}/{total_frames} frames assigned ({assigned_count/total_frames*100:.1f}%)")

        return frame_labels

    except Exception as e:
        logger.error(f"Error during direct parsing of lab file {label_path}: {e}")
        logger.error(traceback.format_exc())
        return ['N'] * total_frames # Return 'N' on error


def count_files_in_subdirectories(directory, file_pattern):
    """Count files in a directory and all its subdirectories matching a pattern."""
    if not directory or not os.path.exists(directory):
        return 0
    count = 0
    # Use Path.rglob for recursive search
    for file_path in Path(directory).rglob(file_pattern):
        if file_path.is_file():
            count += 1
    return count

def find_sample_files(directory, file_pattern, max_samples=5):
    """Find sample files in a directory and all its subdirectories matching a pattern."""
    if not directory or not os.path.exists(directory):
        return []
    samples = []
    # Use Path.rglob for recursive search
    for file_path in Path(directory).rglob(file_pattern):
        if file_path.is_file():
            samples.append(str(file_path))
            if len(samples) >= max_samples:
                break
    return samples

def resolve_path(path, storage_root=None, project_root=None):
    """
    Resolve a path that could be absolute, relative to storage_root, or relative to project_root.
    """
    if not path:
        return None
    if os.path.isabs(path):
        return path
    if storage_root:
        storage_path = os.path.join(storage_root, path)
        # Check existence for storage_root resolution
        if os.path.exists(storage_path):
            return storage_path
    if project_root:
        project_path = os.path.join(project_root, path)
        # Check existence for project_root resolution
        if os.path.exists(project_path):
            return project_path
    # Fallback: prefer storage_root if provided, otherwise project_root
    if storage_root:
        return os.path.join(storage_root, path)
    return os.path.join(project_root, path) if project_root else path

def load_normalization_from_checkpoint(path, storage_root=None, project_root=None):
    """Load mean and std from a teacher checkpoint, or return (0.0, 1.0) if unavailable."""
    if not path:
        logger.warning("No teacher checkpoint specified for normalization. Using defaults (0.0, 1.0).")
        return 0.0, 1.0
    resolved_path = resolve_path(path, storage_root, project_root)
    if not os.path.exists(resolved_path):
        logger.warning(f"Teacher checkpoint for normalization not found at {resolved_path}. Using defaults (0.0, 1.0).")
        return 0.0, 1.0
    try:
        checkpoint = torch.load(resolved_path, map_location='cpu')
        mean = checkpoint.get('mean', 0.0)
        std = checkpoint.get('std', 1.0)
        mean = float(mean.item()) if hasattr(mean, 'item') else float(mean)
        std = float(std.item()) if hasattr(std, 'item') else float(std)
        if std == 0:
            logger.warning("Teacher checkpoint std is zero, using 1.0 instead.")
            std = 1.0
        logger.info(f"Loaded normalization from teacher checkpoint: mean={mean:.4f}, std={std:.4f}")
        return mean, std
    except Exception as e:
        logger.error(f"Error loading normalization from teacher checkpoint: {e}")
        logger.warning("Using default normalization parameters (mean=0.0, std=1.0).")
        return 0.0, 1.0

def log_dataset_chord_mapping(label_dirs, chord_mapping, master_mapping, logger):
    """
    Scans label files, processes unique raw labels, and logs their mapping
    to the final vocabulary index and label.
    """
    logger.info("\n=== Analyzing Dataset Chord Label Mapping ===")
    unique_raw_labels = set()
    processed_files = 0
    skipped_files = 0

    if not label_dirs:
        logger.warning("No label directories provided for mapping analysis.")
        return

    logger.info(f"Scanning label directories: {label_dirs}")
    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            logger.warning(f"Label directory not found: {label_dir}")
            continue
        
        # Use rglob to find all .lab and .txt files recursively
        label_files = list(Path(label_dir).rglob('*.lab')) + list(Path(label_dir).rglob('*.txt'))
        
        if not label_files:
            logger.warning(f"No .lab or .txt files found in {label_dir}")
            continue

        logger.info(f"Found {len(label_files)} label files in {label_dir}. Processing...")
        
        for label_path in tqdm(label_files, desc=f"Scanning {os.path.basename(label_dir)}", leave=False):
            try:
                with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split(maxsplit=2)
                        if len(parts) == 3:
                            raw_label = parts[2]
                            unique_raw_labels.add(raw_label)
                processed_files += 1
            except Exception as e:
                logger.warning(f"Skipping file {label_path} due to error: {e}")
                skipped_files += 1

    logger.info(f"Scan complete. Processed {processed_files} files, skipped {skipped_files}.")
    logger.info(f"Found {len(unique_raw_labels)} unique raw chord labels in the dataset.")

    if not unique_raw_labels:
        logger.warning("No unique raw labels found.")
        return

    # Map raw labels to final vocabulary index and label
    mapping_details = defaultdict(lambda: {'raw_labels': set(), 'standardized': set()})
    unknown_standardized = set()
    n_index = chord_mapping.get('N', None) # Get the index for 'N'

    if n_index is None:
        logger.error("Could not find index for 'N' in chord_mapping. Cannot proceed with mapping analysis.")
        # Try to find it by iterating (less efficient)
        for label, idx in chord_mapping.items():
            if label == 'N':
                n_index = idx
                logger.warning("Found 'N' index by iteration.")
                break
        if n_index is None: return # Still not found

    logger.info("Processing unique raw labels for mapping...")
    for raw_label in tqdm(sorted(list(unique_raw_labels)), desc="Mapping labels", leave=False):
        standardized_label = standardize_chord_label_mir(raw_label)
        
        # Get the index for the standardized label, default to N's index if not found
        final_idx = chord_mapping.get(standardized_label, n_index)
        
        # Check if the standardized label itself was unknown to the mapping
        if standardized_label not in chord_mapping and standardized_label != 'N':
             unknown_standardized.add(standardized_label)
             # Log immediately if a standardized label maps unexpectedly to N
             logger.debug(f"Raw label '{raw_label}' -> Standardized '{standardized_label}' -> Mapped to Index {final_idx} ('N') because '{standardized_label}' not in chord_mapping.")


        # Store details grouped by the final index
        mapping_details[final_idx]['raw_labels'].add(raw_label)
        mapping_details[final_idx]['standardized'].add(standardized_label)

    # Log the results
    logger.info("\n--- Chord Mapping Details (Dataset Raw Labels -> Vocabulary Index) ---")
    for index in sorted(mapping_details.keys()):
        final_label = master_mapping.get(index, f'Unknown Index {index}')
        raw_set = mapping_details[index]['raw_labels']
        std_set = mapping_details[index]['standardized']
        
        # Limit the number of raw labels shown for brevity
        raw_labels_display = sorted(list(raw_set))
        if len(raw_labels_display) > 500:
            raw_labels_display = raw_labels_display[:500] + ['...']
            
        logger.info(f"Index {index} ({final_label}):")
        logger.info(f"  Standardized As -> {sorted(list(std_set))}")
        logger.info(f"  From Raw Labels -> {raw_labels_display} (Total Raw: {len(raw_set)})")

    if unknown_standardized:
        logger.warning("\n--- Standardized Labels Not Found in Mapping (Mapped to 'N') ---")
        for std_label in sorted(list(unknown_standardized)):
            logger.warning(f"  Standardized label '{std_label}' was not in chord_mapping.")
    logger.info("--- End Chord Mapping Details ---")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a student model using pre-computed features/logits and real ground truth labels.")
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config value)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides config value)')
    parser.add_argument('--pretrained', type=str, required=False, # Required for ChordNet, optional for BTC
                        help='Path to pretrained model checkpoint for fine-tuning')
    parser.add_argument('--storage_root', type=str, default=None,
                        help='Root directory for data storage (overrides config value)')
    parser.add_argument('--use_warmup', action='store_true',
                       help='Use warm-up learning rate scheduling')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                       help='Number of warm-up epochs (default: from config)')
    parser.add_argument('--warmup_start_lr', type=float, default=None,
                       help='Initial learning rate for warm-up (default: 1/10 of base LR)')
    parser.add_argument('--lr_schedule', type=str,
                        choices=['cosine', 'linear_decay', 'one_cycle', 'cosine_warm_restarts', 'validation', 'none'],
                        default=None,
                        help='Learning rate schedule type (default: validation-based)')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss to handle class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss (default: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=None,
                       help='Alpha parameter for focal loss (default: None)')
    parser.add_argument('--use_kd_loss', action='store_true',
                       help='Use knowledge distillation loss (requires teacher logits)')
    parser.add_argument('--kd_alpha', type=float, default=None,
                       help='Weight for knowledge distillation loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Temperature for softening distributions (default: 2.0)')
    # No teacher model needed for offline KD
    # parser.add_argument('--teacher_model', type=str, default=None, ...)
    parser.add_argument('--model_scale', type=float, default=None,
                       help='Scaling factor for model capacity (0.5=half, 1.0=base, 2.0=double)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout probability (0-1)')
    parser.add_argument('--disable_cache', action='store_true',
                      help='Disable dataset caching to reduce memory usage')
    parser.add_argument('--metadata_cache', action='store_true', # Changed from type=str
                      help='Only cache metadata (not spectrograms) to reduce memory usage')
    parser.add_argument('--cache_fraction', type=float, default=0.1,
                      help='Fraction of dataset to cache (default: 0.1 = 10%%)')
    parser.add_argument('--lazy_init', action='store_true', # Changed from type=str
                      help='Lazily initialize dataset components to save memory')

    # Data directories for SynthDataset (using real labels)
    parser.add_argument('--spectrograms_dir', type=str, required=False, # Make optional, rely on ENV/config
                      help='Directory containing pre-computed spectrograms (e.g., from LabeledDataset_synth)')
    parser.add_argument('--logits_dir', type=str, required=False, # Make optional, rely on ENV/config
                      help='Directory containing pre-computed teacher logits (e.g., from LabeledDataset_synth)')
    parser.add_argument('--label_dirs', type=str, nargs='+', required=False, # Make optional, rely on ENV/config
                      help='List of directories containing REAL ground truth label files (.lab, .txt). For LabeledDataset, point to the root Labels dir (e.g., /mnt/storage/data/LabeledDataset/Labels)')
    parser.add_argument('--cache_dir', type=str, default=None,
                      help='Directory to cache dataset metadata/features')
    parser.add_argument('--dataset_type', type=str, default=None, # Add dataset_type argument
                        choices=['fma', 'maestro', 'dali_synth', 'labeled_synth', 'combined'],
                        help='Type of dataset structure (overrides auto-detection)')

    # GPU acceleration options
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.9,
                      help='Fraction of GPU memory to use (default: 0.9)')
    parser.add_argument('--batch_gpu_cache', action='store_true', # Changed from type=str
                      help='Cache batches on GPU for repeated access patterns')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                      help='Number of batches to prefetch (default: 2)')
    parser.add_argument('--small_dataset', type=float, default=None,
                      help='Use only a small percentage of dataset for quick testing (e.g., 0.01 for 1%%)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Base learning rate (overrides config value)')
    parser.add_argument('--min_learning_rate', type=float, default=None,
                        help='Minimum learning rate for schedulers (overrides config value)')
    parser.add_argument('--warmup_end_lr', type=float, default=None,
                       help='Target learning rate at the end of warm-up (default: base LR)')
    parser.add_argument('--freeze_feature_extractor', action='store_true',
                       help='Freeze the feature extraction part of the model')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (overrides config value)')
    parser.add_argument('--reset_epoch', action='store_true',
                      help='Start from epoch 1 when loading pretrained model')
    parser.add_argument('--reset_scheduler', action='store_true',
                      help='Reset learning rate scheduler when loading pretrained model')
    parser.add_argument('--timeout_minutes', type=int, default=30,
                      help='Timeout in minutes for distributed operations (default: 30)')
    parser.add_argument('--force_num_classes', type=int, default=None,
                        help='Force the model to use this number of output classes (e.g., 170 or 205)')
    parser.add_argument('--partial_loading', action='store_true',
                        help='Allow partial loading of output layer when model sizes differ')
    parser.add_argument('--use_voca', action='store_true',
                        help='Use large vocabulary (170 chord types instead of standard 25)')
    parser.add_argument('--model_type', type=str, choices=['ChordNet', 'BTC'], default='ChordNet',
                        help='Type of model to use (ChordNet or BTC)')
    parser.add_argument('--btc_checkpoint', type=str, default=None,
                        help='Path to BTC model checkpoint for finetuning (if model_type=BTC)')
    parser.add_argument('--btc_config', type=str, default='./config/btc_config.yaml', # Add BTC config argument
                        help='Path to the BTC model configuration file (if model_type=BTC)')
    parser.add_argument('--log_chord_details', action='store_true',
                       help='Enable detailed logging of chords during MIR evaluation')
    parser.add_argument('--teacher_checkpoint', type=str, default=None, # Renamed from --teacher_checkpoint_for_norm
                        help='Path to the teacher model checkpoint to load normalization parameters (mean, std)')


    args = parser.parse_args()

    # Load configuration from YAML first
    config = HParams.load(args.config) # This is the student/main config

    # --- Environment variable override block ---
    logger.info("Checking for environment variable overrides...")
    config.training['use_warmup'] = os.environ.get('USE_WARMUP', str(config.training.get('use_warmup', False))).lower() == 'true'
    config.training['use_focal_loss'] = os.environ.get('USE_FOCAL_LOSS', str(config.training.get('use_focal_loss', False))).lower() == 'true'
    config.training['use_kd_loss'] = os.environ.get('USE_KD_LOSS', str(config.training.get('use_kd_loss', False))).lower() == 'true'
    config.feature['large_voca'] = os.environ.get('USE_VOCA', str(config.feature.get('large_voca', False))).lower() == 'true'

    if 'MODEL_SCALE' in os.environ: config.model['scale'] = float(os.environ['MODEL_SCALE'])
    if 'LEARNING_RATE' in os.environ: config.training['learning_rate'] = float(os.environ['LEARNING_RATE'])
    if 'MIN_LEARNING_RATE' in os.environ: config.training['min_learning_rate'] = float(os.environ['MIN_LEARNING_RATE'])
    if 'WARMUP_EPOCHS' in os.environ: config.training['warmup_epochs'] = int(os.environ['WARMUP_EPOCHS'])
    if 'WARMUP_START_LR' in os.environ: config.training['warmup_start_lr'] = float(os.environ['WARMUP_START_LR'])
    if 'WARMUP_END_LR' in os.environ: config.training['warmup_end_lr'] = float(os.environ['WARMUP_END_LR'])
    if 'LR_SCHEDULE' in os.environ: config.training['lr_schedule'] = os.environ['LR_SCHEDULE']
    if 'FOCAL_GAMMA' in os.environ: config.training['focal_gamma'] = float(os.environ['FOCAL_GAMMA'])
    if 'FOCAL_ALPHA' in os.environ: config.training['focal_alpha'] = float(os.environ['FOCAL_ALPHA'])
    if 'KD_ALPHA' in os.environ: config.training['kd_alpha'] = float(os.environ['KD_ALPHA'])
    if 'TEMPERATURE' in os.environ: config.training['temperature'] = float(os.environ['TEMPERATURE'])
    if 'DROPOUT' in os.environ: config.model['dropout'] = float(os.environ['DROPOUT'])
    if 'EPOCHS' in os.environ: config.training['num_epochs'] = int(os.environ['EPOCHS'])
    if 'BATCH_SIZE' in os.environ: config.training['batch_size'] = int(os.environ['BATCH_SIZE'])
    if 'DATA_ROOT' in os.environ: config.paths['storage_root'] = os.environ['DATA_ROOT']
    if 'SPECTROGRAMS_DIR' in os.environ: args.spectrograms_dir = os.environ['SPECTROGRAMS_DIR']
    if 'LOGITS_DIR' in os.environ: args.logits_dir = os.environ['LOGITS_DIR']
    if 'LABEL_DIRS' in os.environ: args.label_dirs = os.environ['LABEL_DIRS'].split() # Split space-separated string
    if 'PRETRAINED_MODEL' in os.environ: args.pretrained = os.environ['PRETRAINED_MODEL']
    if 'BTC_CHECKPOINT' in os.environ: args.btc_checkpoint = os.environ['BTC_CHECKPOINT']
    if 'MODEL_TYPE' in os.environ: args.model_type = os.environ['MODEL_TYPE']
    if 'FREEZE_FEATURE_EXTRACTOR' in os.environ: args.freeze_feature_extractor = os.environ['FREEZE_FEATURE_EXTRACTOR'].lower() == 'true'
    if 'SMALL_DATASET' in os.environ: args.small_dataset = float(os.environ['SMALL_DATASET'])
    if 'DISABLE_CACHE' in os.environ: args.disable_cache = os.environ['DISABLE_CACHE'].lower() == 'true'
    # Add teacher checkpoint for norm override
    if 'TEACHER_CHECKPOINT' in os.environ: args.teacher_checkpoint = os.environ['TEACHER_CHECKPOINT'] # Renamed from TEACHER_CHECKPOINT_FOR_NORM
    # Correctly handle boolean flags from ENV vars after argparse
    if 'METADATA_CACHE' in os.environ and os.environ['METADATA_CACHE'].lower() == 'true':
        args.metadata_cache = True
    if 'LAZY_INIT' in os.environ and os.environ['LAZY_INIT'].lower() == 'true':
        args.lazy_init = True
    if 'BATCH_GPU_CACHE' in os.environ and os.environ['BATCH_GPU_CACHE'].lower() == 'true':
        args.batch_gpu_cache = True

    logger.info(f"Config after potential ENV overrides - use_warmup: {config.training.get('use_warmup')}")
    # --- END Environment variable override block ---

    # Override with command line args (These take precedence over ENV vars and config file)
    if args.log_chord_details:
        if 'misc' not in config: config['misc'] = {}
        config.misc['log_chord_details'] = True
        logger.info("Detailed chord logging during evaluation ENABLED via command line.")
    elif config.misc.get('log_chord_details'):
        logger.info("Detailed chord logging during evaluation ENABLED via config/env.")

    # Then check device availability
    if config.misc.get('use_cuda') and is_cuda_available():
        device = get_device()
        logger.info(f"CUDA available. Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available or not requested. Using CPU.")

    # Override config values with command line arguments if provided
    config.misc['seed'] = args.seed if args.seed is not None else config.misc.get('seed', 42)
    config.paths['checkpoints_dir'] = args.save_dir if args.save_dir else config.paths.get('checkpoints_dir', 'checkpoints/finetune')
    config.paths['storage_root'] = args.storage_root if args.storage_root else config.paths.get('storage_root', None)

    # Handle KD loss setting
    use_kd_loss = args.use_kd_loss or str(config.training.get('use_kd_loss', False)).lower() == 'true'
    if use_kd_loss:
        logger.info("Knowledge Distillation Loss ENABLED")
    else:
        logger.info("Knowledge Distillation Loss DISABLED")

    # Set large vocabulary config if specified
    if args.use_voca or str(config.feature.get('large_voca', False)).lower() == 'true':
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        logger.info("Using large vocabulary with 170 chord classes")
    else:
        config.feature['large_voca'] = False
        config.model['num_chords'] = 25 # Default to small voca if not specified
        logger.info("Using small vocabulary with 25 chord classes")

    # Handle learning rate and warmup parameters
    config.training['learning_rate'] = float(args.learning_rate) if args.learning_rate is not None else float(config.training.get('learning_rate', 0.0001))
    config.training['min_learning_rate'] = float(args.min_learning_rate) if args.min_learning_rate is not None else float(config.training.get('min_learning_rate', 5e-6))
    if args.warmup_epochs is not None: config.training['warmup_epochs'] = int(args.warmup_epochs)
    use_warmup_final = args.use_warmup or str(config.training.get('use_warmup', False)).lower() == 'true'
    config.training['use_warmup'] = use_warmup_final
    logger.info(f"Final warm-up setting: {use_warmup_final}")
    if args.warmup_start_lr is not None: config.training['warmup_start_lr'] = float(args.warmup_start_lr)
    elif 'warmup_start_lr' not in config.training: config.training['warmup_start_lr'] = config.training['learning_rate']/10
    if args.warmup_end_lr is not None: config.training['warmup_end_lr'] = float(args.warmup_end_lr)
    elif 'warmup_end_lr' not in config.training: config.training['warmup_end_lr'] = config.training['learning_rate']

    # Override epochs and batch size
    if args.epochs is not None: config.training['num_epochs'] = int(args.epochs)
    if args.batch_size is not None: config.training['batch_size'] = int(args.batch_size)

    # Log parameters
    logger.info(f"Using learning rate: {config.training['learning_rate']}")
    logger.info(f"Using minimum learning rate: {config.training['min_learning_rate']}")
    if use_warmup_final:
        logger.info(f"Using warmup_epochs: {config.training.get('warmup_epochs', 10)}")
        logger.info(f"Using warmup_start_lr: {config.training.get('warmup_start_lr')}")
        logger.info(f"Using warmup_end_lr: {config.training.get('warmup_end_lr')}")
    logger.info(f"Using {config.training.get('num_epochs', 50)} epochs for fine-tuning")
    logger.info(f"Using batch size: {config.training.get('batch_size', 16)}")

    # Log fine-tuning configuration
    logger.info("\n=== Fine-tuning Configuration ===")
    model_scale = float(args.model_scale) if args.model_scale is not None else float(config.model.get('scale', 1.0))
    logger.info(f"Model scale: {model_scale}")
    logger.info(f"Pretrained model (ChordNet): {args.pretrained}")
    logger.info(f"Pretrained model (BTC): {args.btc_checkpoint}")
    if args.freeze_feature_extractor:
        logger.info("Feature extraction layers will be frozen during fine-tuning")

    # Log KD settings
    kd_alpha = args.kd_alpha if args.kd_alpha is not None else float(config.training.get('kd_alpha', 0.5))
    temperature = args.temperature if args.temperature is not None else float(config.training.get('temperature', 2.0))
    if use_kd_loss:
        logger.info("\n=== Knowledge Distillation Enabled ===")
        logger.info(f"KD alpha: {kd_alpha}")
        logger.info(f"Temperature: {temperature}")
        logger.info("Using offline KD with pre-computed logits")
    else:
        logger.info("Knowledge distillation is disabled, using standard loss")

    # Log Focal Loss settings
    use_focal_loss = args.use_focal_loss or str(config.training.get('use_focal_loss', False)).lower() == 'true'
    focal_gamma = args.focal_gamma if args.focal_gamma is not None else float(config.training.get('focal_gamma', 2.0))
    focal_alpha = args.focal_alpha if args.focal_alpha is not None else config.training.get('focal_alpha') # Keep None if not specified
    if use_focal_loss:
        logger.info("\n=== Focal Loss Enabled ===")
        logger.info(f"Gamma: {focal_gamma}")
        if focal_alpha is not None: logger.info(f"Alpha: {focal_alpha}")
        else: logger.info("Alpha: None (using uniform weighting)")
    else:
        logger.info("Using standard cross-entropy loss (Focal Loss disabled)")

    # Final Loss Configuration Summary
    logger.info("\n=== Final Loss Configuration ===")
    if use_kd_loss and use_focal_loss:
        logger.info(f"Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha}) combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * focal_loss")
    elif use_kd_loss:
        logger.info(f"Using standard Cross Entropy combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * cross_entropy")
    elif use_focal_loss:
        logger.info(f"Using only Focal Loss with gamma={focal_gamma}, alpha={focal_alpha}")
    else:
        logger.info("Using only standard Cross Entropy Loss")

    # Set random seed
    seed = int(config.misc['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

    # Set up logging
    logger.logging_verbosity(config.misc.get('logging_level', 'INFO'))

    # Get project root and storage root
    project_root = os.path.dirname(os.path.abspath(__file__))
    storage_root = config.paths.get('storage_root', None)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Storage root: {storage_root}")

    # Resolve data paths
    spec_dir_arg = args.spectrograms_dir or config.paths.get('spectrograms_dir')
    logits_dir_arg = args.logits_dir or config.paths.get('logits_dir')
    label_dirs_arg = args.label_dirs or config.paths.get('label_dirs') # Get from args or config

    spec_dir = resolve_path(spec_dir_arg, storage_root, project_root)
    logits_dir = resolve_path(logits_dir_arg, storage_root, project_root) if logits_dir_arg else None # Handle None case
    if isinstance(label_dirs_arg, str): label_dirs_arg = [label_dirs_arg] # Convert single string to list
    label_dirs = [resolve_path(d, storage_root, project_root) for d in label_dirs_arg] if label_dirs_arg else []

    cache_dir = resolve_path(args.cache_dir or config.paths.get('cache_dir'), storage_root, project_root)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
    else:
        logger.info("Cache directory not specified.")

    # Log data paths and counts
    logger.info("\n=== Dataset Paths ===")
    logger.info(f"Spectrograms: {spec_dir}")
    logger.info(f"Logits: {logits_dir}")
    logger.info(f"Labels: {label_dirs}")

    spec_count = count_files_in_subdirectories(spec_dir, "*.npy")
    logits_count = count_files_in_subdirectories(logits_dir, "*.npy")
    label_count = sum(count_files_in_subdirectories(ld, "*.lab") + count_files_in_subdirectories(ld, "*.txt") for ld in label_dirs)

    logger.info(f"Found {spec_count} spectrogram files")
    logger.info(f"Found {logits_count} logit files")
    logger.info(f"Found {label_count} label files")

    if spec_count == 0 or label_count == 0:
        logger.error("Missing spectrograms or label files. Cannot proceed.")
        return
    if use_kd_loss and logits_count == 0:
        logger.error("Knowledge distillation enabled, but no logit files found. Cannot proceed.")
        return

    # Use the mapping defined in chords.py
    master_mapping = idx2voca_chord()
    chord_mapping = {chord: idx for idx, chord in master_mapping.items()}
    voca_chords_set = set(master_mapping.values()) # Get the set of valid chords

    # Initialize Chords class for processing (still needed?)
    # SynthDataset now uses standardize_chord_label_mir internally via its config
    # chord_processor = Chords()
    # chord_processor.set_chord_mapping(chord_mapping)
    # chord_processor.initialize_chord_mapping()

    # Log mapping info
    logger.info(f"\nUsing chord mapping from chords.py with {len(chord_mapping)} unique chords")
    logger.info(f"Sample chord mapping: {dict(list(chord_mapping.items())[:5])}")
    logger.info(f"Master mapping (idx -> label) size: {len(master_mapping)}")
    logger.info(f"Sample master mapping: {dict(list(master_mapping.items())[:5])} ... {dict(list(master_mapping.items())[-5:])}")


    # --- ADDED: Log dataset chord mapping analysis ---
    log_dataset_chord_mapping(label_dirs, chord_mapping, master_mapping, logger)
    # --- END ADDED ---


    # Resolve checkpoints directory path
    checkpoints_dir_config = config.paths.get('checkpoints_dir', 'checkpoints/finetune')
    checkpoints_dir = resolve_path(checkpoints_dir_config, storage_root, project_root)
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")

    # Initialize SynthDataset with optimized settings
    logger.info("\n=== Creating dataset using SynthDataset ===")

    # --- Determine dataset_type for SynthDataset ---
    dataset_type_for_synth = args.dataset_type # Prioritize command line arg
    if dataset_type_for_synth is None: # If not specified, try to infer
        if spec_dir and "LabeledDataset_synth/spectrograms" in str(spec_dir):
            dataset_type_for_synth = 'labeled_synth'
            logger.info("Inferred dataset type as 'labeled_synth' based on spectrogram path.")
        elif spec_dir and 'fma' in str(spec_dir).lower():
            dataset_type_for_synth = 'fma'
            logger.info("Inferred dataset type as 'fma' based on spectrogram path.")
        elif spec_dir and 'maestro' in str(spec_dir).lower():
            dataset_type_for_synth = 'maestro'
            logger.info("Inferred dataset type as 'maestro' based on spectrogram path.")
        elif spec_dir and 'dali_synth' in str(spec_dir).lower():
             dataset_type_for_synth = 'dali_synth'
             logger.info("Inferred dataset type as 'dali_synth' based on spectrogram path.")
        else:
            dataset_type_for_synth = 'fma' # Default fallback
            logger.warning(f"Could not infer dataset type from paths, defaulting to '{dataset_type_for_synth}'. Specify --dataset_type if needed.")
    else:
        logger.info(f"Using specified dataset type: '{dataset_type_for_synth}'")
    # --- End Determine dataset_type ---


    dataset_args = {
        'spec_dir': spec_dir,
        'label_dir': label_dirs, # Pass the list
        'logits_dir': logits_dir,
        'chord_mapping': chord_mapping, # Pass chord -> idx mapping
        'seq_len': config.training.get('seq_len', 10),
        'stride': config.training.get('seq_stride', 5),
        'frame_duration': config.feature.get('hop_duration', 0.09288),
        'verbose': config.misc.get('logging_level', 'INFO') == 'DEBUG', # Only verbose if debug logging
        'device': 'cpu', # SynthDataset primarily uses CPU loading
        'pin_memory': False, # Force False for SynthDataset
        'prefetch_factor': 1, # Keep 1, DataLoader handles prefetching
        'num_workers': 2, # Force 0 for SynthDataset
        'require_teacher_logits': use_kd_loss,
        'use_cache': not args.disable_cache,
        'metadata_only': args.metadata_cache, # Pass boolean directly
        'cache_fraction': config.data.get('cache_fraction', args.cache_fraction),
        'lazy_init': args.lazy_init, # Pass boolean directly
        'batch_gpu_cache': args.batch_gpu_cache, # Pass boolean directly
        'small_dataset_percentage': args.small_dataset,
        'dataset_type': dataset_type_for_synth # Pass inferred or specified type
    }
    # Add cache_file argument if cache_dir is specified
    # --- Remove this block, cache_file is handled internally by SynthDataset ---
    # if cache_dir:
    #     # Construct a cache file path within the cache directory
    #     cache_key = f"{os.path.basename(spec_dir)}_{os.path.basename(label_dirs[0]) if label_dirs else 'nolabels'}_{dataset_type_for_synth}"
    #     cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    #     dataset_args['cache_file'] = os.path.join(cache_dir, f"synth_dataset_{dataset_type_for_synth}_{cache_hash}.pkl")
    #     logger.info(f"Using cache file path: {dataset_args['cache_file']}")
    # --- End removed block ---


    # Create the dataset
    logger.info("Creating dataset with the following parameters:")
    for key, value in dataset_args.items():
        # Shorten long lists for logging
        if isinstance(value, list) and len(value) > 5:
             log_val = f"[{value[0]}, ..., {value[-1]}] (Total: {len(value)})"
        else:
             log_val = value
        logger.info(f"  {key}: {log_val}")

    try:
        # --- FIX: Pass the correct device object to SynthDataset ---
        dataset_args['device'] = device # Use the torch.device object determined earlier
        # --- End FIX ---
        synth_dataset = SynthDataset(**dataset_args)
    except Exception as e:
        logger.error(f"Failed to initialize SynthDataset: {e}")
        logger.error(traceback.format_exc())
        return

    # Create data loaders for each subset
    batch_size = config.training.get('batch_size', 16)
    logger.info(f"Using batch size: {batch_size}")

    # Use SynthSegmentSubset for efficient subset handling
    train_subset = SynthSegmentSubset(synth_dataset, synth_dataset.train_indices)
    eval_subset = SynthSegmentSubset(synth_dataset, synth_dataset.eval_indices)
    test_subset = SynthSegmentSubset(synth_dataset, synth_dataset.test_indices) # Create test subset

    # Ensure num_workers=0 and pin_memory=False for DataLoaders with SynthDataset
    dataloader_num_workers = 0
    dataloader_pin_memory = False

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
        # Removed prefetch_factor=None
    )
    val_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
        # Removed prefetch_factor=None
    )
    test_loader = DataLoader( # Create test loader
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
        # Removed prefetch_factor=None
    )

    logger.info(f"Training set: {len(train_subset)} samples ({len(train_loader)} batches)")
    logger.info(f"Validation set: {len(eval_subset)} samples ({len(val_loader)} batches)")
    logger.info(f"Test set: {len(test_subset)} samples ({len(test_loader)} batches)")

    # Check data loaders
    logger.info("\n=== Checking data loaders ===")
    try:
        batch = next(iter(train_loader))
        logger.info(f"First train batch loaded successfully:")
        logger.info(f"  Spectro shape: {batch['spectro'].shape}")
        logger.info(f"  Chord idx shape: {batch['chord_idx'].shape}")
        if use_kd_loss:
            if 'teacher_logits' in batch:
                logger.info(f"  Teacher logits shape: {batch['teacher_logits'].shape}")
            else:
                logger.error("KD enabled, but 'teacher_logits' not found in batch!")
                return
    except StopIteration:
        logger.error("ERROR: Train data loader is empty!")
        return
    except Exception as e:
        logger.error(f"ERROR: Failed to load first batch from train_loader: {e}")
        logger.error(traceback.format_exc())
        return

    # Determine the correct number of output classes
    if args.force_num_classes is not None:
        n_classes = args.force_num_classes
        logger.info(f"Forcing model to use {n_classes} output classes as specified by --force_num_classes")
    elif config.feature.get('large_voca', False):
        n_classes = 170
        logger.info(f"Using large vocabulary with {n_classes} output classes")
    else:
        n_classes = 25
        logger.info(f"Using small vocabulary with {n_classes} output classes")

    # Determine model type and pretrained path
    model_type = args.model_type
    pretrained_path = None
    btc_config = None # Initialize btc_config as None

    if model_type == 'BTC':
        # --- Load BTC Config ---
        btc_config_path = resolve_path(args.btc_config, storage_root, project_root)
        if not os.path.exists(btc_config_path):
            logger.error(f"BTC configuration file not found at {btc_config_path}. Cannot initialize BTC model.")
            return
        try:
            btc_config = HParams.load(btc_config_path)
            logger.info(f"Loaded BTC configuration from: {btc_config_path}")
        except Exception as e:
            logger.error(f"Error loading BTC configuration from {btc_config_path}: {e}")
            return
        # --- End Load BTC Config ---

        if args.btc_checkpoint:
            pretrained_path = args.btc_checkpoint
            logger.info(f"\n=== Loading BTC model from {pretrained_path} for fine-tuning ===")
        else:
            logger.info("\n=== No BTC checkpoint specified, will initialize a fresh BTC model ===")
    elif args.pretrained:
        pretrained_path = args.pretrained
        logger.info(f"\n=== Loading ChordNet model from {pretrained_path} for fine-tuning ===")
    else:
        logger.error(f"No pretrained model specified. Please provide --pretrained (for ChordNet) or --btc_checkpoint (for BTC)")
        return

    # Create model instance
    # start_epoch = 1 # REMOVED - Trainer handles its own start epoch internally
    optimizer_state_dict_to_load = None # Initialize optimizer state as None

    # --- Load Normalization from Teacher Checkpoint ---
    # Ensure mean/std are loaded correctly before model creation
    mean_val, std_val = load_normalization_from_checkpoint(
        args.teacher_checkpoint or config.paths.get('teacher_checkpoint'), # Use renamed arg/config key
        storage_root, project_root
    )
    # Keep as float values for now, convert to tensor later if needed by trainer/model
    normalization_params = {'mean': mean_val, 'std': std_val}
    logger.info(f"Using normalization parameters FOR TRAINING (from teacher checkpoint): mean={mean_val:.4f}, std={std_val:.4f}")

    # --- FIX: Convert normalization floats to tensors for Trainer ---
    trainer_normalization = {
        'mean': torch.tensor(normalization_params['mean'], device=device, dtype=torch.float32),
        'std': torch.tensor(normalization_params['std'], device=device, dtype=torch.float32)
    }
    # --- End FIX ---

    try:
        n_freq = getattr(config.feature, 'n_bins', 144) # Use n_bins from config
        logger.info(f"Using frequency dimension (n_bins): {n_freq}")

        n_group = None
        if model_type == 'ChordNet':
            model_scale = float(args.model_scale) if args.model_scale is not None else float(config.model.get('scale', 1.0))
            n_group = max(1, int(config.model.get('n_group', 32) * model_scale))
            logger.info(f"Using n_group={n_group} for ChordNet")

        dropout_rate = args.dropout if args.dropout is not None else config.model.get('dropout', 0.3)
        logger.info(f"Using dropout rate: {dropout_rate}")

        logger.info(f"Creating {model_type} model with {n_classes} output classes")

        if model_type == 'ChordNet':
            model = ChordNet(
                n_freq=n_freq,
                n_classes=n_classes,
                # ... other ChordNet params ...
                dropout=dropout_rate
            ).to(device)
        else: # BTC model
            # --- Pass btc_config.model to BTC_model ---
            if btc_config is None or not hasattr(btc_config, 'model'):
                 logger.error("BTC config or its 'model' section was not loaded correctly. Cannot initialize BTC model.")
                 return

            # --- Use Dictionary Access for btc_config.model ---
            # Get the model config dictionary
            model_config_dict = btc_config.model
            if not isinstance(model_config_dict, dict):
                logger.error(f"Expected btc_config.model to be a dictionary, but got {type(model_config_dict)}. Check btc_config.yaml structure.")
                return

            # Override parameters using dictionary access
            if 'num_chords' in model_config_dict:
                model_config_dict['num_chords'] = n_classes # Update num_chords based on finetuning setting
                logger.info(f"Overriding BTC model num_chords to: {n_classes}")
            else:
                logger.warning("Could not find 'num_chords' key in btc_config.model dictionary to override.")

            # Override dropout if needed using dictionary access
            dropout_keys_found = []
            if 'input_dropout' in model_config_dict:
                model_config_dict['input_dropout'] = dropout_rate
                dropout_keys_found.append('input_dropout')
            if 'layer_dropout' in model_config_dict:
                model_config_dict['layer_dropout'] = dropout_rate
                dropout_keys_found.append('layer_dropout')
            if 'attention_dropout' in model_config_dict:
                model_config_dict['attention_dropout'] = dropout_rate
                dropout_keys_found.append('attention_dropout')
            if 'relu_dropout' in model_config_dict:
                model_config_dict['relu_dropout'] = dropout_rate
                dropout_keys_found.append('relu_dropout')

            if dropout_keys_found:
                logger.info(f"Overriding BTC model dropout rates ({', '.join(dropout_keys_found)}) to: {dropout_rate}")
            else:
                logger.warning("Could not find any dropout parameter keys (input_dropout, etc.) in btc_config.model dictionary to override.")

            # Log using dictionary access
            current_num_chords = model_config_dict.get('num_chords', 'N/A') # Use .get for safe access in log
            logger.info(f"Initializing BTC model with num_chords={current_num_chords}, dropout={dropout_rate}")

            # Pass the potentially modified dictionary to the constructor
            # IMPORTANT: Ensure BTC_model constructor can handle a dictionary OR update BTC_model to use dict access.
            # Assuming BTC_model was updated previously to handle HParams object correctly,
            # we might need to convert the dict back to HParams or adjust BTC_model.
            # Let's try passing the dict first, assuming BTC_model can handle it or was adjusted.
            model = BTC_model(config=model_config_dict).to(device)
            # --- End Dictionary Access ---
            # --- End Pass btc_config.model ---

        # Attach chord mapping and normalization to model (for inference/evaluation)
        model.idx_to_chord = master_mapping
        model.normalization_mean = torch.tensor(normalization_params['mean'], device=device, dtype=torch.float32)
        model.normalization_std = torch.tensor(normalization_params['std'], device=device, dtype=torch.float32)
        logger.info("Attached chord mapping and normalization parameters to model")

        # Load pretrained weights AND potentially optimizer state for resuming
        if pretrained_path:
            try:
                checkpoint = torch.load(pretrained_path, map_location=device)
                if 'n_classes' in checkpoint:
                    pretrained_classes = checkpoint['n_classes']
                    logger.info(f"Pretrained model has {pretrained_classes} output classes")
                    if pretrained_classes != n_classes:
                        logger.warning(f"Mismatch in class count: pretrained={pretrained_classes}, current={n_classes}")
                        if not args.partial_loading:
                            logger.warning("Loading may fail. Use --partial_loading to attempt partial weights loading.")

                if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint: state_dict = checkpoint['model']
                else: state_dict = checkpoint

                # Handle 'module.' prefix
                if all(k.startswith('module.') for k in state_dict.keys()):
                    logger.info("Detected 'module.' prefix in state dict keys. Removing prefix.")
                    state_dict = {k[7:]: v for k, v in state_dict.items()}

                model.load_state_dict(state_dict, strict=not args.partial_loading)
                logger.info(f"Successfully loaded model weights from {pretrained_path}")

                # --- Load optimizer state if resuming ---
                # Note: Normalization is now loaded separately from the teacher checkpoint above
                if not args.reset_epoch:
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer_state_dict_to_load = checkpoint['optimizer_state_dict']
                        logger.info(f"Found optimizer state in {pretrained_path}. Will load if trainer doesn't load its own state.")
                    else:
                        logger.warning(f"Resuming requested (reset_epoch=False), but no optimizer state found in {pretrained_path}.")
                    # We no longer load epoch here, trainer handles it internally based on its own checkpoints
                else:
                    logger.info("Reset flags active (--reset_epoch). Ignoring optimizer state from pretrained file.")
                # --- End loading optimizer state ---

                # Clean up checkpoint memory
                del checkpoint, state_dict
                gc.collect()

            except Exception as e:
                logger.error(f"Error loading pretrained model/state from {pretrained_path}: {e}")
                if model_type == 'BTC':
                    logger.info("Continuing with freshly initialized BTC model")
                else:
                    logger.error("Cannot continue without pretrained weights for ChordNet model")
                    return
        else:
            if model_type == 'BTC':
                logger.info("No pretrained weights provided. Using freshly initialized BTC model.")
            else: # Should not happen due to earlier check, but safety first
                logger.error("Cannot continue without pretrained weights for ChordNet model")
                return

        # Freeze feature extraction layers if requested
        if args.freeze_feature_extractor:
            logger.info("Freezing feature extraction layers:")
            frozen_count = 0
            for name, param in model.named_parameters():
                # Adjust freezing logic based on model type if necessary
                freeze_condition = False
                if model_type == 'ChordNet' and ('frequency_net' in name or 'prenet' in name):
                    freeze_condition = True
                elif model_type == 'BTC' and ('conv1' in name or 'conv_layers' in name): # Example for BTC
                    freeze_condition = True
                # Add more specific conditions based on actual layer names

                if freeze_condition:
                    param.requires_grad = False
                    logger.info(f"  Frozen: {name}")
                    frozen_count += 1

            if frozen_count == 0:
                logger.warning("Freeze feature extractor requested, but no layers matched the freezing criteria.")
            else:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")

    except Exception as e:
        logger.error(f"Error creating or loading model: {e}")
        logger.error(traceback.format_exc())
        return

    # Create optimizer - only optimize unfrozen parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training['learning_rate'],
        weight_decay=config.training.get('weight_decay', 0.0)
    )

    # Load the optimizer state dict if it was found and resuming is intended
    # This state might be overwritten if the trainer loads its own checkpoint later
    if optimizer_state_dict_to_load:
        try:
            optimizer.load_state_dict(optimizer_state_dict_to_load)
            logger.info("Successfully loaded optimizer state from pretrained checkpoint.")
            # Optional: Move optimizer state to the correct device if necessary
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            logger.info("Moved optimizer state to current device.")
        except Exception as e:
            logger.error(f"Error loading optimizer state from pretrained checkpoint: {e}. Using fresh optimizer state.")
            optimizer_state_dict_to_load = None # Ensure we know it wasn't loaded

    # Clean up GPU memory before training
    if torch.cuda.is_available():
        logger.info("Performing CUDA memory cleanup before training")
        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"CUDA memory stats (GB): allocated={allocated:.2f}, reserved={reserved:.2f}")

    # Final memory cleanup before training
    if torch.cuda.is_available():
        logger.info("Final CUDA memory cleanup before training")
        torch.cuda.empty_cache()

    # Handle LR schedule
    lr_schedule_type = args.lr_schedule or config.training.get('lr_schedule', 'validation') # Default to validation
    if lr_schedule_type in ['validation', 'none']:
        lr_schedule_type = None # Trainer handles these internally or disables scheduler

    # Create trainer
    use_warmup_value = config.training.get('use_warmup', False)
    warmup_epochs = int(config.training.get('warmup_epochs', 10)) if use_warmup_value else None
    warmup_start_lr = float(config.training.get('warmup_start_lr')) if use_warmup_value else None
    warmup_end_lr = float(config.training.get('warmup_end_lr')) if use_warmup_value else None

    logger.info(f"Creating trainer with use_warmup={use_warmup_value}")
    if use_warmup_value:
        logger.info(f"Warmup configuration: {warmup_epochs} epochs from {warmup_start_lr} to {warmup_end_lr}")

    trainer = StudentTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=int(config.training.get('num_epochs', 50)),
        logger=logger,
        checkpoint_dir=checkpoints_dir,
        class_weights=None, # Add class weights later if needed
        idx_to_chord=master_mapping,
        normalization=trainer_normalization, # Pass the checkpoint-based normalization tensors
        early_stopping_patience=int(config.training.get('early_stopping_patience', 10)), # Increased patience
        lr_decay_factor=float(config.training.get('lr_decay_factor', 0.95)),
        min_lr=float(config.training.get('min_learning_rate', 5e-6)),
        use_warmup=use_warmup_value,
        warmup_epochs=warmup_epochs,
        warmup_start_lr=warmup_start_lr,
        warmup_end_lr=warmup_end_lr,
        lr_schedule_type=lr_schedule_type,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        use_kd_loss=use_kd_loss,
        kd_alpha=kd_alpha,
        temperature=temperature,
        timeout_minutes=args.timeout_minutes,
        reset_epoch=args.reset_epoch, # Pass reset flags
        reset_scheduler=args.reset_scheduler
        # REMOVED: start_epoch=start_epoch
    )

    # Attach chord mapping to trainer (chord -> idx) if needed
    trainer.set_chord_mapping(chord_mapping)

    # Log checkpoint loading status for resuming by checking for the latest state file
    # This check informs the user about the trainer's likely behavior
    latest_checkpoint_path = os.path.join(checkpoints_dir, "trainer_state_latest.pth")
    will_trainer_load_internal_state = os.path.exists(latest_checkpoint_path) and not args.reset_epoch

    if will_trainer_load_internal_state:
         logger.info(f"Trainer found existing internal checkpoint '{latest_checkpoint_path}'. Attempting to resume training (will load epoch, optimizer, etc.).")
         if optimizer_state_dict_to_load:
             logger.warning("Optimizer state loaded from pretrained checkpoint might be overwritten by trainer's internal checkpoint.")
    else:
        if not os.path.exists(latest_checkpoint_path):
            logger.info(f"No suitable internal trainer checkpoint found at '{latest_checkpoint_path}'.")
        if args.reset_epoch:
            logger.info("Reset flags active (--reset_epoch).")
        logger.info("Starting training from scratch (epoch 1) after loading pretrained weights.")
        if optimizer_state_dict_to_load:
            logger.info("Using optimizer state loaded from the pretrained checkpoint.")
        else:
            logger.info("Using a fresh optimizer state.")


    # Run training
    logger.info(f"\n=== Starting fine-tuning ===")
    try:
        # Verify KD setup if enabled
        if use_kd_loss:
            logger.info("Verifying offline knowledge distillation setup...")
            try:
                sample_batch = next(iter(train_loader))
                if 'teacher_logits' in sample_batch:
                    logger.info(f"✅ Teacher logits found in batch with shape: {sample_batch['teacher_logits'].shape}")
                    # Optional: Run one batch to check loss calculation
                    # batch_metrics = trainer.train_batch(sample_batch)
                    # logger.info(f"Sample batch metrics: {batch_metrics}")
                else:
                    logger.warning("⚠️ No teacher logits found in the batch. KD will not work.")
            except Exception as e:
                logger.error(f"Error verifying KD setup: {e}")

        # Start training - Trainer will handle loading its state internally
        trainer.train(train_loader, val_loader)
        logger.info("Fine-tuning completed successfully!")
    except KeyboardInterrupt:
        logger.info("Fine-tuning interrupted by user")
    except Exception as e:
        logger.error(f"ERROR during fine-tuning: {e}")
        logger.error(traceback.format_exc())

    # Final evaluation on test set
    logger.info("\n=== Testing ===")
    try:
        if trainer.load_best_model():
            logger.info("Evaluating using best model checkpoint.")

            # Basic testing with Tester class
            tester = Tester(
                model=model,
                test_loader=test_loader, # Use the test_loader created earlier
                device=device,
                idx_to_chord=master_mapping,
                normalization=trainer_normalization, # Pass the correct normalization dict
                output_dir=checkpoints_dir,
                logger=logger
            )
            test_metrics = tester.evaluate(save_plots=True)

            # Save test metrics
            try:
                metrics_path = os.path.join(checkpoints_dir, "test_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Test metrics saved to {metrics_path}")
            except Exception as e:
                logger.error(f"Error saving test metrics: {e}")

            # Advanced MIR evaluation using frame-level data from SynthDataset
            logger.info("\n=== MIR evaluation (Test Set) ===")
            all_preds_idx = []
            all_targets_idx = []
            all_target_labels_orig = [] # Collect original labels for debugging if needed
            model.eval()
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="MIR Eval"):
                    inputs = batch['spectro'].to(device)
                    targets = batch['chord_idx'].to(device) # Shape: (batch, seq_len)
                    # Store original labels if available in batch (SynthDataset might not include them)
                    if 'chord_label' in batch:
                         # Assuming batch['chord_label'] is a list of lists/strings
                         for label_seq in batch['chord_label']:
                              if isinstance(label_seq, (list, tuple)):
                                   all_target_labels_orig.extend(label_seq)
                              else:
                                   all_target_labels_orig.append(label_seq)


                    # Apply normalization (using model's attached norm params)
                    if hasattr(model, 'normalization_mean') and hasattr(model, 'normalization_std'):
                        inputs = (inputs - model.normalization_mean) / model.normalization_std
                    else:
                         # Fallback to trainer's norm if model doesn't have it
                         if trainer_normalization and 'mean' in trainer_normalization and 'std' in trainer_normalization:
                              inputs = (inputs - trainer_normalization['mean']) / trainer_normalization['std']


                    outputs = model(inputs) # Shape: (batch, seq_len, n_classes)
                    if isinstance(outputs, tuple): logits = outputs[0]
                    else: logits = outputs

                    preds = logits.argmax(dim=-1) # Shape: (batch, seq_len)

                    # Flatten batch and sequence dimensions
                    all_preds_idx.extend(preds.view(-1).cpu().numpy())
                    all_targets_idx.extend(targets.view(-1).cpu().numpy())

            # Convert indices to chord labels using master_mapping
            # These predictions are already based on the standardized vocabulary
            all_prediction_labels_std = [master_mapping.get(idx, 'N') for idx in all_preds_idx] # Use N for unknown preds
            # Targets from dataset should already be mapped to indices of known chords or N's index
            all_reference_labels_std = [master_mapping.get(idx, 'N') for idx in all_targets_idx] # Use N for unknown targets

            mir_eval_results = {}
            if all_reference_labels_std and all_prediction_labels_std:
                logger.info(f"Calculating final MIR scores using {len(all_reference_labels_std)} aggregated frames...")
                try:
                    # Create dummy timestamps and use calculate_chord_scores
                    frame_duration = config.feature.get('hop_duration', 0.09288)
                    num_frames = len(all_reference_labels_std)
                    timestamps = np.arange(num_frames) * frame_duration

                    # Use the refactored calculate_chord_scores which calls mir_eval.evaluate
                    scores_tuple = calculate_chord_scores(
                        timestamps, frame_duration,
                        all_reference_labels_std, all_prediction_labels_std
                    )

                    # Map tuple back to dictionary
                    score_names = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
                    mir_eval_results = {name: score for name, score in zip(score_names, scores_tuple)}

                    logger.info(f"Detailed MIR scores: {mir_eval_results}")

                    # Calculate frame-wise accuracy using standardized labels
                    correct_frames = sum(1 for ref, pred in zip(all_reference_labels_std, all_prediction_labels_std) if ref == pred)
                    total_frames = len(all_reference_labels_std)
                    frame_accuracy = correct_frames / total_frames if total_frames > 0 else 0
                    mir_eval_results['frame_accuracy'] = frame_accuracy
                    logger.info(f"Frame-wise Accuracy (Standardized): {frame_accuracy:.4f}")

                    # --- Add Individual Chord Quality Accuracy ---
                    logger.info("\n--- Final Test Set Chord Quality Accuracy ---")
                    # Pass the standardized labels to the analysis function
                    ind_acc, quality_stats = compute_individual_chord_accuracy(
                        all_reference_labels_std,
                        all_prediction_labels_std
                    )
                    # Optionally add ind_acc to mir_eval_results (might make JSON large)
                    # mir_eval_results['quality_accuracy'] = ind_acc
                    # mir_eval_results['quality_stats'] = quality_stats # Even larger

                except Exception as mir_calc_error:
                    logger.error(f"Failed to calculate detailed MIR scores: {mir_calc_error}")
                    logger.error(traceback.format_exc())
                    mir_eval_results['error'] = f"MIR calculation failed: {mir_calc_error}"
            else:
                logger.warning("No reference or prediction labels collected for MIR evaluation.")
                mir_eval_results = {'error': 'No labels collected'}

            # Log label statistics (using standardized labels)
            n_count_ref = sum(1 for label in all_reference_labels_std if label == 'N')
            x_count_ref = 0 # X is mapped to N
            n_count_pred = sum(1 for label in all_prediction_labels_std if label == 'N')
            x_count_pred = 0 # X is mapped to N
            logger.info(f"\nChord label statistics (Standardized):")
            if all_reference_labels_std: logger.info(f"Reference labels: {len(all_reference_labels_std)} total, {n_count_ref} 'N' ({n_count_ref/len(all_reference_labels_std)*100:.1f}%)")
            else: logger.warning("No reference labels collected.")
            if all_prediction_labels_std: logger.info(f"Predicted labels: {len(all_prediction_labels_std)} total, {n_count_pred} 'N' ({n_count_pred/len(all_prediction_labels_std)*100:.1f}%)")
            else: logger.warning("No prediction labels collected.")

            # ... save MIR eval metrics ...

        else:
            logger.warning("Could not load best model for testing")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.error(traceback.format_exc())

    # Save the final model
    try:
        save_path = os.path.join(checkpoints_dir, "finetuned_model_final.pth")
        # Ensure normalization params are saved as numpy arrays or floats
        mean_to_save = normalization_params['mean']
        std_to_save = normalization_params['std']
        if hasattr(mean_to_save, 'cpu'): mean_to_save = mean_to_save.cpu().numpy()
        if hasattr(std_to_save, 'cpu'): std_to_save = std_to_save.cpu().numpy()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'chord_mapping': chord_mapping,
            'idx_to_chord': master_mapping,
            'mean': float(mean_to_save), # Save as float
            'std': float(std_to_save),   # Save as float
            'n_classes': n_classes, # Save number of classes
            'model_type': model_type # Save model type
        }, save_path)
        logger.info(f"Final fine-tuned model saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")

    logger.info("Fine-tuning and evaluation complete!")

if __name__ == '__main__':
    # Set start method for multiprocessing if necessary
    try:
        if sys.platform.startswith('win'):
             # 'spawn' is default on Windows, but set explicitly for clarity
             multiprocessing.set_start_method('spawn', force=True)
        else:
             # Use 'fork' on Unix-like systems if available and desired,
             # otherwise 'spawn' is safer for CUDA but might be slower.
             # Let's stick to 'spawn' for consistency across platforms.
             multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set or cannot be changed
        current_method = multiprocessing.get_start_method(allow_none=True)
        logger.info(f"Multiprocessing start method already set to '{current_method}'.")
        pass
    except Exception as e:
        logger.warning(f"Could not set multiprocessing start method: {e}")

    main()