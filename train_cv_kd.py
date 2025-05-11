import json
import traceback
import multiprocessing
import sys
import os
import torch
import numpy as np
import argparse
import glob
import gc
import random
import time
import hashlib # Add hashlib import
from pathlib import Path
from torch.utils.data import DataLoader, Subset # Add Subset
from collections import Counter, defaultdict # Add defaultdict
from tqdm import tqdm
import torch.nn.functional as F # Add F for padding
import matplotlib.pyplot as plt # Import matplotlib

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation, audio_file_to_features, calculate_chord_scores, lab_file_error_modify as standardize_chord_label_mir, compute_individual_chord_accuracy # Import compute_individual_chord_accuracy
from modules.utils.device import get_device, is_cuda_available, clear_gpu_cache
from modules.data.CrossValidationDataset import CrossValidationDataset # Use CrossValidationDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.models.Transformer.btc_model import BTC_model  # Import BTC model
from modules.training.StudentTrainer import StudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord, Chords
from modules.training.Tester import Tester
from modules.utils.file_utils import count_files_in_subdirectories, find_sample_files, resolve_path, load_normalization_from_checkpoint
# REMOVED: from modules.utils.teacher_utils import load_btc_model, extract_logits_from_teacher, generate_teacher_predictions

# --- Using utility functions from modules.utils.file_utils ---

def log_dataset_chord_mapping(label_dirs, chord_mapping, master_mapping, logger, small_dataset_percentage=None):
    """
    Scans label files, processes unique raw labels, and logs their mapping
    to the final vocabulary index and label.

    Args:
        label_dirs: List of directories containing label files
        chord_mapping: Dictionary mapping chord labels to indices
        master_mapping: Dictionary mapping indices to chord labels
        logger: Logger instance
        small_dataset_percentage: If set, only scan a subset of files
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

        # Apply small_dataset_percentage if specified
        if small_dataset_percentage is not None and 0.0 < small_dataset_percentage < 1.0:
            # Calculate how many files to sample
            sample_size = max(1, int(len(label_files) * small_dataset_percentage))
            # Use a fixed seed for reproducibility
            random.seed(42)
            # Sample a subset of files
            label_files = random.sample(label_files, sample_size)
            logger.info(f"Using small_dataset_percentage={small_dataset_percentage}, analyzing {sample_size} of {len(label_files)} label files")

        logger.info(f"Processing {len(label_files)} label files from {label_dir}...")

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

# --- End Helper Functions ---


def main():
    # Parse command line arguments (aligned with train_finetune.py, keeping CV args)
    parser = argparse.ArgumentParser(description="Train a chord recognition model with cross-validation and knowledge distillation")
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config value)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides config value)')
    parser.add_argument('--load_checkpoint', type=str, default=None, # Renamed from --pretrained for consistency
                        help='Path to a specific checkpoint to load for fine-tuning')
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
                       help='Use knowledge distillation loss (requires teacher logits in dataset)')
    parser.add_argument('--kd_alpha', type=float, default=None,
                       help='Weight for knowledge distillation loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Temperature for softening distributions (default: 2.0)')
    parser.add_argument('--model_scale', type=float, default=None,
                       help='Scaling factor for model capacity (0.5=half, 1.0=base, 2.0=double)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout probability (0-1)')
    parser.add_argument('--disable_cache', action='store_true',
                      help='Disable dataset caching to reduce memory usage')
    parser.add_argument('--metadata_cache', action='store_true',
                      help='Only cache metadata (not spectrograms) to reduce memory usage')
    parser.add_argument('--cache_fraction', type=float, default=1.0, # Default to full cache for CV
                      help='Fraction of dataset to cache (default: 1.0 = 100%%)')
    parser.add_argument('--lazy_init', action='store_true',
                      help='Lazily initialize dataset components to save memory')

    # Data directories for CrossValidationDataset (using LabeledDataset_synth structure)
    parser.add_argument('--spectrograms_dir', type=str, required=False,
                      help='Directory containing pre-computed spectrograms (e.g., LabeledDataset_synth/spectrograms)')
    parser.add_argument('--logits_dir', type=str, required=False,
                      help='Directory containing pre-computed teacher logits (e.g., LabeledDataset_synth/logits)')
    parser.add_argument('--label_dir', type=str, required=False, # Single base dir for labels (e.g., LabeledDataset/Labels)
                      help='Base directory containing REAL ground truth label files (.lab, .txt) in subdirs (e.g., LabeledDataset/Labels)')
    parser.add_argument('--cache_dir', type=str, default=None,
                      help='Directory to cache dataset metadata/features')

    # GPU acceleration options
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.9,
                      help='Fraction of GPU memory to use (default: 0.9)')
    parser.add_argument('--batch_gpu_cache', action='store_true',
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
                        help='Force the model to use this number of output classes (e.g., 170 or 26)')
    parser.add_argument('--partial_loading', action='store_true',
                        help='Allow partial loading of output layer when model sizes differ')
    parser.add_argument('--use_voca', action='store_true',
                        help='Use large vocabulary (170 chord types instead of standard 25)')
    parser.add_argument('--model_type', type=str, choices=['ChordNet', 'BTC'], default='ChordNet',
                        help='Type of model to use (ChordNet or BTC)')
    parser.add_argument('--initial_model_ckpt_path', type=str, default=None,
                        help='Path to initial model checkpoint for both BTC and ChordNet models')
    parser.add_argument('--btc_config', type=str, default='./config/btc_config.yaml',
                        help='Path to the BTC model configuration file (if model_type=BTC)')
    parser.add_argument('--log_chord_details', action='store_true',
                       help='Enable detailed logging of chords during MIR evaluation')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to the teacher model checkpoint to load normalization parameters (mean, std)')

    # CV specific arguments
    parser.add_argument('--kfold', type=int, default=0,
                        help='Which fold to use for validation (0 to total_folds-1)')
    parser.add_argument('--total_folds', type=int, default=4, # Default to 4 folds
                        help='Total number of folds for cross-validation')

    args = parser.parse_args()

    # Load configuration from YAML first
    config = HParams.load(args.config)

    # --- Environment variable override block (Copied from train_finetune.py) ---
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
    if 'LABEL_DIR' in os.environ: args.label_dir = os.environ['LABEL_DIR'] # Single dir for CV
    if 'LOAD_CHECKPOINT' in os.environ: args.load_checkpoint = os.environ['LOAD_CHECKPOINT'] # Use LOAD_CHECKPOINT
    if 'INITIAL_MODEL_CKPT_PATH' in os.environ: args.initial_model_ckpt_path = os.environ['INITIAL_MODEL_CKPT_PATH'] # Path for initial model weights
    if 'MODEL_TYPE' in os.environ: args.model_type = os.environ['MODEL_TYPE']
    if 'FREEZE_FEATURE_EXTRACTOR' in os.environ: args.freeze_feature_extractor = os.environ['FREEZE_FEATURE_EXTRACTOR'].lower() == 'true'
    if 'SMALL_DATASET' in os.environ: args.small_dataset = float(os.environ['SMALL_DATASET'])
    if 'DISABLE_CACHE' in os.environ: args.disable_cache = os.environ['DISABLE_CACHE'].lower() == 'true'
    if 'TEACHER_CHECKPOINT' in os.environ: args.teacher_checkpoint = os.environ['TEACHER_CHECKPOINT']
    if 'METADATA_CACHE' in os.environ and os.environ['METADATA_CACHE'].lower() == 'true': args.metadata_cache = True
    if 'LAZY_INIT' in os.environ and os.environ['LAZY_INIT'].lower() == 'true': args.lazy_init = True
    if 'BATCH_GPU_CACHE' in os.environ and os.environ['BATCH_GPU_CACHE'].lower() == 'true': args.batch_gpu_cache = True
    if 'KFOLD' in os.environ: args.kfold = int(os.environ['KFOLD'])
    if 'TOTAL_FOLDS' in os.environ: args.total_folds = int(os.environ['TOTAL_FOLDS'])
    if 'SAVE_DIR' in os.environ: args.save_dir = os.environ['SAVE_DIR']
    if 'RESET_EPOCH' in os.environ: args.reset_epoch = os.environ['RESET_EPOCH'].lower() == 'true'
    if 'RESET_SCHEDULER' in os.environ: args.reset_scheduler = os.environ['RESET_SCHEDULER'].lower() == 'true'

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
    if config.misc.get('use_cuda', True) and is_cuda_available():
        device = get_device()
        logger.info(f"CUDA available. Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available or not requested. Using CPU.")

    # Override config values with command line arguments if provided
    config.misc['seed'] = args.seed if args.seed is not None else config.misc.get('seed', 42)
    # Modify save dir to include fold info by default
    default_save_dir = f'./checkpoints/cv_kd_fold{args.kfold}'
    config.paths['checkpoints_dir'] = args.save_dir if args.save_dir else config.paths.get('checkpoints_dir', default_save_dir)
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
    logger.info(f"Using {config.training.get('num_epochs', 50)} epochs for training")
    logger.info(f"Using batch size: {config.training.get('batch_size', 16)}")

    # Log fine-tuning configuration
    logger.info("\n=== Cross-Validation KD Configuration ===")
    logger.info(f"Current Fold: {args.kfold} / {args.total_folds}")
    model_scale = float(args.model_scale) if args.model_scale is not None else float(config.model.get('scale', 1.0))
    logger.info(f"Model scale: {model_scale}")
    logger.info(f"Load checkpoint: {args.load_checkpoint}")
    logger.info(f"Initial model checkpoint path: {args.initial_model_ckpt_path}")
    if args.freeze_feature_extractor:
        logger.info("Feature extraction layers will be frozen during training")

    # Log KD settings
    kd_alpha = args.kd_alpha if args.kd_alpha is not None else float(config.training.get('kd_alpha', 0.5))
    temperature = args.temperature if args.temperature is not None else float(config.training.get('temperature', 2.0))
    if use_kd_loss:
        logger.info("\n=== Knowledge Distillation Enabled ===")
        logger.info(f"KD alpha: {kd_alpha}")
        logger.info(f"Temperature: {temperature}")
        logger.info("Using offline KD with pre-computed logits from dataset")
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

    # Resolve data paths for CrossValidationDataset (LabeledDataset_synth structure)
    spec_dir_arg = args.spectrograms_dir or config.paths.get('spectrograms_dir')
    logits_dir_arg = args.logits_dir or config.paths.get('logits_dir')
    label_dir_arg = args.label_dir or config.paths.get('label_dir') # Single base dir

    spec_dir = resolve_path(spec_dir_arg, storage_root, project_root)
    logits_dir = resolve_path(logits_dir_arg, storage_root, project_root) if logits_dir_arg else None
    label_dir = resolve_path(label_dir_arg, storage_root, project_root) # Resolve single base dir

    cache_dir = resolve_path(args.cache_dir or config.paths.get('cache_dir'), storage_root, project_root)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
    else:
        logger.info("Cache directory not specified.")

    # Log data paths and counts
    logger.info("\n=== Dataset Paths (LabeledDataset_synth structure expected) ===")
    logger.info(f"Spectrograms Base: {spec_dir}")
    logger.info(f"Logits Base: {logits_dir}")
    logger.info(f"Labels Base: {label_dir}")

    spec_count = count_files_in_subdirectories(spec_dir, "*_spec.npy")
    logits_count = count_files_in_subdirectories(logits_dir, "*_logits.npy")
    label_count = count_files_in_subdirectories(label_dir, "*.lab") + count_files_in_subdirectories(label_dir, "*.txt")

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
    voca_chords_set = set(master_mapping.values())

    # Log mapping info
    logger.info(f"\nUsing chord mapping from chords.py with {len(chord_mapping)} unique chords")
    logger.info(f"Sample chord mapping: {dict(list(chord_mapping.items())[:5])}")
    logger.info(f"Master mapping (idx -> label) size: {len(master_mapping)}")
    logger.info(f"Sample master mapping: {dict(list(master_mapping.items())[:5])} ... {dict(list(master_mapping.items())[-5:])}")

    # --- ADDED: Log dataset chord mapping analysis ---
    # Pass the single base label directory to the analysis function
    # Also pass small_dataset_percentage to limit the number of files scanned
    log_dataset_chord_mapping([label_dir] if label_dir else [], chord_mapping, master_mapping, logger,
                             small_dataset_percentage=args.small_dataset)
    # --- END ADDED ---

    # Resolve checkpoints directory path (already includes fold info by default)
    checkpoints_dir = config.paths['checkpoints_dir']
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")

    # Initialize CrossValidationDataset
    logger.info("\n=== Creating dataset using CrossValidationDataset ===")
    logger.info(f"Fold {args.kfold} / {args.total_folds}")

    dataset_args = {
        'spec_dir': spec_dir,
        'label_dir': label_dir, # Pass the single base dir
        'logits_dir': logits_dir,
        'chord_mapping': chord_mapping,
        'seq_len': config.training.get('seq_len', 10),
        'stride': config.training.get('seq_stride', 5),
        'frame_duration': config.feature.get('hop_duration', 0.09288),
        'num_folds': args.total_folds,
        'current_fold': args.kfold,
        'verbose': config.misc.get('logging_level', 'INFO') == 'DEBUG',
        'device': device, # Pass the determined device
        'pin_memory': False, # Keep False for CV dataset internal loading
        'prefetch_factor': 1, # Keep 1 for CV dataset internal loading
        'num_workers': 12, # Keep 0 for CV dataset internal loading
        'require_teacher_logits': use_kd_loss,
        'use_cache': not args.disable_cache,
        'metadata_only': args.metadata_cache,
        # 'cache_fraction': config.data.get('cache_fraction', args.cache_fraction), # CV dataset doesn't support fraction
        'lazy_init': args.lazy_init,
        'batch_gpu_cache': args.batch_gpu_cache,
        'small_dataset_percentage': args.small_dataset,
        # 'dataset_type': 'labeled_synth' # Handled internally by CV dataset
        'cache_file_prefix': os.path.join(cache_dir, "cv_cache") if cache_dir else "cv_cache" # Use cache_dir
    }

    logger.info("Creating CrossValidationDataset with the following parameters:")
    for key, value in dataset_args.items():
        log_val = value
        logger.info(f"  {key}: {log_val}")

    try:
        cv_dataset = CrossValidationDataset(**dataset_args)
    except Exception as e:
        logger.error(f"Failed to initialize CrossValidationDataset: {e}")
        logger.error(traceback.format_exc())
        return

    # Create data loaders using the dataset's methods
    batch_size = config.training.get('batch_size', 16)
    logger.info(f"Using batch size: {batch_size}")

    # Use dataset's iterators which handle subsets and correct getitem calls
    # Ensure num_workers=0 and pin_memory=False for DataLoaders with this dataset structure
    dataloader_num_workers = 0
    dataloader_pin_memory = False

    train_loader = cv_dataset.get_train_iterator(
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
    )
    val_loader = cv_dataset.get_val_iterator(
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory
    )
    # Use val_loader as the test_loader for this fold
    test_loader = val_loader

    logger.info(f"Training set (Fold {args.kfold}): {len(cv_dataset.train_segment_indices)} segments ({len(train_loader)} batches)")
    logger.info(f"Validation set (Fold {args.kfold}): {len(cv_dataset.val_segment_indices)} segments ({len(val_loader)} batches)")

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
        n_classes = 25 # Default small
        logger.info(f"Using small vocabulary with {n_classes} output classes")

    # Determine model type and pretrained path
    model_type = args.model_type
    btc_config = None

    # Determine the path for initial model weights
    # First check if initial_model_ckpt_path is provided
    pretrained_path = args.initial_model_ckpt_path

    # If not, fall back to load_checkpoint if it's a valid file path (not a keyword)
    if not pretrained_path and args.load_checkpoint and args.load_checkpoint not in ["auto", "never", "required"]:
        pretrained_path = args.load_checkpoint
        logger.info(f"Using --load_checkpoint ({pretrained_path}) as initial model weights path.")

    if model_type == 'BTC':
        # Load BTC Config
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

        if pretrained_path:
            logger.info(f"\n=== Loading BTC model from {pretrained_path} for training ===")
        else:
            logger.info("\n=== No BTC checkpoint specified, will initialize a fresh BTC model ===")
    elif pretrained_path:
        logger.info(f"\n=== Loading ChordNet model from {pretrained_path} for training ===")
    else:
        logger.error(f"No checkpoint specified via --initial_model_ckpt_path. Please provide a checkpoint to load.")
        return

    # Create model instance
    optimizer_state_dict_to_load = None

    # Load Normalization from Teacher Checkpoint
    mean_val, std_val = load_normalization_from_checkpoint(
        args.teacher_checkpoint or config.paths.get('teacher_checkpoint'),
        storage_root, project_root
    )
    normalization_params = {'mean': mean_val, 'std': std_val}
    logger.info(f"Using normalization parameters FOR TRAINING (from teacher checkpoint): mean={mean_val:.4f}, std={std_val:.4f}")

    # Convert normalization floats to tensors for Trainer
    trainer_normalization = {
        'mean': torch.tensor(normalization_params['mean'], device=device, dtype=torch.float32),
        'std': torch.tensor(normalization_params['std'], device=device, dtype=torch.float32)
    }

    try:
        n_freq = getattr(config.feature, 'n_bins', 144)
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
            # --- ChordNet Creation (aligned with train_finetune) ---
            f_layer = config.model.get('f_layer', 3)
            f_head = config.model.get('f_head', 6)
            t_layer = config.model.get('t_layer', 3)
            t_head = config.model.get('t_head', 6)
            d_layer = config.model.get('d_layer', 3)
            d_head = config.model.get('d_head', 6)
            logger.info(f"ChordNet params: f_layer={f_layer}, f_head={f_head}, t_layer={t_layer}, t_head={t_head}, d_layer={d_layer}, d_head={d_head}")
            model = ChordNet(
                n_freq=n_freq,
                n_classes=n_classes,
                n_group=n_group,
                f_layer=f_layer,
                f_head=f_head,
                t_layer=t_layer,
                t_head=t_head,
                d_layer=d_layer,
                d_head=d_head,
                dropout=dropout_rate
            ).to(device)
            # --- End ChordNet Creation ---
        else: # BTC model
            # --- BTC Creation (aligned with train_finetune) ---
            if btc_config is None or not hasattr(btc_config, 'model'):
                 logger.error("BTC config or its 'model' section was not loaded correctly. Cannot initialize BTC model.")
                 return

            model_config_dict = btc_config.model
            if not isinstance(model_config_dict, dict):
                logger.error(f"Expected btc_config.model to be a dictionary, but got {type(model_config_dict)}. Check btc_config.yaml structure.")
                return

            # Override parameters
            model_config_dict['num_chords'] = n_classes
            logger.info(f"Overriding BTC model num_chords to: {n_classes}")

            dropout_keys_found = []
            if 'input_dropout' in model_config_dict: model_config_dict['input_dropout'] = dropout_rate; dropout_keys_found.append('input_dropout')
            if 'layer_dropout' in model_config_dict: model_config_dict['layer_dropout'] = dropout_rate; dropout_keys_found.append('layer_dropout')
            if 'attention_dropout' in model_config_dict: model_config_dict['attention_dropout'] = dropout_rate; dropout_keys_found.append('attention_dropout')
            if 'relu_dropout' in model_config_dict: model_config_dict['relu_dropout'] = dropout_rate; dropout_keys_found.append('relu_dropout')

            if dropout_keys_found: logger.info(f"Overriding BTC model dropout rates ({', '.join(dropout_keys_found)}) to: {dropout_rate}")
            else: logger.warning("Could not find dropout keys in btc_config.model dictionary.")

            # Apply model scale to hidden_size for BTC
            hidden_size_base = model_config_dict.get('hidden_size', 128)
            model_scale_btc = float(args.model_scale) if args.model_scale is not None else float(config.model.get('scale', 1.0))
            hidden_size = max(32, int(hidden_size_base * model_scale_btc))
            model_config_dict['hidden_size'] = hidden_size
            logger.info(f"Applying scale {model_scale_btc} to BTC hidden_size: {hidden_size_base} -> {hidden_size}")

            # Ensure feature_size matches n_freq
            model_config_dict['feature_size'] = n_freq
            logger.info(f"Setting BTC feature_size to: {n_freq}")

            logger.info(f"Initializing BTC model with num_chords={model_config_dict.get('num_chords', 'N/A')}, hidden_size={hidden_size}, dropout={dropout_rate}")
            model = BTC_model(config=model_config_dict).to(device)
            # --- End BTC Creation ---

        # Attach chord mapping and normalization to model
        model.idx_to_chord = master_mapping
        model.normalization_mean = torch.tensor(normalization_params['mean'], device=device, dtype=torch.float32)
        model.normalization_std = torch.tensor(normalization_params['std'], device=device, dtype=torch.float32)
        logger.info("Attached chord mapping and normalization parameters to model")

        # Load pretrained weights AND potentially optimizer state
        if pretrained_path: # This condition checks if a path was determined
            resolved_load_path = resolve_path(pretrained_path, storage_root, project_root)
            if os.path.exists(resolved_load_path):
                logger.info(f"Loading checkpoint from: {resolved_load_path}")
                try:
                    checkpoint = torch.load(resolved_load_path, map_location=device) # Actual model loading
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

                    if all(k.startswith('module.') for k in state_dict.keys()):
                        logger.info("Detected 'module.' prefix in state dict keys. Removing prefix.")
                        state_dict = {k[7:]: v for k, v in state_dict.items()}

                    model.load_state_dict(state_dict, strict=not args.partial_loading)
                    logger.info(f"Successfully loaded model weights from {resolved_load_path}")

                    if not args.reset_epoch:
                        if 'optimizer_state_dict' in checkpoint:
                            optimizer_state_dict_to_load = checkpoint['optimizer_state_dict']
                            logger.info(f"Found optimizer state in {resolved_load_path}. Will load if trainer doesn't load its own state.")
                        else:
                            logger.warning(f"Resuming requested (reset_epoch=False), but no optimizer state found in {resolved_load_path}.")
                    else:
                        logger.info("Reset flags active (--reset_epoch). Ignoring optimizer state from checkpoint file.")

                    del checkpoint, state_dict
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error loading checkpoint: {e}")
                    # Decide if we should continue with random weights (maybe only for BTC?)
                    if model_type == 'BTC':
                        logger.warning("Continuing with freshly initialized BTC model due to loading error.")
                    else:
                        logger.error("Cannot continue without loading weights for ChordNet model.")
                        return
            else:
                logger.warning(f"Specified checkpoint not found: {resolved_load_path}. Starting from scratch (if BTC) or failing (if ChordNet).")
                if model_type != 'BTC': return # Fail if ChordNet needs weights

        # Freeze feature extraction layers if requested
        if args.freeze_feature_extractor:
            logger.info("Freezing feature extraction layers:")
            frozen_count = 0
            for name, param in model.named_parameters():
                freeze_condition = False
                if model_type == 'ChordNet' and ('frequency_net' in name or 'prenet' in name): freeze_condition = True
                elif model_type == 'BTC' and ('conv1' in name or 'conv_layers' in name): freeze_condition = True # Example

                if freeze_condition:
                    param.requires_grad = False
                    logger.info(f"  Frozen: {name}")
                    frozen_count += 1

            if frozen_count == 0: logger.warning("Freeze feature extractor requested, but no layers matched criteria.")
            else:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")

    except Exception as e:
        logger.error(f"Error creating or loading model: {e}")
        logger.error(traceback.format_exc())
        return

    # Create optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training['learning_rate'],
        weight_decay=config.training.get('weight_decay', 0.0)
    )

    # Load optimizer state if found and resuming
    if optimizer_state_dict_to_load:
        try:
            optimizer.load_state_dict(optimizer_state_dict_to_load)
            logger.info("Successfully loaded optimizer state from checkpoint.")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor): state[k] = v.to(device)
            logger.info("Moved optimizer state to current device.")
        except Exception as e:
            logger.error(f"Error loading optimizer state from checkpoint: {e}. Using fresh optimizer state.")
            optimizer_state_dict_to_load = None

    # Clean up GPU memory before training
    if torch.cuda.is_available():
        logger.info("Performing CUDA memory cleanup before training")
        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"CUDA memory stats (GB): allocated={allocated:.2f}, reserved={reserved:.2f}")

    # Handle LR schedule
    lr_schedule_type = args.lr_schedule or config.training.get('lr_schedule', 'validation')
    if lr_schedule_type in ['validation', 'none']: lr_schedule_type = None

    # Create trainer
    use_warmup_value = config.training.get('use_warmup', False)
    warmup_epochs = int(config.training.get('warmup_epochs', 10)) if use_warmup_value else None
    warmup_start_lr = float(config.training.get('warmup_start_lr')) if use_warmup_value else None
    warmup_end_lr = float(config.training.get('warmup_end_lr')) if use_warmup_value else None

    logger.info(f"Creating trainer with use_warmup={use_warmup_value}")
    if use_warmup_value: logger.info(f"Warmup configuration: {warmup_epochs} epochs from {warmup_start_lr} to {warmup_end_lr}")

    trainer = StudentTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=int(config.training.get('num_epochs', 50)),
        logger=logger,
        checkpoint_dir=checkpoints_dir,
        class_weights=None,
        idx_to_chord=master_mapping,
        normalization=trainer_normalization, # Pass the checkpoint-based normalization tensors
        early_stopping_patience=int(config.training.get('early_stopping_patience', 10)),
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
        reset_epoch=args.reset_epoch,
        reset_scheduler=args.reset_scheduler
    )

    # Attach chord mapping to trainer (chord -> idx)
    trainer.set_chord_mapping(chord_mapping)

    # Log checkpoint loading status for resuming
    latest_checkpoint_path = os.path.join(checkpoints_dir, "trainer_state_latest.pth")
    will_trainer_load_internal_state = os.path.exists(latest_checkpoint_path) and not args.reset_epoch

    if will_trainer_load_internal_state:
         logger.info(f"Trainer found existing internal checkpoint '{latest_checkpoint_path}'. Attempting to resume training.")
         if optimizer_state_dict_to_load: logger.warning("Optimizer state loaded from external checkpoint might be overwritten.")
    else:
        if not os.path.exists(latest_checkpoint_path): logger.info(f"No suitable internal trainer checkpoint found at '{latest_checkpoint_path}'.")
        if args.reset_epoch: logger.info("Reset flags active (--reset_epoch).")
        logger.info("Starting training from scratch (epoch 1) after loading external weights.")
        if optimizer_state_dict_to_load: logger.info("Using optimizer state loaded from the external checkpoint.")
        else: logger.info("Using a fresh optimizer state.")

    # Run training
    logger.info(f"\n=== Starting training for Fold {args.kfold} ===")
    try:
        # Verify KD setup if enabled
        if use_kd_loss:
            logger.info("Verifying offline knowledge distillation setup...")
            try:
                sample_batch = next(iter(train_loader))
                if 'teacher_logits' in sample_batch:
                    logger.info(f"✅ Teacher logits found in batch with shape: {sample_batch['teacher_logits'].shape}")
                else:
                    logger.warning("⚠️ No teacher logits found in the batch. KD will not work.")
            except Exception as e:
                logger.error(f"Error verifying KD setup: {e}")

        # Start training
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"ERROR during training: {e}")
        logger.error(traceback.format_exc())

    # Final evaluation on validation set (which is the test set for this fold)
    logger.info(f"\n=== Testing (Validation Fold {args.kfold}) ===")
    try:
        if trainer.load_best_model():
            logger.info("Evaluating using best model checkpoint for this fold.")

            # Basic testing with Tester class
            tester = Tester(
                model=model,
                test_loader=test_loader, # Use the validation loader for this fold
                device=device,
                idx_to_chord=master_mapping,
                normalization=trainer_normalization,
                output_dir=checkpoints_dir,
                logger=logger
            )
            test_metrics = tester.evaluate(save_plots=True)

            # Save test metrics for this fold
            try:
                metrics_path = os.path.join(checkpoints_dir, f"test_metrics_fold{args.kfold}.json")
                with open(metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Test metrics for fold {args.kfold} saved to {metrics_path}")
            except Exception as e:
                logger.error(f"Error saving test metrics for fold {args.kfold}: {e}")

            # Advanced MIR evaluation using frame-level data from CrossValidationDataset
            logger.info(f"\n=== MIR evaluation (Validation Fold {args.kfold}) ===")
            all_preds_idx = []
            all_targets_idx = []
            model.eval()
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"MIR Eval Fold {args.kfold}"):
                    inputs = batch['spectro'].to(device)
                    targets = batch['chord_idx'].to(device) # Shape: (batch, seq_len)

                    # Apply normalization (using model's attached norm params)
                    if hasattr(model, 'normalization_mean') and hasattr(model, 'normalization_std'):
                        inputs = (inputs - model.normalization_mean) / model.normalization_std
                    elif trainer_normalization and 'mean' in trainer_normalization and 'std' in trainer_normalization:
                         inputs = (inputs - trainer_normalization['mean']) / trainer_normalization['std']

                    outputs = model(inputs) # Shape: (batch, seq_len, n_classes)
                    if isinstance(outputs, tuple): logits = outputs[0]
                    else: logits = outputs

                    preds = logits.argmax(dim=-1) # Shape: (batch, seq_len)

                    # Flatten batch and sequence dimensions
                    all_preds_idx.extend(preds.view(-1).cpu().numpy())
                    all_targets_idx.extend(targets.view(-1).cpu().numpy())

            # Convert indices to chord labels using master_mapping
            all_prediction_labels_std = [master_mapping.get(idx, 'N') for idx in all_preds_idx]
            all_reference_labels_std = [master_mapping.get(idx, 'N') for idx in all_targets_idx]

            mir_eval_results = {}
            if all_reference_labels_std and all_prediction_labels_std:
                logger.info(f"Calculating final MIR scores using {len(all_reference_labels_std)} aggregated frames for Fold {args.kfold}...")
                try:
                    # Create dummy timestamps and use calculate_chord_scores
                    frame_duration = config.feature.get('hop_duration', 0.09288)
                    num_frames = len(all_reference_labels_std)
                    timestamps = np.arange(num_frames) * frame_duration

                    scores_tuple = calculate_chord_scores(
                        timestamps, frame_duration,
                        all_reference_labels_std, all_prediction_labels_std
                    )

                    score_names = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
                    mir_eval_results = {name: score for name, score in zip(score_names, scores_tuple)}
                    logger.info(f"Detailed MIR scores (Fold {args.kfold}): {mir_eval_results}")

                    # Calculate frame-wise accuracy
                    correct_frames = sum(1 for ref, pred in zip(all_reference_labels_std, all_prediction_labels_std) if ref == pred)
                    total_frames = len(all_reference_labels_std)
                    frame_accuracy = correct_frames / total_frames if total_frames > 0 else 0
                    mir_eval_results['frame_accuracy'] = frame_accuracy
                    logger.info(f"Frame-wise Accuracy (Standardized, Fold {args.kfold}): {frame_accuracy:.4f}")

                    # --- Add Individual Chord Quality Accuracy ---
                    logger.info(f"\n--- Chord Quality Accuracy (Validation Fold {args.kfold}) ---")
                    ind_acc, quality_stats = compute_individual_chord_accuracy(
                        all_reference_labels_std,
                        all_prediction_labels_std
                    )

                except Exception as mir_calc_error:
                    logger.error(f"Failed to calculate detailed MIR scores for Fold {args.kfold}: {mir_calc_error}")
                    logger.error(traceback.format_exc())
                    mir_eval_results['error'] = f"MIR calculation failed: {mir_calc_error}"
            else:
                logger.warning(f"No reference or prediction labels collected for MIR evaluation (Fold {args.kfold}).")
                mir_eval_results = {'error': 'No labels collected'}

            # Save MIR-eval metrics for this fold
            try:
                mir_eval_path = os.path.join(checkpoints_dir, f"mir_eval_metrics_fold{args.kfold}.json")
                with open(mir_eval_path, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    serializable_results = {}
                    for key, value in mir_eval_results.items():
                        if isinstance(value, (np.float32, np.float64)):
                            serializable_results[key] = float(value)
                        elif isinstance(value, (np.int32, np.int64)):
                             serializable_results[key] = int(value)
                        else:
                            serializable_results[key] = value
                    json.dump(serializable_results, f, indent=2)
                logger.info(f"MIR evaluation metrics for fold {args.kfold} saved to {mir_eval_path}")
            except Exception as e:
                 logger.error(f"Error saving MIR evaluation metrics for fold {args.kfold}: {e}")

        else:
            logger.warning(f"Could not load best model for testing fold {args.kfold}")
    except Exception as e:
        logger.error(f"Error during testing for fold {args.kfold}: {e}")
        logger.error(traceback.format_exc())

    # Save the final model for the fold
    try:
        save_path = os.path.join(checkpoints_dir, f"final_model_fold{args.kfold}.pth")
        mean_to_save = normalization_params['mean']
        std_to_save = normalization_params['std']
        if hasattr(mean_to_save, 'cpu'): mean_to_save = mean_to_save.cpu().numpy()
        if hasattr(std_to_save, 'cpu'): std_to_save = std_to_save.cpu().numpy()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'chord_mapping': chord_mapping,
            'idx_to_chord': master_mapping,
            'mean': float(mean_to_save),
            'std': float(std_to_save),
            'n_classes': n_classes,
            'model_type': model_type,
            'fold': args.kfold, # Add fold info
            'total_folds': args.total_folds # Add total folds info
        }, save_path)
        logger.info(f"Final model for fold {args.kfold} saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving final model for fold {args.kfold}: {e}")

    logger.info(f"Cross-validation training and evaluation for fold {args.kfold} complete!")

if __name__ == '__main__':
    # Set start method for multiprocessing if necessary
    try:
        if sys.platform.startswith('win'):
             multiprocessing.set_start_method('spawn', force=True)
        else:
             multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        current_method = multiprocessing.get_start_method(allow_none=True)
        logger.info(f"Multiprocessing start method already set to '{current_method}'.")
        pass
    except Exception as e:
        logger.warning(f"Could not set multiprocessing start method: {e}")

    main()
