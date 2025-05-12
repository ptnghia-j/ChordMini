import multiprocessing
import sys
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import argparse
import glob
import gc
import traceback
import json
import datetime
from pathlib import Path
from collections import Counter

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.device import get_device, is_cuda_available, is_gpu_available, clear_gpu_cache
from modules.data.SynthDataset import SynthDataset, SynthSegmentSubset
# Import BTC model instead of ChordNet
from modules.models.Transformer.btc_model import BTC_model
from modules.training.StudentTrainer import StudentTrainer
from modules.training.DistributedStudentTrainer import DistributedStudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord
from modules.training.Tester import Tester
from modules.utils.file_utils import count_files_in_subdirectories, find_sample_files, resolve_path, load_normalization_from_checkpoint

# Using utility functions from modules.utils.file_utils

# Note: find_data_directory is available from modules.utils.file_utils if needed

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a BTC chord recognition model")
    # Point to btc_config.yaml by default
    parser.add_argument('--config', type=str, default='./config/btc_config.yaml',
                        help='Path to the BTC configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config value)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides config value)')
    # Model type defaults to BTC
    parser.add_argument('--model', type=str, default='BTC',
                        help='Model type for evaluation (should be BTC)')
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

    # Add distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--distributed_backend', type=str, default='nccl',
                       help='Distributed backend (nccl, gloo, etc.)')
    parser.add_argument('--world_size', type=int, default=None,
                       help='Number of processes for distributed training')
    parser.add_argument('--rank', type=int, default=None,
                       help='Rank of the current process')
    parser.add_argument('--local_rank', type=int, default=None,
                       help='Local rank of the current process')
    parser.add_argument('--dist_url', type=str, default='env://',
                       help='URL used to set up distributed training')

    # Add focal loss arguments
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss to handle class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss (default: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=None,
                       help='Alpha parameter for focal loss (default: None)')

    # Add knowledge distillation arguments (can still be used if teacher logits are available)
    parser.add_argument('--use_kd_loss', action='store_true',
                       help='Use knowledge distillation loss (teacher logits must be in batch data)')
    parser.add_argument('--kd_alpha', type=float, default=0.5,
                       help='Weight for knowledge distillation loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for softening distributions (default: 1.0)')
    parser.add_argument('--logits_dir', type=str, default=None,
                       help='Directory containing teacher logits (required for KD)')

    # BTC model doesn't use scale, but dropout is in btc_config
    # parser.add_argument('--model_scale', type=float, default=None, ...) # Removed
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout probability (overrides config value, affects input, layer, attention, relu)')

    # Dataset caching behavior
    parser.add_argument('--disable_cache', action='store_true',
                      help='Disable dataset caching to reduce memory usage')
    parser.add_argument('--metadata_cache', action='store_true',
                      help='Only cache metadata (not spectrograms) to reduce memory usage')
    parser.add_argument('--cache_fraction', type=float, default=0.1,
                      help='Fraction of dataset to cache (default: 0.1 = 10%%)')
    parser.add_argument('--lazy_init', action='store_true',
                      help='Use lazy initialization for dataset to reduce memory usage')

    # Data directories override
    parser.add_argument('--spec_dir', type=str, default=None,
                      help='Directory containing spectrograms (overrides config value)')
    parser.add_argument('--label_dir', type=str, default=None,
                      help='Directory containing labels (overrides config value)')

    # GPU acceleration options
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.9,
                      help='Fraction of GPU memory to use (default: 0.9)')
    parser.add_argument('--batch_gpu_cache', action='store_true',
                      help='Cache batches on GPU for repeated access patterns')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                      help='Number of batches to prefetch (default: 2)')

    # Small dataset percentage
    parser.add_argument('--small_dataset', type=float, default=None,
                      help='Use only a small percentage of dataset for quick testing (e.g., 0.01 for 1%%)')

    # Learning rate arguments
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Base learning rate (overrides config value)')
    parser.add_argument('--min_learning_rate', type=float, default=None,
                        help='Minimum learning rate for schedulers (default: 0)') # BTC config doesn't specify min_lr
    parser.add_argument('--warmup_end_lr', type=float, default=None,
                       help='Target learning rate at the end of warm-up (default: base LR)')

    # Modify dataset_type choices to include 'dali_synth', 'combined', and pairwise combinations
    parser.add_argument('--dataset_type', type=str,
                      choices=['fma', 'maestro', 'dali_synth', 'combined',
                               'fma+maestro', 'fma+dali_synth', 'maestro+dali_synth'],
                      default='fma',
                      help='Dataset format type: fma, maestro, dali_synth, combined (all), or pairwise combinations (fma+maestro, fma+dali_synth, maestro+dali_synth)')

    # Checkpoint loading
    parser.add_argument('--load_checkpoint', type=str, default=None,
                      help='Path to checkpoint file to resume training from')
    parser.add_argument('--reset_epoch', action='store_true',
                      help='Start from epoch 1 even when loading from checkpoint')
    parser.add_argument('--reset_scheduler', action='store_true',
                      help='Reset learning rate scheduler when --reset_epoch is used')
    parser.add_argument('--timeout_minutes', type=int, default=90,
                      help='Timeout in minutes for distributed operations (default: 90)')
    # Add teacher checkpoint argument
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to the teacher model checkpoint for loading normalization stats')


    args = parser.parse_args()

    # Load configuration from YAML first (btc_config.yaml)
    config = HParams.load(args.config)

    # --- Merge relevant parts from student_config structure if needed ---
    # Example: Ensure training section exists for compatibility with trainer
    if not hasattr(config, 'training'):
        config.training = {}
    if not hasattr(config, 'misc'):
        config.misc = {}
    if not hasattr(config, 'paths'):
        config.paths = {}
    if not hasattr(config, 'data'):
        config.data = {}

    # Override config with dataset_type if specified
    config.data['dataset_type'] = args.dataset_type

    # Set up distributed training if enabled (same as train_student.py)
    distributed_training = args.distributed
    world_size = 1
    rank = 0
    local_rank = 0

    if distributed_training:
        # Initialize distributed environment
        if args.local_rank is not None:
            # Single-node multi-GPU training with torch.distributed.launch
            local_rank = args.local_rank
            rank = args.local_rank
            world_size = torch.cuda.device_count()
            logger.info(f"Initializing distributed training with local_rank={local_rank}, world_size={world_size}")
            # guard invalid GPU ordinal
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                logger.warning("No CUDA devices available, falling back to CPU")
                device = torch.device('cpu')
            else:
                if local_rank >= gpu_count:
                    logger.warning(f"local_rank {local_rank} ≥ available GPUs ({gpu_count}), defaulting to GPU 0")
                    local_rank = 0
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
            # initialize process group via env:// only
            dist.init_process_group(
                backend=args.distributed_backend,
                init_method=args.dist_url
            )

        elif args.rank is not None and args.world_size is not None:
            rank = args.rank
            world_size = args.world_size
            logger.info(f"Initializing distributed training with rank={rank}, world_size={world_size}")
            dist.init_process_group(backend=args.distributed_backend,
                                   init_method=args.dist_url,
                                   world_size=world_size,
                                   rank=rank)
            # compute and guard local_rank
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                logger.warning("No CUDA devices available, falling back to CPU")
                local_rank = 0
                device = torch.device('cpu')
            else:
                local_rank = rank % gpu_count
                if local_rank >= gpu_count:
                    logger.warning(f"Computed local_rank {local_rank} ≥ available GPUs ({gpu_count}), defaulting to GPU 0")
                    local_rank = 0
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")

        else:
            # Auto-detect number of GPUs for single-node training
            world_size = torch.cuda.device_count()
            if world_size > 1:
                logger.info(f"Initializing distributed training with {world_size} GPUs")
                mp.spawn(distributed_main,
                         args=(world_size, args),
                         nprocs=world_size,
                         join=True)
                return
            else:
                logger.info("Only one GPU detected, disabling distributed training")
                distributed_training = False

    # Then check device availability for non-distributed training
    if not distributed_training:
        # Use CUDA if available, otherwise CPU
        if is_cuda_available():
            device = get_device()
            logger.info(f"Using CUDA for training on device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for training")

    # Override config values with command line arguments if provided
    config.misc['seed'] = args.seed or config.misc.get('seed', 42)

    # Set centralized checkpoint directory outside of ChordMini
    default_checkpoints_dir = '/mnt/storage/checkpoints/btc'
    # Fall back to local path if centralized directory doesn't exist
    if not os.path.exists(os.path.dirname(default_checkpoints_dir)):
        default_checkpoints_dir = 'checkpoints/btc'

    config.paths['checkpoints_dir'] = args.save_dir or config.paths.get('ckpt_path', default_checkpoints_dir)
    config.paths['storage_root'] = args.storage_root or config.paths.get('root_path', None) # Use root_path from btc_config

    # Handle learning rate and warmup parameters from btc_config and args
    # Use experiment section from btc_config
    config.training['learning_rate'] = args.learning_rate or config.experiment.get('learning_rate', 0.0001)
    # btc_config doesn't have min_lr, default to 0 or use arg
    config.training['min_learning_rate'] = args.min_learning_rate or 0.0

    # Warmup args override config (btc_config doesn't have warmup)
    config.training['use_warmup'] = args.use_warmup
    config.training['warmup_epochs'] = args.warmup_epochs or 5 # Default warmup epochs if enabled
    config.training['warmup_start_lr'] = args.warmup_start_lr or config.training['learning_rate'] / 10
    config.training['warmup_end_lr'] = args.warmup_end_lr or config.training['learning_rate']

    # Log parameters that have been overridden
    logger.info(f"Using learning rate: {config.training['learning_rate']}")
    logger.info(f"Using minimum learning rate: {config.training['min_learning_rate']}")
    if config.training['use_warmup']:
        logger.info(f"Using warmup_epochs: {config.training['warmup_epochs']}")
        logger.info(f"Using warmup_start_lr: {config.training.get('warmup_start_lr')}")
        logger.info(f"Using warmup_end_lr: {config.training.get('warmup_end_lr')}")

    # Log training configuration
    logger.info("\n=== Training Configuration ===")
    logger.info(f"Model type: {args.model}")


    # Log knowledge distillation settings
    use_kd = args.use_kd_loss or config.training.get('use_kd_loss', False)
    use_kd = str(use_kd).lower() == "true"
    kd_alpha = args.kd_alpha
    temperature = args.temperature

    if use_kd:
        logger.info("\n=== Knowledge Distillation Enabled ===")
        logger.info(f"KD alpha: {kd_alpha} (weighting between KD and CE loss)")
        logger.info(f"Temperature: {temperature} (for softening distributions)")
    else:
        logger.info("Knowledge distillation is disabled, using standard loss")

    # Log focal loss settings
    # --- MODIFICATION START: Ensure use_focal is boolean ---
    use_focal = args.use_focal_loss or config.training.get('use_focal_loss', False)
    use_focal = str(use_focal).lower() == "true"
    # --- MODIFICATION END ---
    focal_gamma = args.focal_gamma
    focal_alpha = args.focal_alpha
    if use_focal:
        logger.info("\n=== Focal Loss Enabled ===")
        logger.info(f"Gamma: {focal_gamma}")
        if focal_alpha:
            logger.info(f"Alpha: {focal_alpha}")
    else:
        logger.info("Using standard cross-entropy loss")

    # Clear summary of loss function configuration
    if use_kd and use_focal:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha}) combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * focal_loss")
        logger.info(f"Note: When teacher logits are not available for a batch, only focal loss will be used")
    elif use_kd:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using standard Cross Entropy combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * cross_entropy")
        logger.info(f"Note: When teacher logits are not available for a batch, only cross entropy will be used")
    elif use_focal:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using only Focal Loss with gamma={focal_gamma}, alpha={focal_alpha}")
    else:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info("Using only standard Cross Entropy Loss")

    # Initialize dataset_args dictionary
    dataset_args = {}

    dataset_args['small_dataset_percentage'] = args.small_dataset
    if dataset_args['small_dataset_percentage'] is None or (isinstance(dataset_args['small_dataset_percentage'], str) and dataset_args['small_dataset_percentage'].lower() in ["null", "none", ""]):
        dataset_args['small_dataset_percentage'] = None
        logger.info("Using full dataset (small_dataset_percentage is None)")
    else:
        try:
            dataset_args['small_dataset_percentage'] = float(dataset_args['small_dataset_percentage'])
            logger.info(f"Using {dataset_args['small_dataset_percentage']*100:.1f}% of dataset")
        except ValueError:
            logger.error(f"Invalid small_dataset_percentage: {dataset_args['small_dataset_percentage']}")
            dataset_args['small_dataset_percentage'] = None
            logger.info("Falling back to using full dataset")

    # Set dataset type
    dataset_args['dataset_type'] = config.data.get('dataset_type', 'fma')

    # Set random seed for reproducibility
    if hasattr(config.misc, 'seed'):
        seed = config.misc['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")

    # Set up logging
    logger.logging_verbosity(config.misc.get('logging_level', 1))

    # Get project root and storage root
    project_root = os.path.dirname(os.path.abspath(__file__))
    storage_root = config.paths.get('storage_root', None)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Storage root: {storage_root}")

    logger.info(f"Looking for data files:")

    # Define standard paths for all datasets
    data_root = os.environ.get('DATA_ROOT', '/mnt/storage/data')

    # Standard paths for FMA dataset
    fma_spec_dir = os.path.join(data_root, "logits/synth/spectrograms")
    fma_label_dir = os.path.join(data_root, "logits/synth/labels")
    fma_logits_dir = os.path.join(data_root, "logits/synth/logits")

    # Standard paths for Maestro dataset
    maestro_spec_dir = os.path.join(data_root, "logits/maestro_synth/spectrograms")
    maestro_label_dir = os.path.join(data_root, "logits/maestro_synth/labels")
    maestro_logits_dir = os.path.join(data_root, "logits/maestro_synth/logits")

    # Standard paths for DALI dataset
    dali_spec_dir = os.path.join(data_root, "dali_synth/spectrograms")
    dali_label_dir = os.path.join(data_root, "dali_synth/labels")
    dali_logits_dir = os.path.join(data_root, "dali_synth/logits")

    # IMPORTANT: We're intentionally NOT using LabeledDataset_augmented here to match train_student.py behavior
    # The issue is that SynthDataset is still scanning LabeledDataset_augmented even when not explicitly included

    # Determine active dataset types
    active_types = []
    if args.dataset_type == 'combined':
        # Match train_student.py behavior - only use fma, maestro, dali_synth
        # Explicitly NOT including 'labeled' or 'labeled_synth' to avoid LabeledDataset_augmented
        active_types = ['fma', 'maestro', 'dali_synth']
    elif '+' in args.dataset_type:
        # Split the dataset types and filter out any that might include 'labeled' or 'labeled_synth'
        types = args.dataset_type.split('+')
        active_types = [t for t in types if t not in ['labeled', 'labeled_synth']]
        # If we filtered out any types, log a warning
        if len(active_types) != len(types):
            logger.warning(f"Filtered out 'labeled' and 'labeled_synth' from dataset types to avoid LabeledDataset_augmented")
    else:
        # If the dataset type is 'labeled' or 'labeled_synth', use 'fma' instead
        if args.dataset_type in ['labeled', 'labeled_synth']:
            logger.warning(f"Replacing dataset type '{args.dataset_type}' with 'fma' to avoid LabeledDataset_augmented")
            active_types = ['fma']
        else:
            active_types = [args.dataset_type]

    logger.info(f"Active dataset types: {active_types}")

    # Initialize lists for combined paths
    spec_dirs_list = []
    label_dirs_list = []
    logits_dirs_list = [] # Only populated if use_kd is True

    # Collect paths based on active types
    if 'fma' in active_types:
        # If FMA is active, args.spec_dir (etc.) are considered FMA overrides if provided
        effective_fma_spec_dir = args.spec_dir or fma_spec_dir
        effective_fma_label_dir = args.label_dir or fma_label_dir

        resolved_spec_dir = resolve_path(effective_fma_spec_dir, storage_root, project_root)
        resolved_label_dir = resolve_path(effective_fma_label_dir, storage_root, project_root)
        spec_dirs_list.append(resolved_spec_dir)
        label_dirs_list.append(resolved_label_dir)

        if use_kd:
            effective_fma_logits_dir = args.logits_dir or fma_logits_dir
            resolved_logits_dir = resolve_path(effective_fma_logits_dir, storage_root, project_root)
            logits_dirs_list.append(resolved_logits_dir)

        fma_spec_count = count_files_in_subdirectories(resolved_spec_dir, "*_spec.npy")
        fma_label_count = count_files_in_subdirectories(resolved_label_dir, "*.lab")
        logger.info(f"  FMA: {fma_spec_count} specs, {fma_label_count} labels at {resolved_spec_dir}")

    if 'maestro' in active_types:
        # For Maestro, args.spec_dir (etc.) apply only if dataset_type is 'maestro' or 'combined'
        effective_maestro_spec_dir = maestro_spec_dir
        if args.spec_dir and (args.dataset_type == 'maestro' or args.dataset_type == 'combined'):
            effective_maestro_spec_dir = args.spec_dir

        effective_maestro_label_dir = maestro_label_dir
        if args.label_dir and (args.dataset_type == 'maestro' or args.dataset_type == 'combined'):
            effective_maestro_label_dir = args.label_dir

        resolved_spec_dir = resolve_path(effective_maestro_spec_dir, storage_root, project_root)
        resolved_label_dir = resolve_path(effective_maestro_label_dir, storage_root, project_root)
        spec_dirs_list.append(resolved_spec_dir)
        label_dirs_list.append(resolved_label_dir)

        if use_kd:
            effective_maestro_logits_dir = maestro_logits_dir
            if args.logits_dir and (args.dataset_type == 'maestro' or args.dataset_type == 'combined'):
                effective_maestro_logits_dir = args.logits_dir
            resolved_logits_dir = resolve_path(effective_maestro_logits_dir, storage_root, project_root)
            logits_dirs_list.append(resolved_logits_dir)

        maestro_spec_count = count_files_in_subdirectories(resolved_spec_dir, "*_spec.npy")
        maestro_label_count = count_files_in_subdirectories(resolved_label_dir, "*.lab")
        logger.info(f"  Maestro: {maestro_spec_count} specs, {maestro_label_count} labels at {resolved_spec_dir}")

    if 'dali_synth' in active_types:
        # For DALI, args.spec_dir (etc.) apply only if dataset_type is 'dali_synth' or 'combined'
        effective_dali_spec_dir = dali_spec_dir
        if args.spec_dir and (args.dataset_type == 'dali_synth' or args.dataset_type == 'combined'):
            effective_dali_spec_dir = args.spec_dir

        effective_dali_label_dir = dali_label_dir
        if args.label_dir and (args.dataset_type == 'dali_synth' or args.dataset_type == 'combined'):
            effective_dali_label_dir = args.label_dir

        resolved_spec_dir = resolve_path(effective_dali_spec_dir, storage_root, project_root)
        resolved_label_dir = resolve_path(effective_dali_label_dir, storage_root, project_root)
        spec_dirs_list.append(resolved_spec_dir)
        label_dirs_list.append(resolved_label_dir)

        if use_kd:
            effective_dali_logits_dir = dali_logits_dir
            if args.logits_dir and (args.dataset_type == 'dali_synth' or args.dataset_type == 'combined'):
                effective_dali_logits_dir = args.logits_dir
            resolved_logits_dir = resolve_path(effective_dali_logits_dir, storage_root, project_root)
            logits_dirs_list.append(resolved_logits_dir)

        dali_spec_count = count_files_in_subdirectories(resolved_spec_dir, "*_spec.npy")
        dali_label_count = count_files_in_subdirectories(resolved_label_dir, "*.lab")
        logger.info(f"  DALI Synth: {dali_spec_count} specs, {dali_label_count} labels at {resolved_spec_dir}")

    # Check if any data was found
    if not spec_dirs_list or not label_dirs_list:
         logger.error("No valid data directories found for the specified dataset types. Exiting.")
         sys.exit(1)

    # Assign final paths (use lists for SynthDataset)
    # If only one dataset type, use the string path directly for clarity/backward compatibility if needed by dataset
    spec_dir = spec_dirs_list[0] if len(spec_dirs_list) == 1 else spec_dirs_list
    label_dir = label_dirs_list[0] if len(label_dirs_list) == 1 else label_dirs_list
    # Pass None if KD is off or no logits dirs were found/added
    logits_dir = (logits_dirs_list[0] if len(logits_dirs_list) == 1 else logits_dirs_list) if use_kd and logits_dirs_list else None

    logger.info(f"\nFinal Spectrogram Dirs: {spec_dir}")
    logger.info(f"Final Label Dirs: {label_dir}")
    logger.info(f"Final Logits Dirs: {logits_dir}")

    # Use the mapping defined in chords.py
    master_mapping = idx2voca_chord()
    chord_mapping = {chord: idx for idx, chord in master_mapping.items()}

    # Verify mapping of special chords
    logger.info(f"Mapping of special chords:")
    for special_chord in ["N", "X"]:
        if special_chord in chord_mapping:
            logger.info(f"  {special_chord} chord is mapped to index {chord_mapping[special_chord]}")
        else:
            logger.info(f"  {special_chord} chord is not in the mapping - this may cause issues")

    # Log mapping info
    logger.info(f"\nUsing chord mapping from chords.py with {len(chord_mapping)} unique chords")
    logger.info(f"Sample chord mapping: {dict(list(chord_mapping.items())[:5])}")

    # Resolve checkpoints directory path
    checkpoints_dir_config = config.paths.get('ckpt_path', 'checkpoints/btc') # Use ckpt_path from btc_config
    checkpoints_dir = resolve_path(checkpoints_dir_config, storage_root, project_root)
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")

    # Initialize SynthDataset with optimized settings
    logger.info("\n=== Creating dataset ===")

    # Use values from btc_config where appropriate
    dataset_args.update({
        'spec_dir': spec_dir, # Use the resolved list or string
        'label_dir': label_dir, # Use the resolved list or string
        'logits_dir': logits_dir, # Use the resolved list or string or None
        'chord_mapping': chord_mapping,
        'seq_len': config.model.get('seq_len', 108), # From btc_config.model
        'stride': config.model.get('stride', 108), # From btc_config.model
        'frame_duration': config.model.get('frame_duration', 0.09288), # From btc_config.model
        'verbose': True,
        'device': device,
        'pin_memory': False,
        'prefetch_factor': float(args.prefetch_factor) if args.prefetch_factor else 1,
        'num_workers': 10,
        'require_teacher_logits': use_kd,
        'use_cache': not args.disable_cache, # Use arg directly
        'metadata_only': args.metadata_cache, # Use arg directly
        'cache_fraction': args.cache_fraction, # Use arg directly
        'lazy_init': args.lazy_init, # Use arg directly
        'batch_gpu_cache': args.batch_gpu_cache, # Use arg directly
        'dataset_type': args.dataset_type # Pass the potentially combined type string
    })

    # Create the dataset
    logger.info("Creating dataset with the following parameters:")
    for key, value in dataset_args.items():
        # --- MODIFICATION START ---
        # Add detailed logging for directory paths being passed
        if key in ['spec_dir', 'label_dir', 'logits_dir']:
            logger.info(f"  {key}: {value} (Type: {type(value)})")
        else:
            logger.info(f"  {key}: {value}")
        # --- MODIFICATION END ---

    synth_dataset = SynthDataset(**dataset_args)

    # Create data loaders for each subset
    batch_size = config.experiment.get('batch_size', 128) # From btc_config.experiment
    logger.info(f"Using batch size: {batch_size}")

    if distributed_training:
        # Create train subset using SynthSegmentSubset
        train_subset = SynthSegmentSubset(synth_dataset, synth_dataset.train_indices)
        eval_subset = SynthSegmentSubset(synth_dataset, synth_dataset.eval_indices)

        # Create distributed samplers
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        # Create data loaders with samplers
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle here, the sampler will do it
            sampler=train_sampler,
            num_workers=0,  # Force single worker for GPU optimization
            pin_memory=False
        )

        val_loader = torch.utils.data.DataLoader(
            eval_subset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,  # Force single worker for GPU optimization
            pin_memory=False
        )
    else:
        train_loader = synth_dataset.get_train_iterator(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Force single worker for GPU optimization
            pin_memory=False
        )

        val_loader = synth_dataset.get_eval_iterator(
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Force single worker for GPU optimization
            pin_memory=False
        )

    logger.info("\n=== Checking data loaders ===")
    try:
        batch = next(iter(train_loader))
        # Always expect a dictionary from the DataLoader
        if isinstance(batch, dict) and 'spectro' in batch and 'chord_idx' in batch:
            # --- MODIFICATION: Use correct key 'spectro' ---
            inputs = batch['spectro']
            targets = batch['chord_idx']
            logger.info(f"First batch loaded successfully: inputs shape {inputs.shape}, targets shape {targets.shape}")

            # Check device placement (optional but good practice)
            if torch.cuda.is_available():
                if inputs.device.type == 'cuda':
                    logger.info("Success: Batch tensors are already on GPU (as expected from SynthDataset)")
                else:
                    logger.warning("Warning: Batch tensors are on CPU, expected GPU. Check dataset device setting.")
            else:
                 logger.info("Running on CPU, batch tensors are on CPU.")

            # Check for teacher logits if KD is enabled
            if use_kd:
                if 'teacher_logits' in batch:
                    logger.info(f"Teacher logits found in first batch with shape: {batch['teacher_logits'].shape}")
                else:
                    logger.warning("KD is enabled, but 'teacher_logits' key is missing in the first batch.")

        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
             # Fallback for unexpected tuple/list format (shouldn't happen with current dataset)
             inputs, targets = batch[0], batch[1]
             logger.warning("DataLoader yielded a tuple/list, expected dict. Using first two elements.")
             logger.info(f"First batch loaded (tuple/list format): inputs shape {inputs.shape}, targets shape {targets.shape}")
        else:
             logger.error(f"ERROR: Unexpected batch format received from DataLoader: {type(batch)}")
             raise TypeError("Unexpected batch format")

    except StopIteration:
        logger.error("ERROR: DataLoader is empty. Cannot load first batch.")
        logger.error("Cannot proceed with training due to data loading issue.")
        return
    except Exception as e:
        logger.error(f"ERROR: Failed to load or process first batch from train_loader: {e}")
        logger.error(traceback.format_exc()) # Add traceback for more details
        logger.error("Cannot proceed with training due to data loading issue.")
        return

    # Initialize model
    logger.info("\n=== Creating model ===")

    # Get frequency dimension and class count from btc_config
    n_freq = config.model.get('feature_size', 144)
    n_classes = config.model.get('num_chords', 170)
    logger.info(f"Using frequency dimension: {n_freq}")
    logger.info(f"Output classes: {n_classes}")

    # Override dropout if provided via args
    if args.dropout is not None:
        config.model['input_dropout'] = args.dropout
        config.model['layer_dropout'] = args.dropout
        config.model['attention_dropout'] = args.dropout
        config.model['relu_dropout'] = args.dropout
        logger.info(f"Overriding dropout rates with: {args.dropout}")

    # Create model instance using the model sub-config
    model = BTC_model(config=config.model).to(device)

    # Wrap model with DistributedDataParallel if using distributed training
    if distributed_training:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False # Set to True if encountering issues with unused params
        )
        logger.info(f"Model wrapped with DistributedDataParallel (rank {rank})")

    # Attach chord mapping to model
    if distributed_training:
        # Access the underlying model instance
        model.module.idx_to_chord = master_mapping
    else:
        model.idx_to_chord = master_mapping
    logger.info("Attached chord mapping to model for correct MIR evaluation")

    # Create optimizer using parameters from btc_config.experiment
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training['learning_rate'], # Use potentially overridden LR
        weight_decay=config.experiment.get('weight_decay', 0.0)
    )

    # Clean up GPU memory before training
    if torch.cuda.is_available():
        logger.info("Performing CUDA memory cleanup before training")
        gc.collect()
        torch.cuda.empty_cache()
        # Print memory stats
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"CUDA memory stats (GB): allocated={allocated:.2f}, reserved={reserved:.2f}")

    # --- Remove normalization recalculation ---
    # Calculate dataset statistics efficiently
    mean, std = None, None
    if args.teacher_checkpoint:
        mean, std = load_normalization_from_checkpoint(args.teacher_checkpoint, storage_root, project_root)

    if mean is None or std is None:
        logger.warning("Could not load normalization from teacher checkpoint or path not provided.")
        # Fallback to default values if loading fails or no checkpoint is given
        mean, std = 0.0, 1.0
        logger.warning(f"Using default normalization: mean={mean}, std={std}")
    else:
        logger.info(f"Using normalization loaded from teacher: mean={mean:.4f}, std={std:.4f}")

    # Create normalized tensors on device
    mean_tensor = torch.tensor(mean, device=device)
    std_tensor = torch.tensor(std, device=device)
    normalization = {'mean': mean_tensor, 'std': std_tensor}

    # Final memory cleanup before training
    if torch.cuda.is_available():
        logger.info("Final CUDA memory cleanup before training")
        torch.cuda.empty_cache()

    # Handle LR schedule
    lr_schedule_type = None
    if args.lr_schedule in ['validation', 'none']:
        lr_schedule_type = None
    else:
        # Use arg if provided, otherwise btc_config doesn't specify schedule
        lr_schedule_type = args.lr_schedule

    # Create trainer based on whether we're using distributed training
    trainer_class = DistributedStudentTrainer if distributed_training else StudentTrainer
    trainer_args = {
        'model': model,
        'optimizer': optimizer,
        'device': device,
        'num_epochs': config.experiment.get('max_epoch', 100), # From btc_config.experiment
        'logger': logger,
        'checkpoint_dir': checkpoints_dir,
        'class_weights': None, # Focal loss or standard CE handles weights
        'idx_to_chord': master_mapping,
        'normalization': normalization,
        'early_stopping_patience': config.training.get('early_stopping_patience', 10), # Use value from merged config
        'lr_decay_factor': config.training.get('lr_decay_factor', 0.95), # Use value from merged config
        'min_lr': config.training.get('min_learning_rate', 0.0), # Use value from merged config
        'use_warmup': config.training['use_warmup'],
        'warmup_epochs': config.training.get('warmup_epochs'),
        'warmup_start_lr': config.training.get('warmup_start_lr'),
        'warmup_end_lr': config.training.get('warmup_end_lr'),
        'lr_schedule_type': lr_schedule_type,
        'use_focal_loss': use_focal,
        'focal_gamma': focal_gamma,
        'focal_alpha': focal_alpha,
        'use_kd_loss': use_kd,
        'kd_alpha': kd_alpha,
        'temperature': temperature,
        'timeout_minutes': args.timeout_minutes,
    }
    if distributed_training:
        trainer_args.update({'rank': rank, 'world_size': world_size})

    trainer = trainer_class(**trainer_args)

    # Set chord mapping in trainer
    trainer.set_chord_mapping(chord_mapping)

    # Load checkpoint if specified
    start_epoch = 1
    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            try:
                logger.info(f"\n=== Loading checkpoint from {args.load_checkpoint} ===")
                checkpoint = torch.load(args.load_checkpoint, map_location=device)

                # Load model weights
                # Handle potential DDP prefix mismatch
                state_dict = checkpoint['model_state_dict']
                if distributed_training and not list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
                elif not distributed_training and list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                logger.info("Model state loaded successfully")

                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded successfully")

                # Determine starting epoch
                if not args.reset_epoch and 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"Resuming from epoch {start_epoch}")
                else:
                    start_epoch = 1
                    logger.info(f"Reset epoch flag set - Starting from epoch 1")

                    # Handle scheduler reset if requested
                    if args.reset_scheduler:
                        logger.info("Reset scheduler flag set - Starting with fresh learning rate schedule")
                        if config.training['use_warmup']:
                            warmup_start_lr = config.training.get('warmup_start_lr')
                            if warmup_start_lr is not None:
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = warmup_start_lr
                                logger.info(f"Set LR to warmup start: {warmup_start_lr}")
                        else:
                            # Reset to base learning rate
                            base_lr = config.training['learning_rate']
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = base_lr
                            logger.info(f"Set LR to base value: {base_lr}")

                # Load scheduler state if available
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and trainer.smooth_scheduler:
                    if not (args.reset_epoch and args.reset_scheduler):
                        try:
                            trainer.smooth_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                            logger.info("Scheduler state loaded successfully")
                        except Exception as e:
                            logger.warning(f"Could not load scheduler state: {e}")
                    else:
                        logger.info("Skipped scheduler state due to reset flags")

                # Set best validation accuracy if available
                if hasattr(trainer, 'best_val_acc') and 'accuracy' in checkpoint:
                    trainer.best_val_acc = checkpoint['accuracy']
                    logger.info(f"Set best validation accuracy to {trainer.best_val_acc:.4f}")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Starting from scratch due to checkpoint error")
                start_epoch = 1
        else:
            logger.warning(f"Checkpoint file not found: {args.load_checkpoint}")
            logger.warning("Starting from scratch")
    else:
        logger.info("No checkpoint specified, starting from scratch")

    # Run training
    logger.info(f"\n=== Starting training from epoch {start_epoch}/{config.experiment.get('max_epoch', 100)} ===")
    try:
        logger.info("Preparing data (this may take a while for large datasets)...")
        trainer.train(train_loader, val_loader, start_epoch=start_epoch)
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"ERROR during training: {e}")
        logger.error(traceback.format_exc())

    # Final evaluation on test set - only run on main process if distributed
    if not distributed_training or (distributed_training and rank == 0):
        logger.info("\n=== Testing ===")
        try:
            # Explicitly set the best model path to use btc_model_best.pth
            best_model_filename = "btc_model_best.pth"
            trainer.best_model_path = os.path.join(checkpoints_dir, best_model_filename)
            logger.info(f"Looking for best model at: {trainer.best_model_path}")

            # Try to find the model with student prefix if btc prefix doesn't exist
            if not os.path.exists(trainer.best_model_path):
                alt_filename = "student_model_best.pth"
                alt_path = os.path.join(checkpoints_dir, alt_filename)
                if os.path.exists(alt_path):
                    logger.info(f"Best model not found with 'btc' prefix, but found with 'student' prefix at: {alt_path}")
                    logger.info(f"Copying {alt_path} to {trainer.best_model_path}")
                    import shutil
                    shutil.copy2(alt_path, trainer.best_model_path)

            if trainer.load_best_model():
                # Create test loader with distributed sampler if needed
                if distributed_training:
                    test_subset = SynthSegmentSubset(synth_dataset, synth_dataset.test_indices)
                    test_sampler = torch.utils.data.distributed.DistributedSampler(
                        test_subset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=False
                    )
                    test_loader = torch.utils.data.DataLoader(
                        test_subset,
                        batch_size=batch_size, # Use calculated batch_size
                        shuffle=False,
                        sampler=test_sampler,
                        num_workers=0,
                        pin_memory=False
                    )
                else:
                    test_loader = synth_dataset.get_test_iterator(
                        batch_size=batch_size, # Use calculated batch_size
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False
                    )

                # Basic testing with Tester class
                tester = Tester(
                    model=model,
                    test_loader=test_loader,  # Pass the test_loader parameter
                    device=device,
                    idx_to_chord=master_mapping,
                    normalization=normalization,
                    output_dir=checkpoints_dir  # Also add output directory for saving results
                )

                # Run test using the evaluate method
                metrics = tester.evaluate(save_plots=True)
                test_acc = metrics.get('accuracy', 0.0)
                test_loss = metrics.get('loss', 0.0)
                logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

                # Save test results
                test_results = {
                    'test_loss': test_loss,
                    'test_accuracy': test_acc
                }

                test_results_path = os.path.join(checkpoints_dir, "test_results.json")
                with open(test_results_path, 'w') as f:
                    json.dump(test_results, f, indent=2)
                logger.info(f"Test results saved to {test_results_path}")

                # Visualize chord quality distribution and accuracy
                try:
                    from modules.utils.visualize import plot_chord_quality_distribution_accuracy

                    # Collect all predictions and targets from test set
                    all_preds = []
                    all_targets = []
                    model.eval()
                    with torch.no_grad():
                        for batch in test_loader:
                            if distributed_training:
                                inputs, targets = batch
                            else:
                                inputs, targets = batch['spectro'], batch['chord_idx']
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            if normalization:
                                inputs = (inputs - normalization['mean']) / normalization['std']

                            # Forward pass
                            outputs = model(inputs)

                            # Handle different output formats
                            if isinstance(outputs, tuple):
                                logits = outputs[0]
                            else:
                                logits = outputs

                            # BTC model outputs [batch, time, classes], average over time for eval
                            if logits.ndim == 3:
                                logits = logits.mean(dim=1)

                            preds = logits.argmax(dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())

                    # Define focus qualities
                    focus_qualities = ["maj", "min", "dim", "aug", "min6", "maj6", "min7",
                                      "min-maj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]

                    # Create distribution and accuracy visualization
                    quality_dist_path = os.path.join(checkpoints_dir, "chord_quality_distribution_accuracy.png")
                    plot_chord_quality_distribution_accuracy(
                        all_preds, all_targets, master_mapping,
                        save_path=quality_dist_path,
                        title="Chord Quality Distribution and Accuracy",
                        focus_qualities=focus_qualities
                    )
                    logger.info(f"Chord quality distribution and accuracy plot saved to {quality_dist_path}")
                except Exception as e:
                    logger.error(f"Error creating visualization: {e}")

                # Advanced testing with mir_eval module
                logger.info("\n=== MIR evaluation ===")

                score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
                dataset_length = len(synth_dataset.samples)

                if dataset_length < 3:
                    logger.warning("Dataset too small for MIR evaluation")
                else:
                    # Synchronize all processes before MIR evaluation with timeout handling
                    if distributed_training:
                        logger.info(f"Rank {rank}: Synchronizing before MIR evaluation")
                        try:
                            # Set a timeout for the barrier operation
                            prev_timeout = os.environ.get('NCCL_BLOCKING_WAIT', None)
                            os.environ['NCCL_BLOCKING_WAIT'] = '0'  # Non-blocking mode

                            # Use a timeout for the barrier
                            barrier_timeout = datetime.timedelta(seconds=60)
                            dist.barrier(timeout=barrier_timeout)

                            # Restore previous timeout setting
                            if prev_timeout is not None:
                                os.environ['NCCL_BLOCKING_WAIT'] = prev_timeout
                            else:
                                os.environ.pop('NCCL_BLOCKING_WAIT', None)
                        except Exception as e:
                            logger.warning(f"Rank {rank}: Barrier synchronization failed: {e}. Continuing anyway.")

                    # Split dataset into 3 parts for evaluation
                    all_samples = synth_dataset.samples

                    # In distributed mode, each rank evaluates a different split
                    # In non-distributed mode, evaluate all splits sequentially
                    if distributed_training:
                        # Determine which split this rank should process
                        if rank % 3 == 0:
                            split_idx = 1
                            split_samples = all_samples[:dataset_length//3]
                        elif rank % 3 == 1:
                            split_idx = 2
                            split_samples = all_samples[dataset_length//3:2*dataset_length//3]
                        else:
                            split_idx = 3
                            split_samples = all_samples[2*dataset_length//3:]

                        logger.info(f"Rank {rank}: Evaluating split {split_idx} with {len(split_samples)} samples")

                        # Each rank evaluates only its assigned split
                        score_list_dict, song_length_list, average_score_dict = large_voca_score_calculation(
                            valid_dataset=split_samples, config=config, model=model, model_type=args.model,
                            mean=mean, std=std, device=device)

                        # Gather results from all ranks
                        if world_size > 1:
                            # Use more efficient tensor-based gathering instead of object gathering
                            # Define metrics to gather
                            metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']

                            # Convert local results to tensor
                            local_results = torch.tensor([average_score_dict.get(m, 0.0) for m in metrics],
                                                        dtype=torch.float32, device=device)

                            # Create tensor to gather results (world_size x num_metrics)
                            gathered_tensor = torch.zeros(world_size, len(metrics), dtype=torch.float32, device=device)

                            # Use all_gather with tensor instead of all_gather_object
                            try:
                                # Set a timeout for the gather operation
                                prev_timeout = os.environ.get('NCCL_BLOCKING_WAIT', None)
                                os.environ['NCCL_BLOCKING_WAIT'] = '0'  # Non-blocking mode

                                # Gather tensors with timeout handling
                                logger.info(f"Rank {rank}: Starting tensor gathering for MIR evaluation results")
                                dist.all_gather_into_tensor(gathered_tensor, local_results)
                                logger.info(f"Rank {rank}: Completed tensor gathering for MIR evaluation results")

                                # Restore previous timeout setting
                                if prev_timeout is not None:
                                    os.environ['NCCL_BLOCKING_WAIT'] = prev_timeout
                                else:
                                    os.environ.pop('NCCL_BLOCKING_WAIT', None)
                            except Exception as e:
                                logger.error(f"Rank {rank}: Error during tensor gathering: {e}")
                                # Fall back to point-to-point communication
                                if rank == 0:
                                    # Master process receives from all workers
                                    gathered_tensor[0] = local_results  # Add own results
                                    for src_rank in range(1, world_size):
                                        try:
                                            logger.info(f"Rank {rank}: Receiving results from rank {src_rank}")
                                            dist.recv(gathered_tensor[src_rank], src=src_rank)
                                        except Exception as recv_err:
                                            logger.error(f"Rank {rank}: Error receiving from rank {src_rank}: {recv_err}")
                                            # Fill with zeros if receive fails
                                            gathered_tensor[src_rank].zero_()
                                else:
                                    # Worker processes send to master
                                    try:
                                        logger.info(f"Rank {rank}: Sending results to rank 0")
                                        dist.send(local_results, dst=0)
                                    except Exception as send_err:
                                        logger.error(f"Rank {rank}: Error sending to rank 0: {send_err}")

                            # Also gather split indices and sizes using simple point-to-point communication
                            split_info = torch.tensor([split_idx, len(split_samples)], dtype=torch.int64, device=device)
                            gathered_split_info = torch.zeros(world_size, 2, dtype=torch.int64, device=device)

                            try:
                                dist.all_gather_into_tensor(gathered_split_info, split_info)
                            except Exception as e:
                                logger.error(f"Rank {rank}: Error gathering split info: {e}")
                                # Fall back to point-to-point communication
                                if rank == 0:
                                    gathered_split_info[0] = split_info  # Add own results
                                    for src_rank in range(1, world_size):
                                        try:
                                            dist.recv(gathered_split_info[src_rank], src=src_rank)
                                        except Exception:
                                            # Fill with zeros if receive fails
                                            gathered_split_info[src_rank].zero_()
                                else:
                                    try:
                                        dist.send(split_info, dst=0)
                                    except Exception:
                                        pass  # Ignore send errors from workers

                            # Process gathered results on rank 0
                            if rank == 0:
                                # Initialize dictionaries for each split
                                score_list_dict1, score_list_dict2, score_list_dict3 = {}, {}, {}
                                song_length_list1, song_length_list2, song_length_list3 = [], [], []
                                average_score_dict1, average_score_dict2, average_score_dict3 = {}, {}, {}

                                # Process gathered results
                                for worker_rank in range(world_size):
                                    # Get split index and size
                                    worker_split_idx = int(gathered_split_info[worker_rank][0].item())
                                    worker_split_size = int(gathered_split_info[worker_rank][1].item())

                                    # Get metrics
                                    worker_results = gathered_tensor[worker_rank].tolist()

                                    # Convert results back to dict
                                    split_dict = {metrics[i]: worker_results[i] for i in range(len(metrics))}

                                    if worker_split_idx == 1:
                                        average_score_dict1 = split_dict
                                        song_length_list1 = [1.0] * worker_split_size  # Placeholder
                                    elif worker_split_idx == 2:
                                        average_score_dict2 = split_dict
                                        song_length_list2 = [1.0] * worker_split_size  # Placeholder
                                    elif worker_split_idx == 3:
                                        average_score_dict3 = split_dict
                                        song_length_list3 = [1.0] * worker_split_size  # Placeholder

                        # Synchronize after gathering results with timeout handling
                        if distributed_training:
                            try:
                                # Set a timeout for the barrier operation
                                prev_timeout = os.environ.get('NCCL_BLOCKING_WAIT', None)
                                os.environ['NCCL_BLOCKING_WAIT'] = '0'  # Non-blocking mode

                                # Use a timeout for the barrier
                                barrier_timeout = datetime.timedelta(seconds=60)
                                dist.barrier(timeout=barrier_timeout)

                                # Restore previous timeout setting
                                if prev_timeout is not None:
                                    os.environ['NCCL_BLOCKING_WAIT'] = prev_timeout
                                else:
                                    os.environ.pop('NCCL_BLOCKING_WAIT', None)
                            except Exception as e:
                                logger.warning(f"Rank {rank}: Post-gathering barrier synchronization failed: {e}. Continuing anyway.")
                    else:
                        # Non-distributed mode: evaluate all splits sequentially
                        valid_dataset1 = all_samples[:dataset_length//3]
                        valid_dataset2 = all_samples[dataset_length//3:2*dataset_length//3]
                        valid_dataset3 = all_samples[2*dataset_length//3:]

                        # Evaluate each split
                        logger.info(f"Evaluating model on {len(valid_dataset1)} samples in split 1...")
                        # --- MODIFICATION START: Define sampled_song_ids1 ---
                        sampled_song_ids1 = None
                        small_dataset_pct = dataset_args.get('small_dataset_percentage')
                        if small_dataset_pct is not None and small_dataset_pct != '' and float(small_dataset_pct) < 1.0:
                            # Extract unique song IDs from the samples
                            sampled_song_ids1 = set(sample.get('song_id', '') for sample in valid_dataset1 if 'song_id' in sample)
                            logger.info(f"Using {len(sampled_song_ids1)} sampled song IDs for MIR evaluation (split 1)")
                        # --- MODIFICATION END ---
                        score_list_dict1, song_length_list1, average_score_dict1 = large_voca_score_calculation(
                            valid_dataset=valid_dataset1, config=config, model=model, model_type=args.model,
                            mean=mean, std=std, device=device, sampled_song_ids=sampled_song_ids1)

                        logger.info(f"Evaluating model on {len(valid_dataset2)} samples in split 2...")
                        # Get sampled_song_ids if using small dataset percentage
                        sampled_song_ids2 = None
                        # small_dataset_pct = dataset_args.get('small_dataset_percentage') # Already defined above
                        if small_dataset_pct is not None and small_dataset_pct != '' and float(small_dataset_pct) < 1.0:
                            # Extract unique song IDs from the samples
                            sampled_song_ids2 = set(sample.get('song_id', '') for sample in valid_dataset2 if 'song_id' in sample)
                            logger.info(f"Using {len(sampled_song_ids2)} sampled song IDs for MIR evaluation (split 2)")

                        score_list_dict2, song_length_list2, average_score_dict2 = large_voca_score_calculation(
                            valid_dataset=valid_dataset2, config=config, model=model, model_type=args.model,
                            mean=mean, std=std, device=device, sampled_song_ids=sampled_song_ids2)

                        logger.info(f"Evaluating model on {len(valid_dataset3)} samples in split 3...")
                        # --- MODIFICATION START: Define sampled_song_ids3 ---
                        sampled_song_ids3 = None
                        # small_dataset_pct = dataset_args.get('small_dataset_percentage') # Already defined above
                        if small_dataset_pct is not None and small_dataset_pct != '' and float(small_dataset_pct) < 1.0:
                            # Extract unique song IDs from the samples
                            sampled_song_ids3 = set(sample.get('song_id', '') for sample in valid_dataset3 if 'song_id' in sample)
                            logger.info(f"Using {len(sampled_song_ids3)} sampled song IDs for MIR evaluation (split 3)")
                        # --- MODIFICATION END ---
                        score_list_dict3, song_length_list3, average_score_dict3 = large_voca_score_calculation(
                            valid_dataset=valid_dataset3, config=config, model=model, model_type=args.model,
                            mean=mean, std=std, device=device, sampled_song_ids=sampled_song_ids3) # Pass sampled_song_ids3

                    # Combine results
                    mir_eval_results = {}
                    for metric in score_metrics:
                        mir_eval_results[metric] = {
                            'split1': average_score_dict1.get(metric, 0.0),
                            'split2': average_score_dict2.get(metric, 0.0),
                            'split3': average_score_dict3.get(metric, 0.0),
                            'average': (average_score_dict1.get(metric, 0.0) +
                                        average_score_dict2.get(metric, 0.0) +
                                        average_score_dict3.get(metric, 0.0)) / 3
                        }

                    # Log results with range
                    logger.info("\nMIR Evaluation Results:")
                    for metric, values in mir_eval_results.items():
                        # Calculate min and max values across the three splits
                        split_values = [
                            values['split1'] * 100,
                            values['split2'] * 100,
                            values['split3'] * 100
                        ]
                        min_val = min(split_values)
                        max_val = max(split_values)
                        avg_val = values['average'] * 100

                        # Display average with min-max range
                        logger.info(f"{metric}: {avg_val:.2f}% (avg), range: [{min_val:.2f}% - {max_val:.2f}%]")

                    # Save results to file
                    mir_eval_path = os.path.join(checkpoints_dir, "mir_eval_results.json")
                    with open(mir_eval_path, 'w') as f:
                        json.dump(mir_eval_results, f, indent=2)
                    logger.info(f"MIR evaluation metrics saved to {mir_eval_path}")
            else:
                logger.warning("Could not load best model for testing")
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            logger.error(traceback.format_exc())

    # Save the final model
    try:
        # Use btc_model_final.pth
        final_model_filename = "btc_model_final.pth"
        save_path = os.path.join(checkpoints_dir, final_model_filename)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the underlying model if using DDP
        model_to_save = model.module if distributed_training else model

        # Save model with all necessary information
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'chord_mapping': chord_mapping,
            'idx_to_chord': master_mapping,
            'mean': normalization['mean'].cpu().numpy() if hasattr(normalization['mean'], 'cpu') else normalization['mean'],
            'std': normalization['std'].cpu().numpy() if hasattr(normalization['std'], 'cpu') else normalization['std']
        }, save_path)

        logger.info(f"Final BTC model saved to {save_path}")

        # Also save a copy in the ChordMini/checkpoints/btc directory for backward compatibility
        local_checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/btc")
        if local_checkpoints_dir != checkpoints_dir:  # Only if they're different
            os.makedirs(local_checkpoints_dir, exist_ok=True)
            local_save_path = os.path.join(local_checkpoints_dir, final_model_filename)
            import shutil
            shutil.copy2(save_path, local_save_path)
            logger.info(f"Also saved a copy to {local_save_path} for backward compatibility")

    except Exception as e:
        logger.error(f"Error saving final BTC model: {e}")
        logger.error(traceback.format_exc())

    logger.info("BTC training and evaluation complete!")

    # Clear GPU cache before exiting
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU cache: {e}")

def distributed_main(local_rank, world_size, args):
    """Main function for distributed training."""
    # Set up distributed environment
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500' # Use a different port if running simultaneously with student training
    dist.init_process_group(backend=args.distributed_backend or 'nccl',
                           world_size=world_size,
                           rank=local_rank)

    # Set device for this process
    torch.cuda.set_device(local_rank)

    # Update args for the worker process
    args.local_rank = local_rank
    args.rank = local_rank
    args.world_size = world_size

    # Still distributed so we pick the DDP trainer
    args.distributed = True

    # Call main function
    main()

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    main()
