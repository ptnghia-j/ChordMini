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
from pathlib import Path
from collections import Counter

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.device import get_device, is_cuda_available, is_gpu_available, clear_gpu_cache
# Removed unused get_quick_dataset_stats
from modules.data.SynthDataset import SynthDataset, SynthSegmentSubset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.StudentTrainer import StudentTrainer
from modules.training.DistributedStudentTrainer import DistributedStudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord
from modules.training.Tester import Tester
from modules.utils.file_utils import count_files_in_subdirectories, resolve_path, load_normalization_from_checkpoint

# Using utility functions from modules.utils.file_utils

def main(rank=0, world_size=1):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a chord recognition student model using synthesized data")
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config value)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides config value)')
    parser.add_argument('--model', type=str, default='ChordNet',
                        help='Model type for evaluation')
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

    # Add knowledge distillation arguments
    parser.add_argument('--use_kd_loss', action='store_true',
                       help='Use knowledge distillation loss (teacher logits must be in batch data)')
    parser.add_argument('--kd_alpha', type=float, default=0.5,
                       help='Weight for knowledge distillation loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for softening distributions (default: 1.0)')
    parser.add_argument('--logits_dir', type=str, default=None,
                       help='Directory containing teacher logits (required for KD)')

    # Add model scale argument
    parser.add_argument('--model_scale', type=float, default=None,
                       help='Scaling factor for model capacity (0.5=half, 1.0=base, 2.0=double)')

    # Add dropout argument
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout probability (0-1)')

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
                        help='Minimum learning rate for schedulers (overrides config value)')
    parser.add_argument('--warmup_end_lr', type=float, default=None,
                       help='Target learning rate at the end of warm-up (default: base LR)')

    # Modify dataset_type choices to include 'labeled' and combinations
    parser.add_argument('--dataset_type', type=str,
                      choices=['fma', 'maestro', 'dali_synth', 'labeled', 'combined',
                               'fma+maestro', 'fma+dali_synth', 'fma+labeled',
                               'maestro+dali_synth', 'maestro+labeled',
                               'dali_synth+labeled',
                               'fma+maestro+dali_synth', 'fma+maestro+labeled', # etc.
                               'maestro+dali_synth+labeled', 'fma+dali_synth+labeled'],
                      default='fma',
                      help='Dataset format type: fma, maestro, dali_synth, labeled, combined (all), or specific combinations (e.g., fma+labeled)')

    # Checkpoint loading
    parser.add_argument('--load_checkpoint', type=str, default=None,
                      help='Path to checkpoint file to resume training from')
    parser.add_argument('--reset_epoch', action='store_true',
                      help='Start from epoch 1 even when loading from checkpoint')
    parser.add_argument('--reset_scheduler', action='store_true',
                      help='Reset learning rate scheduler when --reset_epoch is used')
    parser.add_argument('--timeout_minutes', type=int, default=90,
                      help='Timeout in minutes for distributed operations (default: 90)')

    # New argument for teacher checkpoint
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to the teacher model checkpoint to load normalization parameters (mean, std)')

    args = parser.parse_args()

    # Load configuration from YAML first
    config = HParams.load(args.config)

    # Override config with dataset_type if specified
    if not hasattr(config, 'data'):
        config.data = {}
    config.data['dataset_type'] = args.dataset_type

    # Set up distributed training if enabled
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
        if config.misc['use_cuda'] and is_cuda_available():
            device = get_device()
            logger.info(f"Using CUDA for training on device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for training")

    # Override config values with command line arguments if provided
    config.misc['seed'] = args.seed or config.misc.get('seed', 42)
    config.paths['checkpoints_dir'] = args.save_dir or config.paths.get('checkpoints_dir', 'checkpoints')
    config.paths['storage_root'] = args.storage_root or config.paths.get('storage_root', None)

    # Handle learning rate and warmup parameters correctly - FIX FOR WARMUP EPOCHS ISSUE
    config.training['learning_rate'] = args.learning_rate or config.training.get('learning_rate', 0.0001)
    config.training['min_learning_rate'] = args.min_learning_rate or config.training.get('min_learning_rate', 5e-6)

    # FIX: Use args.warmup_epochs only if explicitly provided, otherwise keep the config value as is
    if args.warmup_epochs is not None:
        config.training['warmup_epochs'] = args.warmup_epochs
    # Don't set a default if not in config

    # Similarly for other warmup parameters - avoid overriding config with hardcoded defaults
    if args.warmup_start_lr is not None:
        config.training['warmup_start_lr'] = args.warmup_start_lr
    elif 'warmup_start_lr' not in config.training:
        config.training['warmup_start_lr'] = config.training['learning_rate']/10

    if args.warmup_end_lr is not None:
        config.training['warmup_end_lr'] = args.warmup_end_lr
    elif 'warmup_end_lr' not in config.training:
        config.training['warmup_end_lr'] = config.training['learning_rate']

    # Log parameters that have been overridden
    logger.info(f"Using learning rate: {config.training['learning_rate']}")
    logger.info(f"Using minimum learning rate: {config.training['min_learning_rate']}")
    if 'warmup_epochs' in config.training:
        logger.info(f"Using warmup_epochs: {config.training['warmup_epochs']}")
    logger.info(f"Using warmup_start_lr: {config.training.get('warmup_start_lr')}")
    logger.info(f"Using warmup_end_lr: {config.training.get('warmup_end_lr')}")

    # Log training configuration
    logger.info("\n=== Training Configuration ===")
    logger.info(f"Model type: {args.model}")
    model_scale = args.model_scale or config.model.get('scale', 1.0)
    logger.info(f"Model scale: {model_scale}")

    # Log knowledge distillation settings
    use_kd = args.use_kd_loss if args.use_kd_loss else config.training.get('use_kd_loss', False)
    use_kd = str(use_kd).lower() == "true"

    kd_alpha = args.kd_alpha or config.training.get('kd_alpha', 0.5)
    temperature = args.temperature or config.training.get('temperature', 1.0)

    if use_kd:
        logger.info("\n=== Knowledge Distillation Enabled ===")
        logger.info(f"KD alpha: {kd_alpha} (weighting between KD and CE loss)")
        logger.info(f"Temperature: {temperature} (for softening distributions)")
        if args.logits_dir:
            logger.info(f"Using teacher logits from directory: {args.logits_dir}")
        else:
            logger.info("No logits directory specified - teacher logits must be in batch data")
    else:
        logger.info("Knowledge distillation is disabled, using standard loss")

    # Log focal loss settings
    if args.use_focal_loss or config.training.get('use_focal_loss', False):
        logger.info("\n=== Focal Loss Enabled ===")
        logger.info(f"Gamma: {args.focal_gamma or config.training.get('focal_gamma', 2.0)}")
        if args.focal_alpha or config.training.get('focal_alpha'):
            logger.info(f"Alpha: {args.focal_alpha or config.training.get('focal_alpha')}")
    else:
        logger.info("Using standard cross-entropy loss")

    # Clear summary of loss function configuration
    if use_kd and (args.use_focal_loss or config.training.get('use_focal_loss', False)):
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using Focal Loss (gamma={args.focal_gamma or config.training.get('focal_gamma', 2.0)}, alpha={args.focal_alpha or config.training.get('focal_alpha')}) combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * focal_loss")
        logger.info(f"Note: When teacher logits are not available for a batch, only focal loss will be used")
    elif use_kd:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using standard Cross Entropy combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * cross_entropy")
        logger.info(f"Note: When teacher logits are not available for a batch, only cross entropy will be used")
    elif args.use_focal_loss or config.training.get('use_focal_loss', False):
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using only Focal Loss with gamma={args.focal_gamma or config.training.get('focal_gamma', 2.0)}, alpha={args.focal_alpha or config.training.get('focal_alpha')}")
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
    logger.logging_verbosity(config.misc['logging_level'])

    # Get project root and storage root
    project_root = os.path.dirname(os.path.abspath(__file__))
    storage_root = config.paths.get('storage_root', None)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Storage root: {storage_root}")

    # --- Load Normalization from Teacher Checkpoint ---
    mean_val, std_val = load_normalization_from_checkpoint(
        args.teacher_checkpoint or config.paths.get('teacher_checkpoint'),
        storage_root, project_root
    )
    mean_tensor = torch.tensor(mean_val, device=device, dtype=torch.float32)
    std_tensor = torch.tensor(std_val, device=device, dtype=torch.float32)
    normalization = {'mean': mean_tensor, 'std': std_tensor}
    logger.info(f"Using normalization parameters FOR TRAINING (from teacher checkpoint): mean={mean_val:.4f}, std={std_val:.4f}")


    # Resolve primary paths from config, with CLI override
    # spec_dir_config = resolve_path(args.spec_dir or config.paths.get('spec_dir', 'data/logits/synth/spectrograms'),
    #                               storage_root, project_root)
    # label_dir_config = resolve_path(args.label_dir or config.paths.get('label_dir', 'data/logits/synth/labels'),
    #                                storage_root, project_root)

    # Resolve alternative paths if available
    # alt_spec_dir = resolve_path(config.paths.get('alt_spec_dir'), storage_root, project_root) if config.paths.get('alt_spec_dir') else None
    # alt_label_dir = resolve_path(config.paths.get('alt_label_dir'), storage_root, project_root) if config.paths.get('alt_label_dir') else None

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

    # Standard paths for LabeledDataset (Real Labels)
    # Assuming features are pre-computed in a parallel structure
    labeled_spec_dir = os.path.join(data_root, "LabeledDataset_synth/spectrograms")
    labeled_label_dir = os.path.join(data_root, "LabeledDataset/Labels") # Real labels path
    labeled_logits_dir = os.path.join(data_root, "LabeledDataset_synth/logits") # Optional teacher logits

    # Determine active dataset types
    active_types = []
    if args.dataset_type == 'combined':
        active_types = ['fma', 'maestro', 'dali_synth', 'labeled']
    elif '+' in args.dataset_type:
        active_types = args.dataset_type.split('+')
    else:
        active_types = [args.dataset_type]

    logger.info(f"Active dataset types: {active_types}")

    # Initialize lists for combined paths
    spec_dirs_list = []
    label_dirs_list = []
    logits_dirs_list = []

    # Collect paths based on active types
    if 'fma' in active_types:
        spec_dirs_list.append(resolve_path(args.spec_dir or fma_spec_dir, storage_root, project_root))
        label_dirs_list.append(resolve_path(args.label_dir or fma_label_dir, storage_root, project_root))
        logits_dirs_list.append(resolve_path(args.logits_dir or fma_logits_dir, storage_root, project_root))
        fma_spec_count = count_files_in_subdirectories(spec_dirs_list[-1], "*_spec.npy")
        fma_label_count = count_files_in_subdirectories(label_dirs_list[-1], "*.lab") # Assuming .lab for FMA synth labels
        logger.info(f"  FMA: {fma_spec_count} specs, {fma_label_count} labels")

    if 'maestro' in active_types:
        spec_dirs_list.append(resolve_path(maestro_spec_dir, storage_root, project_root))
        label_dirs_list.append(resolve_path(maestro_label_dir, storage_root, project_root))
        logits_dirs_list.append(resolve_path(maestro_logits_dir, storage_root, project_root))
        maestro_spec_count = count_files_in_subdirectories(spec_dirs_list[-1], "*_spec.npy")
        maestro_label_count = count_files_in_subdirectories(label_dirs_list[-1], "*.lab") # Assuming .lab for Maestro synth labels
        logger.info(f"  Maestro: {maestro_spec_count} specs, {maestro_label_count} labels")

    if 'dali_synth' in active_types:
        spec_dirs_list.append(resolve_path(dali_spec_dir, storage_root, project_root))
        label_dirs_list.append(resolve_path(dali_label_dir, storage_root, project_root))
        logits_dirs_list.append(resolve_path(dali_logits_dir, storage_root, project_root))
        dali_spec_count = count_files_in_subdirectories(spec_dirs_list[-1], "*_spec.npy")
        dali_label_count = count_files_in_subdirectories(label_dirs_list[-1], "*.lab") # Assuming .lab for DALI synth labels
        logger.info(f"  DALI Synth: {dali_spec_count} specs, {dali_label_count} labels")

    if 'labeled' in active_types:
        spec_dirs_list.append(resolve_path(labeled_spec_dir, storage_root, project_root))
        label_dirs_list.append(resolve_path(labeled_label_dir, storage_root, project_root)) # Use the real label path
        logits_dirs_list.append(resolve_path(labeled_logits_dir, storage_root, project_root))
        labeled_spec_count = count_files_in_subdirectories(spec_dirs_list[-1], "*_spec.npy")
        # Count .lab files recursively within the LabeledDataset/Labels structure
        labeled_label_count = count_files_in_subdirectories(label_dirs_list[-1], "*.lab")
        logger.info(f"  Labeled: {labeled_spec_count} specs, {labeled_label_count} labels (real)")

    # Check if any data was found
    if not spec_dirs_list or not label_dirs_list:
         logger.error("No valid data directories found for the specified dataset types. Exiting.")
         sys.exit(1)

    # Assign final paths (use lists for SynthDataset)
    spec_dir = spec_dirs_list
    label_dir = label_dirs_list
    logits_dir = logits_dirs_list if any(logits_dirs_list) else None # Pass None if no logits dirs were added

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
    checkpoints_dir_config = config.paths.get('checkpoints_dir', 'checkpoints')
    checkpoints_dir = resolve_path(checkpoints_dir_config, storage_root, project_root)
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")

    # Initialize SynthDataset with optimized settings
    logger.info("\n=== Creating dataset ===")


    dataset_args.update({
        'spec_dir': spec_dir,
        'label_dir': label_dir, # Pass the list of directories
        'logits_dir': logits_dir, # Pass the list or None
        'chord_mapping': chord_mapping,
        'seq_len': config.training.get('seq_len', 10),
        'stride': config.training.get('seq_stride', 5),
        'frame_duration': config.feature.get('hop_duration', 0.1),
        'verbose': True,
        'device': device,
        'pin_memory': False,
        'prefetch_factor': float(args.prefetch_factor) if args.prefetch_factor else 1,
        'num_workers': 10,
        # debug area
        'require_teacher_logits': use_kd,
        'use_cache': not config.data.get('disable_cache', False),
        'metadata_only': str(args.metadata_cache).lower() == "true",
        'cache_fraction': config.data.get('cache_fraction', 0.1),
        'lazy_init': str(args.lazy_init).lower() == "true",
        'batch_gpu_cache': str(args.batch_gpu_cache).lower() == "true",
        'dataset_type': args.dataset_type # Pass the potentially combined type string
    })

    # Create the dataset
    logger.info("Creating dataset with the following parameters:")
    for key, value in dataset_args.items():
        logger.info(f"  {key}: {value}")

    synth_dataset = SynthDataset(**dataset_args)

    # Create data loaders for each subset
    batch_size = config.training.get('batch_size', 16)
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
            inputs = batch['spectro'] # <--- ERROR HERE
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

    # Get frequency dimension and class count
    n_freq = getattr(config.feature, 'freq_bins', 144)
    n_classes = len(chord_mapping)
    logger.info(f"Using frequency dimension: {n_freq}")
    logger.info(f"Output classes: {n_classes}")

    # Always use n_group=2 for all inputs
    n_group = 2
    feature_dim = n_freq // n_group
    logger.info(f"Using fixed n_group=2, resulting in feature dimension: {feature_dim}")

    # Get dropout value
    dropout_rate = args.dropout if args.dropout is not None else config.model.get('dropout', 0.3)
    logger.info(f"Using dropout rate: {dropout_rate}")

    # Scale model parameters based on model_scale
    def scale_model_params(config, scale_factor):
        """Scale model parameters based on the scale factor"""
        # Get base configuration for the model
        base_config = config.model.get('base_config', {})

        # If base_config is not specified, fall back to direct model parameters
        if not base_config:
            base_config = {
                'f_layer': config.model.get('f_layer', 3),
                'f_head': config.model.get('f_head', 2),
                't_layer': config.model.get('t_layer', 4),
                't_head': config.model.get('t_head', 4),
                'd_layer': config.model.get('d_layer', 3),
                'd_head': config.model.get('d_head', 4)
            }

        # Apply scale to model parameters
        f_layer = max(1, int(round(base_config.get('f_layer', 3) * scale_factor)))
        t_layer = max(1, int(round(base_config.get('t_layer', 3) * scale_factor)))
        d_layer = max(1, int(round(base_config.get('d_layer', 3) * scale_factor)))
        f_head = 2
        t_head = 4
        d_head = 4

        # Ensure f_head is compatible with feature_dim (must be a divisor)
        if feature_dim % f_head != 0:
            # Find the largest divisor of feature_dim that's <= f_head
            for h in range(f_head, 0, -1):
                if feature_dim % h == 0:
                    logger.info(f"Adjusted f_head from {f_head} to {h} to ensure compatibility with feature_dim={feature_dim}")
                    f_head = h
                    break

        return {
            'f_layer': f_layer,
            'f_head': f_head,
            't_layer': t_layer,
            't_head': t_head,
            'd_layer': d_layer,
            'd_head': d_head
        }

    # Apply model scaling
    scaled_params = scale_model_params(config, model_scale)
    logger.info(f"Scaled model parameters (scale={model_scale}):")
    logger.info(f"  Frequency encoder: {scaled_params['f_layer']} layers, {scaled_params['f_head']} heads")
    logger.info(f"  Time encoder: {scaled_params['t_layer']} layers, {scaled_params['t_head']} heads")
    logger.info(f"  Decoder: {scaled_params['d_layer']} layers, {scaled_params['d_head']} heads")

    # Create model instance with scaled parameters
    model = ChordNet(
        n_freq=n_freq,
        n_classes=n_classes,
        n_group=n_group,
        f_layer=scaled_params['f_layer'],
        f_head=scaled_params['f_head'],
        t_layer=scaled_params['t_layer'],
        t_head=scaled_params['t_head'],
        d_layer=scaled_params['d_layer'],
        d_head=scaled_params['d_head'],
        dropout=dropout_rate
    ).to(device)

    # Wrap model with DistributedDataParallel if using distributed training
    if distributed_training:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        logger.info(f"Model wrapped with DistributedDataParallel (rank {rank})")

    # Attach chord mapping to model
    if distributed_training:
        model.module.idx_to_chord = master_mapping
    else:
        model.idx_to_chord = master_mapping
    logger.info("Attached chord mapping to model for correct MIR evaluation")

    # Create optimizer with different learning rates for ReZero parameters
    # Separate parameters into two groups: ReZero alpha parameters and all other parameters
    rezero_params = []
    other_params = []

    # Helper function to identify ReZero parameters
    def is_rezero_param(name):
        return 'alpha' in name

    # Collect parameters based on their names
    if distributed_training:
        # For distributed training, access the module
        for name, param in model.module.named_parameters():
            if is_rezero_param(name):
                rezero_params.append(param)
                logger.info(f"ReZero parameter found: {name}")
            else:
                other_params.append(param)
    else:
        # For non-distributed training
        for name, param in model.named_parameters():
            if is_rezero_param(name):
                rezero_params.append(param)
                logger.info(f"ReZero parameter found: {name}")
            else:
                other_params.append(param)

    # Create optimizer with parameter groups
    base_lr = config.training['learning_rate']
    rezero_lr = base_lr * 0.1  # 10x smaller learning rate for ReZero parameters

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': base_lr},
        {'params': rezero_params, 'lr': rezero_lr}
    ], weight_decay=config.training.get('weight_decay', 0.0))

    logger.info(f"Using learning rate {base_lr} for main parameters and {rezero_lr} for ReZero parameters")

    # Clean up GPU memory before training
    if torch.cuda.is_available():
        logger.info("Performing CUDA memory cleanup before training")
        gc.collect()
        torch.cuda.empty_cache()
        # Print memory stats
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"CUDA memory stats (GB): allocated={allocated:.2f}, reserved={reserved:.2f}")

    # Final memory cleanup before training
    if torch.cuda.is_available():
        logger.info("Final CUDA memory cleanup before training")
        torch.cuda.empty_cache()

    # Handle LR schedule
    lr_schedule_type = None
    if args.lr_schedule in ['validation', 'none']:
        lr_schedule_type = None
    else:
        lr_schedule_type = args.lr_schedule or config.training.get('lr_schedule', None)

    # Create trainer based on whether we're using distributed training
    if distributed_training:
        # We don't need to wrap the model's forward method anymore since we're using SynthSegmentSubset
        # which already returns the correct format (spectro, chord_idx)

        trainer = DistributedStudentTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            num_epochs=config.training.get('num_epochs', config.training.get('max_epochs', 100)),
            logger=logger,
            checkpoint_dir=checkpoints_dir,
            class_weights=None,
            idx_to_chord=master_mapping,
            normalization=normalization, # Use the normalization loaded from teacher checkpoint
            early_stopping_patience=config.training.get('early_stopping_patience', 5),
            lr_decay_factor=config.training.get('lr_decay_factor', 0.95),
            min_lr=config.training.get('min_learning_rate', 5e-6),
            use_warmup=args.use_warmup or config.training.get('use_warmup', False),
            # IMPORTANT: Use config.training.get() to avoid None errors if not set in config
            warmup_epochs=config.training.get('warmup_epochs'),
            warmup_start_lr=config.training.get('warmup_start_lr'),
            warmup_end_lr=config.training.get('warmup_end_lr'),
            lr_schedule_type=lr_schedule_type,
            use_focal_loss=args.use_focal_loss or config.training.get('use_focal_loss', False),
            focal_gamma=args.focal_gamma or config.training.get('focal_gamma', 2.0),
            focal_alpha=args.focal_alpha or config.training.get('focal_alpha', None),
            use_kd_loss=use_kd,
            kd_alpha=kd_alpha,
            temperature=temperature,
            rank=rank,
            world_size=world_size,
            timeout_minutes=args.timeout_minutes
        )
    else:
        trainer = StudentTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            num_epochs=config.training.get('num_epochs', config.training.get('max_epochs', 100)),
            logger=logger,
            checkpoint_dir=checkpoints_dir,
            class_weights=None,
            idx_to_chord=master_mapping,
            normalization=normalization, # Use the normalization loaded from teacher checkpoint
            #max_grad_norm=0.5,
            early_stopping_patience=config.training.get('early_stopping_patience', 5),
            lr_decay_factor=config.training.get('lr_decay_factor', 0.95),
            min_lr=config.training.get('min_learning_rate', 5e-6),
            use_warmup=args.use_warmup or config.training.get('use_warmup', False),
            # IMPORTANT: Use config.training.get() to avoid None errors if not set in config
            warmup_epochs=config.training.get('warmup_epochs'),
            warmup_start_lr=config.training.get('warmup_start_lr'),
            warmup_end_lr=config.training.get('warmup_end_lr'),
            lr_schedule_type=lr_schedule_type,
            use_focal_loss=args.use_focal_loss or config.training.get('use_focal_loss', False),
            focal_gamma=args.focal_gamma or config.training.get('focal_gamma', 2.0),
            focal_alpha=args.focal_alpha or config.training.get('focal_alpha', None),
            use_kd_loss=use_kd,
            kd_alpha=kd_alpha,
            temperature=temperature,
        )

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
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model state loaded successfully")

                # Load idx_to_chord mapping if available
                if 'idx_to_chord' in checkpoint:
                    if distributed_training:
                        model.module.idx_to_chord = checkpoint['idx_to_chord']
                    else:
                        model.idx_to_chord = checkpoint['idx_to_chord']
                    logger.info("Chord mapping loaded from checkpoint")
                else:
                    logger.warning("Checkpoint does not contain idx_to_chord mapping")
                    # Set it from master_mapping anyway
                    if distributed_training:
                        model.module.idx_to_chord = master_mapping
                    else:
                        model.idx_to_chord = master_mapping
                    logger.info("Using default chord mapping")

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
                        if args.use_warmup or config.training.get('use_warmup', False):
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
    logger.info(f"\n=== Starting training from epoch {start_epoch}/{config.training.get('num_epochs', 100)} ===")
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
                        batch_size=config.training['batch_size'],
                        shuffle=False,
                        sampler=test_sampler,
                        num_workers=0,
                        pin_memory=False
                    )
                else:
                    test_loader = synth_dataset.get_test_iterator(
                        batch_size=config.training['batch_size'],
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
                                # In distributed mode with direct DataLoader, batch is a tuple of (inputs, targets)
                                inputs, targets = batch
                            else:
                                # In non-distributed mode with SynthDataset's iterator, batch is a dict
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

                            if logits.ndim == 3 and targets.ndim <= 2:
                                logits = logits.mean(dim=1)  # Average over time dimension

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
                    # Synchronize all processes before MIR evaluation
                    if distributed_training:
                        logger.info(f"Rank {rank}: Synchronizing before MIR evaluation")
                        dist.barrier()

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
                        # Get sampled_song_ids if using small dataset percentage
                        sampled_song_ids = None
                        small_dataset_pct = dataset_args.get('small_dataset_percentage')
                        if small_dataset_pct is not None and small_dataset_pct != '' and float(small_dataset_pct) < 1.0:
                            # Extract unique song IDs from the samples
                            sampled_song_ids = set(sample.get('song_id', '') for sample in split_samples if 'song_id' in sample)
                            logger.info(f"Using {len(sampled_song_ids)} sampled song IDs for MIR evaluation")

                        score_list_dict, song_length_list, average_score_dict = large_voca_score_calculation(
                            valid_dataset=split_samples, config=config, model=model, model_type=args.model,
                            mean=mean_val, std=std_val, device=device, sampled_song_ids=sampled_song_ids)

                        # Gather results from all ranks
                        if world_size > 1:
                            # Create placeholder tensors for gathering
                            gathered_results = [None for _ in range(world_size)]

                            # Convert dict to tensor for gathering
                            metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
                            local_results = torch.tensor([average_score_dict.get(m, 0.0) for m in metrics],
                                                        dtype=torch.float32, device=device)

                            # Gather results from all processes
                            dist.all_gather_object(gathered_results,
                                                 (split_idx, local_results.tolist(), len(split_samples)))

                            # Process gathered results on rank 0
                            if rank == 0:
                                # Initialize dictionaries for each split
                                score_list_dict1, score_list_dict2, score_list_dict3 = {}, {}, {}
                                song_length_list1, song_length_list2, song_length_list3 = [], [], []
                                average_score_dict1, average_score_dict2, average_score_dict3 = {}, {}, {}

                                # Process gathered results
                                for split_idx, results, split_size in gathered_results:
                                    # Convert results back to dict
                                    split_dict = {metrics[i]: results[i] for i in range(len(metrics))}

                                    if split_idx == 1:
                                        average_score_dict1 = split_dict
                                        song_length_list1 = [1.0] * split_size  # Placeholder
                                    elif split_idx == 2:
                                        average_score_dict2 = split_dict
                                        song_length_list2 = [1.0] * split_size  # Placeholder
                                    elif split_idx == 3:
                                        average_score_dict3 = split_dict
                                        song_length_list3 = [1.0] * split_size  # Placeholder

                        # Synchronize after gathering results
                        if distributed_training:
                            dist.barrier()
                    else:
                        # Non-distributed mode: evaluate all splits sequentially
                        valid_dataset1 = all_samples[:dataset_length//3]
                        valid_dataset2 = all_samples[dataset_length//3:2*dataset_length//3]
                        valid_dataset3 = all_samples[2*dataset_length//3:]

                        # Evaluate each split
                        logger.info(f"Evaluating model on {len(valid_dataset1)} samples in split 1...")
                        # --- Define sampled_song_ids1 ---
                        sampled_song_ids1 = None
                        small_dataset_pct = dataset_args.get('small_dataset_percentage')
                        if small_dataset_pct is not None and small_dataset_pct != '' and float(small_dataset_pct) < 1.0:
                            # Extract unique song IDs from the samples
                            sampled_song_ids1 = set(sample.get('song_id', '') for sample in valid_dataset1 if 'song_id' in sample)
                            logger.info(f"Using {len(sampled_song_ids1)} sampled song IDs for MIR evaluation (split 1)")
                        # --- MODIFICATION START: Use mean_val and std_val ---
                        score_list_dict1, song_length_list1, average_score_dict1 = large_voca_score_calculation(
                            valid_dataset=valid_dataset1, config=config, model=model, model_type=args.model,
                            mean=mean_val, std=std_val, device=device, sampled_song_ids=sampled_song_ids1)
                        # --- MODIFICATION END ---

                        logger.info(f"Evaluating model on {len(valid_dataset2)} samples in split 2...")
                        # Get sampled_song_ids if using small dataset percentage
                        sampled_song_ids2 = None
                        # small_dataset_pct = dataset_args.get('small_dataset_percentage') # Already defined above
                        if small_dataset_pct is not None and small_dataset_pct != '' and float(small_dataset_pct) < 1.0:
                            # Extract unique song IDs from the samples
                            sampled_song_ids2 = set(sample.get('song_id', '') for sample in valid_dataset2 if 'song_id' in sample)
                            logger.info(f"Using {len(sampled_song_ids2)} sampled song IDs for MIR evaluation (split 2)")

                        # --- MODIFICATION START: Use mean_val and std_val ---
                        score_list_dict2, song_length_list2, average_score_dict2 = large_voca_score_calculation(
                            valid_dataset=valid_dataset2, config=config, model=model, model_type=args.model,
                            mean=mean_val, std=std_val, device=device, sampled_song_ids=sampled_song_ids2)
                        # --- MODIFICATION END ---

                        logger.info(f"Evaluating model on {len(valid_dataset3)} samples in split 3...")
                        # --- Define sampled_song_ids3 ---
                        sampled_song_ids3 = None
                        # small_dataset_pct = dataset_args.get('small_dataset_percentage') # Already defined above
                        if small_dataset_pct is not None and small_dataset_pct != '' and float(small_dataset_pct) < 1.0:
                            # Extract unique song IDs from the samples
                            sampled_song_ids3 = set(sample.get('song_id', '') for sample in valid_dataset3 if 'song_id' in sample)
                            logger.info(f"Using {len(sampled_song_ids3)} sampled song IDs for MIR evaluation (split 3)")
                        # --- MODIFICATION START: Use mean_val and std_val ---
                        score_list_dict3, song_length_list3, average_score_dict3 = large_voca_score_calculation(
                            valid_dataset=valid_dataset3, config=config, model=model, model_type=args.model,
                            mean=mean_val, std=std_val, device=device, sampled_song_ids=sampled_song_ids3) # Pass sampled_song_ids3
                        # --- MODIFICATION END ---

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
        save_path = os.path.join(checkpoints_dir, "student_model_final.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'chord_mapping': chord_mapping,
            'idx_to_chord': master_mapping,
            'mean': normalization['mean'].cpu().numpy() if hasattr(normalization['mean'], 'cpu') else normalization['mean'],
            'std': normalization['std'].cpu().numpy() if hasattr(normalization['std'], 'cpu') else normalization['std']
        }, save_path)
        logger.info(f"Final model saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")

    logger.info("Student training and evaluation complete!")

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
    os.environ['MASTER_PORT'] = '29500'
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