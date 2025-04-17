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
from modules.data.SynthDataset import SynthDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.StudentTrainer import StudentTrainer
from modules.training.DistributedStudentTrainer import DistributedStudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord
from modules.training.Tester import Tester

class ListSampler:
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def count_files_in_subdirectories(directory, file_pattern):
    """Count files in a directory and all its subdirectories matching a pattern."""
    if not os.path.exists(directory):
        return 0

    count = 0
    # Count files directly in the directory
    for file in glob.glob(os.path.join(directory, file_pattern)):
        if os.path.isfile(file):
            count += 1

    # Count files in subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_pattern.replace("*", "")):
                count += 1

    return count

def find_sample_files(directory, file_pattern, max_samples=5):
    """Find sample files in a directory and all its subdirectories matching a pattern."""
    if not os.path.exists(directory):
        return []

    samples = []
    # Find files in all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_pattern.replace("*", "")):
                samples.append(os.path.join(root, file))
                if len(samples) >= max_samples:
                    return samples

    return samples

def resolve_path(path, storage_root=None, project_root=None):
    """
    Resolve a path that could be absolute, relative to storage_root, or relative to project_root.

    Args:
        path (str): The path to resolve
        storage_root (str): The storage root path
        project_root (str): The project root path

    Returns:
        str: The resolved absolute path
    """
    if not path:
        return None

    # If it's already absolute, return it directly
    if os.path.isabs(path):
        return path

    # Try as relative to storage_root first
    if storage_root:
        storage_path = os.path.join(storage_root, path)
        if os.path.exists(storage_path):
            return storage_path

    # Then try as relative to project_root
    if project_root:
        project_path = os.path.join(project_root, path)
        if os.path.exists(project_path):
            return project_path

    # If neither exists but a storage_root was specified, prefer that resolution
    if storage_root:
        return os.path.join(storage_root, path)

    # Otherwise default to project_root resolution
    return os.path.join(project_root, path) if project_root else path

def find_data_directory(primary_path, alt_path, file_type, description):
    """Find directories containing data files with comprehensive fallback options."""
    paths_to_check = [
        primary_path,                          # Primary path from config/args
        alt_path,                              # Alternative path from config
        f"/mnt/storage/data/synth/{description}s",  # Common fallback location
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data/synth/{description}s")  # Project fallback
    ]

    # Filter out None paths
    paths_to_check = [p for p in paths_to_check if p]

    for path in paths_to_check:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Count files
        count = count_files_in_subdirectories(path, file_type)
        if count > 0:
            sample_files = find_sample_files(path, file_type, 3)
            logger.info(f"  {description.capitalize()}: {path} ({count} files)")
            if sample_files:
                logger.info(f"  Example files: {sample_files}")
            return path, count

    return paths_to_check[0] if paths_to_check else None, 0

def get_quick_dataset_stats(data_loader, device, max_batches=10):
    """Calculate statistics from a small subset of data without blocking."""
    try:
        # Process in chunks with progress
        mean_sum = 0.0
        square_sum = 0.0
        sample_count = 0

        logger.info(f"Processing subset of data for quick statistics...")
        for i, batch in enumerate(data_loader):
            if i >= max_batches:  # Limit batches processed
                break

            # Move to CPU explicitly
            features = batch['spectro'].to('cpu')

            # Calculate stats
            batch_mean = torch.mean(features).item()
            batch_square_mean = torch.mean(features.pow(2)).item()

            # Update running sums
            mean_sum += batch_mean
            square_sum += batch_square_mean
            sample_count += 1

            # Free memory
            del features

            # Clear GPU cache periodically
            if i % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate final statistics
        if sample_count > 0:
            mean = mean_sum / sample_count
            square_mean = square_sum / sample_count
            std = np.sqrt(square_mean - mean**2)
            logger.info(f"Quick stats from {sample_count} batches: mean={mean:.4f}, std={std:.4f}")
            return mean, std
        else:
            logger.warning("No samples processed for statistics")
            return 0.0, 1.0
    except Exception as e:
        logger.error(f"Error in quick stats calculation: {e}")
        return 0.0, 1.0

def main():
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
                        help='Minimum learning rate for schedulers (overrides config value)')
    parser.add_argument('--warmup_end_lr', type=float, default=None,
                       help='Target learning rate at the end of warm-up (default: base LR)')

    # Modify dataset_type choices to include 'combined'
    parser.add_argument('--dataset_type', type=str, choices=['fma', 'maestro', 'combined'], default='fma',
                      help='Dataset format type: fma (numeric IDs), maestro (arbitrary filenames), or combined (both)')

    # Checkpoint loading
    parser.add_argument('--load_checkpoint', type=str, default=None,
                      help='Path to checkpoint file to resume training from')
    parser.add_argument('--reset_epoch', action='store_true',
                      help='Start from epoch 1 even when loading from checkpoint')
    parser.add_argument('--reset_scheduler', action='store_true',
                      help='Reset learning rate scheduler when --reset_epoch is used')

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
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=args.distributed_backend)
            device = torch.device(f"cuda:{local_rank}")
        elif args.rank is not None and args.world_size is not None:
            # Multi-node training with manual setup
            rank = args.rank
            world_size = args.world_size
            logger.info(f"Initializing distributed training with rank={rank}, world_size={world_size}")
            dist.init_process_group(backend=args.distributed_backend,
                                   init_method=args.dist_url,
                                   world_size=world_size,
                                   rank=rank)
            local_rank = rank % torch.cuda.device_count()
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

    # Resolve primary paths from config, with CLI override
    spec_dir_config = resolve_path(args.spec_dir or config.paths.get('spec_dir', 'data/logits/synth/spectrograms'),
                                  storage_root, project_root)
    label_dir_config = resolve_path(args.label_dir or config.paths.get('label_dir', 'data/logits/synth/labels'),
                                   storage_root, project_root)

    # Resolve alternative paths if available
    alt_spec_dir = resolve_path(config.paths.get('alt_spec_dir'), storage_root, project_root) if config.paths.get('alt_spec_dir') else None
    alt_label_dir = resolve_path(config.paths.get('alt_label_dir'), storage_root, project_root) if config.paths.get('alt_label_dir') else None

    logger.info(f"Looking for data files:")

    # For combined mode, also find Maestro paths using standardized directory structure
    if args.dataset_type == 'combined':
        # Use standardized paths based on the DATA_ROOT
        data_root = os.environ.get('DATA_ROOT', '/mnt/storage/data')

        # Standard paths for Maestro dataset
        maestro_spec_dir = os.path.join(data_root, "logits/maestro_synth/spectrograms")
        maestro_label_dir = os.path.join(data_root, "logits/maestro_synth/labels")
        maestro_logits_dir = os.path.join(data_root, "logits/maestro_synth/logits")

        # Find FMA data directories
        fma_spec_dir, fma_spec_count = find_data_directory(spec_dir_config, alt_spec_dir, "*.npy", "FMA spectrogram")
        fma_label_dir, fma_label_count = find_data_directory(label_dir_config, alt_label_dir, "*.lab", "FMA label")
        fma_logits_dir = None
        if args.logits_dir:
            fma_logits_dir, _ = find_data_directory(args.logits_dir, None, "*.npy", "FMA logits")
        else:
            # Use standard path for FMA logits
            fma_logits_dir = os.path.join(data_root, "logits/synth/logits")

        # Find Maestro data directories
        maestro_spec_count = count_files_in_subdirectories(maestro_spec_dir, "*.npy")
        maestro_label_count = count_files_in_subdirectories(maestro_label_dir, "*.lab")

        # Log combined dataset files
        logger.info(f"\n=== Combined Dataset Files ===")
        logger.info(f"FMA: {fma_spec_count} spectrograms, {fma_label_count} labels")
        logger.info(f"Maestro: {maestro_spec_count} spectrograms, {maestro_label_count} labels (in {maestro_spec_dir})")

        # Final check - fail if we have no data from either dataset
        total_spec_count = fma_spec_count + maestro_spec_count
        total_label_count = fma_label_count + maestro_label_count

        if total_spec_count == 0 or total_label_count == 0:
            raise RuntimeError(f"ERROR: Missing spectrogram or label files in combined mode. Found {total_spec_count} total spectrogram files and {total_label_count} total label files.")

        # Use lists for spec_dir, label_dir, and logits_dir in combined mode
        spec_dir = [fma_spec_dir, maestro_spec_dir]
        label_dir = [fma_label_dir, maestro_label_dir]
        logits_dir = [fma_logits_dir, maestro_logits_dir]

    else:
        # Original single dataset mode
        synth_spec_dir, spec_count = find_data_directory(spec_dir_config, alt_spec_dir, "*.npy", "spectrogram")
        synth_label_dir, label_count = find_data_directory(label_dir_config, alt_label_dir, "*.lab", "label")

        # Final check - fail if we still don't have data
        if spec_count == 0 or label_count == 0:
            raise RuntimeError(f"ERROR: Missing spectrogram or label files. Found {spec_count} spectrogram files and {label_count} label files.")

        # Use single directories for spec_dir, label_dir
        spec_dir = synth_spec_dir
        label_dir = synth_label_dir
        logits_dir = args.logits_dir

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
        'label_dir': label_dir,
        'logits_dir': logits_dir,
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
        # Create distributed samplers
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            synth_dataset.train_indices,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            synth_dataset.eval_indices,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        train_loader = synth_dataset.get_train_iterator(
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle here, the sampler will do it
            sampler=train_sampler,
            num_workers=0,  # Force single worker for GPU optimization
            pin_memory=False
        )

        val_loader = synth_dataset.get_eval_iterator(
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
        logger.info(f"First batch loaded successfully: {batch['spectro'].shape}")

        if torch.cuda.is_available():
            if batch['spectro'].device.type == 'cuda':
                logger.info("Success: Batch tensors are already on GPU")
            else:
                logger.info("Note: Batch tensors are on CPU and will be moved to GPU during training")
    except Exception as e:
        logger.error(f"ERROR: Failed to load first batch from train_loader: {e}")
        logger.error("Cannot proceed with training due to data loading issue.")
        return

    # Initialize model
    logger.info("\n=== Creating model ===")

    # Get frequency dimension and class count
    n_freq = getattr(config.feature, 'freq_bins', 144)
    n_classes = len(chord_mapping)
    logger.info(f"Using frequency dimension: {n_freq}")
    logger.info(f"Output classes: {n_classes}")

    # Apply model scale factor
    if model_scale != 1.0:
        n_group = max(1, int(32 * model_scale))
        logger.info(f"Using n_group={n_group}, resulting in feature dimension: {n_freq // n_group}")
    else:
        n_group = config.model.get('n_group', 32)

    # Get dropout value
    dropout_rate = args.dropout if args.dropout is not None else config.model.get('dropout', 0.3)
    logger.info(f"Using dropout rate: {dropout_rate}")

    # Create model instance
    model = ChordNet(
        n_freq=n_freq,
        n_classes=n_classes,
        n_group=n_group,
        f_layer=config.model.get('base_config', {}).get('f_layer', 3),
        f_head=config.model.get('base_config', {}).get('f_head', 6),
        t_layer=config.model.get('base_config', {}).get('t_layer', 3),
        t_head=config.model.get('base_config', {}).get('t_head', 6),
        d_layer=config.model.get('base_config', {}).get('d_layer', 3),
        d_head=config.model.get('base_config', {}).get('d_head', 6),
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

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training['learning_rate'],
        weight_decay=config.training.get('weight_decay', 0.0)
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

    # Calculate dataset statistics efficiently
    try:
        logger.info("Calculating global mean and std for normalization...")

        # Create stats loader
        stats_batch_size = min(16, config.training['batch_size'])
        stats_loader = torch.utils.data.DataLoader(
            synth_dataset,
            batch_size=stats_batch_size,
            sampler=torch.utils.data.RandomSampler(
                synth_dataset,
                replacement=True,
                num_samples=min(1000, len(synth_dataset))
            ),
            num_workers=0,
            pin_memory=False
        )

        mean, std = get_quick_dataset_stats(stats_loader, device)
        logger.info(f"Using statistics: mean={mean:.4f}, std={std:.4f}")
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        mean, std = 0.0, 1.0
        logger.warning("Using default mean=0.0, std=1.0 due to calculation error")

    # Create normalized tensors on device
    mean = torch.tensor(mean, device=device)
    std = torch.tensor(std, device=device)
    normalization = {'mean': mean, 'std': std}

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
        trainer = DistributedStudentTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            num_epochs=config.training.get('num_epochs', config.training.get('max_epochs', 100)),
            logger=logger,
            checkpoint_dir=checkpoints_dir,
            class_weights=None,
            idx_to_chord=master_mapping,
            normalization=normalization,
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
            world_size=world_size
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
            normalization=normalization,
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
                    test_sampler = torch.utils.data.distributed.DistributedSampler(
                        synth_dataset.test_indices,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=False
                    )
                    test_loader = synth_dataset.get_test_iterator(
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
                    device=device,
                    idx_to_chord=master_mapping,
                    normalization=normalization
                )

                # Run test
                test_loss, test_acc = tester.test(test_loader)
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
                    # Split dataset into 3 parts for evaluation
                    valid_dataset1 = synth_dataset.samples[:dataset_length//3]
                    valid_dataset2 = synth_dataset.samples[dataset_length//3:2*dataset_length//3]
                    valid_dataset3 = synth_dataset.samples[2*dataset_length//3:]

                    # Evaluate each split
                    logger.info(f"Evaluating model on {len(valid_dataset1)} samples in split 1...")
                    score_list_dict1, song_length_list1, average_score_dict1 = large_voca_score_calculation(
                        valid_dataset=valid_dataset1, config=config, model=model, model_type=args.model,
                        mean=mean, std=std, device=device)

                    logger.info(f"Evaluating model on {len(valid_dataset2)} samples in split 2...")
                    score_list_dict2, song_length_list2, average_score_dict2 = large_voca_score_calculation(
                        valid_dataset=valid_dataset2, config=config, model=model, model_type=args.model,
                        mean=mean, std=std, device=device)

                    logger.info(f"Evaluating model on {len(valid_dataset3)} samples in split 3...")
                    score_list_dict3, song_length_list3, average_score_dict3 = large_voca_score_calculation(
                        valid_dataset=valid_dataset3, config=config, model=model, model_type=args.model,
                        mean=mean, std=std, device=device)

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

                    # Log results
                    logger.info("\nMIR Evaluation Results:")
                    for metric, values in mir_eval_results.items():
                        logger.info(f"{metric}: {values['average']*100:.2f}% (avg)")

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

    # Call main function
    main()

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    main()