import json # Add json import if not already present
import traceback # Add traceback import if not already present
import multiprocessing # Add multiprocessing import
import sys # Add sys import
import os # Add os import
import torch # Add torch import
import numpy as np # Add numpy import
import argparse # Add argparse import
import glob # Add glob import
import gc # Add gc import
import random # Add random import
from pathlib import Path
from collections import Counter

# Define the standard pitch classes for chord notation
PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.device import get_device, is_cuda_available, is_gpu_available, clear_gpu_cache
from modules.data.LabeledDataset import LabeledDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.models.Transformer.btc_model import BTC_model  # Import BTC model
from modules.training.StudentTrainer import StudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord, Chords
from modules.training.Tester import Tester
from modules.utils.teacher_utils import load_btc_model, extract_logits_from_teacher

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
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained chord recognition model on real labeled data")
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config value)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides config value)')
    parser.add_argument('--pretrained', type=str, required=False,
                        help='Path to pretrained model checkpoint (required for ChordNet, optional for BTC)')
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
    parser.add_argument('--teacher_model', type=str, default=None,
                       help='Path to teacher model for knowledge distillation')
    parser.add_argument('--kd_debug_mode', action='store_true',
                       help='Enable debug mode for teacher logit extraction')

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

    # Data directories for LabeledDataset
    parser.add_argument('--audio_dirs', type=str, nargs='+', default=None,
                      help='Directories containing audio files')
    parser.add_argument('--label_dirs', type=str, nargs='+', default=None,
                      help='Directories containing label files')
    parser.add_argument('--cache_dir', type=str, default=None,
                      help='Directory to cache extracted features')

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

    # Fine-tuning specific options
    parser.add_argument('--freeze_feature_extractor', action='store_true',
                       help='Freeze the feature extraction part of the model')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (overrides config value)')

    # Checkpoint loading
    parser.add_argument('--reset_epoch', action='store_true',
                      help='Start from epoch 1 when loading pretrained model')
    parser.add_argument('--reset_scheduler', action='store_true',
                      help='Reset learning rate scheduler when loading pretrained model')
    parser.add_argument('--timeout_minutes', type=int, default=30,
                      help='Timeout in minutes for distributed operations (default: 30)')

    # Add parameters to handle model loading/saving
    parser.add_argument('--force_num_classes', type=int, default=None,
                        help='Force the model to use this number of output classes (e.g., 170 or 205)')
    parser.add_argument('--partial_loading', action='store_true',
                        help='Allow partial loading of output layer when model sizes differ')

    # Add option for large vocabulary
    parser.add_argument('--use_voca', action='store_true',
                        help='Use large vocabulary (170 chord types instead of standard 25)')

    # Add model type argument
    parser.add_argument('--model_type', type=str, choices=['ChordNet', 'BTC'], default='ChordNet',
                        help='Type of model to use (ChordNet or BTC)')

    # Add BTC model checkpoint path
    parser.add_argument('--btc_checkpoint', type=str, default=None,
                        help='Path to BTC model checkpoint for finetuning (if model_type=BTC)')

    # Add detailed chord logging argument
    parser.add_argument('--log_chord_details', action='store_true',
                       help='Enable detailed logging of chords during MIR evaluation')

    args = parser.parse_args()

    # Load configuration from YAML first
    config = HParams.load(args.config)

    # --- REMOVED Environment variable override block ---

    # Override with command line args
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

    # Handle KD loss setting with proper type conversion
    use_kd_loss = args.use_kd_loss

    # Be more explicit about logging KD settings
    if use_kd_loss:
        logger.info("Knowledge Distillation Loss ENABLED via command line argument")
    else:
        # Check if any config value might be enabling KD
        config_kd = config.training.get('use_kd_loss', False)
        if isinstance(config_kd, str):
            config_kd = config_kd.lower() == "true"
        else:
            config_kd = bool(config_kd)

        if config_kd:
            logger.info("Knowledge Distillation Loss ENABLED via config file")
            use_kd_loss = True
        else:
            logger.info("Knowledge Distillation Loss DISABLED - will use standard loss")
            use_kd_loss = False

    # Set large vocabulary config if specified - with proper boolean handling
    if args.use_voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170  # 170 chord types
        logger.info("Using large vocabulary with 170 chord classes")
    elif hasattr(config.feature, 'large_voca'):
        # Handle string "true"/"false" in config
        if isinstance(config.feature.large_voca, str):
            config.feature['large_voca'] = config.feature.large_voca.lower() == "true"
        if config.feature.get('large_voca', False):
            config.model['num_chords'] = 170
            logger.info("Using large vocabulary with 170 chord classes (from config)")

    # Handle learning rate and warmup parameters
    config.training['learning_rate'] = float(args.learning_rate) if args.learning_rate is not None else float(config.training.get('learning_rate', 0.0001))
    config.training['min_learning_rate'] = float(args.min_learning_rate) if args.min_learning_rate is not None else float(config.training.get('min_learning_rate', 5e-6))

    # Only set warmup_epochs if explicitly provided
    if args.warmup_epochs is not None:
        config.training['warmup_epochs'] = int(args.warmup_epochs)

    # Handle warmup_start_lr properly
    if args.warmup_start_lr is not None:
        config.training['warmup_start_lr'] = float(args.warmup_start_lr)
    elif 'warmup_start_lr' not in config.training:
        config.training['warmup_start_lr'] = config.training['learning_rate']/10

    # Handle warmup_end_lr properly
    if args.warmup_end_lr is not None:
        config.training['warmup_end_lr'] = float(args.warmup_end_lr)
    elif 'warmup_end_lr' not in config.training:
        config.training['warmup_end_lr'] = config.training['learning_rate']

    # Override number of epochs if specified
    if args.epochs is not None:
        config.training['num_epochs'] = int(args.epochs)

    # Override batch size if specified
    if args.batch_size is not None:
        config.training['batch_size'] = int(args.batch_size)

    # Log parameters that have been overridden
    logger.info(f"Using learning rate: {config.training['learning_rate']}")
    logger.info(f"Using minimum learning rate: {config.training['min_learning_rate']}")
    if 'warmup_epochs' in config.training:
        logger.info(f"Using warmup_epochs: {config.training['warmup_epochs']}")
    logger.info(f"Using warmup_start_lr: {config.training.get('warmup_start_lr')}")
    logger.info(f"Using warmup_end_lr: {config.training.get('warmup_end_lr')}")
    logger.info(f"Using {config.training.get('num_epochs')} epochs for fine-tuning")
    logger.info(f"Using batch size: {config.training.get('batch_size', 16)}")

    # Log training configuration with proper type conversions
    logger.info("\n=== Fine-tuning Configuration ===")
    model_scale = float(args.model_scale) if args.model_scale is not None else float(config.model.get('scale', 1.0))
    logger.info(f"Model scale: {model_scale}")
    logger.info(f"Pretrained model: {args.pretrained}")
    if args.freeze_feature_extractor:
        logger.info("Feature extraction layers will be frozen during fine-tuning")

    # Log knowledge distillation settings with proper type handling
    kd_alpha = float(args.kd_alpha) if args.kd_alpha is not None else float(config.training.get('kd_alpha', 0.5))
    temperature = float(args.temperature) if args.temperature is not None else float(config.training.get('temperature', 1.0))

    if use_kd_loss:
        logger.info("\n=== Knowledge Distillation Enabled ===")
        logger.info(f"KD alpha: {kd_alpha} (weighting between KD and CE loss)")
        logger.info(f"Temperature: {temperature} (for softening distributions)")
        if args.teacher_model:
            logger.info(f"Using teacher model from: {args.teacher_model}")
        else:
            logger.info("No teacher model specified - teacher logits must be generated separately")
    else:
        logger.info("Knowledge distillation is disabled, using standard loss")

    # Handle focal loss settings with proper type conversion
    use_focal_loss = args.use_focal_loss
    if not use_focal_loss:
        config_focal = config.training.get('use_focal_loss', False)
        if isinstance(config_focal, str):
            use_focal_loss = config_focal.lower() == "true"
        else:
            use_focal_loss = bool(config_focal)

    focal_gamma = float(args.focal_gamma) if args.focal_gamma is not None else float(config.training.get('focal_gamma', 2.0))

    if args.focal_alpha is not None:
        focal_alpha = float(args.focal_alpha)
    elif config.training.get('focal_alpha') is not None:
        focal_alpha = float(config.training.get('focal_alpha'))
    else:
        focal_alpha = None

    if use_focal_loss:
        logger.info("\n=== Focal Loss Enabled ===")
        logger.info(f"Gamma: {focal_gamma}")
        if focal_alpha is not None:
            logger.info(f"Alpha: {focal_alpha}")
    else:
        logger.info("Using standard cross-entropy loss")

    # Clear summary of loss function configuration
    if use_kd_loss and use_focal_loss:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha}) combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * focal_loss")
    elif use_kd_loss:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using standard Cross Entropy combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * cross_entropy")
    elif use_focal_loss:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using only Focal Loss with gamma={focal_gamma}, alpha={focal_alpha}")
    else:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info("Using only standard Cross Entropy Loss")

    # Set random seed for reproducibility
    seed = int(config.misc['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

    # Initialize dataset_args dictionary
    dataset_args = {}

    # Set dataset type - use small dataset for quick testing if specified
    dataset_args['small_dataset_percentage'] = args.small_dataset
    if dataset_args['small_dataset_percentage'] is None:
        logger.info("Using full dataset")
    elif dataset_args['small_dataset_percentage'] <= 0 or dataset_args['small_dataset_percentage'] > 1.0:
        logger.warning(f"Invalid small_dataset value: {dataset_args['small_dataset_percentage']}. Must be between 0 and 1. Using full dataset.")
        dataset_args['small_dataset_percentage'] = 1.0
    else:
        logger.info(f"Using {dataset_args['small_dataset_percentage']*100:.1f}% of dataset")

    # Set up logging
    logger.logging_verbosity(config.misc.get('logging_level', 'INFO'))

    # Get project root and storage root
    project_root = os.path.dirname(os.path.abspath(__file__))
    storage_root = config.paths.get('storage_root', None)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Storage root: {storage_root}")

    # Resolve paths for LabeledDataset
    if args.audio_dirs:
        audio_dirs = [resolve_path(d, storage_root, project_root) for d in args.audio_dirs]
    else:
        # Default audio directories in standard location
        default_audio_dirs = [
            os.path.join('/mnt/storage/data/LabeledDataset/Audio', subdir)
            for subdir in ['billboard', 'caroleKing', 'queen', 'theBeatles']
        ]
        audio_dirs = default_audio_dirs
        logger.info(f"Using default audio directories: {audio_dirs}")

    if args.label_dirs:
        label_dirs = [resolve_path(d, storage_root, project_root) for d in args.label_dirs]
    else:
        # Default label directories in standard location
        default_label_dirs = [
            os.path.join('/mnt/storage/data/LabeledDataset/Labels', subdir)
            for subdir in ['billboardLabels', 'caroleKingLabels', 'queenLabels', 'theBeatlesLabels']
        ]
        label_dirs = default_label_dirs
        logger.info(f"Using default label directories: {label_dirs}")

    # Resolve cache directory
    cache_dir = resolve_path(args.cache_dir or '/mnt/storage/data/LabeledDataset/cache', storage_root, project_root)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Using feature cache directory: {cache_dir}")

    # Check for audio files in each directory
    logger.info("\n=== Checking for audio and label files ===")
    for audio_dir, label_dir in zip(audio_dirs, label_dirs):
        audio_count = count_files_in_subdirectories(audio_dir, "*.mp3") + count_files_in_subdirectories(audio_dir, "*.wav")
        label_count = count_files_in_subdirectories(label_dir, "*.lab") + count_files_in_subdirectories(label_dir, "*.txt")
        logger.info(f"Found {audio_count} audio files in {audio_dir}")
        logger.info(f"Found {label_count} label files in {label_dir}")

    # Set up chord processing using the Chords class
    logger.info("\n=== Setting up chord mapping ===")
    # Get the mapping from idx2voca_chord - THIS IS THE SOURCE OF TRUTH
    master_mapping = idx2voca_chord()
    # Create a reverse mapping for dataset initialization if needed
    chord_mapping = {chord: idx for idx, chord in master_mapping.items()}

    # Initialize Chords class (optional, if needed elsewhere, but mapping is key)
    chord_processor = Chords()
    chord_processor.set_chord_mapping(chord_mapping) # Use the reverse mapping here
    chord_processor.initialize_chord_mapping() # Initialize variants

    # Log mapping info
    logger.info(f"\nUsing idx->chord mapping from idx2voca_chord with {len(master_mapping)} entries")
    logger.info(f"Sample idx->chord mapping: {dict(list(master_mapping.items())[:5])}")
    logger.info(f"Reverse chord->idx mapping created with {len(chord_mapping)} entries")
    logger.info(f"Sample chord->idx mapping: {dict(list(chord_mapping.items())[:5])}")

    # Always use the external checkpoint path in /mnt/storage/checkpoints
    external_dir = "/mnt/storage/checkpoints/btc"
    try:
        os.makedirs(external_dir, exist_ok=True)
        checkpoints_dir = external_dir
        logger.info(f"Using external checkpoint directory: {external_dir}")
    except Exception as e:
        # Fall back to config path if external directory can't be created
        checkpoints_dir_config = config.paths.get('checkpoints_dir', 'checkpoints/finetune')
        checkpoints_dir = resolve_path(checkpoints_dir_config, storage_root, project_root)
        logger.info(f"Could not use external directory {external_dir}: {e}")
        logger.info(f"Using fallback checkpoint directory: {checkpoints_dir}")

    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")

    # Setup feature configuration for LabeledDataset
    feature_config = {
        'mp3': {
            'song_hz': config.feature.get('sample_rate', 22050),
            'inst_len': config.mp3.get('inst_len', 10.0),
            'skip_interval': config.mp3.get('skip_interval', 5.0)
        },
        'feature': {
            'n_fft': config.feature.get('n_fft', 512),
            'hop_length': config.feature.get('hop_length', 2048),
            'n_bins': config.feature.get('n_bins', 144),
            'bins_per_octave': config.feature.get('bins_per_octave', 24),
            'hop_duration': config.feature.get('hop_duration', 0.09288)
        }
    }

    # Initialize LabeledDataset
    logger.info("\n=== Creating dataset ===")
    labeled_dataset = LabeledDataset(
        audio_dirs=audio_dirs,
        label_dirs=label_dirs,
        chord_mapping=chord_mapping,  # Pass the mapping, not the Chords instance
        seq_len=config.training.get('seq_len', 10),
        stride=config.training.get('seq_stride', 5),
        cache_features=not args.disable_cache,
        cache_dir=cache_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=seed,
        feature_config=feature_config,
        device=device,
        small_dataset_percentage=dataset_args.get('small_dataset_percentage', 1.0)
    )

    # Sample a few label files to analyze
    sample_songs = random.sample(labeled_dataset.audio_label_pairs, min(5, len(labeled_dataset.audio_label_pairs)))
    logger.info("Analyzing sample label files for diagnostic purposes...")
    for audio_path, label_path in sample_songs:
        labeled_dataset.analyze_label_file(label_path)

    # Create data loaders for each subset
    batch_size = config.training.get('batch_size', 16)
    logger.info(f"Using batch size: {batch_size}")

    train_loader = labeled_dataset.get_train_iterator(
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = labeled_dataset.get_val_iterator(
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Check train and validation dataset sizes
    logger.info(f"Training set: {len(labeled_dataset.train_indices)} samples")
    logger.info(f"Validation set: {len(labeled_dataset.val_indices)} samples")
    logger.info(f"Test set: {len(labeled_dataset.test_indices)} samples")

    # Try to load the first batch to check data loader works
    logger.info("\n=== Checking data loaders ===")
    try:
        batch = next(iter(train_loader))
        logger.info(f"First batch loaded successfully: {batch['spectro'].shape}")
    except Exception as e:
        logger.error(f"ERROR: Failed to load first batch from train_loader: {e}")
        logger.error("Cannot proceed with training due to data loading issue.")
        return

    # Determine the correct number of output classes
    if args.force_num_classes is not None:
        n_classes = args.force_num_classes
        logger.info(f"Forcing model to use {n_classes} output classes as specified by --force_num_classes")
    elif args.use_voca or config.training.get('use_voca', False) or config.feature.get('large_voca', False):
        n_classes = 170  # Standard number for large vocabulary
        logger.info(f"Using large vocabulary with {n_classes} output classes")
    else:
        n_classes = 25   # Standard number for small vocabulary (major/minor only)
        logger.info(f"Using small vocabulary with {n_classes} output classes")

    # Determine model type
    model_type = args.model_type
    if model_type not in ['ChordNet', 'BTC']:
        logger.warning(f"Unknown model type: {model_type}. Defaulting to ChordNet.")
        model_type = 'ChordNet'

    logger.info(f"Using model type: {model_type}")

    # Check for BTC checkpoint if model_type is BTC
    if model_type == 'BTC':
        if args.btc_checkpoint:
            logger.info(f"\n=== Loading BTC model from {args.btc_checkpoint} ===")
            pretrained_path = args.btc_checkpoint
        else:
            # For BTC model, we can proceed without a checkpoint (will initialize a fresh model)
            logger.info("\n=== No BTC checkpoint specified, will initialize a fresh BTC model ===")
            pretrained_path = None
    elif args.pretrained:
        # Use the standard pretrained path for ChordNet
        pretrained_path = args.pretrained
        logger.info(f"\n=== Loading ChordNet model from {pretrained_path} ===")
    else:
        # No pretrained model specified for ChordNet
        logger.error("No pretrained model specified. Please provide --pretrained for ChordNet model")
        return

    try:
        # Get frequency dimension
        n_freq = getattr(config.feature, 'freq_bins', 144)
        logger.info(f"Using frequency dimension: {n_freq}")

        # Apply model scale factor (only for ChordNet)
        if model_type == 'ChordNet':
            if model_scale != 1.0:
                n_group = max(1, int(32 * model_scale))
                logger.info(f"Using n_group={n_group}, resulting in feature dimension: {n_freq // n_group}")
            else:
                n_group = config.model.get('n_group', 32)
        else:
            # For BTC, n_group is not used
            n_group = None
            logger.info("n_group parameter not used for BTC model")

        # Get dropout value
        dropout_rate = args.dropout if args.dropout is not None else config.model.get('dropout', 0.3)
        logger.info(f"Using dropout rate: {dropout_rate}")

        # Create fresh model instance based on model type
        model_type = args.model_type
        if model_type not in ['ChordNet', 'BTC']:
            logger.warning(f"Unknown model type: {model_type}. Defaulting to ChordNet.")
            model_type = 'ChordNet'

        logger.info(f"Creating {model_type} model with {n_classes} output classes")

        if model_type == 'ChordNet':
            model = ChordNet(
                n_freq=n_freq,
                n_classes=n_classes,  # Use determined number of classes
                n_group=n_group,
                f_layer=config.model.get('base_config', {}).get('f_layer', 3),
                f_head=config.model.get('base_config', {}).get('f_head', 6),
                t_layer=config.model.get('base_config', {}).get('t_layer', 3),
                t_head=config.model.get('base_config', {}).get('t_head', 6),
                d_layer=config.model.get('base_config', {}).get('d_layer', 3),
                d_head=config.model.get('base_config', {}).get('d_head', 6),
                dropout=dropout_rate
            ).to(device)
        else:  # BTC model
            # BTC specific parameters
            # Create a config dictionary for BTC model
            btc_config = {
                'feature_size': n_freq,
                'hidden_size': config.model.get('hidden_size', 128),
                'num_layers': config.model.get('num_layers', 8),
                'num_heads': config.model.get('num_heads', 4),
                'total_key_depth': config.model.get('total_key_depth', 128),
                'total_value_depth': config.model.get('total_value_depth', 128),
                'filter_size': config.model.get('filter_size', 128),
                'seq_len': config.model.get('timestep', 108),
                'input_dropout': config.model.get('input_dropout', 0.2),
                'layer_dropout': config.model.get('layer_dropout', 0.2),
                'attention_dropout': config.model.get('attention_dropout', 0.2),
                'relu_dropout': config.model.get('relu_dropout', 0.2),
                'num_chords': n_classes
            }

            logger.info(f"Using BTC model with parameters:")
            for key, value in btc_config.items():
                logger.info(f"  {key}: {value}")

            model = BTC_model(config=btc_config).to(device)

        # Attach chord mapping to model
        model.idx_to_chord = master_mapping
        logger.info("Attached chord mapping to model for correct MIR evaluation")

        # Load pretrained weights if available
        if pretrained_path:
            try:
                checkpoint = torch.load(pretrained_path, map_location=device)

                # Check if the checkpoint contains model dimensions info
                if 'n_classes' in checkpoint:
                    pretrained_classes = checkpoint['n_classes']
                    logger.info(f"Pretrained model has {pretrained_classes} output classes")

                    # Warn if there's a mismatch
                    if pretrained_classes != n_classes:
                        logger.warning(f"Mismatch in class count: pretrained={pretrained_classes}, current={n_classes}")

                        if not args.partial_loading:
                            logger.warning("Loading may fail. Use --partial_loading to attempt partial weights loading.")

                # Extract the state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # Handle the case where the model was saved with DataParallel or DistributedDataParallel
                # which adds 'module.' prefix to all keys
                if model_type == 'BTC' and all(k.startswith('module.') for k in state_dict.keys()):
                    logger.info("Detected 'module.' prefix in state dict keys. Removing prefix for compatibility.")
                    # Create a new state dict with the 'module.' prefix removed from all keys
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        # Remove the 'module.' prefix
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    state_dict = new_state_dict

                # Load weights with partial loading option
                model.load_state_dict(state_dict, strict=not args.partial_loading)
                logger.info("Successfully loaded pretrained weights")
            except Exception as e:
                logger.error(f"Error loading pretrained model from {pretrained_path}: {e}")
                if model_type == 'BTC':
                    logger.info("Continuing with freshly initialized BTC model")
                else:
                    logger.error("Cannot continue without pretrained weights for ChordNet model")
                    return
        else:
            if model_type == 'BTC':
                logger.info("No pretrained weights provided. Using freshly initialized BTC model.")
            else:
                logger.error("Cannot continue without pretrained weights for ChordNet model")
                return

        # Freeze feature extraction layers if requested
        if args.freeze_feature_extractor:
            logger.info("Freezing feature extraction layers:")
            for name, param in model.named_parameters():
                if 'frequency_net' in name:
                    param.requires_grad = False
                    logger.info(f"  Frozen: {name}")

            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")

    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        logger.error(traceback.format_exc())
        return

    # Load teacher model if provided
    teacher_model = None
    teacher_mean = None
    teacher_std = None

    # Check for teacher model - either from args or try to find btc_model_large_voca.pt
    teacher_model_path = args.teacher_model
    if not teacher_model_path:
        # Try to find the BTC model in the external storage path
        external_btc_path = "/mnt/storage/checkpoints/btc/btc_model_large_voca.pt"
        if os.path.exists(external_btc_path):
            logger.info(f"Found BTC teacher model at external path: {external_btc_path}")
            teacher_model_path = external_btc_path
            # Enable KD loss automatically if we found a teacher model
            if not use_kd_loss:
                use_kd_loss = True
                logger.info("Automatically enabling knowledge distillation with found teacher model")
                kd_alpha = args.kd_alpha if args.kd_alpha is not None else float(config.training.get('kd_alpha', 0.5))
                temperature = args.temperature if args.temperature is not None else float(config.training.get('temperature', 1.0))
                logger.info(f"Using KD alpha: {kd_alpha}, temperature: {temperature}")

    if teacher_model_path:
        logger.info(f"\n=== Loading teacher model from {teacher_model_path} ===")
        try:
            # Determine vocabulary size based on args and config
            use_voca = args.use_voca or config.feature.get('large_voca', False)

            # Load the teacher model with enhanced error handling
            teacher_model, teacher_mean, teacher_std, teacher_status = load_btc_model(
                teacher_model_path,
                device,
                use_voca=use_voca
            )

            # Check if loading was successful
            if teacher_status["success"] and teacher_model is not None:
                logger.info(f"Teacher model loaded successfully: {teacher_status['message']}")
                logger.info(f"Model implementation: {teacher_status['implementation']}")
                logger.info(f"Model validation: {teacher_status['model_validated']}")

                # If we don't have explicit KD loss but loaded a teacher, enable it
                if not use_kd_loss:
                    use_kd_loss = True
                    logger.info("Automatically enabling knowledge distillation with loaded teacher model")
                    kd_alpha = args.kd_alpha if args.kd_alpha is not None else float(config.training.get('kd_alpha', 0.5))
                    temperature = args.temperature if args.temperature is not None else float(config.training.get('temperature', 1.0))
                    logger.info(f"Using KD alpha: {kd_alpha}, temperature: {temperature}")

                # Check if we have mean/std for normalization
                if not teacher_status["has_mean_std"]:
                    logger.warning("Teacher model does not have mean/std values for normalization")
                    logger.warning("Using default values: mean=0.0, std=1.0")
            else:
                logger.error(f"Failed to load teacher model: {teacher_status['message']}")
                teacher_model = None
                use_kd_loss = False
                logger.warning("Knowledge distillation disabled due to teacher model loading failure")

        except Exception as e:
            logger.error(f"Error loading teacher model: {e}")
            logger.error(traceback.format_exc())
            teacher_model = None
            use_kd_loss = False
            logger.warning("Knowledge distillation disabled due to exception")

    # Create optimizer - only optimize unfrozen parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
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
        stats_batch_size = min(16, int(config.training.get('batch_size', 16)))
        stats_loader = torch.utils.data.DataLoader(
            labeled_dataset,
            batch_size=stats_batch_size,
            sampler=torch.utils.data.RandomSampler(
                labeled_dataset,
                replacement=True,
                num_samples=min(1000, len(labeled_dataset))
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
    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32)
    normalization = {'mean': mean_tensor, 'std': std_tensor}
    logger.info(f"Normalization tensors created on device: {device}")

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

    # Create trainer with KD support
    trainer = StudentTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=int(config.training.get('num_epochs', 50)),
        logger=logger,
        checkpoint_dir=checkpoints_dir,
        class_weights=None,
        idx_to_chord=master_mapping,
        normalization=normalization,
        early_stopping_patience=int(config.training.get('early_stopping_patience', 5)),
        lr_decay_factor=float(config.training.get('lr_decay_factor', 0.95)),
        min_lr=float(config.training.get('min_learning_rate', 5e-6)),
        use_warmup=args.use_warmup or bool(config.training.get('use_warmup', False)),
        warmup_epochs=int(config.training.get('warmup_epochs')) if config.training.get('warmup_epochs') is not None else None,
        warmup_start_lr=float(config.training.get('warmup_start_lr')) if config.training.get('warmup_start_lr') is not None else None,
        warmup_end_lr=float(config.training.get('warmup_end_lr')) if config.training.get('warmup_end_lr') is not None else None,
        lr_schedule_type=lr_schedule_type,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        use_kd_loss=use_kd_loss,
        kd_alpha=kd_alpha,
        temperature=temperature,
        teacher_model=teacher_model,
        teacher_normalization={'mean': teacher_mean, 'std': teacher_std} if teacher_mean is not None else None,
        timeout_minutes=args.timeout_minutes,
    )

    # Attach chord mapping to trainer (chord -> idx) if needed by trainer internals
    trainer.set_chord_mapping(chord_mapping)

    # Run training
    logger.info(f"\n=== Starting fine-tuning ===")
    try:
        logger.info("Preparing data (this may take a while for large datasets)...")

        # If using teacher model but not within trainer, pre-compute logits
        if teacher_model is not None and use_kd_loss and not trainer.teacher_model:
            logger.info("Pre-computing teacher logits for knowledge distillation...")
            logits_dir = os.path.join(checkpoints_dir, "teacher_logits")
            os.makedirs(logits_dir, exist_ok=True)

            from modules.utils.teacher_utils import generate_teacher_predictions

            # Use debug mode if specified
            debug_mode = args.kd_debug_mode
            if debug_mode:
                logger.info("Debug mode for teacher logit extraction is ENABLED")
            else:
                logger.info("Debug mode for teacher logit extraction is disabled")

            # Generate predictions with enhanced error handling
            teacher_preds, generation_status = generate_teacher_predictions(
                teacher_model,
                train_loader,
                teacher_mean,
                teacher_std,
                device,
                save_dir=logits_dir,
                debug_mode=debug_mode
            )

            # Check if generation was successful
            if generation_status["success"]:
                logger.info(f"Generated teacher predictions for {len(teacher_preds)} samples")
                logger.info(f"Success rate: {generation_status['successful_samples']}/{generation_status['total_samples']} samples ({generation_status['successful_samples']/generation_status['total_samples']*100:.2f}%)")
                logger.info(f"Extraction methods used: {generation_status['extraction_methods_used']}")

                # Set the predictions in the trainer
                trainer.teacher_predictions = teacher_preds

                # Save generation status for reference
                status_path = os.path.join(logits_dir, "generation_status.json")
                try:
                    with open(status_path, 'w') as f:
                        # Convert any non-serializable values to strings
                        serializable_status = {k: str(v) if not isinstance(v, (dict, list, int, float, bool, str, type(None))) else v
                                              for k, v in generation_status.items()}
                        json.dump(serializable_status, f, indent=2)
                    logger.info(f"Saved generation status to {status_path}")
                except Exception as e:
                    logger.warning(f"Failed to save generation status: {e}")
            else:
                logger.error(f"Failed to generate teacher predictions: {generation_status['message']}")
                logger.warning("Continuing without knowledge distillation")
                use_kd_loss = False
                trainer.use_kd_loss = False

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
            # Use the test iterator from labeled_dataset
            test_loader = labeled_dataset.get_test_iterator(
                batch_size=config.training.get('batch_size', 16),
                shuffle=False,
                num_workers=0, # Use 0 workers for evaluation consistency
                pin_memory=False
            )

            # Basic testing with Tester class
            tester = Tester(
                model=model,
                test_loader=test_loader,
                device=device,
                idx_to_chord=master_mapping, # Use master_mapping (idx->chord)
                normalization=normalization,
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

            # NEW: Generate chord quality distribution and accuracy visualization
            logger.info("\n=== Generating Chord Quality Distribution and Accuracy Graph ===")
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

                        # Apply normalization if available
                        if normalization and 'mean' in normalization and 'std' in normalization:
                            inputs = (inputs - normalization['mean']) / normalization['std']

                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs

                        # Handle 3D logits (sequence data) - Average over time dimension for frame-level eval
                        if logits.ndim == 3 and targets.ndim == 2:
                             # Flatten logits and targets for frame-level comparison
                            logits = logits.view(-1, logits.size(-1)) # (batch*seq_len, n_classes)
                            targets = targets.view(-1) # (batch*seq_len)
                        elif logits.ndim == 3 and targets.ndim == 1:
                             # If targets are already flat, just flatten logits
                            logits = logits.view(-1, logits.size(-1))

                        preds = logits.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())

                # Define focus qualities
                focus_qualities = ["maj", "min", "dim", "aug", "min6", "maj6", "min7",
                                  "min-maj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]

                # Create distribution and accuracy visualization
                quality_dist_path = os.path.join(checkpoints_dir, "chord_quality_distribution_accuracy.png")
                plot_chord_quality_distribution_accuracy(
                    all_preds, all_targets, master_mapping, # Use master_mapping (idx->chord)
                    save_path=quality_dist_path,
                    title="Chord Quality Distribution and Accuracy (Test Set)",
                    focus_qualities=focus_qualities
                )
                logger.info(f"Chord quality distribution and accuracy graph saved to {quality_dist_path}")
            except ImportError:
                 logger.warning("Could not import plot_chord_quality_distribution_accuracy. Skipping visualization.")
                 logger.warning("Install matplotlib and seaborn: pip install matplotlib seaborn")
            except Exception as e:
                logger.error(f"Error creating chord quality distribution graph: {e}")
                logger.error(traceback.format_exc())

            # Advanced testing with mir_eval module
            logger.info("\n=== MIR evaluation (Test Set) ===")
            score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']

            # Get the actual test samples using the indices
            test_samples = [labeled_dataset.samples[i] for i in labeled_dataset.test_indices]
            dataset_length = len(test_samples)

            if dataset_length == 0:
                logger.info("No test samples available for MIR evaluation.")
                mir_eval_results = {}
            else:
                # Import the custom evaluation functions from modules.utils.mir_eval_modules
                from modules.utils.mir_eval_modules import calculate_chord_scores

                # Prepare to collect all reference and prediction labels
                all_reference_labels = []
                all_prediction_labels = []
                all_durations = []

                # Process each test sample
                logger.info(f"Evaluating {dataset_length} test samples...")

                # Create a function to process a batch of samples and return their scores
                def evaluate_batch(samples):
                    batch_scores = {
                        'root': [], 'thirds': [], 'triads': [], 'sevenths': [],
                        'tetrads': [], 'majmin': [], 'mirex': []
                    }
                    batch_lengths = []
                    batch_refs = []
                    batch_preds = []

                    for sample in samples:
                        try:
                            # Get the feature and label data
                            feature = sample.get('feature')
                            if feature is None:
                                # If feature is not cached, load it from the audio file
                                audio_path = sample.get('audio_path')
                                if audio_path and os.path.exists(audio_path):
                                    from modules.utils.mir_eval_modules import audio_file_to_features
                                    feature, _, _ = audio_file_to_features(audio_path, config)
                                    feature = feature.T  # Transpose to [time, features]
                                else:
                                    logger.warning(f"Skipping sample {sample.get('song_id', 'unknown')}: missing feature and audio")
                                    continue

                            # Get the reference labels
                            reference_labels = sample.get('chord_label', [])
                            if not reference_labels:
                                logger.warning(f"Skipping sample {sample.get('song_id', 'unknown')}: missing reference labels")
                                continue

                            # Normalize the feature
                            if isinstance(mean, torch.Tensor):
                                mean_np = mean.cpu().numpy()
                            else:
                                mean_np = mean

                            if isinstance(std, torch.Tensor):
                                std_np = std.cpu().numpy()
                            else:
                                std_np = std

                            feature_norm = (feature - mean_np) / std_np

                            # Convert to tensor and move to device
                            feature_tensor = torch.tensor(feature_norm, dtype=torch.float32).unsqueeze(0).to(device)

                            # Get model predictions
                            model.eval()
                            with torch.no_grad():
                                outputs = model(feature_tensor)

                                # Handle different output formats
                                if isinstance(outputs, tuple):
                                    logits = outputs[0]
                                else:
                                    logits = outputs

                                # Get predicted class indices
                                predictions = logits.argmax(dim=-1).squeeze().cpu().numpy()

                            # Convert predictions to chord labels
                            pred_labels = []
                            for pred_idx in predictions:
                                chord = master_mapping.get(pred_idx, "N")
                                pred_labels.append(chord)

                            # Calculate frame duration
                            frame_duration = config.feature.get('hop_duration', 0.1)
                            feature_length = len(reference_labels)
                            timestamps = np.arange(feature_length) * frame_duration
                            durations = np.diff(np.append(timestamps, [timestamps[-1] + frame_duration]))

                            # Ensure reference and prediction labels have the same length
                            min_len = min(len(reference_labels), len(pred_labels))
                            reference_labels = reference_labels[:min_len]
                            pred_labels = pred_labels[:min_len]
                            durations = durations[:min_len]

                            # Add to batch collections
                            batch_refs.extend(reference_labels)
                            batch_preds.extend(pred_labels)

                            # Calculate scores
                            root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score = calculate_chord_scores(
                                timestamps[:min_len], durations, reference_labels, pred_labels)

                            # Add scores to batch results
                            batch_scores['root'].append(root_score)
                            batch_scores['thirds'].append(thirds_score)
                            batch_scores['triads'].append(triads_score)
                            batch_scores['sevenths'].append(sevenths_score)
                            batch_scores['tetrads'].append(tetrads_score)
                            batch_scores['majmin'].append(majmin_score)
                            batch_scores['mirex'].append(mirex_score)

                            # Add song length for weighted average
                            song_length = min_len * frame_duration
                            batch_lengths.append(song_length)

                            logger.info(f"Sample {sample.get('song_id', 'unknown')}: length={song_length:.1f}s, root={root_score:.4f}, mirex={mirex_score:.4f}")

                        except Exception as e:
                            logger.error(f"Error evaluating sample {sample.get('song_id', 'unknown')}: {str(e)}")
                            logger.error(traceback.format_exc())

                    return batch_scores, batch_lengths, batch_refs, batch_preds

                # Evaluate all samples
                if dataset_length < 3:
                    # Evaluate all samples together
                    score_list_dict, song_length_list, refs, preds = evaluate_batch(test_samples)
                    all_reference_labels.extend(refs)
                    all_prediction_labels.extend(preds)

                    # Calculate average scores
                    average_score_dict = {}
                    total_length = sum(song_length_list) if song_length_list else 0

                    if total_length > 0:
                        for metric in score_metrics:
                            weighted_sum = sum(score * length for score, length in zip(score_list_dict[metric], song_length_list))
                            average_score_dict[metric] = weighted_sum / total_length
                    else:
                        for metric in score_metrics:
                            average_score_dict[metric] = 0.0

                    # Store results
                    mir_eval_results = {m: float(average_score_dict[m]) for m in score_metrics}
                    logger.info(f"MIR evaluation results (all test samples): {mir_eval_results}")

                else:
                    # Create balanced splits from the test samples
                    split = dataset_length // 3
                    test_dataset1 = test_samples[:split]
                    test_dataset2 = test_samples[split:2*split]
                    test_dataset3 = test_samples[2*split:]

                    # Evaluate each split
                    logger.info(f"Evaluating {len(test_dataset1)} test samples in split 1...")
                    score_list_dict1, song_length_list1, refs1, preds1 = evaluate_batch(test_dataset1)
                    all_reference_labels.extend(refs1)
                    all_prediction_labels.extend(preds1)

                    logger.info(f"Evaluating {len(test_dataset2)} test samples in split 2...")
                    score_list_dict2, song_length_list2, refs2, preds2 = evaluate_batch(test_dataset2)
                    all_reference_labels.extend(refs2)
                    all_prediction_labels.extend(preds2)

                    logger.info(f"Evaluating {len(test_dataset3)} test samples in split 3...")
                    score_list_dict3, song_length_list3, refs3, preds3 = evaluate_batch(test_dataset3)
                    all_reference_labels.extend(refs3)
                    all_prediction_labels.extend(preds3)

                    # Calculate average scores for each split
                    average_score_dict1 = {}
                    average_score_dict2 = {}
                    average_score_dict3 = {}

                    total_length1 = sum(song_length_list1) if song_length_list1 else 0
                    total_length2 = sum(song_length_list2) if song_length_list2 else 0
                    total_length3 = sum(song_length_list3) if song_length_list3 else 0

                    for metric in score_metrics:
                        if total_length1 > 0:
                            weighted_sum1 = sum(score * length for score, length in zip(score_list_dict1[metric], song_length_list1))
                            average_score_dict1[metric] = weighted_sum1 / total_length1
                        else:
                            average_score_dict1[metric] = 0.0

                        if total_length2 > 0:
                            weighted_sum2 = sum(score * length for score, length in zip(score_list_dict2[metric], song_length_list2))
                            average_score_dict2[metric] = weighted_sum2 / total_length2
                        else:
                            average_score_dict2[metric] = 0.0

                        if total_length3 > 0:
                            weighted_sum3 = sum(score * length for score, length in zip(score_list_dict3[metric], song_length_list3))
                            average_score_dict3[metric] = weighted_sum3 / total_length3
                        else:
                            average_score_dict3[metric] = 0.0

                    # Calculate weighted averages across all splits
                    mir_eval_results = {}
                    total_length = total_length1 + total_length2 + total_length3

                    if total_length > 0:
                        for m in score_metrics:
                            # Calculate weighted average based on song lengths
                            avg = (total_length1 * average_score_dict1[m] +
                                   total_length2 * average_score_dict2[m] +
                                   total_length3 * average_score_dict3[m]) / total_length

                            # Calculate min and max values across the three splits
                            split_values = [
                                average_score_dict1[m] * 100,
                                average_score_dict2[m] * 100,
                                average_score_dict3[m] * 100
                            ]
                            min_val = min(split_values)
                            max_val = max(split_values)
                            avg_val = avg * 100

                            # Log individual split scores with range
                            logger.info(f"==== {m}: {avg_val:.2f}% (avg), range: [{min_val:.2f}% - {max_val:.2f}%]")

                            # Store in results dictionary
                            mir_eval_results[m] = {
                                'split1': float(average_score_dict1[m]),
                                'split2': float(average_score_dict2[m]),
                                'split3': float(average_score_dict3[m]),
                                'weighted_avg': float(avg)
                            }
                    else:
                        logger.warning("Total song length is zero, cannot calculate weighted average MIR scores.")
                        mir_eval_results = {'error': 'Total song length zero'}

                # Count 'N' (no chord) and 'X' (unknown chord) labels
                n_count_ref = sum(1 for label in all_reference_labels if label == 'N')
                x_count_ref = sum(1 for label in all_reference_labels if label == 'X')
                n_count_pred = sum(1 for label in all_prediction_labels if label == 'N')
                x_count_pred = sum(1 for label in all_prediction_labels if label == 'X')

                logger.info(f"\nChord label statistics:")
                logger.info(f"Reference labels: {len(all_reference_labels)} total, {n_count_ref} 'N' ({n_count_ref/len(all_reference_labels)*100:.1f}%), {x_count_ref} 'X' ({x_count_ref/len(all_reference_labels)*100:.1f}%)")
                logger.info(f"Predicted labels: {len(all_prediction_labels)} total, {n_count_pred} 'N' ({n_count_pred/len(all_prediction_labels)*100:.1f}%), {x_count_pred} 'X' ({x_count_pred/len(all_prediction_labels)*100:.1f}%)")

            # Save MIR-eval metrics
            try:
                mir_eval_path = os.path.join(checkpoints_dir, "mir_eval_metrics.json")
                with open(mir_eval_path, 'w') as f:
                    json.dump(mir_eval_results, f, indent=2)
                logger.info(f"MIR evaluation metrics saved to {mir_eval_path}")
            except Exception as e:
                 logger.error(f"Error saving MIR evaluation metrics: {e}")

        else:
            logger.warning("Could not load best model for testing")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.error(traceback.format_exc())

    # Save the final model (keep this part)
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

    # --- REMOVE Redundant/Incorrect MIR Evaluation Block ---
    # The following block uses 'val_dataset' which is not defined here
    # and duplicates evaluation already done on the test set. It has been removed.
    # logger.info("\n=== Advanced MIR Evaluation ===")
    # try:
    #     # Make sure we're using the best model
    #     if trainer.load_best_model():
    #         # ... (code using val_dataset) ...
    # except Exception as e:
    #     logger.error(f"Error during MIR evaluation: {e}")
    #     logger.error(traceback.format_exc())
    # --- End REMOVAL ---

    logger.info("Fine-tuning and evaluation complete!")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    main()
