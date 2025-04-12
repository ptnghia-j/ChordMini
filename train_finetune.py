import multiprocessing
import sys
import os
import torch  
import numpy as np
import argparse
import glob
import gc
import traceback
import json
import random
from pathlib import Path
from collections import Counter

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.device import get_device, is_cuda_available, is_gpu_available, clear_gpu_cache
from modules.data.LabeledDataset import LabeledDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.StudentTrainer import StudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord, Chords
from modules.training.Tester import Tester

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
    parser.add_argument('--pretrained', type=str, required=True, 
                        help='Path to pretrained model checkpoint')
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
    
    # Add parameters to handle model loading/saving
    parser.add_argument('--force_num_classes', type=int, default=None,
                        help='Force the model to use this number of output classes (e.g., 170 or 205)')
    parser.add_argument('--partial_loading', action='store_true',
                        help='Allow partial loading of output layer when model sizes differ')
    
    # Add option for large vocabulary
    parser.add_argument('--use_voca', action='store_true',
                        help='Use large vocabulary (170 chord types instead of standard 25)')
    
    args = parser.parse_args()

    # Load configuration from YAML first
    config = HParams.load(args.config)
    
    # Then check device availability
    if config.misc['use_cuda'] and is_cuda_available():
        device = get_device()
        logger.info(f"Using CUDA for training on device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training")
    
    # Override config values with command line arguments if provided
    config.misc['seed'] = args.seed or config.misc.get('seed', 42)
    config.paths['checkpoints_dir'] = args.save_dir or config.paths.get('checkpoints_dir', 'checkpoints/finetune')
    config.paths['storage_root'] = args.storage_root or config.paths.get('storage_root', None)
    
    # Set large vocabulary config if specified
    if args.use_voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170  # 170 chord types (12 roots × 14 qualities + 2 special chords)
        logger.info("Using large vocabulary with 170 chord classes")
    
    # Handle learning rate and warmup parameters
    config.training['learning_rate'] = args.learning_rate or config.training.get('learning_rate', 0.0001)
    config.training['min_learning_rate'] = args.min_learning_rate or config.training.get('min_learning_rate', 5e-6)
    
    if args.warmup_epochs is not None:
        config.training['warmup_epochs'] = args.warmup_epochs
    
    if args.warmup_start_lr is not None:
        config.training['warmup_start_lr'] = args.warmup_start_lr
    elif 'warmup_start_lr' not in config.training:
        config.training['warmup_start_lr'] = config.training['learning_rate']/10
    
    if args.warmup_end_lr is not None:
        config.training['warmup_end_lr'] = args.warmup_end_lr
    elif 'warmup_end_lr' not in config.training:
        config.training['warmup_end_lr'] = config.training['learning_rate']

    # Override number of epochs if specified
    if args.epochs is not None:
        config.training['num_epochs'] = args.epochs

    # Override batch size if specified
    if args.batch_size is not None:
        config.training['batch_size'] = args.batch_size

    # Log parameters that have been overridden
    logger.info(f"Using learning rate: {config.training['learning_rate']}")
    logger.info(f"Using minimum learning rate: {config.training['min_learning_rate']}")
    if 'warmup_epochs' in config.training:
        logger.info(f"Using warmup_epochs: {config.training['warmup_epochs']}")
    logger.info(f"Using warmup_start_lr: {config.training.get('warmup_start_lr')}")
    logger.info(f"Using warmup_end_lr: {config.training.get('warmup_end_lr')}")
    logger.info(f"Using {config.training.get('num_epochs')} epochs for fine-tuning")
    logger.info(f"Using batch size: {config.training.get('batch_size', 16)}")
    
    # Log training configuration
    logger.info("\n=== Fine-tuning Configuration ===")
    model_scale = args.model_scale or config.model.get('scale', 1.0)
    logger.info(f"Model scale: {model_scale}")
    logger.info(f"Pretrained model: {args.pretrained}")
    if args.freeze_feature_extractor:
        logger.info("Feature extraction layers will be frozen during fine-tuning")
    
    # Log knowledge distillation settings
    use_kd = args.use_kd_loss if args.use_kd_loss else config.training.get('use_kd_loss', False)
    use_kd = str(use_kd).lower() == "true"
    
    kd_alpha = args.kd_alpha or config.training.get('kd_alpha', 0.5)
    temperature = args.temperature or config.training.get('temperature', 1.0)
    
    if use_kd:
        logger.info("\n=== Knowledge Distillation Enabled ===")
        logger.info(f"KD alpha: {kd_alpha} (weighting between KD and CE loss)")
        logger.info(f"Temperature: {temperature} (for softening distributions)")
        if args.teacher_model:
            logger.info(f"Using teacher model from: {args.teacher_model}")
        else:
            logger.info("No teacher model specified - teacher logits must be generated separately")
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
    elif use_kd:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using standard Cross Entropy combined with KD Loss")
        logger.info(f"KD formula: final_loss = {kd_alpha} * KL_div_loss + {1-kd_alpha} * cross_entropy")
    elif args.use_focal_loss or config.training.get('use_focal_loss', False):
        logger.info("\n=== Final Loss Configuration ===")
        logger.info(f"Using only Focal Loss with gamma={args.focal_gamma or config.training.get('focal_gamma', 2.0)}, alpha={args.focal_alpha or config.training.get('focal_alpha')}")
    else:
        logger.info("\n=== Final Loss Configuration ===")
        logger.info("Using only standard Cross Entropy Loss")
    
    # Set random seed for reproducibility
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
    # First get the mapping from idx2voca_chord
    master_mapping = idx2voca_chord()
    # Then create a reverse mapping
    chord_mapping = {chord: idx for idx, chord in master_mapping.items()}
    
    # Initialize Chords class with the mapping
    chord_processor = Chords()
    chord_processor.set_chord_mapping(chord_mapping)
    
    # Verify and initialize chord mapping
    chord_processor.initialize_chord_mapping(chord_mapping)
    
    # Log a few chord mappings for verification
    chord_examples = ["C", "C:min", "D", "F#:7", "G:maj7", "A:min7", "N", "X"]
    logger.info("Example chord mappings:")
    for chord in chord_examples:
        if chord in chord_mapping:
            logger.info(f"  {chord} -> {chord_mapping[chord]}")
        else:
            logger.info(f"  {chord} -> Not in mapping")
    
    # Log mapping info
    logger.info(f"\nUsing chord mapping from chords.py with {len(chord_mapping)} unique chords")
    logger.info(f"Sample chord mapping: {dict(list(chord_mapping.items())[:5])}")
    
    # Resolve checkpoints directory path
    checkpoints_dir_config = config.paths.get('checkpoints_dir', 'checkpoints/finetune')
    checkpoints_dir = resolve_path(checkpoints_dir_config, storage_root, project_root)
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
        device=device
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
    
    # Load pretrained model
    logger.info(f"\n=== Loading pretrained model from {args.pretrained} ===")
    try:
        # Get frequency dimension
        n_freq = getattr(config.feature, 'freq_bins', 144)
        logger.info(f"Using frequency dimension: {n_freq}")
        
        # Apply model scale factor
        if model_scale != 1.0:
            n_group = max(1, int(32 * model_scale))
            logger.info(f"Using n_group={n_group}, resulting in feature dimension: {n_freq // n_group}")
        else:
            n_group = config.model.get('n_group', 32)
        
        # Get dropout value
        dropout_rate = args.dropout if args.dropout is not None else config.model.get('dropout', 0.3)
        logger.info(f"Using dropout rate: {dropout_rate}")
        
        # Create fresh model instance
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
        
        # Attach chord mapping to model
        model.idx_to_chord = master_mapping
        logger.info("Attached chord mapping to model for correct MIR evaluation")
        
        # Load pretrained weights
        checkpoint = torch.load(args.pretrained, map_location=device)
        
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
        
        # Load weights with partial loading option
        model.load_state_dict(state_dict, strict=not args.partial_loading)
        logger.info("Successfully loaded pretrained weights")
        
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
        stats_batch_size = min(16, config.training['batch_size'])
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

    # Create trainer with KD support
    trainer = StudentTrainer(
        model=model,
        optimizer=optimizer, 
        device=device,
        num_epochs=config.training.get('num_epochs', 50),
        logger=logger,
        checkpoint_dir=checkpoints_dir,
        class_weights=None,
        idx_to_chord=master_mapping,
        normalization=normalization,
        early_stopping_patience=config.training.get('early_stopping_patience', 5),
        lr_decay_factor=config.training.get('lr_decay_factor', 0.95),
        min_lr=config.training.get('min_learning_rate', 5e-6),
        use_warmup=args.use_warmup or config.training.get('use_warmup', False),
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
    
    # Attach chord processor to trainer
    trainer.set_chord_mapping(chord_mapping)
    
    # Run training
    logger.info(f"\n=== Starting fine-tuning ===")
    try:
        logger.info("Preparing data (this may take a while for large datasets)...")
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
            test_loader = labeled_dataset.get_test_iterator(
                batch_size=config.training['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # Basic testing with Tester class
            tester = Tester(
                model=model,
                test_loader=test_loader,
                device=device,
                idx_to_chord=master_mapping,
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
        else:
            logger.warning("Could not load best model for testing")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        logger.error(traceback.format_exc())
    
    # Save the final model
    try:
        save_path = os.path.join(checkpoints_dir, "finetune_model_final.pth")
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
    
    logger.info("Fine-tuning and evaluation complete!")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    main()
