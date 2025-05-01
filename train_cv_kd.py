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
import random # Add random for seed setting
from pathlib import Path
from torch.utils.data import DataLoader # Add DataLoader

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.device import get_device, is_cuda_available, clear_gpu_cache
from modules.data.CrossValidationDataset import CrossValidationDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.models.Transformer.btc_model import BTC_model  # Import BTC model
from modules.training.StudentTrainer import StudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord, Chords
from modules.training.Tester import Tester
from modules.utils.teacher_utils import load_btc_model, extract_logits_from_teacher, generate_teacher_predictions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a chord recognition model with cross-validation and knowledge distillation")
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=None, # Changed default to None
                        help='Random seed (overrides config value)')
    parser.add_argument('--save_dir', type=str, default=None, # Changed default to None
                        help='Directory to save checkpoints')
    parser.add_argument('--kfold', type=int, default=0,
                        help='Which fold to use for validation (0-4)')
    parser.add_argument('--total_folds', type=int, default=5,
                        help='Total number of folds')
    parser.add_argument('--storage_root', type=str, default=None,
                        help='Root directory for data storage')
    parser.add_argument('--use_voca', action='store_true',
                       help='Use large vocabulary (170 chord types)')
    parser.add_argument('--model_type', type=str, choices=['ChordNet', 'BTC'], default='ChordNet',
                        help='Type of model to use (ChordNet or BTC)')
    parser.add_argument('--btc_checkpoint', type=str, default=None,
                        help='Path to BTC model checkpoint for finetuning (if model_type=BTC)')
    parser.add_argument('--teacher_model', type=str, default=None,
                        help='Path to teacher model for knowledge distillation')
    parser.add_argument('--use_kd_loss', action='store_true',
                        help='Use knowledge distillation loss')
    parser.add_argument('--kd_alpha', type=float, default=None, # Changed default to None
                        help='Weight for knowledge distillation loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=None, # Changed default to None
                        help='Temperature for softening distributions (default: 1.0)')
    parser.add_argument('--kd_debug_mode', action='store_true',
                        help='Enable debug mode for teacher logit extraction')
    parser.add_argument('--audio_dirs', type=str, nargs='+', default=None,
                      help='Directories containing audio files')
    parser.add_argument('--label_dirs', type=str, nargs='+', default=None,
                      help='Directories containing label files')
    parser.add_argument('--cache_dir', type=str, default=None,
                      help='Directory to cache extracted features')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Base learning rate (overrides config value)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config value)')
    parser.add_argument('--preprocess', action='store_true',
                      help='Run preprocessing step to generate all features')
    parser.add_argument('--log_chord_details', action='store_true',
                       help='Enable detailed logging of chords during MIR evaluation')

    # Add arguments similar to train_finetune.py
    parser.add_argument('--model_scale', type=float, default=None,
                       help='Scaling factor for model capacity (0.5=half, 1.0=base, 2.0=double)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout probability (0-1)')
    parser.add_argument('--disable_cache', action='store_true',
                      help='Disable dataset caching (overrides CrossValidationDataset default)')
    parser.add_argument('--metadata_cache', type=str, default="false",
                      help='Only cache metadata (not spectrograms) to reduce memory usage')
    parser.add_argument('--cache_fraction', type=float, default=1.0, # Default to full cache for CV
                      help='Fraction of dataset to cache (default: 1.0 = 100%%)')
    parser.add_argument('--lazy_init', type=str, default="false",
                      help='Lazily initialize dataset components to save memory')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.9,
                      help='Fraction of GPU memory to use (default: 0.9)')
    parser.add_argument('--batch_gpu_cache', type=str, default="false",
                      help='Cache batches on GPU for repeated access patterns')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                      help='Number of batches to prefetch (default: 2)')
    parser.add_argument('--small_dataset', type=float, default=None,
                      help='Use only a small percentage of dataset for quick testing (e.g., 0.01 for 1%%)')
    parser.add_argument('--min_learning_rate', type=float, default=None,
                        help='Minimum learning rate for schedulers (overrides config value)')
    parser.add_argument('--use_warmup', action='store_true',
                       help='Use warm-up learning rate scheduling')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                       help='Number of warm-up epochs (default: from config)')
    parser.add_argument('--warmup_start_lr', type=float, default=None,
                       help='Initial learning rate for warm-up (default: 1/10 of base LR)')
    parser.add_argument('--warmup_end_lr', type=float, default=None,
                       help='Target learning rate at the end of warm-up (default: base LR)')
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
    parser.add_argument('--reset_epoch', action='store_true',
                      help='Start from epoch 1 when loading pretrained model')
    parser.add_argument('--reset_scheduler', action='store_true',
                      help='Reset learning rate scheduler when loading pretrained model')
    parser.add_argument('--timeout_minutes', type=int, default=30,
                      help='Timeout in minutes for distributed operations (default: 30)')
    parser.add_argument('--force_num_classes', type=int, default=None,
                        help='Force the model to use this number of output classes (e.g., 170 or 26)')
    parser.add_argument('--partial_loading', action='store_true', # Added for consistency
                        help='Allow partial loading of output layer when model sizes differ')
    parser.add_argument('--load_checkpoint', type=str, default=None, # Added for consistency
                        help='Path to a specific checkpoint to load')

    args = parser.parse_args()

    # Load configuration
    config = HParams.load(args.config)

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
    if 'EPOCHS' in os.environ: config.training['num_epochs'] = int(os.environ['EPOCHS']) # Added EPOCHS override
    if 'BATCH_SIZE' in os.environ: config.training['batch_size'] = int(os.environ['BATCH_SIZE'])
    if 'DATA_ROOT' in os.environ: config.paths['storage_root'] = os.environ['DATA_ROOT']
    if 'AUDIO_DIRS' in os.environ: args.audio_dirs = os.environ['AUDIO_DIRS'].split()
    if 'LABEL_DIRS' in os.environ: args.label_dirs = os.environ['LABEL_DIRS'].split()
    if 'CACHE_DIR' in os.environ: args.cache_dir = os.environ['CACHE_DIR']
    if 'TEACHER_MODEL' in os.environ: args.teacher_model = os.environ['TEACHER_MODEL']
    if 'BTC_CHECKPOINT' in os.environ: args.btc_checkpoint = os.environ['BTC_CHECKPOINT']
    if 'MODEL_TYPE' in os.environ: args.model_type = os.environ['MODEL_TYPE']
    if 'SMALL_DATASET' in os.environ: args.small_dataset = float(os.environ['SMALL_DATASET'])
    if 'DISABLE_CACHE' in os.environ: args.disable_cache = os.environ['DISABLE_CACHE'].lower() == 'true'
    if 'METADATA_CACHE' in os.environ: args.metadata_cache = os.environ['METADATA_CACHE'].lower() == 'true'
    if 'LAZY_INIT' in os.environ: args.lazy_init = os.environ['LAZY_INIT'].lower() == 'true'
    if 'BATCH_GPU_CACHE' in os.environ: args.batch_gpu_cache = os.environ['BATCH_GPU_CACHE'].lower() == 'true'
    if 'KFOLD' in os.environ: args.kfold = int(os.environ['KFOLD']) # Added KFOLD override
    if 'TOTAL_FOLDS' in os.environ: args.total_folds = int(os.environ['TOTAL_FOLDS']) # Added TOTAL_FOLDS override
    if 'SAVE_DIR' in os.environ: args.save_dir = os.environ['SAVE_DIR'] # Added SAVE_DIR override
    if 'LOAD_CHECKPOINT' in os.environ: args.load_checkpoint = os.environ['LOAD_CHECKPOINT'] # Added LOAD_CHECKPOINT override
    if 'RESET_EPOCH' in os.environ: args.reset_epoch = os.environ['RESET_EPOCH'].lower() == 'true' # Added RESET_EPOCH override
    if 'RESET_SCHEDULER' in os.environ: args.reset_scheduler = os.environ['RESET_SCHEDULER'].lower() == 'true' # Added RESET_SCHEDULER override

    logger.info(f"Config after potential ENV overrides - use_warmup: {config.training.get('use_warmup')}")
    # --- END Environment variable override block ---

    # Override with command line args (These take precedence over ENV vars and config file)
    if args.log_chord_details:
        if 'misc' not in config: config['misc'] = {}
        config.misc['log_chord_details'] = True
        logger.info("Detailed chord logging during evaluation ENABLED via command line.")
    elif config.misc.get('log_chord_details'):
        logger.info("Detailed chord logging during evaluation ENABLED via config/env.")

    # Set up device
    if config.misc.get('use_cuda', True) and is_cuda_available():
        device = get_device()
        logger.info(f"CUDA available. Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available or not requested. Using CPU.")

    # Override config values with command line arguments - IMPROVED CONFIG HANDLING
    config.misc['seed'] = args.seed if args.seed is not None else config.misc.get('seed', 42)
    config.paths['checkpoints_dir'] = args.save_dir if args.save_dir else config.paths.get('checkpoints_dir', './checkpoints/cv_kd')
    config.paths['storage_root'] = args.storage_root if args.storage_root else config.paths.get('storage_root', None)

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
    if 'num_epochs' not in config.training: config.training['num_epochs'] = 50 # Default if not set
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

    # Log KD settings
    use_kd_loss = args.use_kd_loss or str(config.training.get('use_kd_loss', False)).lower() == 'true'
    kd_alpha = args.kd_alpha if args.kd_alpha is not None else float(config.training.get('kd_alpha', 0.5))
    temperature = args.temperature if args.temperature is not None else float(config.training.get('temperature', 1.0))
    if use_kd_loss:
        logger.info("\n=== Knowledge Distillation Enabled ===")
        logger.info(f"KD alpha: {kd_alpha}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Teacher model path: {args.teacher_model}")
    else:
        logger.info("Knowledge distillation is disabled")

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

    # Set up chord mapping
    # Determine if large vocabulary is used
    use_large_voca = args.use_voca or str(config.feature.get('large_voca', False)).lower() == 'true'

    # Determine the correct number of output classes
    if args.force_num_classes is not None:
        n_classes = args.force_num_classes
        logger.info(f"Forcing model to use {n_classes} output classes as specified by --force_num_classes")
    elif use_large_voca:
        n_classes = 170
        logger.info(f"Using large vocabulary with {n_classes} output classes")
    else:
        n_classes = 26 # 25 chords + X
        logger.info(f"Using small vocabulary with {n_classes} output classes")

    # Create chord mappings based on n_classes
    if n_classes == 170:
        logger.info("Using large vocabulary chord mapping (170 chords)")
        master_mapping = idx2voca_chord() # Get idx -> chord mapping
        chord_mapping = {chord: idx for idx, chord in master_mapping.items()} # Create reverse mapping
    else: # Default to small vocabulary (26 classes)
        logger.info("Using standard vocabulary chord mapping (26 chords)")
        chord_mapping = {} # chord -> idx
        master_mapping = {} # idx -> chord
        for i in range(12):
            root = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][i]
            # Major chords
            maj_idx = i * 2
            maj_chord = f"{root}:maj" # Use explicit :maj
            chord_mapping[maj_chord] = maj_idx
            master_mapping[maj_idx] = maj_chord
            # Minor chords
            min_idx = i * 2 + 1
            min_chord = f"{root}:min"
            chord_mapping[min_chord] = min_idx
            master_mapping[min_idx] = min_chord
        # Special chords: N (no chord) and X (unknown)
        chord_mapping["N"] = 24
        master_mapping[24] = "N"
        chord_mapping["X"] = 25 # Index 25 for X
        master_mapping[25] = "X"
        n_classes = 26 # Ensure n_classes is 26

    # Initialize chord class with the reverse mapping (chord -> idx)
    chord_class = Chords()
    chord_class.set_chord_mapping(chord_mapping)
    chord_class.initialize_chord_mapping() # Initialize variants

    # Log mapping info
    logger.info(f"\nUsing idx->chord mapping with {len(master_mapping)} entries")
    logger.info(f"Sample idx->chord mapping: {dict(list(master_mapping.items())[:5])}")
    logger.info(f"Reverse chord->idx mapping created with {len(chord_mapping)} entries")
    logger.info(f"Sample chord->idx mapping: {dict(list(chord_mapping.items())[:5])}")

    # Set random seed for reproducibility - ensure this is an integer
    seed = int(config.misc['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) # Add random seed setting
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

    # Create save directory
    save_dir = config.paths['checkpoints_dir']
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {save_dir}") # Log save dir

    # Set up datasets
    # Resolve project root and storage root
    project_root = os.path.dirname(os.path.abspath(__file__))
    storage_root = config.paths.get('storage_root', None)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Storage root: {storage_root}")

    # Resolve audio_dirs, label_dirs, cache_dir using resolve_path
    def resolve_data_path(p, storage_root, project_root):
        if not p: return None
        if os.path.isabs(p): return p
        if storage_root:
            storage_path = os.path.join(storage_root, p)
            if os.path.exists(storage_path): return storage_path
        if project_root:
            project_path = os.path.join(project_root, p)
            if os.path.exists(project_path): return project_path
        # Fallback
        return os.path.join(storage_root, p) if storage_root else (os.path.join(project_root, p) if project_root else p)

    default_audio_dirs = [
        'isophonic',
        'uspop',
        'robbiewilliams'
    ]
    audio_dirs_resolved = [resolve_data_path(d, storage_root, project_root) for d in (args.audio_dirs or default_audio_dirs)]
    label_dirs_resolved = [resolve_data_path(d, storage_root, project_root) for d in (args.label_dirs or audio_dirs_resolved)] # Default labels same as audio
    cache_dir_resolved = resolve_data_path(args.cache_dir or config.paths.get('cache_dir', 'cache'), storage_root, project_root)

    if cache_dir_resolved:
        os.makedirs(cache_dir_resolved, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir_resolved}")
    else:
        logger.info("Cache directory not specified.")

    logger.info(f"Using audio directories: {audio_dirs_resolved}")
    logger.info(f"Using label directories: {label_dirs_resolved}")

    # Load teacher model if provided
    teacher_model = None
    teacher_mean = None
    teacher_std = None

    # Resolve teacher model path
    resolved_teacher_path = None
    if args.teacher_model:
        resolved_teacher_path = resolve_data_path(args.teacher_model, storage_root, project_root)
        if not os.path.exists(resolved_teacher_path):
             logger.error(f"Teacher model not found at resolved path: {resolved_teacher_path}")
             resolved_teacher_path = None # Reset if not found
             use_kd_loss = False
             logger.warning("Knowledge distillation disabled due to missing teacher model file.")
        else:
             logger.info(f"Resolved teacher model path to: {resolved_teacher_path}")

    if use_kd_loss and resolved_teacher_path:
        logger.info(f"Loading teacher model from {resolved_teacher_path}")
        try:
            # Determine vocabulary size based on args and config
            use_voca = args.use_voca or config.feature.get('large_voca', False)

            # Load the teacher model using our enhanced utility function
            teacher_model, teacher_mean, teacher_std, teacher_status = load_btc_model(
                resolved_teacher_path,
                device,
                use_voca=use_large_voca # Pass vocabulary setting
            )

            # Check if loading was successful
            if teacher_status["success"] and teacher_model is not None:
                logger.info(f"Teacher model loaded successfully: {teacher_status['message']}")
                logger.info(f"Model implementation: {teacher_status['implementation']}")
                logger.info(f"Model validation: {teacher_status['model_validated']}")

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
            use_kd_loss = False # Disable KD if loading fails
            logger.warning("Knowledge distillation disabled due to teacher model loading error.")
    elif use_kd_loss and not resolved_teacher_path:
         logger.warning("Knowledge distillation was enabled, but no valid teacher model path was provided or resolved. Disabling KD.")
         use_kd_loss = False


    # Run preprocessing if specified
    if args.preprocess:
        logger.info("Running preprocessing step to generate all features")
        # Create a dataset that will do all the preprocessing
        preprocess_dataset = CrossValidationDataset(
            config=config,
            audio_dirs=audio_dirs_resolved,
            label_dirs=label_dirs_resolved,
            chord_mapping=chord_mapping,
            train=True,  # Doesn't matter for preprocessing
            kfold=args.kfold,
            total_folds=args.total_folds,
            cache_dir=cache_dir_resolved,
            random_seed=seed,
            device=device
        )

        # Analyze label files before processing
        logger.info("Analyzing label files to understand structure...")
        preprocess_dataset.analyze_label_files(num_files=10)

        # Generate all features
        preprocess_dataset.generate_all_features()
        logger.info("Preprocessing completed")
        return

    # Create datasets for training and validation
    logger.info(f"Creating datasets for fold {args.kfold} of {args.total_folds}")

    # Pass caching arguments to CrossValidationDataset
    dataset_common_args = {
        'config': config,
        'audio_dirs': audio_dirs_resolved,
        'label_dirs': label_dirs_resolved,
        'chord_mapping': chord_mapping,
        'kfold': args.kfold,
        'total_folds': args.total_folds,
        'cache_dir': cache_dir_resolved,
        'random_seed': seed,
        'device': device,
        'teacher_model': None, # Pass later if needed for KD
        'use_cache': not args.disable_cache, # Control caching
        'metadata_only': str(args.metadata_cache).lower() == "true",
        'cache_fraction': args.cache_fraction,
        'lazy_init': str(args.lazy_init).lower() == "true",
        'small_dataset_percentage': args.small_dataset,
    }

    train_dataset = CrossValidationDataset(**dataset_common_args, train=True)
    val_dataset = CrossValidationDataset(**dataset_common_args, train=False)

    # After creating datasets
    logger.info("Analyzing chord distributions:")
    val_dataset.analyze_chord_distribution()
    train_dataset.analyze_chord_distribution()

    # Create data loaders with prefetch_factor
    batch_size = config.training.get('batch_size', 16)
    num_workers = 4 # Or adjust based on system
    pin_memory = torch.cuda.is_available() # Pin memory if using CUDA

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=int(args.prefetch_factor) if args.prefetch_factor > 1 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2), # Fewer workers for validation
        pin_memory=pin_memory,
        prefetch_factor=int(args.prefetch_factor) if args.prefetch_factor > 1 else None
    )

    # Generate teacher predictions for training data if using KD
    teacher_predictions = None
    if args.use_kd_loss and teacher_model is not None:
        logger.info("Generating teacher predictions for knowledge distillation")

        # Set up a directory to save teacher logits
        logits_dir = os.path.join(save_dir, f"teacher_logits_fold{args.kfold}")
        os.makedirs(logits_dir, exist_ok=True)

        # Use debug mode if specified
        debug_mode = args.kd_debug_mode
        if debug_mode:
            logger.info("Debug mode for teacher logit extraction is ENABLED")
        else:
            logger.info("Debug mode for teacher logit extraction is disabled")

        # Generate predictions with enhanced error handling
        teacher_predictions, generation_status = generate_teacher_predictions(
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
            logger.info(f"Generated teacher predictions for {len(teacher_predictions)} samples")
            logger.info(f"Success rate: {generation_status['successful_samples']}/{generation_status['total_samples']} samples ({generation_status['successful_samples']/generation_status['total_samples']*100:.2f}%)")
            logger.info(f"Extraction methods used: {generation_status['extraction_methods_used']}")

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
            teacher_predictions = None

    # Calculate global mean and std
    logger.info("Calculating global mean and std")
    mean = 0.0
    square_mean = 0.0
    k = 0

    mean_std_cache = os.path.join(cache_dir, f'normalization_fold{args.kfold}.pt')
    if os.path.exists(mean_std_cache):
        # Load from cache
        normalization_data = torch.load(mean_std_cache)
        mean = normalization_data.get('mean', 0.0)
        std = normalization_data.get('std', 1.0)
        logger.info(f"Loaded normalization from cache: mean={mean}, std={std}")
    else:
        # Calculate from data
        temp_loader = train_dataset.get_data_loader(batch_size=batch_size, shuffle=True, num_workers=2)
        for i, data in enumerate(temp_loader):
            features = data['spectro'].to('cpu')
            mean += torch.mean(features).item()
            square_mean += torch.mean(features.pow(2)).item()
            k += 1
            if i >= 99:
                break

        if k > 0:
            square_mean = square_mean / k
            mean = mean / k
            std = np.sqrt(max(0, square_mean - mean * mean))
            if std == 0: std = 1.0

            normalization_data = {'mean': mean, 'std': std}
            torch.save(normalization_data, mean_std_cache)
            logger.info(f"Calculated normalization: mean={mean:.4f}, std={std:.4f}")
        else:
            logger.warning("Could not calculate normalization stats (k=0). Using defaults.")
            mean = 0.0
            std = 1.0

    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32)
    normalization = {'mean': mean_tensor, 'std': std_tensor}
    logger.info(f"Normalization tensors created on device: {device}")

    # Create model
    logger.info(f"Creating model with {n_classes} output classes")

    # Get model type from args or config
    model_type = args.model_type
    if model_type not in ['ChordNet', 'BTC']:
        logger.warning(f"Unknown model type: {model_type}. Defaulting to ChordNet.")
        model_type = 'ChordNet'

    logger.info(f"Using model type: {model_type}")

    # Log additional information about feature dimensions
    n_freq = config.feature.get('n_bins', 144)

    if model_type == 'ChordNet':
        # ChordNet specific parameters
        n_group = config.model.get('n_group', 32)
        feature_dim = n_freq // n_group if n_group > 0 else n_freq
        heads = config.model.get('f_head', 6)
        logger.info(f"Using ChordNet feature dimensions: n_freq={n_freq}, n_group={n_group}, feature_dim={feature_dim}, heads={heads}")

        model = ChordNet(
            n_freq=n_freq,
            n_classes=n_classes,
            n_group=n_group,
            f_layer=config.model.get('f_layer', 3),
            f_head=heads,
            t_layer=config.model.get('t_layer', 3),
            t_head=config.model.get('t_head', 6),
            d_layer=config.model.get('d_layer', 3),
            d_head=config.model.get('d_head', 6),
            dropout=config.model.get('dropout', 0.3)
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

    # Apply model scale and dropout
    model_scale = float(args.model_scale) if args.model_scale is not None else float(config.model.get('scale', 1.0))
    dropout_rate = args.dropout if args.dropout is not None else config.model.get('dropout', 0.3)
    logger.info(f"Using model scale: {model_scale}")
    logger.info(f"Using dropout rate: {dropout_rate}")

    if model_type == 'ChordNet':
        # Apply scale to n_group for ChordNet
        n_group_base = config.model.get('n_group', 32)
        n_group = max(1, int(n_group_base * model_scale))
        logger.info(f"Applying scale {model_scale} to n_group: {n_group_base} -> {n_group}")

        model = ChordNet(
            n_freq=n_freq,
            n_classes=n_classes,
            n_group=n_group, # Use scaled n_group
            f_layer=config.model.get('f_layer', 3),
            f_head=heads,
            t_layer=config.model.get('t_layer', 3),
            t_head=config.model.get('t_head', 6),
            d_layer=config.model.get('d_layer', 3),
            d_head=config.model.get('d_head', 6),
            dropout=dropout_rate # Use dropout_rate
        ).to(device)
    else:  # BTC model
        # Apply scale to hidden_size for BTC (example scaling)
        hidden_size_base = config.model.get('hidden_size', 128)
        hidden_size = max(32, int(hidden_size_base * model_scale)) # Ensure minimum size
        logger.info(f"Applying scale {model_scale} to hidden_size: {hidden_size_base} -> {hidden_size}")

        btc_config = {
            'feature_size': n_freq,
            'hidden_size': hidden_size, # Use scaled hidden_size
            'num_layers': config.model.get('num_layers', 8),
            'num_heads': config.model.get('num_heads', 4),
            'total_key_depth': config.model.get('total_key_depth', 128),
            'total_value_depth': config.model.get('total_value_depth', 128),
            'filter_size': config.model.get('filter_size', 128),
            'seq_len': config.model.get('timestep', 108),
            'input_dropout': dropout_rate, # Apply dropout
            'layer_dropout': dropout_rate,
            'attention_dropout': dropout_rate,
            'relu_dropout': dropout_rate,
            'num_chords': n_classes
        }
        # ... (rest of BTC model creation) ...
        model = BTC_model(config=btc_config).to(device)

    # Attach chord mapping to model
    model.idx_to_chord = master_mapping
    logger.info("Attached chord mapping to model for correct MIR evaluation")

    # Load checkpoint if specified (consistent handling)
    load_path = args.load_checkpoint
    if load_path:
        resolved_load_path = resolve_data_path(load_path, storage_root, project_root)
        if os.path.exists(resolved_load_path):
            logger.info(f"Loading checkpoint from: {resolved_load_path}")
            try:
                checkpoint = torch.load(resolved_load_path, map_location=device)
                # ... (Add logic similar to train_finetune to handle state_dict keys, partial loading etc.) ...
                # Example:
                if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint: state_dict = checkpoint['model']
                else: state_dict = checkpoint

                if all(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}

                model.load_state_dict(state_dict, strict=not args.partial_loading)
                logger.info("Successfully loaded checkpoint weights")
                # Optionally load optimizer state etc. if not resetting
                if not args.reset_epoch and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Loaded optimizer state")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
        else:
            logger.warning(f"Specified checkpoint not found: {resolved_load_path}. Starting from scratch.")


    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.get('learning_rate', 0.0001),
        weight_decay=config.training.get('weight_decay', 0.0)
    )

    # IMPROVED: Handle string representations of boolean values
    use_focal_loss = args.use_focal_loss
    if not use_focal_loss:
        config_focal = config.training.get('use_focal_loss', False)
        if isinstance(config_focal, str):
            use_focal_loss = config_focal.lower() == "true"
        else:
            use_focal_loss = bool(config_focal)

    focal_gamma = float(args.focal_gamma) if args.focal_gamma is not None else float(config.training.get('focal_gamma', 2.0))
    focal_alpha = float(args.focal_alpha) if args.focal_alpha is not None else config.training.get('focal_alpha')

    if use_focal_loss:
        logger.info(f"Using Focal Loss with gamma={focal_gamma}")
        if focal_alpha is not None:
            logger.info(f"Focal Loss alpha={focal_alpha}")

    # Create trainer with knowledge distillation - with properly typed parameters
    lr_schedule_type = args.lr_schedule or config.training.get('lr_schedule', 'validation') # Default to validation
    if lr_schedule_type in ['validation', 'none']:
        lr_schedule_type = None # Trainer handles these internally or disables scheduler

    trainer = StudentTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=int(config.training.get('num_epochs', 50)),
        logger=logger,
        checkpoint_dir=save_dir,
        class_weights=None,
        idx_to_chord=master_mapping,
        normalization=normalization,
        early_stopping_patience=int(config.training.get('early_stopping_patience', 10)), # Increased patience
        lr_decay_factor=float(config.training.get('lr_decay_factor', 0.95)),
        min_lr=float(config.training.get('min_learning_rate', 5e-6)),
        use_warmup=use_warmup_final, # Use the final calculated value
        warmup_epochs=int(config.training.get('warmup_epochs', 10)) if use_warmup_final else None,
        warmup_start_lr=float(config.training.get('warmup_start_lr')) if use_warmup_final else None,
        warmup_end_lr=float(config.training.get('warmup_end_lr')) if use_warmup_final else None,
        lr_schedule_type=lr_schedule_type, # Pass schedule type
        use_focal_loss=use_focal_loss, # Use final calculated value
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        use_kd_loss=use_kd_loss, # Use final calculated value
        kd_alpha=kd_alpha,
        temperature=temperature,
        teacher_model=teacher_model,
        teacher_normalization={'mean': teacher_mean, 'std': teacher_std},
        teacher_predictions=teacher_predictions,
        timeout_minutes=args.timeout_minutes, # Pass timeout
        reset_epoch=args.reset_epoch, # Pass reset flags
        reset_scheduler=args.reset_scheduler,
        load_checkpoint_path=resolved_load_path if load_path else None # Pass specific path if provided
    )

    # Set chord mapping
    trainer.set_chord_mapping(chord_mapping)

    # Train the model
    logger.info(f"Starting training for fold {args.kfold}")
    try:
        trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())

    # Final evaluation on validation set (which is the test set for this fold)
    logger.info("\n=== Testing (Validation Fold) ===")
    try:
        if trainer.load_best_model():
            # Use the validation loader (acts as test loader for this fold)
            test_loader = val_loader # Already created earlier

            # Basic testing with Tester class
            tester = Tester(
                model=model,
                test_loader=test_loader, # Use the validation loader here
                device=device,
                idx_to_chord=master_mapping, # Use master_mapping (idx->chord)
                normalization=normalization, # Use the calculated normalization
                output_dir=save_dir, # Use save_dir for this fold
                logger=logger
            )

            test_metrics = tester.evaluate(save_plots=True)

            # Save test metrics for this fold
            try:
                metrics_path = os.path.join(save_dir, f"test_metrics_fold{args.kfold}.json")
                with open(metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Test metrics for fold {args.kfold} saved to {metrics_path}")
            except Exception as e:
                logger.error(f"Error saving test metrics for fold {args.kfold}: {e}")

            # NEW: Generate chord quality distribution and accuracy visualization
            logger.info("\n=== Generating Chord Quality Distribution and Accuracy Graph (Validation Fold) ===")
            try:
                from modules.utils.visualize import plot_chord_quality_distribution_accuracy

                # Collect all predictions and targets from the validation set
                all_preds = []
                all_targets = []
                model.eval()
                with torch.no_grad():
                    for batch in test_loader: # Iterate through validation loader
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

                # Create distribution and accuracy visualization for this fold
                quality_dist_path = os.path.join(save_dir, f"chord_quality_distribution_accuracy_fold{args.kfold}.png")
                plot_chord_quality_distribution_accuracy(
                    all_preds, all_targets, master_mapping, # Use master_mapping (idx->chord)
                    save_path=quality_dist_path,
                    title=f"Chord Quality Distribution and Accuracy (Fold {args.kfold})",
                    focus_qualities=focus_qualities
                )
                logger.info(f"Chord quality distribution graph for fold {args.kfold} saved to {quality_dist_path}")
            except ImportError:
                 logger.warning("Could not import plot_chord_quality_distribution_accuracy. Skipping visualization.")
                 logger.warning("Install matplotlib and seaborn: pip install matplotlib seaborn")
            except Exception as e:
                logger.error(f"Error creating chord quality distribution graph for fold {args.kfold}: {e}")
                logger.error(traceback.format_exc())

            # Advanced testing with mir_eval module on validation set
            logger.info("\n=== MIR evaluation (Validation Fold) ===")
            score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']

            # Use the validation samples for this fold
            validation_samples = val_dataset.samples # These are the samples for the current validation fold
            dataset_length = len(validation_samples)

            if dataset_length < 3:
                logger.info(f"Not enough validation samples ({dataset_length}) in fold {args.kfold} to compute chord metrics with 3 splits.")
                 # Optionally run on the whole validation set if splits aren't possible
                if dataset_length > 0:
                     logger.info(f"Evaluating model on all {dataset_length} validation samples for fold {args.kfold}...")
                     score_list_dict, song_length_list, average_score_dict = large_voca_score_calculation(
                         valid_dataset=validation_samples, config=config, model=model, model_type=model_type,
                         mean=mean, std=std, device=device)
                     mir_eval_results = average_score_dict # Store the single result
                     logger.info(f"MIR evaluation results for fold {args.kfold} (all samples): {mir_eval_results}")
                else:
                     logger.info(f"No validation samples available for MIR evaluation in fold {args.kfold}.")
                     mir_eval_results = {}
            else:
                # Create balanced splits from the validation samples
                split = dataset_length // 3
                valid_dataset1 = validation_samples[:split]
                valid_dataset2 = validation_samples[split:2*split]
                valid_dataset3 = validation_samples[2*split:]

                # Evaluate each split
                logger.info(f"Evaluating model on {len(valid_dataset1)} validation samples in split 1 (Fold {args.kfold})...")
                score_list_dict1, song_length_list1, average_score_dict1 = large_voca_score_calculation(
                    valid_dataset=valid_dataset1, config=config, model=model, model_type=model_type,
                    mean=mean, std=std, device=device)

                logger.info(f"Evaluating model on {len(valid_dataset2)} validation samples in split 2 (Fold {args.kfold})...")
                score_list_dict2, song_length_list2, average_score_dict2 = large_voca_score_calculation(
                    valid_dataset=valid_dataset2, config=config, model=model, model_type=model_type,
                    mean=mean, std=std, device=device)

                logger.info(f"Evaluating model on {len(valid_dataset3)} validation samples in split 3 (Fold {args.kfold})...")
                score_list_dict3, song_length_list3, average_score_dict3 = large_voca_score_calculation(
                    valid_dataset=valid_dataset3, config=config, model=model, model_type=model_type,
                    mean=mean, std=std, device=device)

                # Calculate weighted averages
                mir_eval_results = {}
                # Check if all results are valid before calculating average
                valid_results = (average_score_dict1 and average_score_dict2 and average_score_dict3 and
                                 song_length_list1 and song_length_list2 and song_length_list3)

                if valid_results:
                    total_length = np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3)
                    if total_length > 0:
                        for m in score_metrics:
                             # Ensure metric exists in all dictionaries
                            if m in average_score_dict1 and m in average_score_dict2 and m in average_score_dict3:
                                # Calculate weighted average based on song lengths
                                avg = (np.sum(song_length_list1) * average_score_dict1[m] +
                                       np.sum(song_length_list2) * average_score_dict2[m] +
                                       np.sum(song_length_list3) * average_score_dict3[m]) / total_length

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
                                logger.info(f"==== (Fold {args.kfold}) {m}: {avg_val:.2f}% (avg), range: [{min_val:.2f}% - {max_val:.2f}%]")

                                # Store in results dictionary
                                mir_eval_results[m] = {
                                    'split1': float(average_score_dict1[m]),
                                    'split2': float(average_score_dict2[m]),
                                    'split3': float(average_score_dict3[m]),
                                    'weighted_avg': float(avg)
                                }
                            else:
                                logger.warning(f"(Fold {args.kfold}) Metric '{m}' missing in one or more splits, cannot calculate average.")
                                mir_eval_results[m] = {'error': f'Metric {m} missing in splits'}
                    else:
                        logger.warning(f"(Fold {args.kfold}) Total song length is zero, cannot calculate weighted average MIR scores.")
                        mir_eval_results = {'error': 'Total song length zero'}
                else:
                    logger.warning(f"(Fold {args.kfold}) Could not calculate MIR scores properly for all splits.")
                    mir_eval_results = {'error': 'Calculation failed for one or more splits'}

            # Save MIR-eval metrics for this fold
            try:
                mir_eval_path = os.path.join(save_dir, f"mir_eval_metrics_fold{args.kfold}.json")
                with open(mir_eval_path, 'w') as f:
                    json.dump(mir_eval_results, f, indent=2)
                logger.info(f"MIR evaluation metrics for fold {args.kfold} saved to {mir_eval_path}")
            except Exception as e:
                 logger.error(f"Error saving MIR evaluation metrics for fold {args.kfold}: {e}")

        else:
            logger.warning(f"Could not load best model for testing fold {args.kfold}")
    except Exception as e:
        logger.error(f"Error during testing for fold {args.kfold}: {e}")
        logger.error(traceback.format_exc())

    # Save final model for the fold (keep this part, add more info)
    try:
        final_path = os.path.join(save_dir, f'final_model_fold{args.kfold}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'fold': args.kfold,
            'chord_mapping': chord_mapping, # chord -> idx
            'idx_to_chord': master_mapping, # idx -> chord
            'mean': mean,
            'std': std,
            'n_classes': n_classes, # Save number of classes
            'model_type': model_type, # Save model type
            'config': config.to_dict() # Save config used
        }, final_path)
        logger.info(f"Final model saved to {final_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set or not available
        pass
    main()
