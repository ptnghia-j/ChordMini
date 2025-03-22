import sys
import os
import torch
import numpy as np
import argparse
import glob
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.device import get_device
from modules.data.SynthDataset import SynthDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.StudentTrainer import StudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord

class ListSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class Tester:
    def __init__(self, model, test_loader, device, idx_to_chord=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.idx_to_chord = idx_to_chord

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        # Debug counters
        pred_counter = Counter()
        target_counter = Counter()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                inputs = batch['spectro'].to(self.device)
                targets = batch['chord_idx'].to(self.device)

                # Debug first few batches
                if batch_idx == 0:
                    logger.info(f"Input shape: {inputs.shape}, target shape: {targets.shape}")
                    logger.info(f"First few targets: {targets[:10]}")

                # Get raw logits before prediction
                logits, _ = self.model(inputs)

                # Check if logits have reasonable values
                if batch_idx == 0:
                    logger.info(f"Logits shape: {logits.shape}")
                    logger.info(f"Logits mean: {logits.mean().item()}, std: {logits.std().item()}")
                    logger.info(f"First batch sample logits (max 5 values): {logits[0, :5]}")

                # Use per-frame predictions if targets have a time dimension
                use_per_frame = targets.dim() > 1 and targets.shape[1] > 1
                preds = self.model.predict(inputs, per_frame=use_per_frame)

                # Process predictions based on their dimensions
                if targets.dim() > 1:
                    if preds.dim() > 1:
                        # Both are frame-level: flatten both
                        preds_np = preds.cpu().numpy().flatten()
                        targets_np = targets.cpu().numpy().flatten()
                    else:
                        # Targets are frame-level but preds are segment-level
                        # Use most common target for each sequence
                        seq_targets = []
                        for i in range(targets.shape[0]):
                            labels, counts = torch.unique(targets[i], return_counts=True)
                            most_common_idx = torch.argmax(counts)
                            seq_targets.append(labels[most_common_idx].item())
                        targets_np = np.array(seq_targets)
                        preds_np = preds.cpu().numpy()
                else:
                    # Standard case - both are segment-level
                    preds_np = preds.cpu().numpy().flatten()
                    targets_np = targets.cpu().numpy().flatten()

                # Count distribution using the adjusted arrays
                pred_counter.update(preds_np.tolist())
                target_counter.update(targets_np.tolist())

                all_preds.extend(preds_np)
                all_targets.extend(targets_np)

                # Debug first batch predictions vs targets
                if batch_idx == 0:
                    logger.info(f"First batch - Predictions: {preds_np[:10]}")
                    logger.info(f"First batch - Targets: {targets_np[:10]}")

        # Print distribution statistics
        logger.info("\nTarget Distribution (top 10):")
        for idx, count in target_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") if self.idx_to_chord else str(idx)
            logger.info(f"Target {idx} ({chord_name}): {count} occurrences ({count/len(all_targets)*100:.2f}%)")

        logger.info("\nPrediction Distribution (top 10):")
        for idx, count in pred_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") if self.idx_to_chord else str(idx)
            logger.info(f"Prediction {idx} ({chord_name}): {count} occurrences ({count/len(all_preds)*100:.2f}%)")

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        # Print confusion matrix for most common chords (top 10)
        if len(target_counter) > 1:
            logger.info("\nAnalyzing most common predictions vs targets:")
            top_chords = [idx for idx, _ in target_counter.most_common(10)]
            for true_idx in top_chords:
                true_chord = self.idx_to_chord.get(true_idx, str(true_idx))
                pred_indices = [p for t, p in zip(all_targets, all_preds) if t == true_idx]
                if pred_indices:
                    pred_counts = Counter(pred_indices)
                    most_common_pred = pred_counts.most_common(1)[0][0]
                    most_common_pred_chord = self.idx_to_chord.get(most_common_pred, str(most_common_pred))
                    accuracy_for_chord = pred_counts.get(true_idx, 0) / len(pred_indices)
                    logger.info(f"True: {true_chord} -> Most common prediction: {most_common_pred_chord} (Accuracy: {accuracy_for_chord:.2f})")

        logger.info(f"\nTest Metrics:")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")

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
            if file.endswith(file_pattern[1:]):  # Strip the * from pattern
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
            if file.endswith(file_pattern[1:]):  # Strip the * from pattern
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
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warm-up epochs')
    parser.add_argument('--warmup_start_lr', type=float, default=None,
                       help='Initial learning rate for warm-up (default: 1/10 of base LR)')
    # Add new arguments for smooth LR scheduling
    parser.add_argument('--lr_schedule', type=str, choices=['cosine', 'linear_decay', 'one_cycle', 'cosine_warm_restarts'], default=None,
                       help='Learning rate schedule type (default: validation-based)')
    
    # Add focal loss arguments
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss to handle class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss (default: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=None,
                       help='Alpha parameter for focal loss (default: None)')
    
    # Add knowledge distillation arguments (remove teacher_model_path)
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
    
    # Add parameter to control dataset caching behavior
    parser.add_argument('--disable_cache', action='store_true',
                      help='Disable dataset caching to reduce memory usage')
    parser.add_argument('--metadata_cache', action='store_true',
                      help='Only cache metadata (not spectrograms) to reduce memory usage')
    parser.add_argument('--cache_fraction', type=float, default=0.1,
                      help='Fraction of dataset to cache (default: 0.1 = 10%%)')
    
    # Add lazy_init argument
    parser.add_argument('--lazy_init', action='store_true',
                      help='Use lazy initialization for dataset to reduce memory usage')
    
    args = parser.parse_args()

    # Load configuration from YAML
    config = HParams.load(args.config)

    # Override config values with command line arguments if provided
    if args.seed is not None:
        config.misc['seed'] = args.seed

    if args.save_dir is not None:
        config.paths['checkpoints_dir'] = args.save_dir

    if args.storage_root is not None:
        config.paths['storage_root'] = args.storage_root

    # Override model scale from command line if provided
    if args.model_scale is not None:
        if not hasattr(config.model, 'scale'):
            config.model['scale'] = args.model_scale
        else:
            config.model.scale = args.model_scale
        logger.info(f"Overriding model scale from command line: {args.model_scale}")

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

    # Get device
    device = get_device() if config.misc['use_cuda'] else torch.device('cpu')
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # Get project root directory (important for path resolution)
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Project root: {project_root}")

    # Get storage root from config
    storage_root = config.paths.get('storage_root', None)
    logger.info(f"Storage root: {storage_root}")

    # Resolve paths using the new function
    # First, try to get spec_dir and label_dir from config
    spec_dir_config = config.paths.get('spec_dir', 'data/synth/spectrograms')
    label_dir_config = config.paths.get('label_dir', 'data/synth/labels')
    
    # Resolve the primary paths
    synth_spec_dir = resolve_path(spec_dir_config, storage_root, project_root)
    synth_label_dir = resolve_path(label_dir_config, storage_root, project_root)

    # Set up alternative paths if available
    alt_spec_dir = config.paths.get('alt_spec_dir', None)
    alt_label_dir = config.paths.get('alt_label_dir', None)

    if alt_spec_dir:
        alt_spec_dir = resolve_path(alt_spec_dir, storage_root, project_root)
    if alt_label_dir:
        alt_label_dir = resolve_path(alt_label_dir, storage_root, project_root)
    
    # Create directories if they don't exist
    os.makedirs(synth_spec_dir, exist_ok=True)
    os.makedirs(synth_label_dir, exist_ok=True)

    # Check if directories exist and contain files (including in subdirectories)
    spec_count = count_files_in_subdirectories(synth_spec_dir, "*.npy")
    label_count = count_files_in_subdirectories(synth_label_dir, "*.lab")

    # Find sample files to verify we're finding files correctly
    spec_samples = find_sample_files(synth_spec_dir, "*.npy", 5)
    label_samples = find_sample_files(synth_label_dir, "*.lab", 5)

    logger.info(f"Loading data from:")
    logger.info(f"  Spectrograms: {synth_spec_dir} ({spec_count} files)")
    if spec_samples:
        logger.info(f"  Example spectrogram files: {spec_samples}")
    logger.info(f"  Labels: {synth_label_dir} ({label_count} files)")
    if label_samples:
        logger.info(f"  Example label files: {label_samples}")

    # Try alternative paths if specified and primary paths don't have files
    if (spec_count == 0 or label_count == 0) and (alt_spec_dir or alt_label_dir):
        logger.info(f"Primary paths don't have enough data. Checking alternative paths...")
        if alt_spec_dir:
            alt_spec_count = count_files_in_subdirectories(alt_spec_dir, "*.npy")
            logger.info(f"  Alt spectrograms path: {alt_spec_dir} ({alt_spec_count} files)")
            if alt_spec_count > 0 and spec_count == 0:
                synth_spec_dir = alt_spec_dir
                spec_count = alt_spec_count
                logger.info(f"  Using alternative spectrogram path: {synth_spec_dir}")
        if alt_label_dir:
            alt_label_count = count_files_in_subdirectories(alt_label_dir, "*.lab")
            logger.info(f"  Alt labels path: {alt_label_dir} ({alt_label_count} files)")
            if alt_label_count > 0 and label_count == 0:
                synth_label_dir = alt_label_dir
                label_count = alt_label_count
                logger.info(f"  Using alternative label path: {synth_label_dir}")

    # Try one last fallback - check common locations
    if spec_count == 0 or label_count == 0:
        # Check common paths
        common_paths = [
            "/mnt/storage/data/synth/spectrograms",
            "/mnt/storage/data/synth/labels",
            os.path.join(project_root, "data/synth/spectrograms"),
            os.path.join(project_root, "data/synth/labels")
        ]
        logger.info(f"Still missing data. Checking common paths as last resort...")
        for path in common_paths:
            if "spectrogram" in path and spec_count == 0 and os.path.exists(path):
                count = count_files_in_subdirectories(path, "*.npy")
                if count > 0:
                    logger.info(f"  Found {count} spectrogram files at {path}")
                    synth_spec_dir = path
                    spec_count = count
                    break
        for path in common_paths:
            if "label" in path and label_count == 0 and os.path.exists(path):
                count = count_files_in_subdirectories(path, "*.lab")
                if count > 0:
                    logger.info(f"  Found {count} label files at {path}")
                    synth_label_dir = path
                    label_count = count
                    break

    # Final check - fail if we still don't have data
    if spec_count == 0 or label_count == 0:
        raise RuntimeError(f"ERROR: Missing spectrogram or label files. Found {spec_count} spectrogram files and {label_count} label files.")

    # Use the mapping defined in chords.py and invert it so that it is chord->index.
    master_mapping = idx2voca_chord()
    # Invert: keys = chord names, values = unique indices.
    chord_mapping = {chord: idx for idx, chord in master_mapping.items()}
    
    # Verify mapping of special chords - this ensures "N" is properly mapped
    logger.info(f"Mapping of special chords:")
    for special_chord in ["N", "X"]:
        if special_chord in chord_mapping:
            logger.info(f"  '{special_chord}' mapped to index {chord_mapping[special_chord]}")
        else:
            logger.warning(f"  '{special_chord}' not found in mapping!")

    # Log mapping info
    logger.info(f"\nUsing chord mapping from chords.py with {len(chord_mapping)} unique chords")
    logger.info(f"Sample chord mapping: {dict(list(chord_mapping.items())[:5])}")

    # Compute frame_duration from configuration if available,
    # otherwise default to 0.1 s
    frame_duration = config.feature.get('hop_duration', 0.1)
    
    # Resolve checkpoints directory path BEFORE dataset initialization
    checkpoints_dir_config = config.paths.get('checkpoints_dir', 'checkpoints')
    checkpoints_dir = resolve_path(checkpoints_dir_config, storage_root, project_root)
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")
    
    # Now we can safely use checkpoints_dir in the dataset initialization
    # Load synthesized dataset with optimized parameters
    cache_file = os.path.join(checkpoints_dir, "dataset_cache.pkl") if not args.disable_cache else None
    
    # Memory optimization: if we have an OOM issue, prioritize metadata-only caching
    use_metadata_only = True  # Default to metadata-only to avoid OOM
    if args.metadata_cache:
        use_metadata_only = True
        logger.info("Using metadata-only caching to reduce memory usage")
    elif args.disable_cache:
        use_metadata_only = False  # Doesn't matter if cache is disabled
        logger.info("Dataset caching disabled to reduce memory usage")
    
    # Get lazy_init from config or command line
    lazy_init = config.data.get('lazy_init', False) if hasattr(config, 'data') else False
    if args.lazy_init:
        lazy_init = True
    
    if lazy_init:
        logger.info("Using lazy initialization to reduce memory usage at startup")
    
    logger.info(f"Using partial dataset caching: {args.cache_fraction*100:.1f}% of samples")
    
    # Initialize SynthDataset with logits path if KD is enabled
    logits_dir = None
    if args.use_kd_loss and args.logits_dir:
        logits_dir = resolve_path(args.logits_dir, storage_root, project_root)
        logger.info(f"Knowledge distillation enabled - using teacher logits from: {logits_dir}")
        
        # Verify logits directory exists and contains files
        if not os.path.exists(logits_dir):
            logger.warning(f"Logits directory does not exist: {logits_dir}")
            if args.use_kd_loss:
                logger.error("Knowledge distillation requires valid logits directory")
                raise RuntimeError(f"Logits directory not found: {logits_dir}")
        else:
            logits_count = count_files_in_subdirectories(logits_dir, "*.npy")
            logger.info(f"Found {logits_count} teacher logit files in {logits_dir}")
            if logits_count == 0 and args.use_kd_loss:
                logger.error("No teacher logit files found but knowledge distillation is enabled")
                raise RuntimeError(f"No teacher logit files found in: {logits_dir}")
    
    # Initialize dataset with memory optimization options including lazy_init
    synth_dataset = SynthDataset(
        synth_spec_dir,
        synth_label_dir, 
        chord_mapping=chord_mapping, 
        seq_len=config.training['seq_len'], 
        stride=config.training['seq_stride'],
        frame_duration=frame_duration,
        num_workers=min(8, os.cpu_count() or 1),  # Limit to 8 workers maximum
        cache_file=cache_file,
        use_cache=not args.disable_cache,
        metadata_only=use_metadata_only,
        cache_fraction=args.cache_fraction,
        logits_dir=logits_dir,  # Pass logits_dir to the dataset
        lazy_init=lazy_init     # Add lazy initialization parameter
    )
    
    # After loading dataset, verify chord distribution matches expected mapping
    if len(synth_dataset.samples) == 0:
        logger.warning("No samples loaded from SynthDataset (lazy mode). Skipping chord distribution calculation.")
    else:
        chord_counter = Counter([sample['chord_label'] for sample in synth_dataset.samples])
        logger.info("\nChord distribution in loaded dataset:")
        n_chord_count = chord_counter.get("N", 0)
        logger.info(f"'N' chord count: {n_chord_count} ({n_chord_count/len(synth_dataset.samples)*100:.2f}% of total)")
        logger.info(f"Total synthesized samples: {len(synth_dataset.samples)}")

    # Create data loaders with optimized settings
    # Determine optimal num_workers
    num_workers = min(4, os.cpu_count()) if torch.cuda.is_available() else 0
    
    train_loader = synth_dataset.get_train_iterator(
        batch_size=config.training['batch_size'], 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = synth_dataset.get_eval_iterator(
        batch_size=config.training['batch_size'], 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Debug samples - add more detailed shape information
    logger.info("=== Debug: Training set sample ===")
    try:
        train_batch = next(iter(train_loader))
        logger.info(f"Training spectrogram tensor shape: {train_batch['spectro'].shape}")
        # Determine if we're using CQT or STFT based on frequency dimension
        actual_freq_dim = train_batch['spectro'].shape[-1]
        logger.info(f"Detected frequency dimension in data: {actual_freq_dim}")
        # For CQT, frequency dimension is typically around 144
        # For STFT, it's typically much higher (e.g., 1024 or 2048)
        is_cqt = actual_freq_dim <= 256
        if is_cqt:
            logger.info(f"Detected Constant-Q Transform (CQT) input with {actual_freq_dim} frequency bins")
        else:
            logger.info(f"Detected STFT input with {actual_freq_dim} frequency bins")
            
        # Set the frequency dimension to match the actual data
        n_freq = actual_freq_dim
    except Exception as e:
        logger.error(f"Error processing training batch: {e}")
        logger.error("This may indicate problems with the data format. Attempting to continue...")
        n_freq = config.model.get('n_freq', 144)
        logger.info(f"Using default frequency dimension: {n_freq}")

    # Get the number of unique chords in our dataset for the output layer
    num_unique_chords = len(chord_mapping)
    # Initialize n_classes with the number of unique chords
    n_classes = num_unique_chords
    logger.info(f"Output classes: {n_classes}")
    
    # Determine n_group based on frequency dimension for CQT vs STFT
    # For CQT, we'll use n_group=12 to get actual_feature_dim=12 (for 144 bins)
    # For STFT, we'll use the config value or default to 32
    n_group = 12 if is_cqt else config.model.get('n_group', 32)
    
    # Ensure n_freq is divisible by n_group
    if n_freq % n_group != 0:
        # Find a suitable n_group that divides n_freq
        for candidate in [12, 16, 24, 32, 48]:
            if n_freq % candidate == 0:
                n_group = candidate
                break
        logger.warning(f"Adjusted n_group to {n_group} to ensure n_freq ({n_freq}) is divisible")

    # Log the feature dimensions
    actual_feature_dim = n_freq // n_group
    logger.info(f"Using n_group={n_group}, resulting in actual feature dimension: {actual_feature_dim}")

    # Get model scale factor and compute scaled parameters
    model_scale = config.model.get('scale', 1.0)
    
    # Get base configuration for the model
    base_config = config.model.get('base_config', {})
    
    # If base_config is not specified, fall back to direct model parameters
    if not base_config:
        base_config = {
            'f_layer': config.model.get('f_layer', 3),
            'f_head': config.model.get('f_head', 6),
            't_layer': config.model.get('t_layer', 3),
            't_head': config.model.get('t_head', 6),
            'd_layer': config.model.get('d_layer', 3),
            'd_head': config.model.get('d_head', 6)
        }
    
    # Apply scale to model parameters
    f_layer = max(1, int(round(base_config['f_layer'] * model_scale)))
    f_head = max(1, int(round(base_config['f_head'] * model_scale)))
    t_layer = max(1, int(round(base_config['t_layer'] * model_scale)))
    t_head = max(1, int(round(base_config['t_head'] * model_scale)))
    d_layer = max(1, int(round(base_config['d_layer'] * model_scale)))
    d_head = max(1, int(round(base_config['d_head'] * model_scale)))
    
    # Log model scaling information
    logger.info(f"Using model scale: {model_scale}")
    if model_scale != 1.0:
        logger.info("Scaled model configuration:")
        logger.info(f"  f_layer: {base_config['f_layer']} -> {f_layer}")
        logger.info(f"  f_head: {base_config['f_head']} -> {f_head}")
        logger.info(f"  t_layer: {base_config['t_layer']} -> {t_layer}")
        logger.info(f"  t_head: {base_config['t_head']} -> {t_head}")
        logger.info(f"  d_layer: {base_config['d_layer']} -> {d_layer}")
        logger.info(f"  d_head: {base_config['d_head']} -> {d_head}")
    
    # Verify that the scaled parameters maintain dimensional compatibility
    if actual_feature_dim % f_head != 0:
        logger.warning(f"Feature dimension {actual_feature_dim} not divisible by scaled f_head={f_head}")
        # Find compatible f_head value
        for head_count in range(f_head, 0, -1):
            if actual_feature_dim % head_count == 0:
                logger.warning(f"Adjusting f_head from {f_head} to {head_count} for compatibility")
                f_head = head_count
                break
            
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
        dropout=config.model['dropout'],
        #ignore_index=chord_mapping.get("N")
    ).to(device)

    # Log model configuration for transparency
    logger.info("Model configuration:")
    logger.info(f"  n_freq: {n_freq}")
    logger.info(f"  n_classes: {n_classes}")
    logger.info(f"  n_group: {n_group}")
    logger.info(f"  f_layer: {f_layer}")
    logger.info(f"  f_head: {f_head}")
    logger.info(f"  t_layer: {t_layer}")
    logger.info(f"  t_head: {t_head}")
    logger.info(f"  d_layer: {d_layer}")
    logger.info(f"  d_head: {d_head}")
    logger.info(f"  dropout: {config.model['dropout']}")

    if torch.cuda.device_count() > 1 and config.misc['use_cuda']:
        model = torch.nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")

    # Create optimizer - match teacher settings
    optimizer = torch.optim.Adam(model.parameters(), 
                               lr=config.training['learning_rate'],    
                               weight_decay=config.training['weight_decay'], 
                               betas=tuple(config.training['betas']), 
                               eps=config.training['epsilon'])

    # Calculate class weights to handle imbalanced data
    # Only calculate if not using focal loss
    class_weights = None
    if not args.use_focal_loss and not config.training.get('use_focal_loss', False):
        dist_counter = Counter([sample['chord_label'] for sample in synth_dataset.samples])
        sorted_chords = sorted(chord_mapping.keys())
        total_samples = sum(dist_counter.values())
        ignore_index=chord_mapping.get("N")
        logger.info(f"Total chord instances: {total_samples}")
        for ch in sorted_chords[:10]:
            ratio = dist_counter.get(ch, 0) / total_samples * 100
            logger.info(f"Chord: {ch}, Count: {dist_counter.get(ch, 0)}, Percentage: {ratio:.4f}%")
        
        # The issue is here - using max on a single value. Let's fix with a safer approach:
        class_weights = np.array([
            0.0 if ch not in dist_counter else np.log(total_samples / max(1, dist_counter.get(ch, 1)) + 1)
            for ch in sorted_chords
        ], dtype=np.float32)
        logger.info(f"Generated class weights for {len(class_weights)} classes using log(inverse frequency)")
    else:
        logger.info("Using focal loss instead of class weights for handling imbalance")

    # Create idx_to_chord mapping for the loss function
    idx_to_chord = {v: k for k, v in chord_mapping.items()}
    
    # Calculate global mean and std for normalization (as done in teacher model)
    mean = 0
    square_mean = 0
    k = 0
    logger.info("Calculating global mean and std for normalization...")
    for i, data in enumerate(train_loader):
        features = data['spectro'].to(device)
        mean += torch.mean(features).item()
        square_mean += torch.mean(features.pow(2)).item()
        k += 1

    if k > 0:
        square_mean = square_mean / k
        mean = mean / k
        std = np.sqrt(square_mean - mean * mean)
        logger.info(f"Global mean: {mean}, std: {std}")
    else:
        mean = 0.0
        std = 1.0
        logger.warning("Could not calculate mean and std, using defaults")

    # Create our StudentTrainer with all parameters (remove teacher_model parameter)
    trainer = StudentTrainer(
        model, 
        optimizer, 
        device=device, 
        num_epochs=config.training['max_epochs'],
        class_weights=class_weights,  # Will be None if using focal loss
        idx_to_chord=idx_to_chord,
        normalization={'mean': mean, 'std': std},
        early_stopping_patience=config.training.get('early_stopping_patience', 5),
        lr_decay_factor=config.training.get('lr_decay_factor', 0.95),
        min_lr=config.training.get('min_lr', 5e-6),
        checkpoint_dir=checkpoints_dir,
        logger=logger,
        use_warmup=args.use_warmup,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        warmup_end_lr=config.training['learning_rate'],  # Use config LR as warmup end point
        lr_schedule_type=args.lr_schedule,  # Pass the scheduler type
        use_focal_loss=args.use_focal_loss or config.training.get('use_focal_loss', False),
        focal_gamma=args.focal_gamma or config.training.get('focal_gamma', 2.0),
        focal_alpha=args.focal_alpha or config.training.get('focal_alpha', None),
        use_kd_loss=args.use_kd_loss,
        kd_alpha=args.kd_alpha,
        temperature=args.temperature
    )

    # Set chord mapping for saving with the checkpoint
    trainer.set_chord_mapping(chord_mapping)

    # Check for existing checkpoints to resume training
    start_epoch = 1
    if os.path.exists(checkpoints_dir):
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "student_model_epoch_*.pth"))
        if checkpoint_files:
            # Extract epoch numbers and find the highest one
            epoch_numbers = []
            for f in checkpoint_files:
                try:
                    epoch_num = int(os.path.basename(f).replace("student_model_epoch_", "").replace(".pth", ""))
                    epoch_numbers.append((epoch_num, f))
                except ValueError:
                    continue
            
            if epoch_numbers:
                # Find the most recent checkpoint
                latest_epoch, latest_checkpoint = max(epoch_numbers, key=lambda x: x[0])
                logger.info(f"Found checkpoint from epoch {latest_epoch}. Attempting to resume training...")
                
                try:
                    # Load checkpoint
                    checkpoint = torch.load(latest_checkpoint, map_location=device)
                    
                    # Restore model state
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Restore optimizer state
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # Restore scheduler if it exists in the checkpoint
                    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and hasattr(trainer, 'smooth_scheduler'):
                        trainer.smooth_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                    # Set starting epoch to the next one
                    start_epoch = latest_epoch + 1
                    logger.info(f"Successfully loaded checkpoint. Training will resume from epoch {start_epoch}.")
                    
                    # If we have validation stats, restore those too
                    if 'accuracy' in checkpoint:
                        trainer.best_val_acc = checkpoint.get('accuracy', 0)
                        logger.info(f"Restored best validation accuracy: {trainer.best_val_acc:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {e}")
                    logger.info("Training will start from scratch.")
                    start_epoch = 1
    
    # Modify train method call to include starting epoch
    trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    # Test the model
    logger.info("Starting testing phase...")
    # Load the best model for testing
    trainer.load_best_model()
    test_loader = synth_dataset.get_test_iterator(batch_size=config.training['batch_size'])
    
    logger.info("=== Debug: Test set sample ===")
    test_batch = next(iter(test_loader))
    logger.info(f"Test spectrogram tensor shape: {test_batch['spectro'].shape}")
    logger.info(f"Test labels: {test_batch['chord_idx'][:10]}")

    # Evaluate on test set using the Tester class
    tester = Tester(model, test_loader, device, idx_to_chord=idx_to_chord)
    tester.evaluate()
    
    score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
    dataset_length = len(synth_dataset.samples)
    if dataset_length < 3:
        logger.info("Not enough validation samples to compute chord metrics.")
    else:
        # Create balanced splits of the samples
        split = dataset_length // 3
        valid_dataset1 = synth_dataset.samples[:split]
        valid_dataset2 = synth_dataset.samples[split:2*split]
        valid_dataset3 = synth_dataset.samples[2*split:]
        
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
        
        # Calculate and report the weighted average scores
        for m in score_metrics:
            if song_length_list1 and song_length_list2 and song_length_list3:
                avg = (np.sum(song_length_list1) * average_score_dict1[m] +
                       np.sum(song_length_list2) * average_score_dict2[m] +
                       np.sum(song_length_list3) * average_score_dict3[m]) / (
                       np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3))
                logger.info(f"==== {m} score 1 is {average_score_dict1[m]:.4f}")
                logger.info(f"==== {m} score 2 is {average_score_dict2[m]:.4f}")
                logger.info(f"==== {m} score 3 is {average_score_dict3[m]:.4f}")
                logger.info(f"==== {m} mix average score is {avg:.4f}")
            else:
                logger.info(f"==== {m} scores couldn't be calculated properly")

    # Save the final model
    save_path = os.path.join(checkpoints_dir, "student_model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'chord_mapping': chord_mapping,
        'idx_to_chord': idx_to_chord,
        'mean': mean,
        'std': std
    }, save_path)
    
    logger.info(f"Final model saved to {save_path}")

if __name__ == '__main__':
    main()