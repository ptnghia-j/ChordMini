import sys
import os
import torch  # Ensure torch is imported at the top level
import numpy as np
import argparse
import glob
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
# Use more comprehensive imports from device module, is_cuda_available, is_gpu_available, clear_gpu_cache
from modules.utils.device import get_device, is_cuda_available, is_gpu_available, clear_gpu_cache
from modules.data.SynthDataset import SynthDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.StudentTrainer import StudentTrainer
from modules.utils import logger
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord
from modules.training.Tester import Tester

class ListSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class Tester:
    def __init__(self, model, test_loader, device, idx_to_chord=None, normalization=None, output_dir=None, logger=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.idx_to_chord = idx_to_chord
        self.normalization = normalization
        self.output_dir = output_dir
        self.logger = logger

    def evaluate(self, save_plots=False):
        self.model.eval()
        all_preds = []
        all_targets = []

        # Debug counters
        pred_counter = Counter()
        target_counter = Counter()
        
        # Track batch shapes for debugging
        batch_shapes = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if isinstance(batch, dict):
                    inputs = batch['spectro'].to(self.device)
                    targets = batch['chord_idx'].to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                # Record input shapes for debugging
                batch_shapes.append(f"Batch {batch_idx}: inputs {inputs.shape}, targets {targets.shape}")
                
                if self.normalization:
                    mean = self.normalization['mean'].to(self.device)
                    std = self.normalization['std'].to(self.device)
                    inputs = (inputs - mean) / std
                
                # Reshape inputs if needed
                if inputs.dim() == 2:  # [batch, features]
                    inputs = inputs.unsqueeze(1)  # Add time dimension
                    
                # Get frame-by-frame predictions
                outputs = self.model.predict(inputs)
                
                # Move to CPU for processing
                preds_np = outputs.cpu().numpy()
                
                # Flatten targets if needed
                if targets.dim() > 1:
                    targets_np = targets.cpu().numpy().flatten()
                else:
                    targets_np = targets.cpu().numpy()
                
                # Log shapes to debug dimension mismatches
                if batch_idx == 0:
                    self._log(f"First batch shapes - Predictions: {preds_np.shape}, Targets: {targets_np.shape}")
                
                # Ensure predictions and targets have the same length
                # If they don't, this indicates a bug in our model.predict implementation
                if len(preds_np) != len(targets_np):
                    self._log(f"WARNING: Prediction length ({len(preds_np)}) != Target length ({len(targets_np)})")
                    # Find the minimum length and truncate both arrays
                    min_len = min(len(preds_np), len(targets_np))
                    preds_np = preds_np[:min_len]
                    targets_np = targets_np[:min_len]
                
                # Add to lists
                all_preds.extend(preds_np)
                all_targets.extend(targets_np)
                
                # Count occurrences for analysis
                for pred in preds_np:
                    pred_counter[pred] += 1
                for target in targets_np:
                    target_counter[target] += 1

        # Log batch shapes if there were problems
        if len(all_preds) != len(all_targets):
            self._log("WARNING: Final prediction and target counts don't match!")
            self._log(f"Predictions: {len(all_preds)}, Targets: {len(all_targets)}")
            self._log("Batch shapes:")
            for shape_info in batch_shapes[:5]:  # Show first 5 batches
                self._log(shape_info)
            
            # Ensure lengths match for metrics calculation
            min_len = min(len(all_preds), len(all_targets))
            all_preds = all_preds[:min_len]
            all_targets = all_targets[:min_len]

        # Print distribution statistics
        self._log("\nTarget Distribution (top 10):")
        for idx, count in target_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, f"Unknown-{idx}") if self.idx_to_chord else f"Class-{idx}"
            self._log(f"{chord_name}: {count} samples ({count/sum(target_counter.values())*100:.2f}%)")

        self._log("\nPrediction Distribution (top 10):")
        for idx, count in pred_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, f"Unknown-{idx}") if self.idx_to_chord else f"Class-{idx}"
            self._log(f"{chord_name}: {count} samples ({count/sum(pred_counter.values())*100:.2f}%)")

        # Calculate metrics
        self._log(f"\nCalculating metrics on {len(all_targets)} samples")
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        # Print confusion matrix for most common chords (top 10)
        if len(target_counter) > 1:
            self._save_confusion_matrix(all_targets, all_preds)

        self._log(f"\nTest Metrics:")
        self._log(f"Test Accuracy: {accuracy:.4f}")
        self._log(f"Test Precision: {precision:.4f}")
        self._log(f"Test Recall: {recall:.4f}")
        self._log(f"Test F1 Score: {f1:.4f}")
        
        # Also calculate per-segment metrics by majority voting
        self._log("\nCalculating per-segment metrics by majority voting:")
        self._calculate_segment_metrics(self.test_loader)
        
        # Save plots if requested and output directory is provided
        if save_plots and self.output_dir:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                
                # Create a figure directory
                figures_dir = os.path.join(self.output_dir, "figures")
                os.makedirs(figures_dir, exist_ok=True)
                
                # Plot class distribution
                plt.figure(figsize=(12, 6))
                top_classes = [self.idx_to_chord.get(idx, f"Class-{idx}") 
                              for idx, _ in target_counter.most_common(15)]
                counts = [count for _, count in target_counter.most_common(15)]
                
                plt.bar(top_classes, counts)
                plt.title("Class Distribution (Top 15)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(figures_dir, "class_distribution.png"))
                plt.close()
                
                self._log(f"Saved distribution plot to {figures_dir}")
            except Exception as e:
                self._log(f"Error creating plots: {e}")
            
        # Return metrics as a dictionary for further processing
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_segment_metrics(self, test_loader):
        """Calculate metrics at the segment level using majority voting"""
        segment_predictions = []
        segment_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    inputs = batch['spectro'].to(self.device)
                    targets = batch['chord_idx'].to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                if self.normalization:
                    mean = self.normalization['mean'].to(self.device)
                    std = self.normalization['std'].to(self.device)
                    inputs = (inputs - mean) / std
                
                # Make sure inputs have right shape
                if inputs.dim() == 2:  # [batch, features]
                    inputs = inputs.unsqueeze(1)  # Add time dimension
                
                # Get frame-by-frame predictions
                outputs = self.model.predict(inputs)
                
                # Reshape predictions and targets to [batch, seq_len]
                batch_size = inputs.size(0)
                seq_len = inputs.size(1)
                
                # Reshape predictions to [batch, seq_len]
                predictions_reshaped = outputs.cpu().reshape(batch_size, seq_len).numpy()
                targets_reshaped = targets.cpu().numpy()
                
                # Get majority vote for each segment
                for batch_idx in range(batch_size):
                    pred_segment = predictions_reshaped[batch_idx]
                    target_segment = targets_reshaped[batch_idx]
                    
                    # Majority vote for predictions
                    majority_pred = np.bincount(pred_segment).argmax()
                    
                    # Majority vote for targets (though they should all be the same in a segment)
                    majority_target = np.bincount(target_segment).argmax()
                    
                    segment_predictions.append(majority_pred)
                    segment_targets.append(majority_target)
        
        # Calculate metrics
        accuracy = accuracy_score(segment_targets, segment_predictions)
        precision = precision_score(segment_targets, segment_predictions, average='weighted', zero_division=0)
        recall = recall_score(segment_targets, segment_predictions, average='weighted', zero_division=0)
        f1 = f1_score(segment_targets, segment_predictions, average='weighted', zero_division=0)
        
        self._log(f"Segment-level Accuracy: {accuracy:.4f}")
        self._log(f"Segment-level Precision: {precision:.4f}")
        self._log(f"Segment-level Recall: {recall:.4f}")
        self._log(f"Segment-level F1 Score: {f1:.4f}")
        
        return {
            'segment_accuracy': accuracy,
            'segment_precision': precision,
            'segment_recall': recall,
            'segment_f1': f1
        }
    
    def _log(self, message):
        """Log a message using the logger if available, otherwise print"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
            
    def _save_confusion_matrix(self, targets, predictions):
        """Save confusion matrix visualization if matplotlib is available"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from sklearn.metrics import confusion_matrix
            
            # Get the most common classes in the targets
            target_counter = Counter(targets)
            top_classes = [idx for idx, _ in target_counter.most_common(10)]
            
            # Create class name mapping
            class_names = {}
            if self.idx_to_chord:
                for cls in top_classes:
                    class_names[cls] = self.idx_to_chord.get(cls, f"Class-{cls}")
            else:
                class_names = {cls: f"Class-{cls}" for cls in top_classes}
                
            # Filter data to include only top classes
            mask = np.isin(targets, top_classes)
            filtered_targets = np.array(targets)[mask]
            filtered_preds = np.array(predictions)[mask]
            
            # Compute confusion matrix
            cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)
            
            # Normalize the confusion matrix - FIX: Handle division by zero and NaN values
            row_sums = cm.sum(axis=1)
            # Add small epsilon to avoid division by zero
            row_sums = np.where(row_sums == 0, 1e-10, row_sums)
            cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
            
            # Replace NaN values with zeros for better visualization
            cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
            
            # Create figure with robust min/max values
            plt.figure(figsize=(10, 8))
            
            # Use robust min/max values to avoid seaborn warnings
            vmin = np.nanmin(cm_normalized[~np.isnan(cm_normalized)]) if np.any(~np.isnan(cm_normalized)) else 0
            vmax = np.nanmax(cm_normalized[~np.isnan(cm_normalized)]) if np.any(~np.isnan(cm_normalized)) else 1
            
            # Create heatmap with robust parameters
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        vmin=vmin, vmax=vmax,
                        xticklabels=[class_names[cls] for cls in top_classes],
                        yticklabels=[class_names[cls] for cls in top_classes])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix (Top 10 Classes)')
            plt.tight_layout()
            
            # Save if output directory is provided
            if self.output_dir:
                figures_dir = os.path.join(self.output_dir, "figures")
                os.makedirs(figures_dir, exist_ok=True)
                plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=300)
                self._log(f"Saved confusion matrix to {figures_dir}")
            
            plt.close()
            
        except Exception as e:
            self._log(f"Error creating confusion matrix: {e}")

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
    
    # Add new arguments for data directories override
    parser.add_argument('--spec_dir', type=str, default=None,
                      help='Directory containing spectrograms (overrides config value)')
    parser.add_argument('--label_dir', type=str, default=None,
                      help='Directory containing labels (overrides config value)')
    
    # Add new GPU acceleration options
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.9,
                      help='Fraction of GPU memory to use (default: 0.9)')
    parser.add_argument('--batch_gpu_cache', action='store_true',
                      help='Cache batches on GPU for repeated access patterns')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                      help='Number of batches to prefetch (default: 2)')
    parser.add_argument('--no_pin_memory', action='store_true',
                      help='Disable pin_memory for DataLoader (not recommended)')
    
    # Add new argument for small dataset percentage
    parser.add_argument('--small_dataset', type=float, default=None,
                      help='Use only a small percentage of dataset for quick testing (e.g., 0.01 for 1%%)')
    
    args = parser.parse_args()

    # Load configuration from YAML first before checking CUDA availability
    config = HParams.load(args.config)
    
    # Then check device availability
    if config.misc['use_cuda'] and is_cuda_available():
        device = get_device()  # This will return cuda if available
        logger.info(f"Using CUDA for training on device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training")
    
    # Override config values with command line arguments if provided
    if args.seed is not None:
        config.misc['seed'] = args.seed

    if args.save_dir is not None:
        config.paths['checkpoints_dir'] = args.save_dir

    if args.storage_root is not None:
        config.paths['storage_root'] = args.storage_root
    
    # Log training configuration
    logger.info("\n=== Training Configuration ===")
    logger.info(f"Model type: {args.model}")
    logger.info(f"Model scale: {args.model_scale or config.model.get('scale', 1.0)}")
    
    # Log knowledge distillation settings
    use_kd = args.use_kd_loss or config.training.get('use_kd_loss', False)
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
    
    # Log learning rate settings
    logger.info("\n=== Learning Rate Configuration ===")
    logger.info(f"Initial LR: {config.training['learning_rate']}")
    if args.lr_schedule or config.training.get('lr_schedule'):
        logger.info(f"LR schedule: {args.lr_schedule or config.training.get('lr_schedule')}")
    if args.use_warmup or config.training.get('use_warmup', False):
        # Fix: Use the warmup_epochs value from the command line or config, but prioritize command line
        warmup_epochs = args.warmup_epochs if args.warmup_epochs != 5 else config.training.get('warmup_epochs', 5)
        logger.info(f"Using LR warm-up for {warmup_epochs} epochs")
        logger.info(f"Warm-up start LR: {args.warmup_start_lr or config.training.get('warmup_start_lr', config.training['learning_rate']/10)}")
    
    # Log small dataset percentage setting if enabled
    small_dataset_percentage = args.small_dataset
    if small_dataset_percentage is None:
        # Check config if argument not provided
        small_dataset_percentage = config.data.get('small_dataset_percentage')
    
    if small_dataset_percentage is not None:
        logger.info(f"\n=== Using Small Dataset ===")
        logger.info(f"Dataset will be limited to {small_dataset_percentage*100:.1f}% of the full size")
        logger.info(f"This is a testing/development feature and should not be used for final training")
    
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
    
    # Get device with more explicit options for GPU
    if config.misc['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA for training on device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training")
    
    # Get project root directory (important for path resolution)
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Project root: {project_root}")
    
    # Get storage root from config
    storage_root = config.paths.get('storage_root', None)
    logger.info(f"Storage root: {storage_root}")
    
    # Resolve paths using the new function
    # First, try to get spec_dir and label_dir from config, but allow CLI override
    spec_dir_config = args.spec_dir or config.paths.get('spec_dir', 'data/synth/spectrograms')
    label_dir_config = args.label_dir or config.paths.get('label_dir', 'data/synth/labels')
    
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
            if alt_spec_count > 0:
                logger.info(f"Found {alt_spec_count} spectrogram files in alternative path {alt_spec_dir}")
                synth_spec_dir = alt_spec_dir
                spec_count = alt_spec_count
        if alt_label_dir:
            alt_label_count = count_files_in_subdirectories(alt_label_dir, "*.lab")
            if alt_label_count > 0:
                logger.info(f"Found {alt_label_count} label files in alternative path {alt_label_dir}")
                synth_label_dir = alt_label_dir
                label_count = alt_label_count
    
    # Try one last fallback - check common locations
    if spec_count == 0 or label_count == 0:
        # Check common paths
        common_paths = [
            "/mnt/storage/data/synth/spectrograms",
            "/mnt/storage/data/synth/labels",
            os.path.join(project_root, "data/synth/spectrograms"),
            os.path.join(project_root, "data/synth/labels"),
        ]
        logger.info(f"Still missing data. Checking common paths as last resort...")
        for path in common_paths:
            if path.endswith("/spectrograms") and spec_count == 0 and os.path.exists(path):
                spec_count = count_files_in_subdirectories(path, "*.npy")
                if spec_count > 0:
                    logger.info(f"Found {spec_count} spectrogram files in fallback path {path}")
                    synth_spec_dir = path
        for path in common_paths:
            if path.endswith("/labels") and label_count == 0 and os.path.exists(path):
                label_count = count_files_in_subdirectories(path, "*.lab")
                if label_count > 0:
                    logger.info(f"Found {label_count} label files in fallback path {path}")
                    synth_label_dir = path
    
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
            logger.info(f"  {special_chord} chord is mapped to index {chord_mapping[special_chord]}")
        else:
            logger.info(f"  {special_chord} chord is not in the mapping - this may cause issues")
    
    # Log mapping info
    logger.info(f"\nUsing chord mapping from chords.py with {len(chord_mapping)} unique chords")
    logger.info(f"Sample chord mapping: {dict(list(chord_mapping.items())[:5])}")
    
    # Compute frame_duration from configuration if available,
    # otherwise default to 0.1
    frame_duration = config.feature.get('hop_duration', 0.1)
    
    # Resolve checkpoints directory path BEFORE dataset initialization
    checkpoints_dir_config = config.paths.get('checkpoints_dir', 'checkpoints')
    checkpoints_dir = resolve_path(checkpoints_dir_config, storage_root, project_root)
    os.makedirs(checkpoints_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")
    
    # Initialize SynthDataset with GPU optimization options
    logger.info("\n=== Creating dataset ===")
    # Set up dataset initialization parameters
    dataset_args = {
        'spec_dir': synth_spec_dir,
        'label_dir': synth_label_dir,
        'chord_mapping': chord_mapping,
        'seq_len': config.training.get('seq_len', 10),  # Add fallback default
        'stride': config.training.get('seq_stride', 5),  # Add fallback default
        'frame_duration': frame_duration,
        'verbose': True,
        'device': device,  # Pass the device for GPU acceleration
        'pin_memory': False,  # IMPORTANT FIX: Disable pin_memory to avoid memory issues
        'prefetch_factor': 1,  # IMPORTANT FIX: Reduce prefetch factor
        'batch_gpu_cache': False  # IMPORTANT FIX: Disable GPU caching to avoid memory issues
    }
    
    # IMPORTANT FIX: Force worker count to 0 for safer operation
    dataset_args['num_workers'] = 0  # Disable workers to avoid shared memory issues
    logger.info("Using 0 workers to avoid CUDA shared memory issues")
    
    # Apply knowledge distillation settings
    if use_kd and args.logits_dir:
        logits_dir = resolve_path(args.logits_dir, storage_root, project_root)
        logger.info(f"Using teacher logits from: {logits_dir}")
        dataset_args['logits_dir'] = logits_dir
    
    # Apply dataset optimization settings from command line
    dataset_args['use_cache'] = not args.disable_cache
    dataset_args['metadata_only'] = args.metadata_cache
    dataset_args['cache_fraction'] = args.cache_fraction
    dataset_args['lazy_init'] = args.lazy_init
    
    # If we're using a small dataset for quick testing, set the percentage
    if args.small_dataset is not None:
        dataset_args['small_dataset_percentage'] = args.small_dataset
        logger.info(f"Using only {args.small_dataset * 100:.1f}% of the dataset for quick testing")
    
    # Update dataset initialization to ensure small_dataset_percentage is passed correctly
    dataset_args['small_dataset_percentage'] = small_dataset_percentage
    
    # Create the dataset with parameters
    logger.info("Creating SynthDataset...")
    synth_dataset = SynthDataset(**dataset_args)
    
    # After loading data, print dataset sizes to confirm data is available
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total segments: {len(synth_dataset)}")
    logger.info(f"Training segments: {len(synth_dataset.train_indices)}")
    logger.info(f"Validation segments: {len(synth_dataset.eval_indices)}")
    logger.info(f"Test segments: {len(synth_dataset.test_indices)}")
    
    # Check if we ended up with empty datasets after filtering
    if len(synth_dataset.train_indices) == 0:
        logger.error("ERROR: Training dataset is empty after processing. Cannot proceed with training.")
        return
    
    if len(synth_dataset.eval_indices) == 0:
        logger.warning("WARNING: Validation dataset is empty. Will skip validation steps.")
    
    # Create data loaders with optimized settings for GPU - MOVE THIS BEFORE CHECKING THE LOADER
    # IMPORTANT FIX: Force zero workers to avoid multiprocessing issues
    train_loader = synth_dataset.get_train_iterator(
        batch_size=config.training['batch_size'], 
        shuffle=True,
        num_workers=0,  # Force 0 to avoid shared memory issues
        pin_memory=False  # Disable pin memory to avoid memory issues
    )
    
    val_loader = synth_dataset.get_eval_iterator(
        batch_size=config.training['batch_size'], 
        shuffle=False,
        num_workers=0,  # Force 0 to avoid shared memory issues
        pin_memory=False  # Disable pin memory
    )
    
    # Check train loader
    logger.info("\n=== Checking data loaders ===")
    try:
        # Check if we can get at least one batch from the loader
        batch = next(iter(train_loader))
        logger.info(f"First batch loaded successfully: {batch['spectro'].shape}")
    except Exception as e:
        logger.error(f"ERROR: Failed to load first batch from train_loader: {e}")
        logger.error("Cannot proceed with training due to data loading issue.")
        return
    
    # Initialize model
    logger.info("\n=== Creating model ===")
    
    # Detect frequency dimension from dataset
    n_freq = getattr(config.feature, 'freq_bins', 144)  # Default to 144 if not specified
    n_classes = len(chord_mapping)
    logger.info(f"Using default frequency dimension: {n_freq}")
    logger.info(f"Output classes: {n_classes}")
    
    # Apply model scale factor (from CLI args or config)
    model_scale = args.model_scale or config.model.get('scale', 1.0)
    
    # Adjust model size based on scale factor
    if model_scale != 1.0:
        n_group = int(32 * model_scale)  # Scale n_group
        if n_group < 1:
            n_group = 1
        logger.info(f"Using n_group={n_group}, resulting in actual feature dimension: {n_freq // n_group}")
    else:
        n_group = config.model.get('n_group', 32)
    
    logger.info(f"Using model scale: {model_scale}")
    
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
        dropout=config.model.get('dropout', 0.5)
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.get('learning_rate', 0.0001),
        weight_decay=config.training.get('weight_decay', 0.0)
    )
    
    # IMPORTANT FIX: Reduce batch size to avoid OOM
    original_batch_size = config.training['batch_size']
    if torch.cuda.is_available():
        try:
            # Get available memory and set a conservative batch size
            free_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)  # in GB
            # Calculate safer batch size
            if free_mem < 8:  # Less than 8GB
                new_batch_size = min(original_batch_size, 16)
            elif free_mem < 16:  # 8-16GB
                new_batch_size = min(original_batch_size, 32)
            else:  # More than 16GB
                new_batch_size = min(original_batch_size, 64)
                
            # Make sure batch size is a multiple of 8 for efficiency
            new_batch_size = (new_batch_size // 8) * 8
            if new_batch_size < 8:
                new_batch_size = 8  # Minimum batch size of 8
                
            if new_batch_size < original_batch_size:
                logger.info(f"Reducing batch size from {original_batch_size} to {new_batch_size} to avoid CUDA OOM errors")
                config.training['batch_size'] = new_batch_size
        except Exception as e:
            logger.warning(f"Error determining batch size: {e}")
    
    # IMPORTANT FIX: Add memory cleanup before training
    if torch.cuda.is_available():
        logger.info("Performing CUDA memory cleanup before training")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # Print memory stats
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"CUDA memory stats (GB): allocated={allocated:.2f}, max_allocated={max_allocated:.2f}")
        logger.info(f"CUDA memory stats (GB): reserved={reserved:.2f}, max_reserved={max_reserved:.2f}")
    
    # Create data loaders with optimized settings for GPU
    # IMPORTANT FIX: Force zero workers to avoid multiprocessing issues
    train_loader = synth_dataset.get_train_iterator(
        batch_size=config.training['batch_size'], 
        shuffle=True,
        num_workers=0,  # Force 0 to avoid shared memory issues
        pin_memory=False  # Disable pin memory to avoid memory issues
    )
    
    val_loader = synth_dataset.get_eval_iterator(
        batch_size=config.training['batch_size'], 
        shuffle=False,
        num_workers=0,  # Force 0 to avoid shared memory issues
        pin_memory=False  # Disable pin memory
    )
    
    # Calculate global mean and std for normalization (as done in teacher model)
    # IMPORTANT FIX: Use safer memory handling when calculating stats
    try:
        # Use a smaller batch size and safer approach
        logger.info("Calculating global mean and std for normalization...")
        stats_batch_size = min(16, config.training['batch_size'])
        # Create a smaller loader just for stats calculation
        stats_loader = synth_dataset.get_train_iterator(
            batch_size=stats_batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        mean = 0
        square_mean = 0
        k = 0
        # Limit the number of batches to process - 100 batches should be enough
        max_stats_batches = 100
        
        for i, data in enumerate(stats_loader):
            if i >= max_stats_batches:
                break
                
            try:
                # Use CPU for calculations to avoid GPU OOM
                features = data['spectro'].to('cpu')
                mean += torch.mean(features).item()
                square_mean += torch.mean(features.pow(2)).item()
                k += 1
                
                # Explicitly delete tensors to free memory
                del features
                # Force garbage collection
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error processing batch {i} for stats: {e}")
                continue
                
        if k > 0:
            square_mean = square_mean / k
            mean = mean / k
            std = np.sqrt(square_mean - mean * mean)
            logger.info(f"Global mean: {mean}, std: {std}")
        else:
            mean = 0.0
            std = 1.0
            logger.warning("Could not calculate mean and std, using defaults")
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        mean = 0.0
        std = 1.0
        logger.warning("Using default mean=0.0, std=1.0 due to calculation error")
    
    # Create normalized tensors on GPU once and reuse
    try:
        mean = torch.tensor(mean, device=device)
        std = torch.tensor(std, device=device)
        normalization = {'mean': mean, 'std': std}
    except Exception as e:
        logger.error(f"Error creating normalization tensors: {e}")
        # Fall back to CPU tensors
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        normalization = {'mean': mean, 'std': std}
    
    # IMPORTANT FIX: Clear memory again before starting training
    if torch.cuda.is_available():
        logger.info("Final CUDA memory cleanup before training")
        torch.cuda.empty_cache()
    
    # Create StudentTrainer with enhanced loss functions
    trainer = StudentTrainer(
        model=model,
        optimizer=optimizer, 
        device=device,
        num_epochs=config.training.get('num_epochs', config.training.get('max_epochs', 100)),
        logger=logger,
        checkpoint_dir=checkpoints_dir,
        class_weights=None,  # Using focal loss instead of class weights
        idx_to_chord=master_mapping,
        normalization=normalization,
        early_stopping_patience=config.training.get('early_stopping_patience', 5),
        lr_decay_factor=config.training.get('lr_decay_factor', 0.95),
        min_lr=config.training.get('min_lr', 5e-6),
        use_warmup=args.use_warmup or config.training.get('use_warmup', False),
        warmup_epochs=args.warmup_epochs or config.training.get('warmup_epochs', 5),
        warmup_start_lr=args.warmup_start_lr or config.training.get('warmup_start_lr', None),
        lr_schedule_type=args.lr_schedule or config.training.get('lr_schedule', None),
        use_focal_loss=args.use_focal_loss or config.training.get('use_focal_loss', False),
        focal_gamma=args.focal_gamma or config.training.get('focal_gamma', 2.0),
        focal_alpha=args.focal_alpha or config.training.get('focal_alpha', None),
        use_kd_loss=use_kd,
        kd_alpha=kd_alpha,
        temperature=temperature
    )
    
    # Set chord mapping in trainer for checkpoint saving
    trainer.set_chord_mapping(chord_mapping)
    
    # After model initialization, log the model structure
    logger.info("\n=== Model Summary ===")
    try:
        # Log basic model information
        parameter_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {parameter_count:,}")
        logger.info(f"Trainable parameters: {trainable_count:,}")
        
        # Log structure of first batch to confirm dimensions
        sample_batch = next(iter(train_loader))
        logger.info(f"Input spectro shape: {sample_batch['spectro'].shape}")
        logger.info(f"Target chord_idx shape: {sample_batch['chord_idx'].shape}")
        
        # Verify we have teacher logits if KD is enabled
        if use_kd and 'teacher_logits' not in sample_batch:
            logger.error("ERROR: Knowledge distillation is enabled but no teacher_logits found in batch.")
            logger.error("Check your dataset configuration or disable KD with --use_kd_loss=false")
            return
            
    except Exception as e:
        logger.error(f"ERROR: Failed during model check: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Run training with error reporting
    logger.info("\n=== Starting training ===")
    try:
        # Wrap the training call with timeout handling to detect hangs
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Training function took too long to start")
        
        # Set a timeout for 60 seconds - if training doesn't start by then, there's an issue
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        trainer.train(train_loader, val_loader)
        # Cancel the alarm if training starts successfully
        signal.alarm(0)
        
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except TimeoutError as e:
        logger.error(f"ERROR: {e}")
        logger.error("Training function did not start within the expected time. This could indicate a deadlock or infinite loop.")
    except Exception as e:
        logger.error(f"ERROR during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Final evaluation on test set
    logger.info("\n=== Testing ===")
    try:
        # Load best model before testing
        if trainer.load_best_model():
            test_loader = synth_dataset.get_test_iterator(
                batch_size=config.training['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            logger.info("=== Debug: Test set sample ===")
            test_batch = next(iter(test_loader))
            logger.info(f"Test spectrogram tensor shape: {test_batch['spectro'].shape}")
            logger.info(f"Test labels: {test_batch['chord_idx'][:10]}")
            
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
                import json
                metrics_path = os.path.join(checkpoints_dir, "test_metrics.json") 
                with open(metrics_path, 'w') as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Test metrics saved to {metrics_path}")
            except Exception as e:
                logger.error(f"Error saving test metrics: {e}")
            
            # Advanced testing with mir_eval module
            logger.info("\n=== MIR evaluation ===")
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
                mir_eval_results = {}
                for m in score_metrics:
                    if song_length_list1 and song_length_list2 and song_length_list3:
                        # Calculate weighted average based on song lengths
                        avg = (np.sum(song_length_list1) * average_score_dict1[m] +
                               np.sum(song_length_list2) * average_score_dict2[m] +
                               np.sum(song_length_list3) * average_score_dict3[m]) / (
                               np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3))
                        
                        # Log individual split scores
                        logger.info(f"==== {m} score 1 is {average_score_dict1[m]:.4f}")
                        logger.info(f"==== {m} score 2 is {average_score_dict2[m]:.4f}")
                        logger.info(f"==== {m} score 3 is {average_score_dict3[m]:.4f}")
                        logger.info(f"==== {m} mix average score is {avg:.4f}")
                        
                        # Store in results dictionary
                        mir_eval_results[m] = {
                            'split1': float(average_score_dict1[m]),
                            'split2': float(average_score_dict2[m]),
                            'split3': float(average_score_dict3[m]),
                            'weighted_avg': float(avg)
                        }
                    else:
                        logger.info(f"==== {m} scores couldn't be calculated properly")
                        mir_eval_results[m] = {'error': 'Calculation failed'}
                
                # Save MIR-eval metrics to a separate file
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
        import traceback
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

if __name__ == '__main__':
    main()