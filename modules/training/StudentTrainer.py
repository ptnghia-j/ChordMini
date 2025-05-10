import os
import torch
import numpy as np
import warnings  # Add import for warnings module
import traceback  # Add import for traceback module

import torch.nn.functional as F
import matplotlib.pyplot as plt  # Add missing matplotlib import

from modules.utils.logger import info, warning, error, debug, logging_verbosity, is_debug # Ensure is_debug is imported
from modules.training.Trainer import BaseTrainer
from modules.training.Schedulers import create_scheduler, create_warmup_scheduler
from collections import Counter
from modules.utils import checkpoint_utils # Added import
from modules.utils.gradient_utils import safe_clip_grad_norm_ # Added import

# Import visualization functions including chord quality mapping
from modules.utils.visualize import (
    plot_confusion_matrix, plot_chord_quality_confusion_matrix,
    plot_learning_curve, calculate_quality_confusion_matrix,
    calculate_confusion_matrix
)

class StudentTrainer(BaseTrainer):
    """
    Extension of BaseTrainer for student model training with early stopping
    and validation-based learning rate adjustment.
    """
    def __init__(self, model, optimizer, scheduler=None, device=None, num_epochs=100,
                 logger=None, use_animator=True, checkpoint_dir="checkpoints", # checkpoint_dir is the intended SAVE_DIR
                 max_grad_norm=1.0, class_weights=None, idx_to_chord=None,
                 normalization=None, early_stopping_patience=5,
                 lr_decay_factor=0.95, min_lr=5e-6,
                 use_warmup=False, warmup_epochs=5, warmup_start_lr=None, warmup_end_lr=None,
                 lr_schedule_type=None, use_focal_loss=False, focal_gamma=2.0, focal_alpha=None,
                 use_kd_loss=False, kd_alpha=0.5, temperature=2.0,
                 timeout_minutes=30, reset_epoch=False, reset_scheduler=False): # Removed teacher_model, teacher_normalization

        # First call the parent's __init__ to set up the logger and other attributes
        super().__init__(model, optimizer, scheduler, device, num_epochs,
                         logger, use_animator, checkpoint_dir, max_grad_norm, # Pass checkpoint_dir to parent
                         None, idx_to_chord, normalization)  # Pass None for class_weights initially

        # Store reset flags separately
        self.reset_epoch = reset_epoch
        self.reset_scheduler = reset_scheduler

        # Focal loss parameters - set these first as they affect class weight handling
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # New KD parameters:
        self.use_kd_loss = use_kd_loss
        self.kd_alpha = kd_alpha
        self.temperature = temperature

        # For offline KD, we only use pre-computed logits from the batch
        # No on-the-fly extraction is needed

        # Now that logger is initialized, we can pad class weights
        self.weight_tensor = None
        self.class_weights = None

        if class_weights is not None and hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            # Handle class imbalance - modify weights before padding

            expected_classes = model.fc.out_features
            info(f"Padding class weights from {len(class_weights)} to {expected_classes}")
            padded_weights = self._pad_class_weights(class_weights, expected_classes)

            # Now set the loss function with padded weights
            self.weight_tensor = torch.tensor(padded_weights, device=self.device)
            self.class_weights = padded_weights

        # Set up standard loss function - use weight tensor only if not using focal loss
        if not self.use_focal_loss:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weight_tensor)
        else:
            # For focal loss, we'll use unweighted cross entropy as base
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=None)

        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.best_val_acc = 0
        self.early_stop_counter = 0

        # Learning rate adjustment parameters
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.before_val_acc = 0
        self.consecutive_no_improve = 0  # Add counter for consecutive epochs without improvement

        # New warm-up parameters
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        # If warmup_start_lr not provided, use 1/10 of the initial learning rate
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else optimizer.param_groups[0]['lr'] / 10.0  # typically 1.0e-5 if initial lr=1.0e-4
        # If warmup_end_lr not provided, use the initial learning rate
        self.warmup_end_lr = warmup_end_lr if warmup_end_lr is not None else optimizer.param_groups[0]['lr']  # typically 1.0e-4
        # Track original learning rate for later use
        self.initial_lr = optimizer.param_groups[0]['lr']

        # Verify warmup values are sensible - fix incomplete warmup swap logic
        if self.use_warmup and self.warmup_start_lr >= self.warmup_end_lr:
            info(f"WARNING: warmup_start_lr ({self.warmup_start_lr}) >= warmup_end_lr ({self.warmup_end_lr})")
            info("This would cause LR to decrease during warmup instead of increasing. Swapping values.")
            temp = self.warmup_start_lr
            self.warmup_start_lr = self.warmup_end_lr
            self.warmup_end_lr = temp
            info(f"After swap: warmup_start_lr ({self.warmup_start_lr}) → warmup_end_lr ({self.warmup_end_lr})")

        # LR schedule type for smooth scheduling
        self.lr_schedule_type = lr_schedule_type
        self.smooth_scheduler = None
        self._last_stepped_epoch = None  # Track last stepped epoch to avoid duplicates

        # Modified to allow both warmup and scheduler to coexist
        # Set up smooth scheduler if requested
        if self.lr_schedule_type:
            self._create_smooth_scheduler()
            info(f"Using smooth '{self.lr_schedule_type}' learning rate schedule")
        else:
            info("Using validation-based learning rate adjustment")

        # Set up warmup independent of scheduler choice
        if self.use_warmup:
            info(f"Using warm-up LR schedule for first {self.warmup_epochs} epochs")
            info(f"Warm-up LR: {self.warmup_start_lr:.6f} → {self.warmup_end_lr:.6f}")
            # Set initial learning rate to warm-up start LR
            self._set_lr(self.warmup_start_lr)

            # If using both warmup and a scheduler, log this combined approach
            if self.lr_schedule_type:
                info(f"Will apply {self.lr_schedule_type} scheduler after warmup completes")

        # Best model tracking
        # Determine model type for proper file naming
        if hasattr(model, '__class__') and hasattr(model.__class__, '__name__'):
            model_type = model.__class__.__name__
            # Use model-specific prefix for best model path
            if 'BTC' in model_type:
                model_prefix = "btc"
            else:
                model_prefix = "student"
        else:
            model_prefix = "student"  # Default prefix

        # Store model prefix for later use
        self.model_prefix = model_prefix

        # --- Corrected Checkpoint Directory Logic ---
        # The 'checkpoint_dir' passed to __init__ is the primary save location (SAVE_DIR from YAML)
        self.primary_checkpoint_dir = checkpoint_dir
        self.fallback_checkpoint_dir = None # No fallback needed if primary is correctly set

        # Ensure the primary directory exists
        try:
            os.makedirs(self.primary_checkpoint_dir, exist_ok=True)
            info(f"Using primary checkpoint directory: {self.primary_checkpoint_dir}")
        except Exception as e:
            # If the specified directory can't be created, log an error
            error(f"Could not create primary checkpoint directory {self.primary_checkpoint_dir}: {e}")
            # Fallback to a default local directory as a last resort
            self.primary_checkpoint_dir = f"./checkpoints_fallback/{model_prefix}"
            os.makedirs(self.primary_checkpoint_dir, exist_ok=True)
            info(f"Using fallback local checkpoint directory: {self.primary_checkpoint_dir}")
        # --- End Correction ---


        # Set best model path using primary checkpoint directory
        self.best_model_path = os.path.join(self.primary_checkpoint_dir, f"{self.model_prefix}_model_best.pth")
        self.chord_mapping = None

        # Log the best model path for debugging
        if logger:
            logger.info(f"Best model will be saved to: {self.best_model_path}")
            # Removed fallback logging as it's not used in the corrected logic

        # Add flag to track and debug scheduler stepping
        self._scheduler_step_count = 0
        # Add flags for reset behavior
        self.reset_epoch = reset_epoch
        self.reset_scheduler = reset_scheduler

    def _get_model_state_dict(self):
        """Helper to get model state_dict, overridden in DDP."""
        return self.model.state_dict()

    def train_batch(self, batch):
        """Train on a single batch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Get input and target
        spectro = batch['spectro'].to(self.device)
        targets = batch['chord_idx'].to(self.device)

        # Normalize input
        if self.normalization:
            spectro = (spectro - self.normalization['mean']) / self.normalization['std']

        # Forward pass
        outputs = self.model(spectro)

        # Handle case where outputs is a tuple (logits, encoder_output)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # --- Get teacher logits BEFORE potential flattening ---
        teacher_logits = None
        if self.use_kd_loss:
            # First check if teacher_logits are already in the batch (pre-computed)
            if 'teacher_logits' in batch and batch['teacher_logits'] is not None:
                teacher_logits = batch['teacher_logits']
                # Log only once
                if not hasattr(self, '_precomputed_logits_logged'):
                    info(f"Using pre-computed teacher logits with shape: {teacher_logits.shape}")
                    self._precomputed_logits_logged = True

        # --- Flatten per-frame logits/targets AND teacher_logits consistently ---
        if logits.ndim == 3 and targets.ndim == 2:
            # Store original shape info if needed elsewhere
            batch_size, time_steps, num_classes = logits.shape
            debug(f"Original shapes: logits {logits.shape}, targets {targets.shape}")

            # Flatten student logits and targets
            logits = logits.reshape(-1, num_classes)
            targets = targets.reshape(-1)
            debug(f"Flattened student: logits {logits.shape}, targets {targets.shape}")

            # Flatten teacher logits if they exist and are 3D
            if teacher_logits is not None and teacher_logits.ndim == 3:
                if teacher_logits.shape[0] == batch_size and teacher_logits.shape[1] == time_steps:
                    teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))
                    debug(f"Flattened teacher logits: {teacher_logits.shape}")
                else:
                    warning(f"Teacher logits shape {teacher_logits.shape} mismatch with student batch/time {batch_size}/{time_steps}. Cannot flatten consistently.")
                    teacher_logits = None # Invalidate teacher logits if shapes don't match for flattening
        # ----------------------------------------------------------------------

        # Calculate standard loss (cross‐entropy or focal)
        if self.use_focal_loss:
            standard_loss = self.focal_loss(logits, targets)
        else:
            standard_loss = self.loss_fn(logits, targets)

        # Calculate knowledge distillation loss if enabled and teacher_logits are valid
        kd_loss = torch.tensor(0.0, device=self.device)
        if self.use_kd_loss and teacher_logits is not None:
            # Ensure shapes match *after* potential flattening
            if logits.shape == teacher_logits.shape:
                kd_loss = self.knowledge_distillation_loss(
                    logits, teacher_logits, self.temperature
                )

                # Log KD loss details (only once)
                if not hasattr(self, '_kd_loss_details_logged'):
                    info(f"KD loss: {kd_loss.item():.4f}, standard loss: {standard_loss.item():.4f}")
                    info(f"Using temperature: {self.temperature}, alpha: {self.kd_alpha}")
                    self._kd_loss_details_logged = True
            else:
                # This should ideally not happen if flattening logic above is correct
                warning(f"Shape mismatch before KD loss calculation: student {logits.shape}, teacher {teacher_logits.shape}. Skipping KD for this batch.")

        # Combine losses
        if self.use_kd_loss and kd_loss.item() != 0.0:
            # Use combination of KD and standard loss
            loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * standard_loss
        else:
            # Use only standard loss
            loss = standard_loss

        # Backward pass and optimizer step
        loss.backward()
        # Use safe clipping
        total_norm = safe_clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=False)
        self.optimizer.step()

        # Get predicted classes
        _, predicted = torch.max(logits, dim=1)
        # Calculate accuracy
        correct = (predicted == targets).sum().item()
        total = targets.size(0)

        return {
            'loss': loss.item(),
            'standard_loss': standard_loss.item(),
            'kd_loss': kd_loss.item() if self.use_kd_loss else 0.0,
            'accuracy': correct / total if total > 0 else 0.0 # Avoid division by zero
        }

    def knowledge_distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        Calculate knowledge distillation loss. Assumes student_logits and teacher_logits
        have the same shape.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            temperature: Temperature for softening distributions

        Returns:
            KD loss
        """
        # Apply temperature scaling
        soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)

        # Calculate KL divergence loss
        # Use batchmean reduction for stability across different batch sizes
        loss = torch.nn.functional.kl_div(log_probs, soft_targets, reduction='batchmean')

        # Apply temperature^2 scaling to match the gradient scale
        return loss * (temperature ** 2)

    def _create_smooth_scheduler(self):
        """Create a smooth learning rate scheduler that works with warmup."""
        # Store total training epochs for scheduler calculations
        self.total_training_epochs = self.num_epochs

        # If warmup is enabled, we'll create a special combined scheduler
        if self.use_warmup and self.warmup_epochs > 0:
            # For warmup + scheduler combo, we create scheduler for post-warmup epochs only
            self.post_warmup_epochs = max(1, self.num_epochs - self.warmup_epochs)
            info(f"Creating scheduler for {self.post_warmup_epochs} post-warmup epochs")
        else:
            # No warmup, scheduler will be used for all epochs
            self.post_warmup_epochs = self.num_epochs

        # Use the factory function from Schedulers.py to create the appropriate scheduler
        self.smooth_scheduler = create_scheduler(
            self.lr_schedule_type,
            self.optimizer,
            post_warmup_epochs=self.post_warmup_epochs,
            num_epochs=self.num_epochs,
            min_lr=self.min_lr,
            lr_decay_factor=self.lr_decay_factor,
            steps_per_epoch=100  # Estimate, will be updated in train()
        )

        # If no scheduler was created (unknown type), set lr_schedule_type to None
        if self.smooth_scheduler is None:
            self.lr_schedule_type = None

    def _update_learning_rate(self, epoch, batch_idx=None, num_batches=None):
        """
        Update learning rate combining warmup and scheduler logic.

        Args:
            epoch: Current training epoch (1-indexed)
            batch_idx: Current batch index within epoch (for resuming)
            num_batches: Total number of batches per epoch (for resuming)

        Returns:
            Current learning rate after update
        """
        # Get current learning rate before any updates
        current_lr = self.optimizer.param_groups[0]['lr']

        # Determine if we're in warmup phase (epoch is 1-indexed)
        in_warmup = self.use_warmup and epoch <= self.warmup_epochs

        if in_warmup:
            # In warmup phase, use warmup-specific LR calculation
            new_lr = self._warmup_learning_rate(epoch, batch_idx, num_batches)
            if abs(new_lr - current_lr) > 1e-7:  # Only log if there's a meaningful change
                info(f"Warmup LR update for epoch {epoch}/{self.warmup_epochs}: {current_lr:.6f} → {new_lr:.6f}")
            return new_lr
        elif self.lr_schedule_type:
            # Import scheduler types here to avoid circular imports
            from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR, CosineAnnealingWarmRestarts

            # After warmup, use the selected scheduler with adjusted epoch numbering
            if self.use_warmup:
                # For fractional updates within an epoch (some schedulers support this)
                if batch_idx is not None and num_batches is not None and isinstance(self.smooth_scheduler, (OneCycleLR, CosineAnnealingWarmRestarts)):
                    fractional_epoch = epoch - self.warmup_epochs - 1 + (batch_idx / num_batches)
                    self.smooth_scheduler.step(fractional_epoch)
                    self._scheduler_step_count += 1
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if batch_idx % 100 == 0:  # Reduce log frequency
                        info(f"Fractional scheduler step at epoch {epoch}, batch {batch_idx}/{num_batches}: LR={new_lr:.7f}")
                else:
                    # Just use regular epoch-based steps (CosineAnnealingLR, LambdaLR)
                    # Only step if we haven't already stepped for this epoch
                    if not hasattr(self, '_last_stepped_epoch') or self._last_stepped_epoch != (epoch - self.warmup_epochs - 1):
                        # Use step() without epoch parameter to avoid deprecation warning
                        self.smooth_scheduler.step()
                        self._last_stepped_epoch = epoch - self.warmup_epochs - 1
                        self._scheduler_step_count += 1
                        new_lr = self.optimizer.param_groups[0]['lr']
                        info(f"Post-warmup scheduler step at adjusted epoch {epoch - self.warmup_epochs - 1}: {current_lr:.6f} → {new_lr:.6f}")
            else:
                # No warmup - use standard scheduler stepping
                if batch_idx is not None and num_batches is not None and isinstance(self.smooth_scheduler, (OneCycleLR, CosineAnnealingWarmRestarts)):
                    fractional_epoch = (epoch - 1) + (batch_idx / num_batches)
                    self.smooth_scheduler.step(fractional_epoch)
                    self._scheduler_step_count += 1
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if batch_idx % 100 == 0:  # Reduce log frequency
                        info(f"Fractional scheduler step (no warmup) at epoch {epoch}, batch {batch_idx}/{num_batches}: LR={new_lr:.7f}")
                else:
                    # Just use regular epoch-based steps without epoch parameter
                    if not hasattr(self, '_last_stepped_epoch') or self._last_stepped_epoch != (epoch - 1):
                        self.smooth_scheduler.step()
                        self._last_stepped_epoch = epoch - 1
                        self._scheduler_step_count += 1
                        new_lr = self.optimizer.param_groups[0]['lr']
                        info(f"Scheduler step at epoch {epoch-1}: {current_lr:.6f} → {new_lr:.6f}")

            return self.optimizer.param_groups[0]['lr']
        else:
            # No scheduler - just return current LR (will be handled by validation-based adjustment)
            # Only log this message once per epoch (when batch_idx is None or 0)
            if batch_idx is None or batch_idx == 0:
                info(f"No active scheduler at epoch {epoch}, maintaining LR={current_lr:.6f} (will be adjusted by validation if needed)")
            return current_lr

    def focal_loss(self, logits, targets, gamma=2.0, alpha=None):
        """
        Compute focal loss to focus more on hard examples.

        Args:
            logits: Predicted logits from the model
            targets: True class labels
            gamma: Focusing parameter (default: 2.0)
            alpha: Optional class weights

        Returns:
            Focal loss value
        """
        if gamma == 0.0:
            return self.loss_fn(logits, targets)

        # Get class probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Get probability for the target class
        batch_size = logits.size(0)
        p_t = probs[torch.arange(batch_size), targets]

        # Calculate focal weight
        focal_weight = (1 - p_t) ** gamma

        # Use standard cross entropy with 'none' reduction
        # Note: We're not passing alpha as weight anymore
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction='none')

        # Apply the focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha as a scalar multiplier if provided
        if alpha is not None:
            focal_loss = alpha * focal_loss

        return focal_loss.mean()

    def _pad_class_weights(self, weights, expected_length):
        """Pad class weights to match the expected number of classes."""
        if len(weights) == expected_length:
            return weights

        # If weights array is too short, pad with zeros or ones
        if len(weights) < expected_length:
            # Use the mean of existing weights for padding
            if len(weights) > 0:
                mean_weight = sum(w for w in weights if w > 0) / max(1, sum(1 for w in weights if w > 0))
                padding_value = mean_weight
            else:
                padding_value = 1.0

            info(f"Using padding value: {padding_value:.4f}")

            import numpy as np
            padded_weights = np.zeros(expected_length, dtype=np.float32)
            padded_weights[:len(weights)] = weights
            padded_weights[len(weights):] = padding_value
            return padded_weights

        # If weights array is too long, truncate it
        if len(weights) > expected_length:
            info(f"Warning: Truncating class weights from {len(weights)} to {expected_length}")
            return weights[:expected_length]

        return weights

    def _set_lr(self, new_lr):
        """Helper to set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def _warmup_learning_rate(self, epoch, batch_idx=None, num_batches=None):
        """
        Calculate and set learning rate during warm-up period (per-step).

        Args:
            epoch: Current epoch (1-indexed)
            batch_idx: Current batch index within epoch (optional)
            num_batches: Total number of batches per epoch (optional)

        Returns:
            New learning rate
        """
        # Create a warmup scheduler if it doesn't exist
        if not hasattr(self, 'warmup_scheduler'):
            self.warmup_scheduler = create_warmup_scheduler(
                self.optimizer,
                True,
                warmup_epochs=self.warmup_epochs,
                warmup_start_lr=self.warmup_start_lr,
                warmup_end_lr=self.warmup_end_lr
            )
            info(f"Created warmup scheduler: {self.warmup_start_lr:.6f} → {self.warmup_end_lr:.6f} over {self.warmup_epochs} epochs")

        # Calculate warmup progress for logging
        if batch_idx is None or num_batches is None:
            # Epoch-based progress
            progress = (epoch - 1) / max(1, self.warmup_epochs - 1) if self.warmup_epochs > 1 else 1.0
            progress_str = f"epoch {epoch}/{self.warmup_epochs}"
        else:
            # Step-based progress
            total_steps = self.warmup_epochs * num_batches
            current_step = (epoch - 1) * num_batches + batch_idx
            progress = current_step / max(1, total_steps - 1)
            progress_str = f"step {current_step}/{total_steps}"

        # Use the warmup scheduler to calculate and set the learning rate
        new_lr = self.warmup_scheduler.step(epoch, batch_idx, num_batches)

        # Only log detailed messages at the beginning of epochs or every 100 batches
        if batch_idx is None or batch_idx == 0 or batch_idx % 100 == 0:
            # Calculate expected final warmup LR for this epoch
            expected_epoch_end_lr = self.warmup_start_lr + (self.warmup_end_lr - self.warmup_start_lr) * (epoch / self.warmup_epochs)
            info(f"Warmup active at {progress_str} ({progress:.1%}): LR={new_lr:.6f}, target end of epoch: {expected_epoch_end_lr:.6f}")

        return new_lr

    def _adjust_learning_rate(self, val_acc, current_epoch=None):
        """
        Adjust learning rate based on validation accuracy.

        Args:
            val_acc: Current validation accuracy
            current_epoch: Current epoch number (optional, for warmup check)

        Returns:
            bool: Whether learning rate was reduced
        """
        # Check if we're in warmup phase
        in_warmup = False
        if hasattr(self, 'use_warmup') and self.use_warmup and current_epoch is not None:
            if current_epoch <= self.warmup_epochs:
                info(f"Warmup active (epoch {current_epoch}/{self.warmup_epochs}), will skip validation-based LR adjustment")
                in_warmup = True

        # Create a validation-based scheduler if it doesn't exist
        if not hasattr(self, 'validation_scheduler'):
            self.validation_scheduler = create_scheduler(
                'validation',
                self.optimizer,
                lr_decay_factor=self.lr_decay_factor,
                min_lr=self.min_lr,
                patience=1  # Default patience value
            )

            # Initialize best_val_acc with current before_val_acc to maintain state
            self.validation_scheduler.best_val_acc = self.before_val_acc

        # Use the validation scheduler to adjust the learning rate
        lr_reduced = self.validation_scheduler.step(val_acc, in_warmup=in_warmup)

        # Update previous validation accuracy
        self.before_val_acc = val_acc

        return lr_reduced

    def set_chord_mapping(self, chord_mapping):
        """Set chord mapping for saving with checkpoints."""
        self.chord_mapping = chord_mapping

    def _save_best_model(self, val_acc, val_loss, epoch):
        """Save the model when validation accuracy improves."""
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.early_stop_counter = 0

            model_state_dict = self._get_model_state_dict()
            optimizer_state_dict = self.optimizer.state_dict()

            data_to_save = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'loss': val_loss,
                'accuracy': val_acc,
                'chord_mapping': self.chord_mapping,
                'idx_to_chord': self.idx_to_chord,
                'normalization': self.normalization,
            }

            save_success = checkpoint_utils.save_checkpoint(self.best_model_path, **data_to_save)

            if save_success:
                info(f"Saved best model with validation accuracy: {val_acc:.4f}")
                return True
            else:
                info("Failed to save best model")
                return False
        else:
            self.early_stop_counter += 1
            return False

    def _check_early_stopping(self):
        """Check if early stopping criteria is met."""
        if self.early_stop_counter >= self.early_stopping_patience:
            info(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement")
            return True
        return False

    def validate_with_metrics(self, val_loader, current_epoch=None):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Add confusion matrix tracking
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._process_batch(batch)
                if self.normalization:
                    inputs = (inputs - self.normalization['mean']) / self.normalization['std']

                try:
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    # If logits are per-frame, flatten them together with targets
                    if logits.ndim == 3 and targets.ndim == 2:
                        batch_size, time_steps, _ = logits.shape
                        logits_flat = logits.reshape(-1, logits.size(-1))
                        targets_flat = targets.reshape(-1)

                        # Use our compute_loss for focal loss support
                        loss = self.compute_loss(logits_flat, targets_flat)

                        preds_flat = torch.argmax(logits_flat, dim=1)
                        val_correct += (preds_flat == targets_flat).sum().item()
                        val_total += targets_flat.size(0)
                        val_loss += loss.item()

                        # Store predictions and targets for confusion matrix
                        all_preds.extend(preds_flat.cpu().numpy())
                        all_targets.extend(targets_flat.cpu().numpy())
                        continue

                    # Standard case
                    loss = self.compute_loss(logits, targets)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)

                    # Store predictions and targets for confusion matrix
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                except Exception as e:
                    info(f"Error during validation: {str(e)}")
                    continue  # Skip this batch if there's an error

        if val_total == 0:
            info("WARNING: No validation samples were correctly processed!")
            return float('inf'), 0.0  # Return worst possible values

        avg_loss = val_loss / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        info(f"Epoch Validation Loss: {avg_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Pass the current epoch to the confusion matrix function
        # Only print detailed class distribution and confusion matrix every 5 epochs or on the last epoch
        is_last_epoch = current_epoch == self.num_epochs
        is_print_epoch = current_epoch % 5 == 0 or current_epoch == 1 or is_last_epoch

        # Log whether we're generating the full confusion matrix this epoch
        # The full confusion matrix (all 170 classes) is generated every 10 epochs
        # This is controlled by the calculate_confusion_matrix function in visualize.py
        if current_epoch is not None and (current_epoch % 10 == 0 or is_last_epoch):
            info(f"Generating full confusion matrix (all 170 classes) for epoch {current_epoch}")

        # The calculate_confusion_matrix function handles the printing logic internally
        # based on the current_epoch value. It will generate the full matrix every 10 epochs.
        calculate_confusion_matrix(all_preds, all_targets, self.idx_to_chord, self.primary_checkpoint_dir, current_epoch)

        self.model.train()
        return avg_loss, val_acc

    def compute_loss(self, logits, targets, teacher_logits=None):
        """Compute loss with optimized GPU operations"""
        try:
            if isinstance(logits, tuple):
                logits = logits[0]

            # Ensure all tensors are on the correct device
            device = getattr(self, 'device', torch.device('cpu'))
            if logits.device != device:
                logits = logits.to(device, non_blocking=True)
            if targets.device != device:
                targets = targets.to(device, non_blocking=True)
            if teacher_logits is not None and teacher_logits.device != device:
                teacher_logits = teacher_logits.to(device, non_blocking=True)

            # Verify and log input shapes for debugging (commented out to reduce log verbosity)
            # orig_logits_shape = logits.shape
            # orig_targets_shape = targets.shape
            # orig_teacher_logits_shape = teacher_logits.shape if teacher_logits is not None else None

            # --- Flattening Logic ---
            # Check if student logits need flattening (e.g., from BTC model)
            if logits.ndim == 3 and targets.ndim == 2:
                batch_size, time_steps, num_classes = logits.shape
                # Flatten student logits and targets
                logits = logits.reshape(-1, num_classes)
                targets = targets.reshape(-1)

                # Flatten teacher logits if they exist and are 3D
                if teacher_logits is not None and teacher_logits.ndim == 3:
                    if teacher_logits.shape[0] == batch_size and teacher_logits.shape[1] == time_steps:
                        teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))
                    else:
                        warning(f"Teacher logits shape {teacher_logits.shape} mismatch with student batch/time {batch_size}/{time_steps} during flattening. Disabling KD for this batch.")
                        teacher_logits = None # Invalidate teacher logits

            # Report reshape results if any occurred (commented out to reduce log verbosity)
            # if orig_logits_shape != logits.shape or \
            #    orig_targets_shape != targets.shape or \
            #    (orig_teacher_logits_shape is not None and teacher_logits is not None and orig_teacher_logits_shape != teacher_logits.shape) or \
            #    (orig_teacher_logits_shape is not None and teacher_logits is None): # Log if teacher logits were invalidated
            #     info(f"Reshaped tensors for loss - Student: {orig_logits_shape} -> {logits.shape}, Targets: {orig_targets_shape} -> {targets.shape}, Teacher: {orig_teacher_logits_shape} -> {teacher_logits.shape if teacher_logits is not None else 'None'}")

            # --- Loss Calculation ---
            # Calculate standard loss (Focal or CE)
            if self.use_focal_loss:
                standard_loss = self.focal_loss(logits, targets,
                                                gamma=self.focal_gamma,
                                                alpha=self.focal_alpha)
            else:
                try:
                    standard_loss = self.loss_fn(logits, targets)
                except RuntimeError as e:
                    info(f"Error in standard loss calculation: {e}")
                    info(f"Logits shape: {logits.shape}, targets shape: {targets.shape}")
                    info(f"Target values: min={targets.min().item()}, max={targets.max().item()}")
                    num_classes = logits.size(-1)
                    if targets.max().item() >= num_classes:
                        info(f"WARNING: Target values exceed output dimension {num_classes}, clamping")
                        targets = torch.clamp(targets, 0, num_classes-1)
                        standard_loss = self.loss_fn(logits, targets)
                    else:
                        info("Using unweighted cross entropy as fallback")
                        standard_loss = F.cross_entropy(logits, targets)

            # Calculate KD loss if enabled and teacher_logits are valid
            kd_loss = torch.tensor(0.0, device=self.device)
            if self.use_kd_loss and teacher_logits is not None:
                # Ensure shapes match *after* potential flattening
                if logits.shape == teacher_logits.shape:
                    kd_loss = self.knowledge_distillation_loss(
                        logits, teacher_logits, self.temperature
                    )
                    # Combine losses
                    loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * standard_loss
                else:
                    warning(f"Final shape mismatch before KD loss: student {logits.shape}, teacher {teacher_logits.shape}. Using standard loss only.")
                    loss = standard_loss
            else:
                # Use only standard loss
                loss = standard_loss


            # Ensure loss is non-negative and finite
            loss = torch.clamp(loss, min=0.0)
            if torch.isnan(loss) or not torch.isfinite(loss):
                info(f"NaN or infinite loss detected - logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
                         f"targets: min={targets.min().item()}, max={targets.max().item()}")
                # Return a default loss value instead of NaN/inf
                return torch.tensor(1.0, device=loss.device, requires_grad=True)

            return loss
        except Exception as e:
            info(f"Unexpected error in compute_loss: {str(e)}")
            import traceback
            info(traceback.format_exc())
            # Last resort fallback - provide a dummy loss to avoid training failure
            return torch.tensor(1.0, device=self.device, requires_grad=True)

    def _process_batch(self, batch):
        """Move batch to device and apply normalization."""
        features = batch['spectro'].to(self.device)
        targets = batch['chord_idx'].to(self.device)

        # Apply normalization if available
        if self.normalization and 'mean' in self.normalization and 'std' in self.normalization:
            # Ensure normalization params are tensors on the correct device
            mean = self.normalization['mean']
            std = self.normalization['std']
            if not isinstance(mean, torch.Tensor): mean = torch.tensor(mean, device=self.device, dtype=features.dtype)
            if not isinstance(std, torch.Tensor): std = torch.tensor(std, device=self.device, dtype=features.dtype)
            # Add channel dimension if necessary for broadcasting (assuming mean/std are scalar or per-feature)
            # Example: if features are [B, T, F] and mean/std are [F]
            # mean = mean.view(1, 1, -1)
            # std = std.view(1, 1, -1)
            # Or if mean/std are scalar:
            # No view needed

            # Check shapes for broadcasting - ADDED DEBUG LOG
            if is_debug(): # Only log if debug level is enabled
                debug(f"Normalizing features shape {features.shape} with mean shape {mean.shape} and std shape {std.shape}")
                # Basic check: Ensure mean/std are scalar or match the feature dimension if not scalar
                if mean.numel() > 1 and mean.shape[-1] != features.shape[-1]:
                    warning(f"Potential shape mismatch: Features last dim {features.shape[-1]}, Mean shape {mean.shape}")
                if std.numel() > 1 and std.shape[-1] != features.shape[-1]:
                    warning(f"Potential shape mismatch: Features last dim {features.shape[-1]}, Std shape {std.shape}")


            features = (features - mean) / (std + 1e-6) # Add epsilon for stability

        teacher_logits = batch.get('teacher_logits')
        if teacher_logits is not None:
            teacher_logits = teacher_logits.to(self.device, non_blocking=True)

        return features, targets

    def load_checkpoint(self):
        """
        Load the trainer's internal state from trainer_state_latest.pth.
        This method handles loading the trainer's state based on reset flags.

        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
        """
        checkpoint_path = os.path.join(self.primary_checkpoint_dir, "trainer_state_latest.pth")
        checkpoint = checkpoint_utils.load_checkpoint(checkpoint_path, device=self.device)

        if checkpoint is None:
            return False

        try:
            # Load model weights
            if 'model_state_dict' in checkpoint:
                checkpoint_utils.apply_model_state(self.model, checkpoint['model_state_dict'])
                info("Loaded model state from trainer checkpoint")
            else:
                info("No model state found in trainer checkpoint")

            # Handle epoch counter based on reset_epoch flag
            if self.reset_epoch:
                info("Reset epoch flag is set, starting from epoch 1")
                self.start_epoch = 1
                # Reset metric histories
                self.train_losses = []
                self.val_losses = []
                self.best_val_acc = 0
                self.early_stop_counter = 0
                self.before_val_acc = 0
                self.consecutive_no_improve = 0
            else:
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                info(f"Resuming from epoch {self.start_epoch}")
                self.train_losses = checkpoint.get('train_losses', [])
                self.val_losses = checkpoint.get('val_losses', [])
                self.best_val_acc = checkpoint.get('best_val_acc', 0)
                self.early_stop_counter = checkpoint.get('early_stop_counter', 0)
                self.before_val_acc = checkpoint.get('before_val_acc', 0)
                self.consecutive_no_improve = checkpoint.get('consecutive_no_improve', 0)
                info(f"Loaded metric histories: {len(self.train_losses)} train losses, {len(self.val_losses)} val losses")
                info(f"Best validation accuracy so far: {self.best_val_acc:.4f}")

            # Handle optimizer state based on reset_epoch flag
            if not self.reset_epoch and 'optimizer_state_dict' in checkpoint:
                checkpoint_utils.apply_optimizer_state(self.optimizer, checkpoint['optimizer_state_dict'], self.device)
                info("Loaded optimizer state from checkpoint")

            # Handle scheduler state based on reset_scheduler flag
            if not self.reset_scheduler and not self.reset_epoch and 'scheduler_state_dict' in checkpoint:
                checkpoint_utils.apply_scheduler_state(self.smooth_scheduler, checkpoint.get('scheduler_state_dict'))
                info("Loaded scheduler state from checkpoint")
            elif self.reset_scheduler and self.smooth_scheduler is not None:
                info("Reset scheduler flag is set, scheduler state not loaded")

            self.chord_mapping = checkpoint.get('chord_mapping')
            self.idx_to_chord = checkpoint.get('idx_to_chord')
            self.normalization = checkpoint.get('normalization')

            info(f"Successfully loaded trainer state from {checkpoint_path}")
            return True

        except Exception as e:
            error(f"Error processing loaded trainer state: {e}")
            import traceback
            error(traceback.format_exc())
            return False

    def _save_trainer_state(self, current_epoch_completed):
        """
        Save the full trainer state to trainer_state_latest.pth.
        """
        checkpoint_path = os.path.join(self.primary_checkpoint_dir, "trainer_state_latest.pth")
        model_state_dict = self._get_model_state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        scheduler_state_dict = self.smooth_scheduler.state_dict() if self.smooth_scheduler else None

        data_to_save = {
            'epoch': current_epoch_completed,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'scheduler_state_dict': scheduler_state_dict,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_acc': self.best_val_acc,
            'early_stop_counter': self.early_stop_counter,
            'before_val_acc': self.before_val_acc,
            'consecutive_no_improve': self.consecutive_no_improve,
            'chord_mapping': self.chord_mapping,
            'idx_to_chord': self.idx_to_chord,
            'normalization': self.normalization,
        }
        return checkpoint_utils.save_checkpoint(checkpoint_path, **data_to_save)

    def _save_epoch_checkpoint(self, epoch, avg_train_loss, train_acc):
        """Saves a model checkpoint for a specific epoch."""
        checkpoint_path = os.path.join(self.primary_checkpoint_dir, f"{self.model_prefix}_model_epoch_{epoch}.pth")
        model_state_dict = self._get_model_state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        scheduler_state_dict = self.smooth_scheduler.state_dict() if self.smooth_scheduler else None

        data_to_save = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'scheduler_state_dict': scheduler_state_dict,
            'loss': avg_train_loss,
            'accuracy': train_acc,
            'chord_mapping': self.chord_mapping,
            'idx_to_chord': self.idx_to_chord,
            'normalization': self.normalization,
        }
        save_success = checkpoint_utils.save_checkpoint(checkpoint_path, **data_to_save)
        if save_success:
            info(f"Saved checkpoint at epoch {epoch}")
        else:
            info(f"Failed to save checkpoint at epoch {epoch}")


    def train(self, train_loader, val_loader=None, start_epoch=1):
        """
        Train the model with optimized GPU usage.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            start_epoch: Epoch to start training from (for resuming from checkpoint)
        """
        try:
            # Initialize start_epoch from the parameter
            self.start_epoch = start_epoch

            # Load checkpoint if available (may update self.start_epoch)
            self.load_checkpoint()

            self.model.train()

            # Get actual steps per epoch for scheduler
            num_batches = len(train_loader)

            # Optimize GPU memory usage by clearing cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Pre-allocate common tensors to avoid repeated allocations
            if torch.cuda.is_available():
                self._zero_tensor = torch.zeros(1, device=self.device)

            # Import scheduler types here to avoid circular imports
            from torch.optim.lr_scheduler import OneCycleLR

            # Update OneCycleLR with actual steps if needed
            if isinstance(self.smooth_scheduler, OneCycleLR):
                # Calculate total steps adjusting for warmup if needed
                if self.use_warmup and start_epoch <= self.warmup_epochs:
                    # If we're still in warmup, scheduler only needs post-warmup steps
                    remaining_epochs = self.num_epochs - self.warmup_epochs
                    total_steps = num_batches * remaining_epochs
                    info(f"OneCycleLR configured for {total_steps} steps after {self.warmup_epochs} warmup epochs")
                elif self.use_warmup and start_epoch > self.warmup_epochs:
                    # Already past warmup, calculate remaining post-warmup epochs
                    remaining_epochs = self.num_epochs - (start_epoch - 1)
                    total_steps = num_batches * remaining_epochs
                    info(f"Resuming OneCycleLR for {total_steps} steps (past warmup)")
                else:
                    # No warmup, scheduler handles all epochs
                    remaining_epochs = self.num_epochs - (start_epoch - 1)
                    total_steps = num_batches * remaining_epochs
                    info(f"OneCycleLR configured for {total_steps} steps")

                # Recreate scheduler with correct total_steps
                self.smooth_scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=self.initial_lr * 10,
                    total_steps=total_steps,
                    pct_start=0.3,
                    div_factor=25,
                    final_div_factor=10000,
                    anneal_strategy='cos'
                )

            # Reset KD warning flag
            self._kd_warning_logged = False

            # Log KD status at the beginning of training
            if self.use_kd_loss:
                info(f"Knowledge Distillation enabled: α={self.kd_alpha}, temperature={self.temperature}")

                # Check first batch for teacher logits - early verification for offline KD
                try:
                    info("Verifying teacher logits availability in first batch...")
                    first_batch = next(iter(train_loader))

                    if 'teacher_logits' in first_batch and first_batch['teacher_logits'] is not None:
                        # Teacher logits found in batch
                        teacher_shape = first_batch['teacher_logits'].shape
                        info(f"Teacher logits verified with shape: {teacher_shape}")
                        # Check if we have non-zero values in teacher logits
                        if first_batch['teacher_logits'].abs().sum().item() == 0:
                            info("WARNING: Teacher logits contain all zeros! Check your logits loading process.")
                    else:
                        # Teacher logits not found in batch
                        info("WARNING: KD is enabled but teacher_logits are missing from the first batch!")
                        info("Ensure your dataset provides 'teacher_logits' in each batch for offline KD.")
                        # Optionally disable KD if verification fails critically
                        # info("AUTOMATICALLY DISABLING KD due to missing teacher logits in first batch.")
                        # self.use_kd_loss = False

                except StopIteration:
                    info("WARNING: Train loader is empty, cannot verify teacher logits.")
                except Exception as e:
                    info(f"Error checking first batch for teacher logits: {e}")
                    info(traceback.format_exc())
                    # Optionally disable KD on error
                    # info("AUTOMATICALLY DISABLING KD due to error checking teacher logits.")
                    # self.use_kd_loss = False
            else:
                if self.use_focal_loss:
                    info("Knowledge Distillation disabled, using focal loss")
                else:
                    info("Knowledge Distillation disabled, using standard cross entropy loss")

            # Handle initial learning rate explicitly (before first epoch)
            if self.use_warmup and start_epoch == 1:
                # Explicitly set to warmup_start_lr to ensure we're starting from the right point
                curr_lr = self.optimizer.param_groups[0]['lr']
                if abs(curr_lr - self.warmup_start_lr) > 1e-7:  # Allow small floating point differences
                    info(f"Setting initial learning rate from {curr_lr:.6f} to warm-up start value: {self.warmup_start_lr:.6f}")
                    self._set_lr(self.warmup_start_lr)
                else:
                    info(f"Initial learning rate already set to warm-up start value: {curr_lr:.6f}")
            elif self.use_warmup and start_epoch > 1 and start_epoch <= self.warmup_epochs:
                # Resuming in the middle of warmup - calculate appropriate warmup LR
                old_lr = self.optimizer.param_groups[0]['lr']
                new_lr = self._warmup_learning_rate(start_epoch)
                info(f"Resuming in warmup phase at epoch {start_epoch} with LR adjusted from {old_lr:.6f} to {new_lr:.6f}")
            elif self.use_warmup and start_epoch > self.warmup_epochs and self.lr_schedule_type:
                # Resuming after warmup, need to advance scheduler to the right point
                effective_epoch = start_epoch - self.warmup_epochs - 1

                # Import scheduler types here to avoid circular imports
                from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

                # For OneCycleLR and CosineAnnealingWarmRestarts that need explicit epoch values
                if isinstance(self.smooth_scheduler, (OneCycleLR, CosineAnnealingWarmRestarts)):
                    self.smooth_scheduler.step(effective_epoch)
                else:
                    # For other schedulers, step multiple times without epoch parameter
                    for _ in range(effective_epoch):
                        self.smooth_scheduler.step()

                info(f"Resuming after warmup at epoch {start_epoch} (scheduler epoch {effective_epoch}) with LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Debug values for warmup
            if self.use_warmup:
                info(f"Warmup configuration: {self.warmup_epochs} epochs from {self.warmup_start_lr:.6f} to {self.warmup_end_lr:.6f}")

            for epoch in range(start_epoch, self.num_epochs + 1):
                # Determine LR source for this epoch for logging
                if self.use_warmup and epoch <= self.warmup_epochs:
                    lr_source = "warm-up schedule"
                elif self.lr_schedule_type:
                    lr_source = f"'{self.lr_schedule_type}' scheduler"
                else:
                    lr_source = "validation-based adjustment"

                # Apply learning rate update - this now properly handles combined warmup+scheduler
                current_lr = self._update_learning_rate(epoch)
                info(f"Epoch {epoch}: LR = {current_lr:.6f} (from {lr_source})")

                self.timer.reset(); self.timer.start()
                epoch_loss = 0.0
                train_correct = 0
                train_total = 0

                # Track KD usage statistics for this epoch
                kd_batches = 0
                total_batches = 0

                # Get number of batches for fractional epoch calculation
                num_batches = len(train_loader)

                # Use torch.cuda.amp.autocast for mixed precision training if available
                if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
                    autocast = torch.cuda.amp.autocast
                    scaler = torch.cuda.amp.GradScaler()
                    using_amp = True
                    if epoch == start_epoch:
                        info("Using mixed precision training with automatic mixed precision")
                else:
                    using_amp = False

                for batch_idx, batch in enumerate(train_loader):
                    # Update learning rate with batch info for smooth updates
                    if self.lr_schedule_type or self.use_warmup:
                        # Allow fractional epoch updates after warmup phase
                        # This call handles fractional updates internally now
                        self._update_learning_rate(epoch, batch_idx, num_batches)
                        # Removed redundant call to _update_smooth_scheduler

                    # Regular training step
                    inputs, targets = self._process_batch(batch)

                    # Extract teacher logits from batch if provided
                    teacher_logits = None
                    if self.use_kd_loss and 'teacher_logits' in batch:
                        teacher_logits = batch['teacher_logits'].to(self.device, non_blocking=True)
                        kd_batches += 1
                    total_batches += 1

                    # Enhanced KD logging and error checking
                    if self.use_kd_loss and teacher_logits is None:
                        if batch_idx == 0 or batch_idx % 100 == 0:
                            info(f"WARNING: KD enabled but no teacher logits found in batch {batch_idx}")

                            # If this keeps happening, report it clearly
                            if batch_idx >= 100 and kd_batches == 0:
                                info("CRITICAL ERROR: Knowledge Distillation is enabled but NO teacher logits found in any batches so far!")
                                info("This indicates a problem with your dataset or logits loading. Training will continue with standard loss.")
                                info("You should stop training and fix the logits loading issue.")

                    if self.normalization:
                        # Create the normalization tensors using clone().detach() if they don't exist
                        if not hasattr(self, '_norm_mean_tensor') or not hasattr(self, '_norm_std_tensor'):
                            # Ensure normalization values are tensors before cloning
                            norm_mean = self.normalization['mean']
                            norm_std = self.normalization['std']
                            if not isinstance(norm_mean, torch.Tensor):
                                norm_mean = torch.as_tensor(norm_mean, device=self.device, dtype=torch.float)
                            if not isinstance(norm_std, torch.Tensor):
                                norm_std = torch.as_tensor(norm_std, device=self.device, dtype=torch.float)

                            # Use clone().detach()
                            self._norm_mean_tensor = norm_mean.clone().detach().to(self.device)
                            self._norm_std_tensor = norm_std.clone().detach().to(self.device)
                            # Ensure std is not zero
                            if self._norm_std_tensor == 0:
                                info("Warning: Normalization std is zero, using 1.0 instead.")
                                self._norm_std_tensor = torch.tensor(1.0, device=self.device, dtype=torch.float)


                        # Apply normalization on GPU with in-place operations where possible
                        inputs = (inputs - self._norm_mean_tensor) / self._norm_std_tensor

                    self.optimizer.zero_grad(set_to_none=True)

                    try:
                        # Use mixed precision if available for faster computation
                        if using_amp:
                            with autocast():
                                outputs = self.model(inputs)

                                if isinstance(outputs, tuple):
                                    logits = outputs[0]
                                else:
                                    logits = outputs

                                # Pass teacher_logits directly to compute_loss, it will handle flattening
                                loss = self.compute_loss(logits, targets, teacher_logits)

                                # Skip invalid losses (avoid NaN propagation)
                                if loss is not None and not torch.isnan(loss) and torch.isfinite(loss):
                                    # Use scaler for mixed precision
                                    scaler.scale(loss).backward()
                                    scaler.unscale_(self.optimizer)  # Unscale before clipping
                                    # Use safe clipping
                                    safe_clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=False)
                                    scaler.step(self.optimizer)
                                    scaler.update()

                                    # Calculate accuracy based on potentially flattened logits/targets
                                    with torch.no_grad():
                                        # Re-flatten if necessary for accuracy calculation consistency
                                        if logits.ndim == 3 and targets.ndim == 2:
                                            logits_acc = logits.reshape(-1, logits.size(-1))
                                            targets_acc = targets.reshape(-1)
                                        else:
                                            logits_acc = logits
                                            targets_acc = targets

                                        preds = logits_acc.argmax(dim=1)
                                        batch_correct = (preds == targets_acc).sum().item()
                                        train_correct += batch_correct
                                        train_total += targets_acc.size(0)
                                        epoch_loss += loss.item()
                                else:
                                    info(f"WARNING: Skipping batch {batch_idx} due to invalid loss: {loss}")
                        else:
                            # Standard full-precision training
                            outputs = self.model(inputs)

                            if isinstance(outputs, tuple):
                                logits = outputs[0]
                            else:
                                logits = outputs

                            # Pass teacher_logits directly to compute_loss, it will handle flattening
                            loss = self.compute_loss(logits, targets, teacher_logits)

                            # Skip invalid losses (avoid NaN propagation)
                            if loss is not None and not torch.isnan(loss) and torch.isfinite(loss):
                                loss.backward()
                                # Use safe clipping
                                safe_clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=False)
                                self.optimizer.step()

                                # Calculate accuracy based on potentially flattened logits/targets
                                # Re-flatten if necessary for accuracy calculation consistency
                                if logits.ndim == 3 and targets.ndim == 2:
                                    logits_acc = logits.reshape(-1, logits.size(-1))
                                    targets_acc = targets.reshape(-1)
                                else:
                                    logits_acc = logits
                                    targets_acc = targets

                                preds = logits_acc.argmax(dim=1)
                                batch_correct = (preds == targets_acc).sum().item()
                                train_correct += batch_correct
                                train_total += targets_acc.size(0)
                                epoch_loss += loss.item()

                                if batch_idx % 100 == 0:
                                    # Log current LR and indicate whether KD is being used for this batch
                                    current_lr = self.optimizer.param_groups[0]['lr']
                                    batch_acc = batch_correct / targets_acc.size(0) if targets_acc.size(0) > 0 else 0.0
                                    kd_status = " (with KD)" if self.use_kd_loss and teacher_logits is not None else ""
                                    info(f"Epoch {epoch}/{self.num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.4f}{kd_status} | LR: {current_lr:.7f}")
                            else:
                                info(f"WARNING: Skipping batch {batch_idx} due to invalid loss: {loss}")

                    except Exception as e:
                        info(f"Error in training batch {batch_idx}: {str(e)}")
                        import traceback
                        info(traceback.format_exc())
                        continue  # Skip this batch if there's an error

                # # After each epoch, clear CUDA cache to prevent memory fragmentation
                # if (epoch + 1) % 5 == 0 and torch.cuda.is_available():
                #     torch.cuda.empty_cache()

                # Log KD usage statistics
                if self.use_kd_loss:
                    kd_percent = (kd_batches / total_batches) * 100 if total_batches > 0 else 0
                    info(f"KD usage: {kd_batches}/{total_batches} batches ({kd_percent:.1f}%)")

                    # Warn if KD usage is low, as this indicates a potential problem
                    if kd_percent < 50 and total_batches > 10:
                        info(f"WARNING: KD is enabled but only {kd_percent:.1f}% of batches had teacher logits!")
                        info("This means most batches are falling back to standard CE loss, which may impact results.")
                        if kd_percent == 0:
                            info("CRITICAL: No batches had teacher logits despite KD being enabled!")

                # Log training metrics for this epoch
                avg_train_loss = epoch_loss / max(1, len(train_loader))
                train_acc = train_correct / max(1, train_total) if train_total > 0 else 0.0
                self.timer.stop()
                info(f"Epoch {epoch}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Time: {self.timer.elapsed_time():.2f} sec")
                self.train_losses.append(avg_train_loss)

                if self.animator:
                    self.animator.add(epoch, avg_train_loss)

                # Import scheduler types here to avoid circular imports
                from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

                # Step epoch-based schedulers
                if self.lr_schedule_type and isinstance(self.smooth_scheduler, (CosineAnnealingLR, LambdaLR)):
                    if not (self.use_warmup and epoch <= self.warmup_epochs):
                        # Use step() without epoch parameter to avoid deprecation warning
                        self.smooth_scheduler.step()
                        current_lr = self.optimizer.param_groups[0]['lr']
                        info(f"Scheduler stepped to LR: {current_lr:.7f}")

                # Run validation
                if val_loader is not None:
                    # Pass current epoch number to validation for confusion matrix
                    val_loss, val_acc = self.validate_with_metrics(val_loader, current_epoch=epoch)
                    self.val_losses.append(val_loss)

                    # First check if we're in the warm-up phase
                    if self.use_warmup and epoch <= self.warmup_epochs:
                        # In warmup phase - don't apply any other LR adjustments
                        info(f"Epoch {epoch}: Maintaining warmup LR = {self.optimizer.param_groups[0]['lr']:.6f} (warmup active)")
                    else:
                        # Not in warmup phase - apply appropriate scheduler
                        if not self.lr_schedule_type:
                            # Only apply validation-based adjustment if no other scheduler is active
                            lr_before = self.optimizer.param_groups[0]['lr']
                            self._adjust_learning_rate(val_acc, current_epoch=epoch)
                            lr_after = self.optimizer.param_groups[0]['lr']
                            if lr_before != lr_after:
                                info(f"Epoch {epoch}: LR adjusted from {lr_before:.6f} to {lr_after:.6f} based on validation accuracy")

                    # Always track the best model and check for early stopping
                    self._save_best_model(val_acc, val_loss, epoch)
                    if self._check_early_stopping():
                        break

                # Save checkpoints periodically
                if epoch % 5 == 0 or epoch == self.num_epochs:
                    # Save the full trainer state
                    self._save_trainer_state(epoch)

                    # Also save a separate epoch-specific checkpoint for compatibility
                    self._save_epoch_checkpoint(epoch, avg_train_loss, train_acc)


            info(f"Training complete! Scheduler steps: {self._scheduler_step_count}")
            self._print_loss_history()
            self._plot_loss_history()

        except Exception as e:
            info(f"Unexpected error during training: {str(e)}")
            import traceback
            info(traceback.format_exc())
            # Try to save an emergency checkpoint to primary directory
            try:
                emergency_path = os.path.join(self.primary_checkpoint_dir, f"{self.model_prefix}_emergency_checkpoint.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'error': str(e)
                }, emergency_path)
                info(f"Saved emergency checkpoint to {emergency_path}")
            except Exception as e1:
                info(f"Failed to save emergency checkpoint to primary path: {str(e1)}")

                # Removed fallback emergency save logic

    def load_best_model(self):
        """Load the best model saved during training."""
        checkpoint = checkpoint_utils.load_checkpoint(self.best_model_path, device=self.device)
        if checkpoint:
            try:
                checkpoint_utils.apply_model_state(self.model, checkpoint['model_state_dict'])

                # Also load idx_to_chord attribute if available in the checkpoint
                if 'idx_to_chord' in checkpoint:
                    # Handle both distributed and non-distributed models
                    if hasattr(self.model, 'module'):
                        self.model.module.idx_to_chord = checkpoint['idx_to_chord']
                    else:
                        self.model.idx_to_chord = checkpoint['idx_to_chord']
                    info("Loaded idx_to_chord mapping from checkpoint for MIR evaluation")
                else:
                    info("Warning: idx_to_chord mapping not found in checkpoint")

                info(f"Loaded best model from primary path (epoch {checkpoint.get('epoch', 'N/A')}) with validation accuracy {checkpoint.get('accuracy', 0.0):.4f}")
                return True
            except Exception as e:
                error(f"Error applying best model state from primary path: {str(e)}")
                return False
        else:
            info(f"No best model found at primary path: {self.best_model_path}")
            return False

        # Removed fallback loading logic

    def _plot_loss_history(self):
        """Plot and save the loss history."""
        try:
            # Check if we have loss data
            if not hasattr(self, 'train_losses') or len(self.train_losses) == 0:
                info("No loss history to plot")
                return

            # Set up save path in primary checkpoint directory
            save_path = os.path.join(self.primary_checkpoint_dir, f"{self.model_prefix}_loss_history.png")

            # Use the visualization function to plot loss history
            val_losses = self.val_losses if hasattr(self, 'val_losses') and len(self.val_losses) > 0 else None
            fig = plot_learning_curve(
                self.train_losses, val_losses,
                title="Training and Validation Loss",
                save_path=save_path
            )

            info(f"Loss history plot saved to {save_path}")
            plt.close(fig)
        except Exception as e:
            info(f"Error plotting loss history: {e}")
