import os
import torch
import numpy as np
from modules.training.Trainer import BaseTrainer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR, CosineAnnealingWarmRestarts
import torch.nn.functional as F
import matplotlib.pyplot as plt  # Add missing matplotlib import
from collections import Counter
import re
from sklearn.metrics import confusion_matrix

# Import visualization functions including chord quality mapping
from modules.utils.visualize import (
    plot_confusion_matrix, plot_chord_quality_confusion_matrix,
    plot_learning_curve, calculate_quality_confusion_matrix,
)

class StudentTrainer(BaseTrainer):
    """
    Extension of BaseTrainer for student model training with early stopping
    and validation-based learning rate adjustment.
    """
    def __init__(self, model, optimizer, scheduler=None, device=None, num_epochs=100,
                 logger=None, use_animator=True, checkpoint_dir="checkpoints",
                 max_grad_norm=1.0, class_weights=None, idx_to_chord=None,
                 normalization=None, early_stopping_patience=5,
                 lr_decay_factor=0.95, min_lr=5e-6, 
                 use_warmup=False, warmup_epochs=5, warmup_start_lr=None, warmup_end_lr=None,
                 lr_schedule_type=None, use_focal_loss=False, focal_gamma=2.0, focal_alpha=None,
                 use_kd_loss=False, kd_alpha=0.5, temperature=1.0):
        
        # First call the parent's __init__ to set up the logger and other attributes
        super().__init__(model, optimizer, scheduler, device, num_epochs,
                         logger, use_animator, checkpoint_dir, max_grad_norm,
                         None, idx_to_chord, normalization)  # Pass None for class_weights initially
        
        # Focal loss parameters - set these first as they affect class weight handling
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        # New KD parameters:
        self.use_kd_loss = use_kd_loss
        self.kd_alpha = kd_alpha
        self.temperature = temperature
        if self.use_kd_loss:
            self._log(f"Using KD loss with α={kd_alpha} and temperature={temperature}")
            self._log("Teacher logits expected to be provided in each batch")
        
        if self.use_focal_loss:
            self._log(f"Using Focal Loss (gamma={focal_gamma}) to handle class imbalance")
            if self.focal_alpha is not None:
                self._log(f"Using alpha={focal_alpha} for additional class weighting")
            # When using focal loss, don't use standard class weights
            class_weights = None
            self._log("Class weights disabled as focal loss is being used")
        
        # Now that logger is initialized, we can pad class weights
        self.weight_tensor = None
        self.class_weights = None
        
        if class_weights is not None and hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            # Handle class imbalance - modify weights before padding
            if idx_to_chord is not None and not self.use_focal_loss:
                class_weights = self._adjust_weights_for_no_chord(class_weights, idx_to_chord)
            
            expected_classes = model.fc.out_features
            self._log(f"Padding class weights from {len(class_weights)} to {expected_classes}")
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
            self._log(f"WARNING: warmup_start_lr ({self.warmup_start_lr}) >= warmup_end_lr ({self.warmup_end_lr})")
            self._log("This would cause LR to decrease during warmup instead of increasing. Swapping values.")
            temp = self.warmup_start_lr
            self.warmup_start_lr = self.warmup_end_lr
            self.warmup_end_lr = temp
            self._log(f"After swap: warmup_start_lr ({self.warmup_start_lr}) → warmup_end_lr ({self.warmup_end_lr})")
        
        # LR schedule type for smooth scheduling
        self.lr_schedule_type = lr_schedule_type
        self.smooth_scheduler = None
        self._last_stepped_epoch = None  # Track last stepped epoch to avoid duplicates
        
        # Modified to allow both warmup and scheduler to coexist
        # Set up smooth scheduler if requested
        if self.lr_schedule_type:
            self._create_smooth_scheduler()
            self._log(f"Using smooth '{self.lr_schedule_type}' learning rate schedule")
        else:
            self._log("Using validation-based learning rate adjustment")
        
        # Set up warmup independent of scheduler choice
        if self.use_warmup:
            self._log(f"Using warm-up LR schedule for first {self.warmup_epochs} epochs")
            self._log(f"Warm-up LR: {self.warmup_start_lr:.6f} → {self.warmup_end_lr:.6f}")
            # Set initial learning rate to warm-up start LR
            self._set_lr(self.warmup_start_lr)
            
            # If using both warmup and a scheduler, log this combined approach
            if self.lr_schedule_type:
                self._log(f"Will apply {self.lr_schedule_type} scheduler after warmup completes")
        
        # Best model tracking
        self.best_model_path = os.path.join(self.checkpoint_dir, "student_model_best.pth")
        self.chord_mapping = None
        
        # Add flag to track and debug scheduler stepping
        self._scheduler_step_count = 0

    def _create_smooth_scheduler(self):
        """Create a smooth learning rate scheduler that works with warmup."""
        # Store total training epochs for scheduler calculations
        self.total_training_epochs = self.num_epochs
        
        # If warmup is enabled, we'll create a special combined scheduler
        if self.use_warmup and self.warmup_epochs > 0:
            # For warmup + scheduler combo, we create scheduler for post-warmup epochs only
            self.post_warmup_epochs = max(1, self.num_epochs - self.warmup_epochs)
            self._log(f"Creating scheduler for {self.post_warmup_epochs} post-warmup epochs")
        else:
            # No warmup, scheduler will be used for all epochs
            self.post_warmup_epochs = self.num_epochs
            
        if self.lr_schedule_type == 'cosine':
            # Cosine annealing from warmup_end_lr (or initial_lr if no warmup) to min_lr
            # Note: We'll start this scheduler from the warmup_end_lr value, not from self.initial_lr
            start_lr = self.warmup_end_lr if self.use_warmup else self.initial_lr
            self.smooth_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.post_warmup_epochs,
                eta_min=self.min_lr
            )
            self._log(f"Cosine annealing from {start_lr:.6f} to {self.min_lr:.6f} after warmup")
        elif self.lr_schedule_type == 'cosine_warm_restarts':
            # Cosine annealing with warm restarts
            # First restart after 5 epochs, then double the period
            self.smooth_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5,  # First restart after 5 epochs
                T_mult=2,  # Double the period after each restart
                eta_min=self.min_lr
            )
            self._log(f"Cosine annealing with warm restarts: min_lr={self.min_lr:.6f} after warmup")
            
        elif self.lr_schedule_type == 'one_cycle':
            # One-cycle learning rate schedule
            steps_per_epoch = 100  # Estimate, will be updated in train()
            self.smooth_scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.initial_lr * 10,  # Peak LR
                total_steps=steps_per_epoch * self.post_warmup_epochs,
                pct_start=0.3,  # Spend 30% ramping up, 70% ramping down
                div_factor=25,  # Initial LR = max_lr/25
                final_div_factor=10000,  # Final LR = max_lr/10000
                anneal_strategy='cos'
            )
            self._log(f"One-cycle LR: {self.initial_lr:.6f} → {self.initial_lr*10:.6f} → {self.initial_lr*10/10000:.8f} after warmup")
            
        elif self.lr_schedule_type == 'linear_decay':
            # Linear decay from initial LR to min_lr
            lambda_fn = lambda epoch: 1 - (1 - self.min_lr / self.initial_lr) * (epoch / self.post_warmup_epochs)
            self.smooth_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_fn)
            self._log(f"Linear decay from {self.initial_lr:.6f} to {self.min_lr:.6f} after warmup")
            
        else:
            self._log(f"Unknown scheduler type: {self.lr_schedule_type}. Using warmup-only with validation-based adjustment")
            self.lr_schedule_type = None
    
    def _update_smooth_scheduler(self, epoch, batch_idx, num_batches):
        """Update learning rate scheduler with fractional epochs."""
        if self.smooth_scheduler is None:
            return
            
        # Ensure we have valid inputs
        if batch_idx is None or num_batches is None or num_batches == 0:
            return
        
        # Only step schedulers that support fractional epochs
        if isinstance(self.smooth_scheduler, (OneCycleLR, CosineAnnealingWarmRestarts)):
            # These can be stepped more frequently for smoother changes
            if batch_idx > 0 and batch_idx % max(1, num_batches // 10) == 0:
                # Calculate fractional epoch
                fractional_epoch = (epoch - 1) + (batch_idx / num_batches)
                self.smooth_scheduler.step(fractional_epoch)
                self._scheduler_step_count += 1
                self._log(f"Fractional scheduler step at epoch {epoch}, batch {batch_idx}/{num_batches}: LR={self.optimizer.param_groups[0]['lr']:.7f}")

    def _update_learning_rate(self, epoch, batch_idx=None, num_batches=None):
        """
        Update learning rate combining warmup and scheduler logic.
        
        Args:
            epoch: Current training epoch (1-indexed)
            batch_idx: Current batch index within epoch (for fractional updates)
            num_batches: Total number of batches per epoch (for fractional updates)
        
        Returns:
            Current learning rate after update
        """
        # Determine if we're in warmup phase (epoch is 1-indexed)
        in_warmup = self.use_warmup and epoch <= self.warmup_epochs
        
        if in_warmup:
            # In warmup phase, use warmup-specific LR calculation
            return self._warmup_learning_rate(epoch)
        elif self.lr_schedule_type:
            # After warmup, use the selected scheduler with adjusted epoch numbering
            if self.use_warmup:
                # For fractional updates within an epoch (some schedulers support this)
                if batch_idx is not None and num_batches is not None and isinstance(self.smooth_scheduler, (OneCycleLR, CosineAnnealingWarmRestarts)):
                    fractional_epoch = epoch - self.warmup_epochs - 1 + (batch_idx / num_batches)
                    self.smooth_scheduler.step(fractional_epoch)
                    self._scheduler_step_count += 1
                    self._log(f"Fractional scheduler step at epoch {epoch}, batch {batch_idx}/{num_batches}: LR={self.optimizer.param_groups[0]['lr']:.7f}")
                else:
                    # Just use regular epoch-based steps (CosineAnnealingLR, LambdaLR)
                    # Only step if we haven't already stepped for this epoch
                    if not hasattr(self, '_last_stepped_epoch') or self._last_stepped_epoch != (epoch - self.warmup_epochs - 1):
                        # Use step() without epoch parameter to avoid deprecation warning
                        self.smooth_scheduler.step()
                        self._last_stepped_epoch = epoch - self.warmup_epochs - 1
                        self._scheduler_step_count += 1
                        self._log(f"Full scheduler step at adjusted epoch {epoch - self.warmup_epochs - 1}: LR={self.optimizer.param_groups[0]['lr']:.7f}")
            else:
                # No warmup - use standard scheduler stepping
                if batch_idx is not None and num_batches is not None and isinstance(self.smooth_scheduler, (OneCycleLR, CosineAnnealingWarmRestarts)):
                    fractional_epoch = (epoch - 1) + (batch_idx / num_batches)
                    self.smooth_scheduler.step(fractional_epoch)
                    self._scheduler_step_count += 1
                    self._log(f"Fractional scheduler step (no warmup) at epoch {epoch}, batch {batch_idx}/{num_batches}: LR={self.optimizer.param_groups[0]['lr']:.7f}")
                else:
                    # Just use regular epoch-based steps without epoch parameter
                    if not hasattr(self, '_last_stepped_epoch') or self._last_stepped_epoch != (epoch - 1):
                        self.smooth_scheduler.step()
                        self._last_stepped_epoch = epoch - 1
                        self._scheduler_step_count += 1
                        self._log(f"Full scheduler step (no warmup) at epoch {epoch-1}: LR={self.optimizer.param_groups[0]['lr']:.7f}")
                        
            return self.optimizer.param_groups[0]['lr']
        else:
            # No scheduler - just return current LR (will be handled by validation-based adjustment)
            return self.optimizer.param_groups[0]['lr']
    
    def _adjust_weights_for_no_chord(self, weights, idx_to_chord):
        """
        Previously adjusted weights for 'N' chord class. Now simply logs chord information
        without modifying weights to ensure balanced training.
        """
        n_chord_idx = None
        
        # Find the index that corresponds to "N" chord (for logging only)
        for idx, chord in idx_to_chord.items():
            if chord == "N":
                n_chord_idx = idx
                break
        
        if n_chord_idx is not None and n_chord_idx < len(weights):
            self._log(f"Found 'N' (no chord) at index {n_chord_idx}")
            self._log(f"Weight for 'N' class: {weights[n_chord_idx]:.4f}")
            self._log("No weight adjustment applied - using original class weights")
        
        # Return weights unmodified
        return weights
    
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

    def distillation_loss(self, student_logits, teacher_logits, targets):
        """
        Combine a KL divergence loss between softened teacher and student predictions with
        standard cross entropy loss on the targets.
        
        Args:
            student_logits (Tensor): Student raw outputs.
            teacher_logits (Tensor): Teacher soft targets.
            targets (Tensor): Target labels.
        Returns:
            Combined loss.
        """
        # Verify dimensions match before proceeding
        if student_logits.shape != teacher_logits.shape:
            # Comment out dimension mismatch logging to reduce confusion
            # self._log(f"Dimension mismatch in distillation_loss: student {student_logits.shape}, teacher {teacher_logits.shape}")
            
            try:
                # Try to adapt dimensions - comment out detailed logging for diagnosis
                s_shape, t_shape = student_logits.shape, teacher_logits.shape
                
                # Handle 4D teacher tensor (batch, seq, time, classes) → needs special handling
                if len(t_shape) == 4 and t_shape[1] == 1:
                    # This is the key fix: squeeze dimension 1 and then reshape
                    # self._log(f"Reshaping 4D teacher tensor with shape {t_shape}")
                    # First squeeze the singleton dimension
                    teacher_logits = teacher_logits.squeeze(1)  # Now (batch, time, classes)
                    
                    # If the student logits are already flattened (batch*time, classes)
                    if len(s_shape) == 2:
                        # Calculate if the batch sizes would match after flattening
                        expected_batch_size = t_shape[0] * t_shape[2]  # batch * time
                        if expected_batch_size == s_shape[0]:
                            # Perfect! Just reshape teacher to match student
                            # self._log(f"Flattening teacher tensor from {teacher_logits.shape} to match student {s_shape}")
                            teacher_logits = teacher_logits.reshape(-1, t_shape[3])  # (batch*time, classes)
                        else:
                            # self._log(f"Batch size mismatch after flattening: expected {expected_batch_size}, got {s_shape[0]}")
                            # If student batch is larger, need to subsample
                            if s_shape[0] > expected_batch_size:
                                pass  # Incomplete code - will be implemented later
                            else:
                                pass  # Incomplete code - will be implemented later
                
                # If that didn't work, try more basic approaches
                if student_logits.shape != teacher_logits.shape:
                    if len(s_shape) != len(t_shape):
                        # self._log(f"Dimension count mismatch: student has {len(s_shape)}, teacher has {len(t_shape)}")
                        pass
                    
                    # Check for batch size mismatch in 2D case
                    if len(s_shape) == 2 and len(t_shape) == 2 and s_shape[1] == t_shape[1]:
                        if s_shape[0] > t_shape[0]:
                            # self._log(f"Batch size mismatch: student={s_shape[0]}, teacher={t_shape[0]}")
                            # self._log(f"Subsampling student batch from {s_shape[0]} to {t_shape[0]}")
                            student_logits = student_logits[:t_shape[0]]
                            targets = targets[:t_shape[0]] if targets.size(0) > t_shape[0] else targets
                        else:
                            # self._log(f"Batch size mismatch: student={s_shape[0]}, teacher={t_shape[0]}")
                            # self._log(f"Truncating teacher batch from {t_shape[0]} to {s_shape[0]}")
                            teacher_logits = teacher_logits[:s_shape[0]]
                
                # Final check once more before continuing
                if student_logits.shape != teacher_logits.shape:
                    raise ValueError(f"Failed to match dimensions after attempted fixes: student {student_logits.shape}, teacher {teacher_logits.shape}")
                
                # self._log(f"Successfully adapted dimensions for KD loss to {student_logits.shape}")
            except Exception as e:
                self._log(f"Error adapting dimensions: {str(e)}")
                self._log("Falling back to standard cross entropy loss")
                return F.cross_entropy(student_logits, targets)
        
        try:
            # Use class attributes for temperature and alpha
            temperature = self.temperature
            alpha = self.kd_alpha


            # student_logits = student_logits - student_logits.mean(dim=-1, keepdim=True)
            # teacher_logits = teacher_logits - teacher_logits.mean(dim=-1, keepdim=True)
            # student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            # teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

            # Apply zero-mean normalization to the logits before softmax
            # This is important for stable knowledge distillation as suggested by research
            student_logits_normalized = student_logits - student_logits.mean(dim=1, keepdim=True)
            teacher_logits_normalized = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)
            
            # KL divergence loss with temperature scaling for soft targets
            student_log_probs = F.log_softmax(student_logits_normalized / temperature, dim=1)
            teacher_probs = F.softmax(teacher_logits_normalized / temperature, dim=1)
            
            # Check for NaN values that could break the loss calculation
            if torch.isnan(student_log_probs).any() or torch.isnan(teacher_probs).any():
                self._log("WARNING: NaN values detected in KD loss inputs")
                student_log_probs = torch.nan_to_num(student_log_probs)
                teacher_probs = torch.nan_to_num(teacher_probs)
            
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
            
            if self.use_focal_loss:
                loss = self.focal_loss(student_logits, targets, gamma=self.focal_gamma, alpha=self.focal_alpha)
            else:
                loss = self.loss_fn(student_logits, targets)
           
            # Combine losses with alpha weighting
            combined_loss = alpha * kl_loss + (1 - alpha) * loss
            
            # Add logging once for diagnostics
            if not hasattr(self, '_kd_loss_logged'):
                self._log(f"KD loss breakdown - KL: {kl_loss.item():.4f}, CE: {loss.item():.4f}, Combined: {combined_loss.item():.4f}")
                self._log(f"Using α={alpha}, temperature={temperature} with zero-mean logit normalization")
                self._kd_loss_logged = True
            
            # NEW: Print teacher distribution details only once during training
            if not hasattr(self, '_teacher_distribution_logged') and teacher_probs.size(0) > 0:
                sample_teacher = teacher_probs[0]
                top_values, top_indices = torch.topk(sample_teacher, 10)
                self._log("Teacher distribution for first sample in batch:")
                for rank, (val, idx) in enumerate(zip(top_values, top_indices), start=1):
                    chord_name = self.idx_to_chord.get(idx.item(), f"Class-{idx.item()}") if self.idx_to_chord else f"Class-{idx.item()}"
                    self._log(f"  Rank {rank}: {chord_name} with probability {val.item():.4f}")
                avg_confidence = teacher_probs.max(dim=1)[0].mean().item()
                self._log(f"Average teacher max probability in batch: {avg_confidence:.4f}")
                # Set flag to prevent future prints
                self._teacher_distribution_logged = True
            
            return combined_loss
        except Exception as e:
            self._log(f"Error in distillation loss calculation: {str(e)}")
            self._log("Falling back to standard cross entropy loss")
            return F.cross_entropy(student_logits, targets)

    def label_smoothed_loss(self, logits, targets, smoothing=0.1):
        """
        Compute label-smoothed cross entropy loss.
        
        Args:
            logits (Tensor): Student predictions.
            targets (Tensor): Pseudo-labels.
            smoothing (float): Smoothing factor.
        
        Returns:
            Scalar loss.
        """
        try:
            n_classes = logits.size(-1)
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                # Construct soft targets
                true_dist = torch.full_like(log_probs, smoothing / (n_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
            loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
            return loss
        except Exception as e:
            self._log(f"Error in label smoothed loss calculation: {str(e)}")
            self._log("Falling back to standard cross entropy loss")
            return F.cross_entropy(logits, targets)

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
                
            self._log(f"Using padding value: {padding_value:.4f}")
            
            import numpy as np
            padded_weights = np.zeros(expected_length, dtype=np.float32)
            padded_weights[:len(weights)] = weights
            padded_weights[len(weights):] = padding_value
            return padded_weights
        
        # If weights array is too long, truncate it
        if len(weights) > expected_length:
            self._log(f"Warning: Truncating class weights from {len(weights)} to {expected_length}")
            return weights[:expected_length]
        
        return weights
    
    def _set_lr(self, new_lr):
        """Helper to set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    
    def _warmup_learning_rate(self, epoch):
        """Calculate and set learning rate during warm-up period"""
        if epoch > self.warmup_epochs:
            # Warm-up complete, set to end LR
            return self._set_lr(self.warmup_end_lr)
        
        # Convert 1-indexed epoch to 0-indexed for correct linear interpolation
        # This fixes the issue where warmup wasn't properly increasing from start_lr
        warmup_progress = (epoch - 1) / max(1, self.warmup_epochs - 1) 
        
        # Clamp to ensure we don't go below start_lr or above end_lr due to rounding
        warmup_progress = max(0.0, min(1.0, warmup_progress))
        
        # Linear interpolation between start_lr and end_lr
        new_lr = self.warmup_start_lr + warmup_progress * (self.warmup_end_lr - self.warmup_start_lr)
        
        # Add more detailed logging for the first few warmup epochs to verify correct ramp-up
        # Log only for first 3 epochs, every 100th epoch, or the final warmup epoch
        # if epoch <= 3 or epoch == self.warmup_epochs or epoch % 100 == 0:
        #     self._log(f"Warm-up epoch {epoch}/{self.warmup_epochs}: progress={warmup_progress:.4f}, LR = {new_lr:.6f}")
        
        return self._set_lr(new_lr)
        
    def _adjust_learning_rate(self, val_acc):
        """Adjust learning rate based on validation accuracy."""
        if self.before_val_acc > val_acc:
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = self._reduce_lr(self.optimizer, self.lr_decay_factor, self.min_lr)
            self._log(f"Decreasing learning rate from {old_lr:.6f} to {new_lr:.6f}")
        
        # Update previous validation accuracy
        self.before_val_acc = val_acc
    
    def _reduce_lr(self, optimizer, factor=0.95, min_lr=5e-6):
        """Reduce learning rate by a factor but ensuring it doesn't go below min_lr."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * factor, min_lr)
        return param_group['lr']
    
    def set_chord_mapping(self, chord_mapping):
        """Set chord mapping for saving with checkpoints."""
        self.chord_mapping = chord_mapping
    
    def _save_best_model(self, val_acc, val_loss, epoch):
        """Save the model when validation accuracy improves."""
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.early_stop_counter = 0
            
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'accuracy': val_acc,
                    'chord_mapping': self.chord_mapping,
                    'idx_to_chord': self.idx_to_chord,
                    'mean': self.normalization['mean'] if self.normalization else None,
                    'std': self.normalization['std'] if self.normalization else None
                }, self.best_model_path)
                
                self._log(f"Saved best model with validation accuracy: {val_acc:.4f}")
                return True
            except Exception as e:
                self._log(f"Error saving model: {str(e)}")
                return False
        else:
            self.early_stop_counter += 1
            return False
    
    def _check_early_stopping(self):
        """Check if early stopping criteria is met."""
        if self.early_stop_counter >= self.early_stopping_patience:
            self._log(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement")
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
                    self._log(f"Error during validation: {str(e)}")
                    continue  # Skip this batch if there's an error
        
        if val_total == 0:
            self._log("WARNING: No validation samples were correctly processed!")
            return float('inf'), 0.0  # Return worst possible values
            
        avg_loss = val_loss / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        self._log(f"Epoch Validation Loss: {avg_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Pass the current epoch to the confusion matrix function
        self._calculate_confusion_matrix(all_preds, all_targets, current_epoch)
        
        self.model.train()
        return avg_loss, val_acc
    
    def _calculate_confusion_matrix(self, predictions, targets, current_epoch=None):
        """
        Calculate, log, and save the confusion matrix.
        Prints chord quality groups for clearer visualization and also saves the
        full class confusion matrix every 10 epochs.
        
        Args:
            predictions: List of predicted class indices
            targets: List of target class indices
            current_epoch: Current training epoch number (for periodic full matrix saving)
        """
        if not predictions or not targets:
            self._log("Cannot calculate confusion matrix: no predictions or targets")
            return
            
        try:
            # Count occurrences of each class in targets
            target_counter = Counter(targets)
            total_samples = len(targets)
            
            # Get the most common classes (up to 10) for standard printing
            most_common_classes = [cls for cls, _ in target_counter.most_common(10)]
            
            # Create a mapping of indices to chord names if available
            chord_names = {}
            if self.idx_to_chord:
                # First, build a reverse mapping from index to chord name
                for idx, chord in self.idx_to_chord.items():
                    chord_names[idx] = chord
                    
                # Also make sure all our most common classes are mapped
                for cls in most_common_classes:
                    if cls not in chord_names:
                        if cls in self.idx_to_chord:
                            chord_names[cls] = self.idx_to_chord[cls]
                        else:
                            # If the class is not in idx_to_chord, use a consistent fallback
                            chord_names[cls] = f"Unknown-{cls}"
            else:
                # If no mapping is available, create generic labels
                for cls in most_common_classes:
                    chord_names[cls] = f"Class-{cls}"
            
            # Log class distribution for top 10 classes (print only)
            self._log("\nClass distribution in validation set (Top 10):")
            for cls in most_common_classes:
                try:
                    count = target_counter.get(cls, 0)
                    percentage = 100 * count / total_samples if total_samples > 0 else 0
                    chord_name = chord_names.get(cls, f"Class-{cls}")
                    self._log(f"  {chord_name}: {count} samples ({percentage:.2f}%)")
                except Exception as e:
                    self._log(f"Error processing class {cls}: {e}")
            
            # Calculate confusion matrix values for top classes (printed to log)
            confusion = {}
            for true_cls in most_common_classes:
                try:
                    # Get indices where true class equals this class
                    true_indices = [i for i, t in enumerate(targets) if t == true_cls]
                    if not true_indices:
                        continue
                        
                    # Get predictions for these indices
                    cls_preds = [predictions[i] for i in true_indices]
                    pred_counter = Counter(cls_preds)
                    
                    # Calculate accuracy for this class
                    correct = pred_counter.get(true_cls, 0)
                    total = len(true_indices)
                    accuracy = correct / total if total > 0 else 0
                    
                    # Get the most commonly predicted class for this true class
                    most_predicted = pred_counter.most_common(1)[0][0] if pred_counter else true_cls
                    
                    # Use the chord_names dictionary consistently
                    true_chord_name = chord_names.get(true_cls, f"Class-{true_cls}")
                    pred_chord_name = chord_names.get(most_predicted, f"Class-{most_predicted}")
                    
                    confusion[true_chord_name] = {
                        'accuracy': accuracy,
                        'most_predicted': pred_chord_name,
                        'correct': correct,
                        'total': total
                    }
                except Exception as e:
                    self._log(f"Error processing confusion data for class {true_cls}: {e}")
            
            # Log confusion matrix information for top classes
            self._log("\nConfusion Matrix Analysis (Top 10 Classes):")
            self._log(f"{'True Class':<20} | {'Accuracy':<10} | {'Most Predicted':<20} | {'Correct/Total'}")
            self._log(f"{'-'*20} | {'-'*10} | {'-'*20} | {'-'*15}")
            
            for true_class, stats in confusion.items():
                self._log(f"{true_class:<20} | {stats['accuracy']:.4f}     | {stats['most_predicted']:<20} | {stats['correct']}/{stats['total']}")
                
            # Calculate overall metrics for these common classes
            common_correct = sum(confusion[cls]['correct'] for cls in confusion)
            common_total = sum(confusion[cls]['total'] for cls in confusion)
            common_acc = common_correct / common_total if common_total > 0 else 0
            self._log(f"\nAccuracy on most common classes: {common_acc:.4f} ({common_correct}/{common_total})")
            
            # NEW: Create and visualize chord quality group confusion matrix using chords.py
            if self.idx_to_chord:
                self._log("\n=== Creating chord quality group confusion matrix ===")
                try:
                    # Use the visualization module to calculate quality statistics
                    quality_cm, quality_counts, quality_accuracy, quality_groups = calculate_quality_confusion_matrix(
                        predictions, targets, self.idx_to_chord
                    )
                    
                    # Log quality distribution
                    self._log("\nChord quality distribution:")
                    for i, quality in enumerate(quality_groups):
                        count = quality_counts.get(i, 0)
                        percentage = 100 * count / len(targets) if targets else 0
                        self._log(f"  {quality}: {count} samples ({percentage:.2f}%)")
                    
                    # Log accuracies by quality
                    self._log("\nAccuracy by chord quality:")
                    for quality, acc in sorted(quality_accuracy.items(), key=lambda x: x[1], reverse=True):
                        self._log(f"  {quality}: {acc:.4f}")
                    
                    # Create and save chord quality confusion matrix
                    title = f"Chord Quality Confusion Matrix - Epoch {current_epoch}"
                    quality_cm_path = os.path.join(
                        self.checkpoint_dir, 
                        f"confusion_matrix_quality_epoch_{current_epoch}.png"
                    )
                    
                    # Plot using the visualization function
                    try:
                        _, _, _, _, _ = plot_chord_quality_confusion_matrix(
                            predictions, targets, self.idx_to_chord,
                            title=title, save_path=quality_cm_path
                        )
                        self._log(f"Saved chord quality confusion matrix to {quality_cm_path}")
                    except Exception as e:
                        self._log(f"Error plotting chord quality confusion matrix: {e}")
                        # Print normalized confusion matrix as text fallback
                        self._log("\nChord Quality Confusion Matrix (normalized):")
                        normalized_cm = quality_cm.astype('float') / quality_cm.sum(axis=1)[:, np.newaxis]
                        normalized_cm = np.nan_to_num(normalized_cm)  # Replace NaN with zero
                        for i, row in enumerate(normalized_cm):
                            self._log(f"{quality_groups[i]:<10}: " + " ".join([f"{x:.2f}" for x in row]))
                        
                except Exception as e:
                    self._log(f"Error creating quality-based confusion matrix: {e}")
                    import traceback
                    self._log(traceback.format_exc())
            
            # Create and save the full confusion matrix every 10 epochs (less frequently)
            if current_epoch is None or current_epoch % 10 == 0:
                self._log(f"\nSaving full class confusion matrix for epoch {current_epoch}")
                
                try:
                    # Create a mapping that includes ALL possible chord indices
                    all_class_mapping = {}
                    if self.idx_to_chord:
                        for idx, chord in self.idx_to_chord.items():
                            all_class_mapping[idx] = chord
                    
                    # Get a list of all unique classes
                    all_classes = set(targets).union(set(predictions))
                    
                    # Ensure the classes are sorted for consistent visualization
                    all_classes_list = sorted(list(all_classes))
                    
                    # Make sure all classes have labels
                    for cls in all_classes:
                        if cls not in all_class_mapping:
                            all_class_mapping[cls] = f"Class-{cls}"
                    
                    # Generate the full confusion matrix using the visualization function
                    full_title = f"Full Confusion Matrix - Epoch {current_epoch}"
                    np_targets_full = np.array(targets)
                    np_preds_full = np.array(predictions)
                    
                    # Set save path
                    full_cm_path = os.path.join(
                        self.checkpoint_dir, 
                        f"confusion_matrix_full_epoch_{current_epoch}.png"
                    )
                    
                    # Generate full confusion matrix plot and save it
                    fig_full = plot_confusion_matrix(
                        np_targets_full, np_preds_full,
                        class_names=all_class_mapping,
                        normalize=True,
                        title=full_title,
                        max_classes=None  # No limit on number of classes
                    )
                    
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    fig_full.savefig(full_cm_path, dpi=300, bbox_inches='tight')
                    self._log(f"Saved full confusion matrix to {full_cm_path}")
                    plt.close(fig_full)
                    
                except Exception as e:
                    self._log(f"Error saving full confusion matrix visualization: {e}")
                    import traceback
                    self._log(traceback.format_exc())
            
        except Exception as e:
            self._log(f"Error calculating confusion matrix: {e}")
            import traceback
            self._log(traceback.format_exc())

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
            
            # Verify and log input shapes for debugging
            orig_logits_shape = logits.shape
            orig_targets_shape = targets.shape
            
            # Check for and handle dimension mismatch between logits and targets
            if logits.ndim == 3 and targets.ndim == 1:
                self._log(f"WARNING: Dimension mismatch - logits: {logits.shape}, targets: {targets.shape}")
                self._log("Averaging logits over time dimension")
                logits = logits.mean(dim=1)  # Average over time dimension
            elif logits.ndim == 3 and targets.ndim == 2:
                if logits.size(1) != targets.size(1):
                    self._log(f"WARNING: Time dimension mismatch - logits: {logits.shape}, targets: {targets.shape}")
                    if logits.size(1) > targets.size(1):
                        self._log(f"Truncating logits time dimension from {logits.size(1)} to {targets.size(1)}")
                        logits = logits[:, :targets.size(1), :]
                    else:
                        self._log(f"Truncating targets time dimension from {targets.size(1)} to {logits.size(1)}")
                        targets = targets[:, :logits.size(1)]
                
                # Reshape for loss calculation
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)
                
            # Report reshape results
            if orig_logits_shape != logits.shape or orig_targets_shape != targets.shape:
                self._log(f"Reshaped tensors for loss calculation - logits: {orig_logits_shape} -> {logits.shape}, "
                         f"targets: {orig_targets_shape} -> {targets.shape}")
            
            # Use focal loss if enabled
            if self.use_focal_loss:
                # Note: focal_loss implementation uses weight=None directly
                # as the weight parameter is controlled by focal_alpha
                loss = self.focal_loss(logits, targets, 
                                      gamma=self.focal_gamma, 
                                      alpha=self.focal_alpha)
            else:
                # Use the standard loss function with class weights
                try:
                    loss = self.loss_fn(logits, targets)
                except RuntimeError as e:
                    self._log(f"Error in loss calculation: {e}")
                    self._log(f"Logits shape: {logits.shape}, targets shape: {targets.shape}")
                    self._log(f"Target values: min={targets.min().item()}, max={targets.max().item()}")
                    
                    # Try to recover by ensuring targets are in valid range
                    num_classes = logits.size(-1)
                    if targets.max().item() >= num_classes:
                        self._log(f"WARNING: Target values exceed output dimension {num_classes}, clamping")
                        targets = torch.clamp(targets, 0, num_classes-1)
                        loss = self.loss_fn(logits, targets)
                    else:
                        # Use a simpler loss function if we can't recover
                        self._log("Using unweighted cross entropy as fallback")
                        loss = F.cross_entropy(logits, targets)
            
            # If KD loss is enabled and teacher_logits are given, compute KD loss and combine.
            if self.use_kd_loss and teacher_logits is not None:
                try:
                    # Standardize teacher logits dimensions - ensure they're compatible
                    # Teacher logits should be [batch, time, classes] or [batch, classes]
                    
                    # Handle 3D teacher logits (batch, time, classes)
                    if teacher_logits.dim() == 3:
                        # If student logits are 2D (batch, classes), then average teacher logits over time
                        if logits.dim() == 2:
                            teacher_logits = teacher_logits.mean(dim=1)  # Average over time dimension
                            self._log(f"Averaged 3D teacher logits over time to match 2D student logits", level='debug')
                    # Handle 1D or 2D teacher logits
                    elif teacher_logits.dim() <= 2:
                        # If student logits are 3D but teacher is 2D, unsqueeze teacher
                        if logits.dim() == 3 and teacher_logits.dim() == 2:
                            if teacher_logits.shape[0] == logits.shape[0]:  # Same batch size
                                # Add time dimension of size 1
                                teacher_logits = teacher_logits.unsqueeze(1)
                                # Expand to match student time dimension
                                teacher_logits = teacher_logits.expand(-1, logits.shape[1], -1)
                                self._log(f"Expanded 2D teacher logits to match 3D student logits", level='debug')
                    
                    # Final dimension check
                    if teacher_logits.shape[-1] != logits.shape[-1]:
                        # Class dimension mismatch, reshape teacher
                        if teacher_logits.shape[-1] > logits.shape[-1]:
                            # Truncate teacher classes
                            if teacher_logits.dim() == 3:
                                teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
                            else:
                                teacher_logits = teacher_logits[..., :logits.shape[-1]]
                        else:
                            # Pad teacher classes
                            pad_size = logits.shape[-1] - teacher_logits.shape[-1]
                            if teacher_logits.dim() == 3:
                                pad_shape = (0, pad_size, 0, 0, 0, 0)  # Padding last dim
                            else:
                                pad_shape = (0, pad_size, 0, 0)  # Padding last dim
                            teacher_logits = F.pad(teacher_logits, pad_shape, "constant", 0)
                    
                    # Now call distillation loss with standardized dimensions
                    kd_loss = self.distillation_loss(logits, teacher_logits, targets)
                    return kd_loss
                except Exception as e:
                    self._log(f"ERROR in KD loss processing - falling back to standard loss: {str(e)}")
                    # Continue with standard loss
            
            # Ensure loss is non-negative (critical fix)
            loss = torch.clamp(loss, min=0.0)
            
            if torch.isnan(loss):
                self._log(f"NaN loss detected - logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
                         f"targets: min={targets.min().item()}, max={targets.max().item()}")
                # Return a default loss value instead of NaN
                return torch.tensor(1.0, device=loss.device, requires_grad=True)
            
            return loss
        except Exception as e:
            self._log(f"Unexpected error in compute_loss: {str(e)}")
            # Last resort fallback - provide a dummy loss to avoid training failure
            return torch.tensor(1.0, device=self.device, requires_grad=True)

    def _process_batch(self, batch):
        """Process a batch to extract inputs and targets with GPU acceleration"""
        if isinstance(batch, dict):
            # Extract spectrograms and targets from dictionary
            inputs = batch.get('spectro')
            targets = batch.get('chord_idx')
        else:
            # Default unpacking
            inputs, targets = batch
            
        # Move to device if not already there
        if inputs.device != self.device:
            inputs = inputs.to(self.device, non_blocking=True)
        if targets.device != self.device:
            targets = targets.to(self.device, non_blocking=True)
            
        return inputs, targets

    def train(self, train_loader, val_loader=None, start_epoch=1):
        """
        Train the model with optimized GPU usage.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            start_epoch: Epoch to start training from (for resuming from checkpoint)
        """
        try:
            self.model.train()
            
            # Get actual steps per epoch for scheduler
            num_batches = len(train_loader)
            
            # Optimize GPU memory usage by clearing cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Pre-allocate common tensors to avoid repeated allocations
            if torch.cuda.is_available():
                self._zero_tensor = torch.zeros(1, device=self.device)
            
            # Update OneCycleLR with actual steps if needed
            if isinstance(self.smooth_scheduler, OneCycleLR):
                # Calculate total steps adjusting for warmup if needed
                if self.use_warmup and start_epoch <= self.warmup_epochs:
                    # If we're still in warmup, scheduler only needs post-warmup steps
                    remaining_epochs = self.num_epochs - self.warmup_epochs
                    total_steps = num_batches * remaining_epochs
                    self._log(f"OneCycleLR configured for {total_steps} steps after {self.warmup_epochs} warmup epochs")
                elif self.use_warmup and start_epoch > self.warmup_epochs:
                    # Already past warmup, calculate remaining post-warmup epochs
                    remaining_epochs = self.num_epochs - (start_epoch - 1)
                    total_steps = num_batches * remaining_epochs
                    self._log(f"Resuming OneCycleLR for {total_steps} steps (past warmup)")
                else:
                    # No warmup, scheduler handles all epochs
                    remaining_epochs = self.num_epochs - (start_epoch - 1)
                    total_steps = num_batches * remaining_epochs
                    self._log(f"OneCycleLR configured for {total_steps} steps")
                    
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
                self._log(f"Knowledge Distillation enabled: α={self.kd_alpha}, temperature={self.temperature}")
                
                # Check first batch for teacher logits - early verification
                try:
                    self._log("Verifying teacher logits availability in first batch...")
                    first_batch = next(iter(train_loader))
                    if 'teacher_logits' not in first_batch:
                        self._log("WARNING: KD is enabled but teacher_logits are missing from the first batch!")
                        self._log("This will cause training to fall back to standard CE loss and may lead to poor results.")
                        self._log("Please ensure your dataset properly includes teacher logits for all samples.")
                    else:
                        teacher_shape = first_batch['teacher_logits'].shape
                        self._log(f"Teacher logits verified with shape: {teacher_shape}")
                        # Check if we have non-zero values in teacher logits
                        if first_batch['teacher_logits'].abs().sum().item() == 0:
                            self._log("WARNING: Teacher logits contain all zeros! Check your logits loading process.")
                except Exception as e:
                    self._log(f"Error checking first batch for teacher logits: {e}")
            else:
                self._log("Knowledge Distillation disabled, using standard loss")
            
            # Handle initial learning rate explicitly (before first epoch)
            if self.use_warmup and start_epoch == 1:
                # Explicitly set to warmup_start_lr to ensure we're starting from the right point
                curr_lr = self.optimizer.param_groups[0]['lr']
                if abs(curr_lr - self.warmup_start_lr) > 1e-7:  # Allow small floating point differences
                    self._log(f"Setting initial learning rate from {curr_lr:.6f} to warm-up start value: {self.warmup_start_lr:.6f}")
                    self._set_lr(self.warmup_start_lr)
                else:
                    self._log(f"Initial learning rate already set to warm-up start value: {curr_lr:.6f}")
            elif self.use_warmup and start_epoch > 1 and start_epoch <= self.warmup_epochs:
                # Resuming in the middle of warmup - calculate appropriate warmup LR
                old_lr = self.optimizer.param_groups[0]['lr']
                new_lr = self._warmup_learning_rate(start_epoch)
                self._log(f"Resuming in warmup phase at epoch {start_epoch} with LR adjusted from {old_lr:.6f} to {new_lr:.6f}")
            elif self.use_warmup and start_epoch > self.warmup_epochs and self.lr_schedule_type:
                # Resuming after warmup, need to advance scheduler to the right point
                effective_epoch = start_epoch - self.warmup_epochs - 1
                
                # For OneCycleLR and CosineAnnealingWarmRestarts that need explicit epoch values
                if isinstance(self.smooth_scheduler, (OneCycleLR, CosineAnnealingWarmRestarts)):
                    self.smooth_scheduler.step(effective_epoch)
                else:
                    # For other schedulers, step multiple times without epoch parameter
                    for _ in range(effective_epoch):
                        self.smooth_scheduler.step()
                        
                self._log(f"Resuming after warmup at epoch {start_epoch} (scheduler epoch {effective_epoch}) with LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
            # Debug values for warmup
            if self.use_warmup:
                self._log(f"Warmup configuration: {self.warmup_epochs} epochs from {self.warmup_start_lr:.6f} to {self.warmup_end_lr:.6f}")
                
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
                self._log(f"Epoch {epoch}: LR = {current_lr:.6f} (from {lr_source})")
                
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
                        self._log("Using mixed precision training with automatic mixed precision")
                else:
                    using_amp = False
                
                for batch_idx, batch in enumerate(train_loader):
                    # Update learning rate with batch info for smooth updates
                    if self.lr_schedule_type or self.use_warmup:
                        # Allow fractional epoch updates after warmup phase
                        self._update_learning_rate(epoch, batch_idx, num_batches)
                    
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
                            self._log(f"WARNING: KD enabled but no teacher logits found in batch {batch_idx}")
                            
                            # If this keeps happening, report it clearly
                            if batch_idx >= 100 and kd_batches == 0:
                                self._log("CRITICAL ERROR: Knowledge Distillation is enabled but NO teacher logits found in any batches so far!")
                                self._log("This indicates a problem with your dataset or logits loading. Training will continue with standard loss.")
                                self._log("You should stop training and fix the logits loading issue.")
                    
                    if self.normalization:
                        # Create the normalization tensors using clone().detach() if they don't exist
                        if not hasattr(self, '_norm_mean_tensor') or not hasattr(self, '_norm_std_tensor'):
                            # Fix: Use clone().detach() instead of torch.tensor()
                            self._norm_mean_tensor = torch.as_tensor(self.normalization['mean'], 
                                                                   device=self.device, dtype=torch.float).clone().detach()
                            self._norm_std_tensor = torch.as_tensor(self.normalization['std'], 
                                                                  device=self.device, dtype=torch.float).clone().detach()
                        
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

                                # For per-frame supervision, flatten logits and targets when needed
                                if logits.ndim == 3 and targets.ndim == 2:
                                    # Store original shape for adapting teacher_logits if needed
                                    orig_logits_shape = logits.shape
                                    
                                    logits = logits.reshape(-1, logits.size(-1))
                                    targets = targets.reshape(-1)
                                    
                                    # If we have teacher logits, reshape them too
                                    if teacher_logits is not None and teacher_logits.ndim == 3:
                                        if teacher_logits.shape[0:2] == orig_logits_shape[0:2]:
                                            teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))
                                        else:
                                            self._log(f"Shape mismatch between student ({orig_logits_shape}) and teacher ({teacher_logits.shape}) logits")
                                            # Try time dimension averaging as fallback 
                                            if teacher_logits.shape[0] == orig_logits_shape[0]:
                                                teacher_logits = teacher_logits.mean(dim=1)
                                            elif teacher_logits.ndim > 2:
                                                teacher_logits = teacher_logits.mean(dim=1).reshape(-1, teacher_logits.size(-1))

                                # Use our custom compute_loss method with teacher_logits if available
                                loss = self.compute_loss(logits, targets, teacher_logits)
                                
                                # Skip invalid losses (avoid NaN propagation)
                                if loss is not None and not torch.isnan(loss) and torch.isfinite(loss):
                                    # Use scaler for mixed precision
                                    scaler.scale(loss).backward()
                                    scaler.unscale_(self.optimizer)  # Unscale before clipping
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                                    scaler.step(self.optimizer)
                                    scaler.update()
                                    
                                    with torch.no_grad():
                                        preds = logits.argmax(dim=1)
                                        batch_correct = (preds == targets).sum().item()
                                        train_correct += batch_correct
                                        train_total += targets.size(0)
                                        epoch_loss += loss.item()
                                else:
                                    self._log(f"WARNING: Skipping batch {batch_idx} due to invalid loss: {loss}")
                        else:
                            # Standard full-precision training
                            outputs = self.model(inputs)
                            
                            if isinstance(outputs, tuple):
                                logits = outputs[0]
                            else:
                                logits = outputs

                            # For per-frame supervision, flatten logits and targets when needed
                            if logits.ndim == 3 and targets.ndim == 2:
                                # Store original shape for adapting teacher_logits if needed
                                orig_logits_shape = logits.shape
                                
                                logits = logits.reshape(-1, logits.size(-1))
                                targets = targets.reshape(-1)
                                
                                # If we have teacher logits, reshape them too
                                if teacher_logits is not None and teacher_logits.ndim == 3:
                                    if teacher_logits.shape[0:2] == orig_logits_shape[0:2]:
                                        teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))
                                    else:
                                        self._log(f"Shape mismatch between student ({orig_logits_shape}) and teacher ({teacher_logits.shape}) logits")
                                        # Try time dimension averaging as fallback 
                                        if teacher_logits.shape[0] == orig_logits_shape[0]:
                                            teacher_logits = teacher_logits.mean(dim=1)
                                        elif teacher_logits.ndim > 2:
                                            teacher_logits = teacher_logits.mean(dim=1).reshape(-1, teacher_logits.size(-1))

                            # Use our custom compute_loss method with teacher_logits if available
                            loss = self.compute_loss(logits, targets, teacher_logits)
                            
                            # Skip invalid losses (avoid NaN propagation)
                            if loss is not None and not torch.isnan(loss) and torch.isfinite(loss):
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                                self.optimizer.step()
                                
                                preds = logits.argmax(dim=1)
                                batch_correct = (preds == targets).sum().item()
                                train_correct += batch_correct
                                train_total += targets.size(0)
                                epoch_loss += loss.item()
                                
                                if batch_idx % 20 == 0:
                                    # Log current LR and indicate whether KD is being used for this batch
                                    current_lr = self.optimizer.param_groups[0]['lr']
                                    batch_acc = batch_correct / targets.size(0) if targets.size(0) > 0 else 0
                                    kd_status = " (with KD)" if teacher_logits is not None else ""
                                    self._log(f"Epoch {epoch}/{self.num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.4f}{kd_status} | LR: {current_lr:.7f}")
                            else:
                                self._log(f"WARNING: Skipping batch {batch_idx} due to invalid loss: {loss}")
                            
                    except Exception as e:
                        self._log(f"Error in training batch {batch_idx}: {str(e)}")
                        import traceback
                        self._log(traceback.format_exc())
                        continue  # Skip this batch if there's an error
                
                # After each epoch, clear CUDA cache to prevent memory fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Log KD usage statistics 
                if self.use_kd_loss:
                    kd_percent = (kd_batches / total_batches) * 100 if total_batches > 0 else 0
                    self._log(f"KD usage: {kd_batches}/{total_batches} batches ({kd_percent:.1f}%)")
                    
                    # Warn if KD usage is low, as this indicates a potential problem 
                    if kd_percent < 50 and total_batches > 10:
                        self._log(f"WARNING: KD is enabled but only {kd_percent:.1f}% of batches had teacher logits!")
                        self._log("This means most batches are falling back to standard CE loss, which may impact results.")
                        if kd_percent == 0:
                            self._log("CRITICAL: No batches had teacher logits despite KD being enabled!")
                
                # Log training metrics for this epoch
                avg_train_loss = epoch_loss / max(1, len(train_loader))
                train_acc = train_correct / max(1, train_total)
                self.timer.stop()
                self._log(f"Epoch {epoch}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Time: {self.timer.elapsed_time():.2f} sec")
                self.train_losses.append(avg_train_loss)
                
                if self.animator:
                    self.animator.add(epoch, avg_train_loss)
                
                # Step epoch-based schedulers
                if self.lr_schedule_type and isinstance(self.smooth_scheduler, (CosineAnnealingLR, LambdaLR)):
                    if not (self.use_warmup and epoch <= self.warmup_epochs):
                        # Use step() without epoch parameter to avoid deprecation warning
                        self.smooth_scheduler.step()
                        current_lr = self.optimizer.param_groups[0]['lr']
                        self._log(f"Scheduler stepped to LR: {current_lr:.7f}")
                
                # Run validation
                if val_loader is not None:
                    # Pass current epoch number to validation for confusion matrix
                    val_loss, val_acc = self.validate_with_metrics(val_loader, current_epoch=epoch)
                    self.val_losses.append(val_loss)
                    
                    # Only apply standard LR adjustment if:
                    # 1. We're not using a smooth scheduler
                    # 2. We're not in the warm-up phase
                    if not self.lr_schedule_type and not (self.use_warmup and epoch <= self.warmup_epochs):
                        self._adjust_learning_rate(val_acc)
                        self._log(f"Epoch {epoch}: LR = {self.optimizer.param_groups[0]['lr']:.6f} (from {lr_source})")
                    
                    # Always track the best model and check for early stopping
                    self._save_best_model(val_acc, val_loss, epoch)
                    if self._check_early_stopping():
                        break
                
                # Save checkpoints periodically
                if epoch % 5 == 0 or epoch == self.num_epochs:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"student_model_epoch_{epoch}.pth")
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.smooth_scheduler.state_dict() if self.smooth_scheduler else None,
                            'loss': avg_train_loss,
                            'accuracy': train_acc,
                            'chord_mapping': self.chord_mapping,
                            'idx_to_chord': self.idx_to_chord,
                            'mean': self.normalization['mean'] if self.normalization else None,
                            'std': self.normalization['std'] if self.normalization else None
                        }, checkpoint_path)
                        self._log(f"Saved checkpoint at epoch {epoch}")
                    except Exception as e:
                        self._log(f"Error saving checkpoint: {str(e)}")

            self._log(f"Training complete! Scheduler steps: {self._scheduler_step_count}")
            self._print_loss_history()
            self._plot_loss_history()
            
        except Exception as e:
            self._log(f"Unexpected error during training: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            # Try to save an emergency checkpoint
            try:
                emergency_path = os.path.join(self.checkpoint_dir, "emergency_checkpoint.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'error': str(e)
                }, emergency_path)
                self._log(f"Saved emergency checkpoint to {emergency_path}")
            except:
                self._log("Failed to save emergency checkpoint")

    def load_best_model(self):
        """Load the best model saved during training."""
        if os.path.exists(self.best_model_path):
            try:
                checkpoint = torch.load(self.best_model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self._log(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['accuracy']:.4f}")
                return True
            except Exception as e:
                self._log(f"Error loading best model: {str(e)}")
                return False
        else:
            self._log("No best model found to load.")
            return False

    def _log(self, message, level=None):
        """
        Log a message with optional level support.
        
        Args:
            message: The message to log
            level: Optional logging level (debug, info, warning, error)
        """
        # If level is provided but not supported by the logger, just use the message
        if self.logger:
            try:
                if level == 'debug':
                    self.logger.debug(message)
                elif level == 'warning' or level == 'warn':
                    self.logger.warning(message)
                elif level == 'error' or level == 'err':
                    self.logger.error(message)
                else:
                    # Default to info level
                    self.logger.info(message)
            except AttributeError:
                # If the logger doesn't support the level method, fall back to info
                self.logger.info(message)
            except Exception as e:
                # Last resort: print directly if logger fails
                print(f"Logger error: {e}")
                print(message)
        else:
            print(message)

    def _plot_loss_history(self):
        """Plot and save the loss history."""
        try:
            # Check if we have loss data
            if not hasattr(self, 'train_losses') or len(self.train_losses) == 0:
                self._log("No loss history to plot")
                return
                
            # Set up save path
            save_path = os.path.join(self.checkpoint_dir, "loss_history.png")
            
            # Use the visualization function to plot loss history
            val_losses = self.val_losses if hasattr(self, 'val_losses') and len(self.val_losses) > 0 else None
            fig = plot_learning_curve(
                self.train_losses, val_losses, 
                title="Training and Validation Loss", 
                save_path=save_path
            )
            
            self._log(f"Loss history plot saved to {save_path}")
            plt.close(fig)
        except Exception as e:
            self._log(f"Error plotting loss history: {e}")
