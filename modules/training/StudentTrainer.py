import os
import torch
import numpy as np
from modules.training.Trainer import BaseTrainer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR, CosineAnnealingWarmRestarts

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
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else optimizer.param_groups[0]['lr'] / 10.0
        # If warmup_end_lr not provided, use the initial learning rate
        self.warmup_end_lr = warmup_end_lr if warmup_end_lr is not None else optimizer.param_groups[0]['lr']
        # Track original learning rate for later use
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        # LR schedule type for smooth scheduling
        self.lr_schedule_type = lr_schedule_type
        self.smooth_scheduler = None
        
        # Create smooth scheduler if requested
        if self.lr_schedule_type:
            self._create_smooth_scheduler()
            self._log(f"Using smooth '{self.lr_schedule_type}' learning rate schedule")
        elif self.use_warmup:
            self._log(f"Using warm-up LR schedule for first {self.warmup_epochs} epochs")
            self._log(f"Warm-up LR: {self.warmup_start_lr:.6f} → {self.warmup_end_lr:.6f}")
            # Set initial learning rate to warm-up start LR
            self._set_lr(self.warmup_start_lr)
        else:
            self._log("Using validation-based learning rate adjustment")
            
        # Best model tracking
        self.best_model_path = os.path.join(self.checkpoint_dir, "student_model_best.pth")
        self.chord_mapping = None
    
    def _create_smooth_scheduler(self):
        """Create a smooth learning rate scheduler."""
        if self.lr_schedule_type == 'cosine':
            # Cosine annealing from initial LR to min_lr over num_epochs
            self.smooth_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.num_epochs,
                eta_min=self.min_lr
            )
            self._log(f"Cosine annealing from {self.initial_lr:.6f} to {self.min_lr:.6f}")
            
        elif self.lr_schedule_type == 'cosine_warm_restarts':
            # Cosine annealing with warm restarts
            # First restart after 5 epochs, then double the period
            self.smooth_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=5,  # First restart after 5 epochs
                T_mult=2,  # Double the period after each restart
                eta_min=self.min_lr
            )
            self._log(f"Cosine annealing with warm restarts: min_lr={self.min_lr:.6f}")
            
        elif self.lr_schedule_type == 'one_cycle':
            # One-cycle learning rate schedule
            steps_per_epoch = 100  # Estimate, will be updated in train()
            self.smooth_scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.initial_lr * 10,  # Peak LR
                total_steps=steps_per_epoch * self.num_epochs,
                pct_start=0.3,  # Spend 30% ramping up, 70% ramping down
                div_factor=25,  # Initial LR = max_lr/25
                final_div_factor=10000,  # Final LR = max_lr/10000
                anneal_strategy='cos'
            )
            self._log(f"One-cycle LR: {self.initial_lr:.6f} → {self.initial_lr*10:.6f} → {self.initial_lr*10/10000:.8f}")
            
        elif self.lr_schedule_type == 'linear_decay':
            # Linear decay from initial LR to min_lr
            lambda_fn = lambda epoch: 1 - (1 - self.min_lr / self.initial_lr) * (epoch / self.num_epochs)
            self.smooth_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_fn)
            self._log(f"Linear decay from {self.initial_lr:.6f} to {self.min_lr:.6f}")
            
        else:
            self._log(f"Unknown scheduler type: {self.lr_schedule_type}. Using validation-based adjustment")
            self.lr_schedule_type = None
    
    def _update_smooth_scheduler(self, epoch, batch_idx, num_batches):
        """Update learning rate scheduler with fractional epochs."""
        if self.smooth_scheduler is None:
            return
            
        if isinstance(self.smooth_scheduler, (CosineAnnealingLR, LambdaLR)):
            # These schedulers work with epoch granularity
            # We'll update them in train() after each epoch
            pass
            
        elif isinstance(self.smooth_scheduler, (OneCycleLR, CosineAnnealingWarmRestarts)):
            # These can be stepped more frequently for smoother changes
            if batch_idx > 0 and batch_idx % max(1, num_batches // 10) == 0:
                self.smooth_scheduler.step(epoch - 1 + batch_idx / num_batches)
    
    def _adjust_weights_for_no_chord(self, weights, idx_to_chord):
        """Adjust weights to handle 'N' (no chord) class imbalance"""
        n_chord_idx = None
        
        # Find the index that corresponds to "N" chord
        for idx, chord in idx_to_chord.items():
            if chord == "N":
                n_chord_idx = idx
                break
        
        if n_chord_idx is not None and n_chord_idx < len(weights):
            self._log(f"Adjusting weights for 'N' (no chord) at index {n_chord_idx}")
            
            # Reduce weight for "N" class to prevent model collapse to majority class
            weights = np.array(weights, dtype=np.float32)
            
            # Option 1: Reduce "N" class weight by a factor (e.g., 0.5)
            n_weight_factor = 0.5
            weights[n_chord_idx] *= n_weight_factor
            
            # Option 2: Boost all other classes to compensate for "N" dominance
            boost_factor = 1.5
            non_n_mask = np.ones_like(weights, dtype=bool)
            non_n_mask[n_chord_idx] = False
            weights[non_n_mask] *= boost_factor
            
            self._log(f"Modified 'N' class weight: {weights[n_chord_idx]:.4f} (reduced by factor {n_weight_factor})")
            self._log(f"Boosted other classes by factor {boost_factor}")
            
            # Normalize weights to maintain overall scale
            if weights.sum() > 0:
                orig_sum = len(weights)  # Original sum (equal weights would be 1.0 each)
                weights = weights * (orig_sum / weights.sum())
                
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
        
        # Use standard cross entropy but weighted by focal weight
        ce_loss = torch.nn.functional.cross_entropy(
            logits, targets, weight=alpha, reduction='none')
        
        # Apply the focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

    def distillation_loss(self, student_logits, teacher_logits, targets, temperature=2.0, alpha=0.7):
        """
        Combine a KL divergence loss between softened teacher and student predictions with
        a label-smoothed cross entropy loss on the pseudo-labels.
        
        Args:
            student_logits (Tensor): Student raw outputs.
            teacher_logits (Tensor): Teacher soft targets.
            targets (Tensor): Pseudo-labels.
            temperature (float): Temperature for softening.
            alpha (float): Weighting factor between distillation and cross entropy.
        Returns:
            Combined loss.
        """
        import torch.nn.functional as F
        
        # Soft targets: apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        
        # Label-smoothed cross entropy
        smoothing = 0.1
        confidence = 1.0 - smoothing
        log_probs = F.log_softmax(student_logits, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        ce_loss = confidence * nll_loss + smoothing * smooth_loss
        
        return alpha * kl_loss + (1 - alpha) * ce_loss

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
        import torch.nn.functional as F
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            # Construct soft targets
            true_dist = torch.full_like(log_probs, smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        return loss

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
        if epoch >= self.warmup_epochs:
            # Warm-up complete, set to end LR
            return self._set_lr(self.warmup_end_lr)
        
        # Linear interpolation between start_lr and end_lr
        progress = epoch / self.warmup_epochs
        new_lr = self.warmup_start_lr + progress * (self.warmup_end_lr - self.warmup_start_lr)
        self._log(f"Warm-up epoch {epoch}/{self.warmup_epochs}: LR = {new_lr:.6f}")
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
        else:
            self.early_stop_counter += 1
            return False
    
    def _check_early_stopping(self):
        """Check if early stopping criteria is met."""
        if self.early_stop_counter >= self.early_stopping_patience:
            self._log(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement")
            return True
        return False
    
    def validate_with_metrics(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._process_batch(batch)
                if self.normalization:
                    inputs = (inputs - self.normalization['mean']) / self.normalization['std']
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
                    continue

                # Standard case
                loss = self.compute_loss(logits, targets)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        avg_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        self._log(f"Epoch Validation Loss: {avg_loss:.4f}, Accuracy: {val_acc:.4f}")
        self.model.train()
        return avg_loss, val_acc

    def compute_loss(self, logits, targets, teacher_logits=None):
        """
        Compute the loss.
        If KD loss is enabled and teacher_logits are provided, combine standard loss and KD loss.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.size(-1))
            if targets.ndim == 2:
                targets = targets.reshape(-1)
        
        # Use focal loss if enabled
        if self.use_focal_loss:
            # Note: focal_loss implementation uses weight=None directly
            # as the weight parameter is controlled by focal_alpha
            loss = self.focal_loss(logits, targets, 
                                  gamma=self.focal_gamma, 
                                  alpha=self.focal_alpha)
        else:
            # Use the standard loss function with class weights
            loss = self.loss_fn(logits, targets)
        
        # If KD loss is enabled and teacher_logits are given, compute KD loss and combine.
        if self.use_kd_loss:
            if teacher_logits is not None:
                temperature = self.temperature
                # Compute softened predictions for student and teacher
                student_log_probs = torch.nn.functional.log_softmax(logits / temperature, dim=1)
                teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
                kd_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
                loss = (1 - self.kd_alpha) * loss + self.kd_alpha * kd_loss
            else:
                # Log warning only once per epoch to avoid spam
                if not hasattr(self, '_kd_warning_logged'):
                    self._log("WARNING: KD loss enabled but teacher logits not provided in batch")
                    self._kd_warning_logged = True
        
        # Ensure loss is non-negative (critical fix)
        loss = torch.clamp(loss, min=0.0)
        
        if torch.isnan(loss):
            self._log(f"NaN loss computed. Outputs: {logits[:5, :5]} | Targets: {targets[:5]}")
        
        return loss

    def train(self, train_loader, val_loader=None, start_epoch=1):
        """Train the model with support for resuming from checkpoints."""
        self.model.train()
        
        # Get actual steps per epoch for scheduler
        num_batches = len(train_loader)
        
        # Update OneCycleLR with actual steps if needed
        if isinstance(self.smooth_scheduler, OneCycleLR) and start_epoch > 1:
            # For resuming training, we need to adjust the total_steps based on starting epoch
            remaining_epochs = self.num_epochs - (start_epoch - 1)
            total_steps = num_batches * remaining_epochs
            # Recreate scheduler with correct total_steps
            self._log(f"Initializing OneCycleLR with {total_steps} total steps (resuming from epoch {start_epoch})")
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
            
        for epoch in range(start_epoch, self.num_epochs + 1):
            # Keep track of current LR source for logging
            if self.lr_schedule_type:
                lr_source = f"'{self.lr_schedule_type}' scheduler"
            elif self.use_warmup and epoch <= self.warmup_epochs:
                lr_source = "warm-up schedule"
            else:
                lr_source = "validation-based adjustment"
            
            # Apply warm-up LR schedule if enabled and in warm-up phase
            # (only if we're not using a smooth scheduler)
            if self.use_warmup and self.lr_schedule_type is None and epoch <= self.warmup_epochs:
                self._warmup_learning_rate(epoch)
                self._log(f"Epoch {epoch}: LR = {self.optimizer.param_groups[0]['lr']:.6f} (from {lr_source})")
            
            self.timer.reset(); self.timer.start()
            epoch_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Get number of batches for fractional epoch calculation
            num_batches = len(train_loader)
            
            for batch_idx, batch in enumerate(train_loader):
                # Update smooth scheduler if applicable
                if self.lr_schedule_type:
                    self._update_smooth_scheduler(epoch, batch_idx, num_batches)
                
                # Regular training step
                inputs, targets = self._process_batch(batch)
                # Extract teacher logits from batch if provided
                teacher_logits = None
                if self.use_kd_loss and 'teacher_logits' in batch:
                    teacher_logits = batch['teacher_logits'].to(self.device)
                
                if self.normalization:
                    inputs = (inputs - self.normalization['mean']) / self.normalization['std']
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # For per-frame supervision, flatten logits and targets when needed
                if logits.ndim == 3 and targets.ndim == 2:
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = targets.reshape(-1)
                    # If we have teacher logits, reshape them too
                    if teacher_logits is not None and teacher_logits.ndim == 3:
                        teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))

                # Use our custom compute_loss method with teacher_logits if available
                loss = self.compute_loss(logits, targets, teacher_logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                preds = logits.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
                epoch_loss += loss.item()
                if batch_idx % 20 == 0:
                    # Log current LR
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self._log(f"Epoch {epoch}/{self.num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.7f}")
            
            # Log training metrics for this epoch
            avg_train_loss = epoch_loss / len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            self.timer.stop()
            self._log(f"Epoch {epoch}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Time: {self.timer.elapsed_time():.2f} sec")
            self.train_losses.append(avg_train_loss)
            if self.animator:
                self.animator.add(epoch, avg_train_loss)
            
            # Step epoch-based schedulers
            if self.lr_schedule_type and isinstance(self.smooth_scheduler, (CosineAnnealingLR, LambdaLR)):
                self.smooth_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self._log(f"Scheduler stepped to LR: {current_lr:.7f}")
            
            # Run validation
            if val_loader is not None:
                val_loss, val_acc = self.validate_with_metrics(val_loader)
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

        self._print_loss_history()
        self._plot_loss_history()

    def load_best_model(self):
        """Load the best model saved during training."""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self._log(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['accuracy']:.4f}")
            return True
        else:
            self._log("No best model found to load.")
            return False
