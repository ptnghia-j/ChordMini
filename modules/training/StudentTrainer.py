import os
import torch
import numpy as np
from modules.training.Trainer import BaseTrainer

class StudentTrainer(BaseTrainer):
    """
    Extension of BaseTrainer for student model training with early stopping
    and validation-based learning rate adjustment.
    """
    def __init__(self, model, optimizer, scheduler=None, device=None, num_epochs=100,
                 logger=None, use_animator=True, checkpoint_dir="checkpoints",
                 max_grad_norm=1.0, class_weights=None, idx_to_chord=None,
                 normalization=None, early_stopping_patience=5,
                 lr_decay_factor=0.95, min_lr=5e-6):
        
        # First call the parent's __init__ to set up the logger and other attributes
        super().__init__(model, optimizer, scheduler, device, num_epochs,
                         logger, use_animator, checkpoint_dir, max_grad_norm,
                         None, idx_to_chord, normalization)  # Pass None for class_weights initially
        
        # Now that logger is initialized, we can pad class weights
        if class_weights is not None and hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            # Handle class imbalance - modify weights before padding
            if idx_to_chord is not None:
                class_weights = self._adjust_weights_for_no_chord(class_weights, idx_to_chord)
            
            expected_classes = model.fc.out_features
            self._log(f"Padding class weights from {len(class_weights)} to {expected_classes}")
            padded_weights = self._pad_class_weights(class_weights, expected_classes)
            
            # Now set the loss function with padded weights
            weight_tensor = torch.tensor(padded_weights, device=self.device)
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            self.class_weights = padded_weights
        
        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.best_val_acc = 0
        self.early_stop_counter = 0
        
        # Learning rate adjustment parameters
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.before_val_acc = 0
        
        # Best model tracking
        self.best_model_path = os.path.join(self.checkpoint_dir, "student_model_best.pth")
        self.chord_mapping = None
        
        # Enable focal loss (optional)
        self.use_focal_loss = False
        if self.use_focal_loss:
            self._log("Using Focal Loss to handle class imbalance")
    
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
        """Run validation and return both loss and accuracy."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._process_batch(batch)
                
                # Apply normalization if specified
                if self.normalization:
                    inputs = (inputs - self.normalization['mean']) / self.normalization['std']
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle temporal data - reshape if needed
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Check for shape mismatch and adjust accordingly
                if logits.dim() > targets.dim():
                    # If logits have more dimensions than targets
                    if logits.dim() == 3:  # [batch, time, classes]
                        # Average over time dimension if targets don't have time dimension
                        if targets.dim() == 1:
                            logits = logits.mean(dim=1)  # [batch, classes]
                        elif targets.dim() == 2 and targets.shape[1] == 1:
                            # If targets are [batch, 1], we squeeze
                            targets = targets.squeeze(1)
                            logits = logits.mean(dim=1)
                        elif targets.dim() == 2 and targets.shape[1] > 1:
                            # Need more sophisticated handling for multi-frame targets
                            # For now, take majority vote across time
                            if targets.shape[1] == logits.shape[1]:
                                # Keep both dimensions but reshape for loss
                                batch_size, seq_len = targets.shape
                                # No need to flatten - compute metrics per frame
                                targets_flat = targets.view(-1)
                                logits_flat = logits.view(-1, logits.size(-1))
                                # Compute loss on flattened data
                                loss = self.loss_fn(logits_flat, targets_flat)
                                # Get predictions
                                preds = logits_flat.argmax(dim=1)
                                # Track metrics
                                val_correct += (preds == targets_flat).sum().item()
                                val_total += targets_flat.size(0)
                                val_loss += loss.item()
                                continue  # Skip the rest of the loop for this special case
                            else:
                                # If time dimensions don't match, take majority vote
                                targets = torch.mode(targets, dim=1)[0]
                                logits = logits.mean(dim=1)
                
                # Calculate loss
                loss = self.loss_fn(logits, targets)
                val_loss += loss.item()
                
                # Get predictions
                preds = logits.argmax(dim=1)
                
                # Track metrics
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        # Calculate average metrics
        avg_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        self._log(f"Validation Loss: {avg_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        self.model.train()
        return avg_loss, val_acc
    
    def train(self, train_loader, val_loader=None):
        """Enhanced training loop with early stopping and LR adjustment."""
        self.model.train()
        
        for epoch in range(1, self.num_epochs + 1):
            self.timer.reset(); self.timer.start()
            epoch_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training phase
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = self._process_batch(batch)
                
                # Apply normalization if specified
                if self.normalization:
                    inputs = (inputs - self.normalization['mean']) / self.normalization['std']
                
                # Forward and backward pass
                self.optimizer.zero_grad(set_to_none=True)
                
                outputs = self.model(inputs)
                
                # Handle temporal data - reshape if needed
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Check for shape mismatch and adjust accordingly
                if logits.dim() > targets.dim():
                    # If logits have more dimensions than targets
                    if logits.dim() == 3:  # [batch, time, classes]
                        # Average over time dimension if targets don't have time dimension
                        if targets.dim() == 1:
                            logits = logits.mean(dim=1)  # [batch, classes]
                        elif targets.dim() == 2 and targets.shape[1] == 1:
                            # If targets are [batch, 1], we squeeze
                            targets = targets.squeeze(1)
                            logits = logits.mean(dim=1)
                        elif targets.dim() == 2 and targets.shape[1] > 1:
                            # Need more sophisticated handling for multi-frame targets
                            # For now, take majority vote across time
                            if targets.shape[1] == logits.shape[1]:
                                # Keep both dimensions but reshape for loss
                                batch_size, seq_len = targets.shape
                                # Flatten for loss calculation
                                targets_flat = targets.reshape(-1)
                                logits_flat = logits.reshape(-1, logits.size(-1))
                                # Compute loss on flattened data
                                loss = self.loss_fn(logits_flat, targets_flat)
                                # For metrics, get predictions on flattened data
                                preds = logits_flat.argmax(dim=1)
                                train_correct += (preds == targets_flat).sum().item()
                                train_total += targets_flat.size(0)
                                epoch_loss += loss.item()
                                # Backward pass
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                                self.optimizer.step()
                                
                                # Log progress
                                if batch_idx % 20 == 0:
                                    self._log(f"Epoch {epoch}/{self.num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                                            f"Loss: {loss.item():.4f}")
                                
                                # Skip the rest of the loop for this special case
                                continue
                            else:
                                # If time dimensions don't match, take majority vote
                                targets = torch.mode(targets, dim=1)[0]
                                logits = logits.mean(dim=1)
                
                # Only use the logits for loss calculation - now properly shaped
                loss = self.loss_fn(logits, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                preds = logits.argmax(dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
                epoch_loss += loss.item()
                
                # Log progress
                if batch_idx % 20 == 0:
                    self._log(f"Epoch {epoch}/{self.num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                            f"Loss: {loss.item():.4f}")
            
            # Calculate average training metrics
            avg_train_loss = epoch_loss / len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            self.timer.stop()
            self._log(f"Epoch {epoch}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f} | "
                    f"Train Accuracy: {train_acc:.4f} | Time: {self.timer.elapsed_time():.2f} sec")
            
            # Store training loss
            self.train_losses.append(avg_train_loss)
            if self.animator:
                self.animator.add(epoch, avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.validate_with_metrics(val_loader)
                self.val_losses.append(val_loss)
                
                # Adjust learning rate based on validation accuracy
                self._adjust_learning_rate(val_acc)
                
                # Save best model
                self._save_best_model(val_acc, val_loss, epoch)
                
                # Check for early stopping
                if self._check_early_stopping():
                    break
            
            # Save periodic checkpoint
            if epoch % 5 == 0 or epoch == self.num_epochs:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"student_model_epoch_{epoch}.pth")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_train_loss,
                    'accuracy': train_acc,
                    'chord_mapping': self.chord_mapping,
                    'idx_to_chord': self.idx_to_chord,
                    'mean': self.normalization['mean'] if self.normalization else None,
                    'std': self.normalization['std'] if self.normalization else None
                }, checkpoint_path)
                
                self._log(f"Saved checkpoint at epoch {epoch}")
        
        # After training finishes
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
