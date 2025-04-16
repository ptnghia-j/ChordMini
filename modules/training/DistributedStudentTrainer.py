import torch
import torch.distributed as dist
import numpy as np
import os
import logging
from modules.training.StudentTrainer import StudentTrainer

class DistributedStudentTrainer(StudentTrainer):
    """
    Extension of StudentTrainer that supports distributed training.
    Handles metric aggregation across ranks and coordinates checkpoint saving.
    """
    
    def __init__(self, model, optimizer, device, num_epochs=100, logger=None, 
                 checkpoint_dir='checkpoints', class_weights=None, idx_to_chord=None,
                 normalization=None, early_stopping_patience=5, lr_decay_factor=0.95,
                 min_lr=5e-6, use_warmup=False, warmup_epochs=None, warmup_start_lr=None,
                 warmup_end_lr=None, lr_schedule_type='cosine', use_focal_loss=False,
                 focal_gamma=2.0, focal_alpha=None, use_kd_loss=False, kd_alpha=0.5,
                 temperature=1.0, rank=0, world_size=1):
        
        super().__init__(model, optimizer, device, num_epochs, logger, 
                        checkpoint_dir, class_weights, idx_to_chord,
                        normalization, early_stopping_patience, lr_decay_factor,
                        min_lr, use_warmup, warmup_epochs, warmup_start_lr,
                        warmup_end_lr, lr_schedule_type, use_focal_loss,
                        focal_gamma, focal_alpha, use_kd_loss, kd_alpha,
                        temperature)
        
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
    def reduce_tensor(self, tensor):
        """Reduce tensor across all ranks."""
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt
    
    def train_epoch(self, train_loader, val_loader, epoch):
        """
        Train for one epoch with distributed support.
        Aggregates metrics across all ranks.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch['spectro'], batch['chord_idx']
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply normalization if provided
            if self.normalization:
                inputs = (inputs - self.normalization['mean']) / self.normalization['std']
            
            # Get teacher logits if using knowledge distillation
            teacher_logits = None
            if self.use_kd_loss and 'teacher_logits' in batch:
                teacher_logits = batch['teacher_logits'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate loss
            loss = self.calculate_loss(logits, targets, teacher_logits)
            
            # Backward pass and optimization
            loss.backward()
            
            # Apply gradient clipping
            if hasattr(self, 'safe_clip_grad_norm_'):
                self.safe_clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update learning rate if using warmup
            if self.use_warmup and self.warmup_scheduler and epoch <= self.warmup_epochs:
                self.warmup_scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            if logits.ndim == 3 and targets.ndim <= 2:
                # Average over time dimension for sequence data
                logits = logits.mean(dim=1)
            
            _, predicted = logits.max(1)
            batch_correct = (predicted == targets).sum().item()
            batch_total = targets.size(0)
            
            correct += batch_correct
            total += batch_total
            
            # Log progress
            if batch_idx % 10 == 0 and self.is_main_process:
                self.logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                               f'Loss: {loss.item():.4f} | Acc: {100.*batch_correct/batch_total:.2f}%')
        
        # Aggregate metrics across all processes
        if self.world_size > 1:
            # Convert to tensors for reduction
            loss_tensor = torch.tensor(total_loss).to(self.device)
            correct_tensor = torch.tensor(correct).to(self.device)
            total_tensor = torch.tensor(total).to(self.device)
            
            # Reduce tensors
            loss_tensor = self.reduce_tensor(loss_tensor)
            correct_tensor = self.reduce_tensor(correct_tensor)
            total_tensor = self.reduce_tensor(total_tensor)
            
            # Convert back to Python values
            total_loss = loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        # Log training metrics
        if self.is_main_process:
            self.logger.info(f'Epoch: {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%')
        
        # Evaluate on validation set
        val_loss, val_acc = self.evaluate(val_loader, epoch)
        
        # Update learning rate based on validation performance
        if self.smooth_scheduler and epoch > (self.warmup_epochs or 0):
            self.smooth_scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.is_main_process:
                self.logger.info(f'Learning rate adjusted to: {current_lr:.6f}')
        
        # Check for early stopping
        if self.early_stopping_counter is not None:
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stopping_counter = 0
                
                # Save best model (only on main process)
                if self.is_main_process:
                    self.save_checkpoint(epoch, val_acc)
            else:
                self.early_stopping_counter += 1
                if self.is_main_process:
                    self.logger.info(f'EarlyStopping counter: {self.early_stopping_counter} out of {self.early_stopping_patience}')
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    if self.is_main_process:
                        self.logger.info('Early stopping triggered')
                    return True  # Signal to stop training
        
        return False  # Continue training
    
    def evaluate(self, val_loader, epoch=None):
        """
        Evaluate the model on the validation set with distributed support.
        Aggregates metrics across all ranks.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs, targets = batch['spectro'], batch['chord_idx']
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Apply normalization if provided
                if self.normalization:
                    inputs = (inputs - self.normalization['mean']) / self.normalization['std']
                
                # Get teacher logits if using knowledge distillation
                teacher_logits = None
                if self.use_kd_loss and 'teacher_logits' in batch:
                    teacher_logits = batch['teacher_logits'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Calculate loss
                loss = self.calculate_loss(logits, targets, teacher_logits)
                total_loss += loss.item()
                
                # Calculate accuracy
                if logits.ndim == 3 and targets.ndim <= 2:
                    # Average over time dimension for sequence data
                    logits = logits.mean(dim=1)
                
                _, predicted = logits.max(1)
                batch_correct = (predicted == targets).sum().item()
                batch_total = targets.size(0)
                
                correct += batch_correct
                total += batch_total
        
        # Aggregate metrics across all processes
        if self.world_size > 1:
            # Convert to tensors for reduction
            loss_tensor = torch.tensor(total_loss).to(self.device)
            correct_tensor = torch.tensor(correct).to(self.device)
            total_tensor = torch.tensor(total).to(self.device)
            
            # Reduce tensors
            loss_tensor = self.reduce_tensor(loss_tensor)
            correct_tensor = self.reduce_tensor(correct_tensor)
            total_tensor = self.reduce_tensor(total_tensor)
            
            # Convert back to Python values
            total_loss = loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        # Log validation metrics
        if self.is_main_process:
            self.logger.info(f'Validation Loss: {avg_loss:.4f} | Validation Acc: {accuracy:.2f}%')
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, accuracy):
        """
        Save a checkpoint of the model.
        Only the main process (rank 0) saves checkpoints.
        """
        if not self.is_main_process:
            return
            
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Determine if this is the best model so far
        is_best = accuracy >= self.best_val_acc
        
        # Get the model state dict (handle DDP wrapper)
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'scheduler_state_dict': self.smooth_scheduler.state_dict() if self.smooth_scheduler else None
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Checkpoint saved to {checkpoint_path}')
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved to {best_path}')
    
    def load_best_model(self):
        """
        Load the best model from the checkpoint directory.
        Only needed on the main process for evaluation.
        """
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        
        if not os.path.exists(best_path):
            if self.is_main_process:
                self.logger.warning(f'Best model checkpoint not found at {best_path}')
            return False
        
        try:
            checkpoint = torch.load(best_path, map_location=self.device)
            
            # Load model weights
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            if self.is_main_process:
                self.logger.info(f'Loaded best model from {best_path} (Accuracy: {checkpoint["accuracy"]:.2f}%)')
            return True
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f'Error loading best model: {e}')
            return False
