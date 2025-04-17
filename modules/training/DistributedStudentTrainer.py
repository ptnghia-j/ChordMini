import torch
import torch.distributed as dist
import os
import numpy as np
from modules.training.StudentTrainer import StudentTrainer
from modules.utils.logger import info, warning, error, debug

class DistributedStudentTrainer(StudentTrainer):
    """
    Extension of StudentTrainer that supports distributed training with PyTorch DDP.
    Handles metric aggregation across ranks and coordinates checkpoint saving.
    """
    
    def __init__(self, model, optimizer, device, num_epochs=100, logger=None, 
                 checkpoint_dir='checkpoints', class_weights=None, idx_to_chord=None,
                 normalization=None, early_stopping_patience=5, lr_decay_factor=0.95,
                 min_lr=5e-6, use_warmup=False, warmup_epochs=None, warmup_start_lr=None,
                 warmup_end_lr=None, lr_schedule_type=None, use_focal_loss=False,
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
        
        if self.is_main_process:
            info(f"Initialized DistributedStudentTrainer with {world_size} processes")
            info(f"This is the main process (rank {rank})")
        
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
        
        # Set epoch for distributed samplers
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Process batch
            batch_result = self.train_batch(batch)
            
            # Update metrics
            total_loss += batch_result['loss']
            
            # Calculate accuracy from batch result
            batch_size = batch['spectro'].size(0)
            correct += batch_result['accuracy'] * batch_size
            total += batch_size
            
            # Log progress
            if batch_idx % 10 == 0 and self.is_main_process:
                info(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                     f'Loss: {batch_result["loss"]:.4f} | Acc: {batch_result["accuracy"]*100:.2f}%')
        
        # Aggregate metrics across all processes
        if self.world_size > 1:
            # Convert to tensors for reduction
            loss_tensor = torch.tensor(total_loss, device=self.device)
            correct_tensor = torch.tensor(correct, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)
            
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
        accuracy = correct / total if total > 0 else 0
        
        # Log training metrics
        if self.is_main_process:
            info(f'Epoch: {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy*100:.2f}%')
        
        # Evaluate on validation set
        val_loss, val_acc = self.evaluate(val_loader, epoch)
        
        # Update learning rate based on validation performance
        if not self.lr_schedule_type:
            # Only adjust learning rate if not using a scheduler
            self._adjust_learning_rate(val_acc)
        else:
            # Update learning rate using scheduler
            self._update_learning_rate(epoch)
        
        # Check for early stopping
        early_stop = False
        if self._save_best_model(val_acc, val_loss, epoch):
            if self.is_main_process:
                info(f'New best model with validation accuracy: {val_acc:.4f}')
        
        if self._check_early_stopping():
            if self.is_main_process:
                info('Early stopping triggered')
            early_stop = True
        
        return early_stop
    
    def evaluate(self, val_loader, epoch=None):
        """
        Evaluate the model on the validation set with distributed support.
        Aggregates metrics across all ranks.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Set epoch for distributed samplers
        if hasattr(val_loader.sampler, 'set_epoch') and epoch is not None:
            val_loader.sampler.set_epoch(epoch)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Get input and target
                spectro = batch['spectro'].to(self.device)
                targets = batch['chord_idx'].to(self.device)
                
                # Normalize input
                if self.normalization:
                    spectro = (spectro - self.normalization['mean']) / self.normalization['std']
                
                # Get teacher logits if using knowledge distillation
                teacher_logits = None
                if self.use_kd_loss and 'teacher_logits' in batch:
                    teacher_logits = batch['teacher_logits'].to(self.device)
                
                # Forward pass
                outputs = self.model(spectro)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Calculate loss
                if self.use_focal_loss:
                    loss = self.focal_loss(logits, targets)
                else:
                    loss = self.loss_fn(logits, targets)
                
                # Add KD loss if enabled and teacher logits are available
                if self.use_kd_loss and teacher_logits is not None:
                    kd_loss = self.knowledge_distillation_loss(logits, teacher_logits, self.temperature)
                    loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * loss
                
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
            loss_tensor = torch.tensor(total_loss, device=self.device)
            correct_tensor = torch.tensor(correct, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)
            
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
        accuracy = correct / total if total > 0 else 0
        
        # Log validation metrics
        if self.is_main_process:
            info(f'Validation Loss: {avg_loss:.4f} | Validation Acc: {accuracy*100:.2f}%')
        
        return avg_loss, accuracy
    
    def _save_best_model(self, val_acc, val_loss, epoch):
        """
        Save the model when validation accuracy improves.
        Only the main process (rank 0) saves checkpoints.
        """
        if not self.is_main_process:
            return False
            
        return super()._save_best_model(val_acc, val_loss, epoch)
    
    def train(self, train_loader, val_loader, start_epoch=1):
        """
        Train the model for multiple epochs with distributed support.
        Only the main process logs progress.
        """
        # Ensure all processes are synchronized before starting training
        if self.world_size > 1:
            dist.barrier()
            
        # Only the main process logs the start of training
        if self.is_main_process:
            info(f"Starting distributed training with {self.world_size} processes")
            info(f"Training for {self.num_epochs} epochs starting from epoch {start_epoch}")
        
        # Train for the specified number of epochs
        for epoch in range(start_epoch, self.num_epochs + 1):
            # Train for one epoch
            early_stop = self.train_epoch(train_loader, val_loader, epoch)
            
            # Check for early stopping
            if early_stop:
                if self.is_main_process:
                    info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Ensure all processes are synchronized at the end of each epoch
            if self.world_size > 1:
                dist.barrier()
        
        # Only the main process logs the end of training
        if self.is_main_process:
            info("Training completed!")
