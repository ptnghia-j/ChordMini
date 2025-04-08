import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from modules.utils import logger
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR

class ChordHMMTrainer:
    """Trainer for ChordHMM model"""
    def __init__(self, hmm_model, optimizer, device='cpu', max_epochs=100, patience=5,
                 scheduler=None, scheduler_type=None, scheduler_params=None):
        self.model = hmm_model
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.best_loss = float('inf')
        self.no_improve_epochs = 0
        
        # Add scheduler support
        self.scheduler_params = scheduler_params or {}
        self.scheduler_type = scheduler_type
        
        # Create scheduler if type specified but no scheduler provided
        if scheduler is None and scheduler_type is not None:
            self.scheduler = self.create_scheduler(scheduler_type)
            logger.info(f"Created {scheduler_type} scheduler with parameters: {self.scheduler_params}")
        else:
            self.scheduler = scheduler
    
    def create_scheduler(self, scheduler_type):
        """
        Create learning rate scheduler based on specified type.
        
        Args:
            scheduler_type: Type of scheduler to use
            
        Returns:
            Learning rate scheduler
        """
        # Get scheduler parameters or use defaults
        min_lr = self.scheduler_params.get('min_lr', 1e-6)
        gamma = self.scheduler_params.get('gamma', 0.5)
        step_size = self.scheduler_params.get('step_size', 10)
        lr_patience = self.scheduler_params.get('lr_patience', 2)
        cycle_mult = self.scheduler_params.get('cycle_mult', 10.0)
        
        # Number of batches per epoch for OneCycleLR
        steps_per_epoch = self.scheduler_params.get('steps_per_epoch', 100)
        
        # Get base learning rate from optimizer
        base_lr = self.optimizer.param_groups[0]['lr']
        
        if scheduler_type == 'cosine':
            # Cosine annealing from initial LR to min LR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=min_lr
            )
            logger.info(f"Using CosineAnnealingLR scheduler from {base_lr} to {min_lr} over {self.max_epochs} epochs")
            
        elif scheduler_type == 'step':
            # Step LR: decay by gamma every step_size epochs
            scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            logger.info(f"Using StepLR scheduler with step size {step_size} and gamma {gamma}")
            
        elif scheduler_type == 'plateau':
            # ReduceLROnPlateau: reduce LR when validation loss plateaus
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=gamma,
                patience=lr_patience,
                min_lr=min_lr,
                verbose=True
            )
            logger.info(f"Using ReduceLROnPlateau scheduler with patience {lr_patience}, " 
                       f"factor {gamma}, and min_lr {min_lr}")
            
        elif scheduler_type == 'onecycle':
            # OneCycleLR: One cycle learning rate policy
            total_steps = steps_per_epoch * self.max_epochs
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=base_lr * cycle_mult,
                total_steps=total_steps,
                pct_start=0.3,  # Spend 30% ramping up, 70% ramping down
                anneal_strategy='cos',
                div_factor=25,  # Initial LR = max_lr/25
                final_div_factor=10000,  # Final LR = max_lr/10000
            )
            logger.info(f"Using OneCycleLR scheduler with max_lr={base_lr*cycle_mult} for {total_steps} total steps")
            
        else:
            # No scheduler
            scheduler = None
            logger.info("Using constant learning rate (no scheduler)")
            
        return scheduler

    def train(self, train_loader, val_loader=None):
        """
        Train the HMM model
        
        Parameters:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
        """
        logger.info("Starting HMM training...")
        
        # Store learning rate history for analysis
        lr_history = []
        train_loss_history = []
        val_loss_history = []
        
        # Calculate steps per epoch for OneCycleLR if not provided
        if self.scheduler_type == 'onecycle' and self.scheduler_params.get('steps_per_epoch') is None:
            steps_per_epoch = len(train_loader)
            # Recreate scheduler with accurate steps_per_epoch
            self.scheduler_params['steps_per_epoch'] = steps_per_epoch
            self.scheduler = self.create_scheduler('onecycle')
            
        for epoch in range(1, self.max_epochs + 1):
            # Training loop
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            # Log current learning rate at start of epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            logger.info(f"Epoch {epoch} starting with LR: {current_lr:.6f}")
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{self.max_epochs}")):
                features = batch['spectro'].to(self.device)
                chord_indices = batch['chord_idx'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                loss = self.model.loss(features, chord_indices)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Step OneCycleLR scheduler if used (needs per-batch updates)
                if self.scheduler_type == 'onecycle' and self.scheduler is not None:
                    self.scheduler.step()
                    
                    # Log LR periodically during onecycle training
                    if batch_idx % max(1, len(train_loader) // 5) == 0:
                        batch_lr = self.optimizer.param_groups[0]['lr']
                        logger.info(f"  Batch {batch_idx}/{len(train_loader)}: LR = {batch_lr:.6f}")
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / max(1, train_batches)
            train_loss_history.append(avg_train_loss)
            # Change from showing raw negative loss to absolute value for clearer interpretation
            logger.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.6f} (raw NLL, lower is better)")
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")
            
            # Validation loop
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                val_loss_history.append(val_loss)
                logger.info(f"Epoch {epoch} - Validation Loss: {val_loss:.6f} (raw NLL, lower is better)")
                
                # Step ReduceLROnPlateau scheduler if used (needs validation metrics)
                if self.scheduler_type == 'plateau' and self.scheduler is not None:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_loss)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        logger.info(f"ReduceLROnPlateau: LR changed from {old_lr:.6f} to {new_lr:.6f}")
                
                # Step other epoch-based schedulers (cosine, step)
                if self.scheduler_type in ['cosine', 'step'] and self.scheduler is not None:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step()
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        logger.info(f"{self.scheduler_type} scheduler: LR changed from {old_lr:.6f} to {new_lr:.6f}")
                
                # Early stopping check - lower values are better (not "more negative")
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.no_improve_epochs = 0
                    logger.info(f"New best validation loss: {val_loss:.6f} (lower is better)")
                    # Save best model
                    self.save_model("best_hmm_model.pth")
                else:
                    self.no_improve_epochs += 1
                    logger.info(f"No improvement for {self.no_improve_epochs} epochs")
                    
                if self.no_improve_epochs >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Add LR history to checkpoint for analysis
        self.lr_history = lr_history
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history if val_loader else None
        
        # Try to plot training history if matplotlib is available
        try:
            self.plot_training_history(save_path="training_history.png")
        except:
            logger.info("Could not plot training history - matplotlib may not be installed")

    def validate(self, val_loader):
        """
        Validate the HMM model
        
        Parameters:
            val_loader: DataLoader for validation data
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                features = batch['spectro'].to(self.device)
                chord_indices = batch['chord_idx'].to(self.device)
                
                loss = self.model.loss(features, chord_indices)
                
                val_loss += loss.item()
                val_batches += 1
        
        return val_loss / max(1, val_batches)
        
    def decode(self, loader):
        """
        Decode data using the trained HMM model
        
        Parameters:
            loader: DataLoader for data to decode
        Returns:
            Predicted chord sequences, true chord sequences, accuracy
        """
        self.model.eval()
        all_preds = []
        all_true = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Decoding"):
                features = batch['spectro'].to(self.device)
                chord_indices = batch['chord_idx'].to(self.device)
                
                # Get predictions
                chord_preds = self.model(features)
                
                # Calculate accuracy
                correct += (chord_preds == chord_indices).sum().item()
                total += chord_indices.numel()
                
                # Collect predictions and true chords
                all_preds.append(chord_preds.cpu())
                all_true.append(chord_indices.cpu())
        
        # Concatenate all predictions and true chords
        all_preds = torch.cat(all_preds, dim=0)
        all_true = torch.cat(all_true, dim=0)
        
        accuracy = correct / max(1, total)
        return all_preds, all_true, accuracy
    
    def save_model(self, path):
        """Save model checkpoint with training history"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
        }
        
        # Save training history if available
        if hasattr(self, 'lr_history'):
            save_dict['lr_history'] = self.lr_history
        if hasattr(self, 'train_loss_history'):
            save_dict['train_loss_history'] = self.train_loss_history
        if hasattr(self, 'val_loss_history'):
            save_dict['val_loss_history'] = self.val_loss_history
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Load training history if available
        if 'lr_history' in checkpoint:
            self.lr_history = checkpoint['lr_history']
        if 'train_loss_history' in checkpoint:
            self.train_loss_history = checkpoint['train_loss_history']
        if 'val_loss_history' in checkpoint:
            self.val_loss_history = checkpoint['val_loss_history']
            
        logger.info(f"Model loaded from {path}")
        
    def plot_training_history(self, save_path=None):
        """Plot learning rate and loss history"""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Plot learning rate
            if hasattr(self, 'lr_history') and self.lr_history:
                epochs = range(1, len(self.lr_history) + 1)
                ax1.plot(epochs, self.lr_history, 'b-')
                ax1.set_ylabel('Learning Rate')
                ax1.set_title('Learning Rate Schedule')
                ax1.set_yscale('log')
                ax1.grid(True)
            
            # Plot loss history
            if hasattr(self, 'train_loss_history') and self.train_loss_history:
                epochs = range(1, len(self.train_loss_history) + 1)
                ax2.plot(epochs, self.train_loss_history, 'b-', label='Train Loss')
                
                if hasattr(self, 'val_loss_history') and self.val_loss_history:
                    ax2.plot(epochs[:len(self.val_loss_history)], self.val_loss_history, 'r-', label='Validation Loss')
                
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Loss')
                ax2.set_title('Training and Validation Loss')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            
            # Save the figure if a path is provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Training history plot saved to {save_path}")
            
            plt.close(fig)
            
        except ImportError:
            logger.info("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
