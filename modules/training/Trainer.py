import os
import torch
import torch.cuda.amp  # NEW: Import AMP
from modules.utils.Timer import Timer
from modules.utils.Animator import Animator
import matplotlib.pyplot as plt  # For plotting loss history
from tqdm import tqdm  # Import tqdm for progress tracking

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler=None, device=None, num_epochs=100, logger=None, use_animator=True, checkpoint_dir="checkpoints", max_grad_norm=1.0, ignore_index=0, class_weights=None, idx_to_chord=None, use_chord_aware_loss=False):  # Added idx_to_chord and use_chord_aware_loss
        """
        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. (Default: a dummy scheduler that does nothing)
            device (torch.device, optional): Device for computation. (Default: device of model parameters or CPU)
            num_epochs (int): Number of training epochs. (Default: 100)
            logger: Logger for logging messages. (Default: None)
            use_animator (bool): If True, instantiate an Animator to graph training progress. (Default: True)
            checkpoint_dir (str): Directory to save checkpoints. (Default: "checkpoints")
            max_grad_norm (float): Maximum norm for gradient clipping. (Default: 1.0)
            ignore_index (int): Index to ignore in the loss computation. (Default: 0)
            class_weights (list, optional): Weights for each class in the loss computation. (Default: None)
            idx_to_chord (dict, optional): Mapping from index to chord. (Default: None)
            use_chord_aware_loss (bool): If True, use chord-aware loss function. (Default: False)
        """
        self.model = model
        self.optimizer = optimizer
        if device is None:
            # Use device of first model parameter if available, else default to CPU.
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        self.device = device
        self.model = self.model.to(self.device)
        if scheduler is None:
            # Create a dummy scheduler that does nothing.
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        else:
            self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.logger = logger
        self.timer = Timer()
        self.animator = Animator(xlabel='Epoch', ylabel='Loss',
                                 legend=['Train Loss'],
                                 xlim=(0, num_epochs), ylim=(0, 10),
                                 figsize=(5, 3)) if use_animator else None
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.max_grad_norm = max_grad_norm
        self.ignore_index = ignore_index
        # NEW: Add assignment of class_weights.
        if class_weights is not None:
            self.class_weights = class_weights
            # Set weight for ignore_index to 0 to avoid predicting it
            if ignore_index < len(self.class_weights):
                self.class_weights[ignore_index] = 0.0
        else:
            self.class_weights = [1.0] * model.fc.out_features  # Default to equal weights
            if ignore_index < len(self.class_weights):
                self.class_weights[ignore_index] = 0.0
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        self.idx_to_chord = idx_to_chord
        self.use_chord_aware_loss = use_chord_aware_loss
        
        # Initialize loss function
        if self.use_chord_aware_loss and self.idx_to_chord:
            from modules.training.ChordLoss import ChordAwareLoss
            self._log("Using chord-aware loss function")
            self.loss_fn = ChordAwareLoss(
                idx_to_chord=self.idx_to_chord,
                ignore_index=self.ignore_index,
                class_weights=self.class_weights,
                device=self.device
            )
        else:
            weight_tensor = torch.tensor(self.class_weights, device=self.device) if class_weights is not None else None
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=self.ignore_index)
        
        # Add lists to track losses across epochs
        self.train_losses = []
        self.val_losses = []

    def _log(self, message):
        # Helper for logging messages.
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _process_batch(self, batch):
        # Factor out extraction of inputs and targets.
        inputs = batch['chroma'].to(self.device)
        targets = batch['chord_idx'].to(self.device)
        return inputs, targets

    def _save_checkpoint(self, epoch):
        # Helper to save checkpoint after each epoch.
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        self._log(f"Epoch {epoch} complete. Checkpoint saved to {checkpoint_path}")

    def train(self, train_loader, val_loader=None):
        self.model.train()
        
        # Disable debug logging for every batch to improve speed
        log_interval = max(1, len(train_loader) // 10)  # Log approximately 10 times per epoch
        
        for epoch in range(1, self.num_epochs + 1):
            self._log(f"Epoch {epoch}/{self.num_epochs} - Starting training")
            self.timer.reset(); self.timer.start()
            epoch_loss = 0.0
            
            # Use tqdm for progress tracking
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit="batch") as progress_bar:
                for batch_idx, batch in enumerate(train_loader):
                    # Process batch faster by moving operations to GPU in one go
                    inputs, targets = self._process_batch(batch)
                    
                    # Optimize the training step
                    loss = self._training_step(inputs, targets)
                    epoch_loss += loss
                    
                    # Update progress bar less frequently
                    if batch_idx % 10 == 0:
                        progress_bar.set_postfix({'loss': f"{loss:.4f}"})
                    progress_bar.update(1)
                    
                    # Only log at intervals to reduce overhead
                    if batch_idx % log_interval == 0:
                        self._log(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} - Loss: {loss:.4f}")
            
            self.timer.stop()
            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            
            self._log(f"Epoch {epoch}/{self.num_epochs}: Loss = {avg_loss:.4f}, Time = {self.timer.elapsed_time():.2f} sec")
            if self.animator:
                self.animator.add(epoch, avg_loss)
            
            # Optionally skip validation on some epochs to speed up training
            if val_loader is not None and (epoch % 1 == 0 or epoch == self.num_epochs):
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
            elif val_loader is not None:
                # Just append the previous value to maintain list length
                self.val_losses.append(self.val_losses[-1] if self.val_losses else 0.0)
                
            self.scheduler.step()
            
            # Only save checkpoint at intervals or final epoch
            if epoch % 5 == 0 or epoch == self.num_epochs:
                self._save_checkpoint(epoch)
        
        # After training completes
        self._print_loss_history()
        self._plot_loss_history()
    
    def _training_step(self, inputs, targets):
        """Optimized single training step."""
        self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Use AMP for mixed precision training
        with torch.amp.autocast(device_type='cuda', enabled=(self.device.type=='cuda')):
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, targets)
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader):
        """Optimized validation function."""
        self.model.eval()
        total_loss = 0.0
        
        # Pre-allocate GPU memory for predictions when possible
        if hasattr(val_loader, 'dataset') and hasattr(val_loader.dataset, '__len__'):
            expected_len = len(val_loader.dataset)
            # Only pre-allocate if dataset size is reasonable to avoid OOM errors
            if expected_len < 100000:  # Reasonable size threshold
                try:
                    # Try to pre-allocate, but fall back if we can't determine shape
                    dummy_batch = next(iter(val_loader))
                    if isinstance(dummy_batch, dict) and 'chord_idx' in dummy_batch:
                        n_classes = self.model.fc.out_features if hasattr(self.model, 'fc') else 0
                        if n_classes > 0:
                            # Create pre-allocated tensors on the correct device
                            all_preds = torch.zeros(expected_len, dtype=torch.long, device=self.device)
                            all_targets = torch.zeros(expected_len, dtype=torch.long, device=self.device)
                            pre_allocated = True
                        else:
                            pre_allocated = False
                    else:
                        pre_allocated = False
                except (StopIteration, RuntimeError):
                    pre_allocated = False
            else:
                pre_allocated = False
        else:
            pre_allocated = False
        
        if not pre_allocated:
            # Fall back to list collection
            all_preds_list = []
            all_targets_list = []
        
        # Use tqdm for progress tracking
        batch_start_idx = 0
        first_batch = True
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                inputs, targets = self._process_batch(batch)
                
                # Fast forward pass with device-appropriate mixed precision
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                with torch.amp.autocast(device_type=device_type, enabled=(self.device.type in ['cuda', 'mps'])):
                    outputs = self.model(inputs)
                    loss = self.compute_loss(outputs, targets)
                
                # Process predictions in a vectorized way
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                if logits.ndim == 3:
                    logits = logits.mean(dim=1)
                
                # Get predictions efficiently
                preds = torch.argmax(logits, dim=1)
                
                # Track batch loss
                total_loss += loss.item()
                
                # Debug info only for first batch
                if first_batch:
                    # Only move a small portion to CPU for debug printing
                    debug_preds = preds[:10].cpu().numpy()
                    debug_targets = targets[:10].cpu().numpy()
                    self._log(f"DEBUG: First batch - Predictions: {debug_preds}")
                    self._log(f"DEBUG: First batch - Targets: {debug_targets}")
                    first_batch = False
                
                # Store predictions and targets efficiently
                if pre_allocated:
                    batch_size = preds.size(0)
                    end_idx = min(batch_start_idx + batch_size, expected_len)
                    # Copy this batch to our pre-allocated tensors
                    all_preds[batch_start_idx:end_idx] = preds
                    all_targets[batch_start_idx:end_idx] = targets
                    batch_start_idx += batch_size
                else:
                    # Keep tensors on GPU until the end
                    all_preds_list.append(preds)
                    all_targets_list.append(targets)
        
        # Process final results
        if pre_allocated:
            # We already have everything in a single tensor on the right device
            # Only move to CPU for analysis at the end
            all_preds_np = all_preds[:batch_start_idx].cpu().numpy()
            all_targets_np = all_targets[:batch_start_idx].cpu().numpy()
        else:
            # Concatenate our collected tensors
            if all_preds_list:
                all_preds_tensor = torch.cat(all_preds_list)
                all_targets_tensor = torch.cat(all_targets_list)
                all_preds_np = all_preds_tensor.cpu().numpy()
                all_targets_np = all_targets_tensor.cpu().numpy()
            else:
                # Handle empty case
                all_preds_np = all_targets_np = np.array([])
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        self._log(f"Validation Loss: {avg_loss:.4f}")
        
        # Analyze results conditionally
        if self.idx_to_chord and len(all_targets_np) > 0:
            self._analyze_validation_results(all_targets_np, all_preds_np)
        
        self.model.train()
        return avg_loss
    
    def _analyze_validation_results(self, all_targets, all_preds):
        """Separate method for analysis to keep validation loop clean."""
        from collections import Counter
        
        # Count distributions
        target_counter = Counter(all_targets)
        pred_counter = Counter(all_preds)
        
        # Print distribution statistics
        self._log("\nDEBUG: Target Distribution (top 10):")
        total_samples = len(all_targets)
        for idx, count in target_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") 
            self._log(f"Target {idx} ({chord_name}): {count} occurrences ({count/total_samples*100:.2f}%)")
            
        self._log("\nDEBUG: Prediction Distribution (top 10):")
        for idx, count in pred_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") if self.idx_to_chord else str(idx)
            self._log(f"Prediction {idx} ({chord_name}): {count} occurrences ({count/total_samples*100:.2f}%)")
        
        # Calculate WCSR if we have the mapping
        if self.idx_to_chord:
            from modules.utils.chord_metrics import weighted_chord_symbol_recall
            wcsr = weighted_chord_symbol_recall(all_targets, all_preds, self.idx_to_chord)
            self._log(f"\nWeighted Chord Symbol Recall (WCSR): {wcsr:.4f}")
            
        # Print confusion matrix for most common chords
        if len(target_counter) > 1:
            self._log("\nAnalyzing most common predictions vs targets:")
            top_chords = [idx for idx, _ in target_counter.most_common(10)]
            for true_idx in top_chords:
                true_chord = self.idx_to_chord.get(true_idx, str(true_idx)) if self.idx_to_chord else str(true_idx)
                pred_indices = [p for t, p in zip(all_targets, all_preds) if t == true_idx]
                if pred_indices:
                    pred_counts = Counter(pred_indices)
                    most_common_pred = pred_counts.most_common(1)[0][0]
                    most_common_pred_chord = self.idx_to_chord.get(most_common_pred, str(most_common_pred)) if self.idx_to_chord else str(most_common_pred)
                    accuracy_for_chord = pred_counts.get(true_idx, 0) / len(pred_indices)
                    self._log(f"True: {true_chord} -> Most common prediction: {most_common_pred_chord} (Accuracy: {accuracy_for_chord:.2f})")
        
    def compute_loss(self, outputs, targets):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if outputs.ndim == 3:
            outputs = outputs.mean(dim=1)
        
        # Diagnostic: Check for NaNs in outputs and targets
        if torch.isnan(outputs).any():
            self._log("NaN detected in model outputs")
        if torch.isnan(targets).any():
            self._log("NaN detected in targets")
        
        # NEW: Add masking to avoid predicting ignore_index
        # Apply a penalty for predicting ignore_index class
        if self.ignore_index < outputs.shape[1]:
            # Create a mask for the ignore_index logits
            ignore_mask = torch.zeros_like(outputs)
            ignore_mask[:, self.ignore_index] = -10.0  # Apply a penalty
            outputs = outputs + ignore_mask
        
        # Use the initialized loss function
        loss = self.loss_fn(outputs, targets)
        
        if torch.isnan(loss):
            self._log(f"NaN loss computed. Outputs: {outputs} | Targets: {targets}")
        
        return loss

    def _print_loss_history(self):
        """Print the training and validation loss history."""
        self._log("\n=== Training Loss History ===")
        for epoch, loss in enumerate(self.train_losses, 1):
            self._log(f"Epoch {epoch}: Training Loss = {loss:.6f}")
            
        if self.val_losses:
            self._log("\n=== Validation Loss History ===")
            for epoch, loss in enumerate(self.val_losses, 1):
                self._log(f"Epoch {epoch}: Validation Loss = {loss:.6f}")
    
    def _plot_loss_history(self):
        """Plot the training and validation loss history."""
        try:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self.train_losses) + 1)
            plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
            
            if self.val_losses:
                plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
                
            plt.title('Loss History')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            loss_plot_path = os.path.join(self.checkpoint_dir, "loss_history.png")
            plt.savefig(loss_plot_path)
            self._log(f"Loss history plot saved to {loss_plot_path}")
            
            # Close the plot to free memory
            plt.close()
        except Exception as e:
            self._log(f"Error plotting loss history: {str(e)}")

    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint, path)
        if self.logger:
            self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.logger:
            self.logger.info(f"Checkpoint loaded from {path}")

    def _display_final_plots(self):
        """Display final training plots after training completes."""
        try:
            if self.animator:
                # If Animator supports a direct plot method
                if hasattr(self.animator, "plot"):
                    self.animator.plot()
                # If Animator has a figure attribute
                elif hasattr(self.animator, "fig"):
                    plt.figure(self.animator.fig.number)
                    plt.show()
                    plt.close()
                else:
                    self._log("Animator does not support a final plot method.")
        except Exception as e:
            self._log(f"Error displaying final plots: {str(e)}")

    def save_model(self, path=None):
        """
        Save the trained model to disk.
        
        Args:
            path (str, optional): Specific path to save the model. If None, 
                                 uses default checkpoint directory.
        """
        if path is None:
            path = os.path.join(self.checkpoint_dir, "final_model.pth")
        
        # Save model in a format suitable for later deployment
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'class_weights': self.class_weights,
            'ignore_index': self.ignore_index
        }, path)
        
        self._log(f"Model saved to {path}")
        
        return path

    def load_model(self, path):
        """
        Load model weights from a saved checkpoint.
        
        Args:
            path (str): Path to the saved model checkpoint
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            # Load optimizer and scheduler states if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self._log(f"Model loaded from {path}")
            return True
        except Exception as e:
            self._log(f"Error loading model: {str(e)}")
            return False