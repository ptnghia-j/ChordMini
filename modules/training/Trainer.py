import os
import torch
import torch.cuda.amp  # NEW: Import AMP
from modules.utils.Timer import Timer
from modules.utils.Animator import Animator
import matplotlib.pyplot as plt  # For plotting loss history

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
        for epoch in range(1, self.num_epochs + 1):
            self._log(f"Epoch {epoch}/{self.num_epochs} - Starting training")
            self.timer.reset(); self.timer.start()
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = self._process_batch(batch)
                self.optimizer.zero_grad()
                # NEW: Use torch.amp.autocast with device_type.
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
                epoch_loss += loss.item()
                if batch_idx % 100 == 0:
                    self._log(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
            self.timer.stop()
            avg_loss = epoch_loss / len(train_loader)
            # Store the average training loss for this epoch
            self.train_losses.append(avg_loss)
            
            self._log(f"Epoch {epoch}/{self.num_epochs}: Loss = {avg_loss:.4f}, Time = {self.timer.elapsed_time():.2f} sec")
            if self.animator:
                self.animator.add(epoch, avg_loss)
            
            # Validate and store validation loss if validation set provided
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
            self.scheduler.step()
            self._save_checkpoint(epoch)
        
        # After training completes, print and plot the loss history
        self._print_loss_history()
        self._plot_loss_history()
        
        # After training, output a final static graph if available.
        if self.animator:
            if hasattr(self.animator, "plot"):
                self.animator.plot()
            elif hasattr(self.animator, "fig"):
                # If Animator has a figure attribute, use pyplot to show the stored figure.
                import matplotlib.pyplot as plt
                plt.ioff()
                self.animator.fig.show()
            else:
                self._log("Animator does not support a final plot method.")

    def compute_loss(self, outputs, targets):
        # When the model returns multiple outputs, we only need the first (logits) for loss computation.
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

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        first_batch = True  # Flag to track first batch for debugging
        
        # Collection for analysis
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._process_batch(batch)
                outputs = self.model(inputs)
                
                # Debug info for first batch
                if first_batch:
                    # Get predictions for first batch
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    if logits.ndim == 3:
                        logits = logits.mean(dim=1)
                        
                    preds = torch.argmax(logits, dim=1)
                    
                    # Convert to numpy for display
                    preds_np = preds.cpu().numpy()
                    targets_np = targets.cpu().numpy()
                    
                    # Print debug information
                    self._log(f"DEBUG: First batch - Predictions: {preds_np[:10]}")
                    self._log(f"DEBUG: First batch - Targets: {targets_np[:10]}")
                    first_batch = False
                
                # Collect predictions and targets for analysis
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                if logits.ndim == 3:
                    logits = logits.mean(dim=1)
                    
                preds = torch.argmax(logits, dim=1)
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                all_preds.extend(preds_np)
                all_targets.extend(targets_np)
                
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(val_loader)
        self._log(f"Validation Loss: {avg_loss:.4f}")
        
        # After processing all batches, analyze results
        from collections import Counter
        import numpy as np
        
        # Count distributions
        target_counter = Counter(all_targets)
        pred_counter = Counter(all_preds)
        
        # Print distribution statistics
        self._log("\nDEBUG: Target Distribution (top 10):")
        total_samples = len(all_targets)
        for idx, count in target_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") if self.idx_to_chord else str(idx)
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
        
        self.model.train()
        return avg_loss  # Return the average validation loss

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