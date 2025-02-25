import os
import torch
import torch.cuda.amp  # NEW: Import AMP
from modules.utils.Timer import Timer
from modules.utils.Animator import Animator

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler=None, device=None, num_epochs=100, logger=None, use_animator=True, checkpoint_dir="checkpoints", max_grad_norm=1.0, ignore_index=0):
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
        # NEW: Initialize GradScaler if using CUDA.
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

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
        # Remove interactive plotting; simply record loss and output final graph.
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            self._log(f"Epoch {epoch}/{self.num_epochs} - Starting training")
            self.timer.reset()
            self.timer.start()
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = self._process_batch(batch)
                self.optimizer.zero_grad()
                # NEW: Use AMP autocast when possible.
                with torch.cuda.amp.autocast(enabled=(self.device.type=='cuda')):
                    outputs = self.model(inputs)
                    loss = self.compute_loss(outputs, targets)
                # NEW: Scale loss and backpropagation.
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
                
                if batch_idx % 10 == 0:
                    self._log(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            self.timer.stop()
            avg_loss = epoch_loss / len(train_loader)
            self._log(f"Epoch {epoch}/{self.num_epochs}: Loss = {avg_loss:.4f}, Time = {self.timer.elapsed_time():.2f} sec")
            
            if self.animator:
                # Record the average loss for later static plotting.
                self.animator.add(epoch, avg_loss)
            
            if val_loader is not None:
                self.validate(val_loader)
            self.scheduler.step()
            
            # Save checkpoint at end of each epoch.
            self._save_checkpoint(epoch)
        
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
        
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        loss = loss_fn(outputs, targets)
        
        if torch.isnan(loss):
            self._log(f"NaN loss computed. Outputs: {outputs} | Targets: {targets}")
        
        return loss

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        first_batch = True  # debug flag for printing validation targets and predictions
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._process_batch(batch)
                # NEW: Debug print shapes and values from the first validation batch.
                if first_batch:
                    self._log(f"[DEBUG] Validation batch - inputs shape: {inputs.shape}, targets shape: {targets.shape}")
                    self._log(f"[DEBUG] Validation batch targets: {targets}")
                    first_batch = False
                outputs = self.model(inputs)
                # NEW: Check if outputs is a tuple
                if isinstance(outputs, tuple):
                    out_shape = outputs[0].shape
                else:
                    out_shape = outputs.shape
                self._log(f"[DEBUG] Model outputs shape: {out_shape}")
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        self._log(f"Validation Loss: {avg_loss:.4f}")
        self.model.train()

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