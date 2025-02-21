import os
import torch
from modules.utils.Timer import Timer
from modules.utils.Animator import Animator

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler=None, device=None, num_epochs=100, logger=None, use_animator=True, checkpoint_dir="checkpoints"):
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
        """
        self.model = model
        self.optimizer = optimizer
        if device is None:
            # Use device of first model parameter if available, else default to CPU.
            device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        self.device = device
        print(f"Using device: {self.device}")  # <-- Added line to verify GPU usage
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

    def train(self, train_loader, val_loader=None):
        # Remove interactive plotting; simply record loss and output final graph.
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs} - Starting training")
            self.timer.reset()
            self.timer.start()
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            self.timer.stop()
            avg_loss = epoch_loss / len(train_loader)
            message = f"Epoch {epoch}/{self.num_epochs}: Loss = {avg_loss:.4f}, Time = {self.timer.elapsed_time():.2f} sec"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)
            
            if self.animator:
                # Record the average loss for later static plotting.
                self.animator.add(epoch, avg_loss)
            
            if val_loader is not None:
                self.validate(val_loader)
            self.scheduler.step()
            
            # Save checkpoint at end of each epoch.
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Epoch {epoch} complete. Checkpoint saved to {checkpoint_path}")
        
        # After training, output a final static graph.
        if self.animator:
            self.animator.plot()

    def compute_loss(self, outputs, targets):
        # If outputs is a tuple, use the first element.
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        # If outputs has shape [B, T, C], permute to [B, C, T].
        if outputs.ndim == 3:
            outputs = outputs.permute(0, 2, 1)
        # Always use CrossEntropyLoss for raw logits with integer targets.
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(outputs, targets)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch  # customize extraction based on data loader output
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        if self.logger:
            self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        else:
            print(f"Validation Loss: {avg_loss:.4f}")
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

    def train_multi_gpu(self, train_loader, val_loader=None):
        if torch.cuda.device_count() < 2:
            if self.logger:
                self.logger.info("Multiple GPUs not detected. Falling back to single GPU training.")
            else:
                print("Multiple GPUs not detected. Falling back to single GPU training.")
            self.train(train_loader, val_loader)
        else:
            self.model = torch.nn.DataParallel(self.model)
            if self.logger:
                self.logger.info(f"Training on {torch.cuda.device_count()} GPUs.")
            else:
                print(f"Training on {torch.cuda.device_count()} GPUs.")
            self.train(train_loader, val_loader)
            self.model = self.model.module
