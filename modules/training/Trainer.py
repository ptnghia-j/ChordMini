import torch
from modules.utils.Timer import Timer
from modules.utils.Animator import Animator

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler=None, device=None, num_epochs=100, logger=None, use_animator=True):
        """
        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. (Default: a dummy scheduler that does nothing)
            device (torch.device, optional): Device for computation. (Default: device of model parameters or CPU)
            num_epochs (int): Number of training epochs. (Default: 100)
            logger: Logger for logging messages. (Default: None)
            use_animator (bool): If True, instantiate an Animator to graph training progress. (Default: True)
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

    def train(self, train_loader, val_loader=None):
        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            self.timer.reset()
            self.timer.start()
            epoch_loss = 0.0
            for batch in train_loader:
                inputs, targets = batch  # customize extraction based on data loader output
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.timer.stop()
            avg_loss = epoch_loss / len(train_loader)
            message = f"Epoch {epoch}/{self.num_epochs}: Loss = {avg_loss:.4f}, Time = {self.timer.elapsed_time():.2f} sec"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)
            
            if self.animator:
                self.animator.add(epoch, avg_loss)
            if val_loader is not None:
                self.validate(val_loader)
            self.scheduler.step()

    def compute_loss(self, outputs, targets):
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
