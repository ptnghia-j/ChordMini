import sys
import os
# Add project root to sys.path so that absolute imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch.optim as optim
import matplotlib.pyplot as plt

import math
import torch
from torch.optim.lr_scheduler import (_LRScheduler, CosineAnnealingWarmRestarts,
                                     CosineAnnealingLR, LambdaLR, OneCycleLR)
from modules.utils.Animator import Animator  # now works after sys.path insertion
from modules.utils.logger import info, warning, error, debug

class ExponentialDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.gamma ** (self.last_epoch + 1)) for base_lr in self.base_lrs]

class PolynomialDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power=1.0, min_lr=0.0, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - min(self.last_epoch + 1, self.max_epochs) / self.max_epochs) ** self.power
        return [ (base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]

# CosineAnnealingWarmRestarts is provided by PyTorch, so we directly use it.
# For consistency, we wrap it in a class alias if needed.
CosineWarmRestartScheduler = CosineAnnealingWarmRestarts

class CosineScheduler(_LRScheduler):
    """
    CosineScheduler implements a cosine learning rate schedule with a single linear warmup phase.
    During warmup (warmup_steps), the learning rate increases linearly from warmup_begin_lr to base_lr.
    After warmup, the learning rate decays following a cosine schedule from base_lr to final_lr over the remaining epochs.
    The learning rate is capped at final_lr.
    """
    def __init__(self, optimizer, max_update, base_lr=0.01, final_lr=0.0, warmup_steps=0, warmup_begin_lr=0.0, last_epoch=-1):
        self.max_update = max_update
        self.base_lr_orig = base_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = max(1, self.max_update - self.warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_warmup_lr(self, epoch):
        # Linear warmup: increase from warmup_begin_lr to base_lr_orig.
        return self.warmup_begin_lr + (self.base_lr_orig - self.warmup_begin_lr) * float(epoch + 1) / float(self.warmup_steps)

    def get_lr(self):
        new_lrs = []
        for base_lr in self.base_lrs:
            # Use custom base_lr from initialization.
            effective_base_lr = self.base_lr_orig
            if self.last_epoch < self.warmup_steps:
                lr = self.get_warmup_lr(self.last_epoch)
            elif self.last_epoch < self.max_update:
                progress = (self.last_epoch - self.warmup_steps + 1) / self.max_steps
                lr = self.final_lr + (effective_base_lr - self.final_lr) * (1 + math.cos(math.pi * progress)) / 2
            else:
                lr = self.final_lr
            new_lrs.append(max(lr, self.final_lr))
        return new_lrs

class WarmupScheduler:
    """
    Warmup scheduler that linearly increases learning rate from start_lr to end_lr over warmup_epochs.
    This is not a PyTorch scheduler but a utility class for manual warmup.
    """
    def __init__(self, optimizer, warmup_epochs, start_lr, end_lr):
        """
        Initialize the warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for warmup
            start_lr: Starting learning rate
            end_lr: Target learning rate after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.end_lr = end_lr

    def get_lr(self, epoch, batch_idx=None, num_batches=None):
        """
        Calculate learning rate for the current epoch/batch.

        Args:
            epoch: Current epoch (1-indexed)
            batch_idx: Current batch index within epoch (optional)
            num_batches: Total number of batches per epoch (optional)

        Returns:
            Learning rate for the current step
        """
        # Ensure we have batch info for per-step calculation
        if batch_idx is None or num_batches is None or num_batches == 0:
            # Fallback to epoch-based if batch info is missing
            warmup_progress = (epoch - 1) / max(1, self.warmup_epochs - 1) if self.warmup_epochs > 1 else 1.0
        else:
            # Calculate total steps in warmup phase
            total_warmup_steps = self.warmup_epochs * num_batches
            # Calculate current step number (0-indexed)
            current_step = (epoch - 1) * num_batches + batch_idx
            # Calculate progress (0.0 to 1.0)
            warmup_progress = current_step / max(1, total_warmup_steps - 1)  # Avoid division by zero

        # Clamp progress
        warmup_progress = max(0.0, min(1.0, warmup_progress))

        # Linear interpolation between start_lr and end_lr
        return self.start_lr + warmup_progress * (self.end_lr - self.start_lr)

    def step(self, epoch, batch_idx=None, num_batches=None):
        """
        Update learning rate for all parameter groups.

        Args:
            epoch: Current epoch (1-indexed)
            batch_idx: Current batch index within epoch (optional)
            num_batches: Total number of batches per epoch (optional)

        Returns:
            Current learning rate
        """
        new_lr = self.get_lr(epoch, batch_idx, num_batches)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


class ValidationBasedScheduler:
    """
    Scheduler that adjusts learning rate based on validation accuracy.
    Reduces learning rate when validation accuracy doesn't improve.
    """
    def __init__(self, optimizer, factor=0.95, min_lr=5e-6, patience=1):
        """
        Initialize the validation-based scheduler.

        Args:
            optimizer: PyTorch optimizer
            factor: Factor to reduce learning rate by (default: 0.95)
            min_lr: Minimum learning rate (default: 5e-6)
            patience: Number of epochs without improvement before reducing LR (default: 1)
        """
        self.optimizer = optimizer
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.best_val_acc = 0
        self.consecutive_no_improve = 0

    def step(self, val_acc, in_warmup=False):
        """
        Adjust learning rate based on validation accuracy.

        Args:
            val_acc: Current validation accuracy
            in_warmup: Whether we're currently in warmup phase (if True, no adjustment will be made)

        Returns:
            bool: Whether learning rate was reduced
        """
        # Skip adjustment during warmup
        if in_warmup:
            info("Validation-based scheduler: Skipping LR adjustment during warmup phase")
            return False

        lr_reduced = False

        if self.best_val_acc > val_acc:
            # Increment counter for consecutive epochs without improvement
            self.consecutive_no_improve += 1

            if self.consecutive_no_improve >= self.patience:
                # Reduce learning rate after patience epochs without improvement
                old_lr = self.optimizer.param_groups[0]['lr']
                new_lr = self._reduce_lr()
                info(f"Decreasing learning rate from {old_lr:.6f} to {new_lr:.6f} after {self.consecutive_no_improve} epochs without improvement")
                self.consecutive_no_improve = 0  # Reset counter after reducing LR
                lr_reduced = True
            else:
                info(f"No improvement for {self.consecutive_no_improve} epoch(s); waiting for {self.patience - self.consecutive_no_improve} more before reducing learning rate")
        else:
            # Reset counter when accuracy improves or stays the same
            if self.consecutive_no_improve > 0:
                info(f"Validation accuracy improved, resetting consecutive non-improvement counter")
            self.consecutive_no_improve = 0

            # Update best validation accuracy if current is better
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

        return lr_reduced

    def _reduce_lr(self):
        """
        Reduce learning rate by factor but ensuring it doesn't go below min_lr.

        Returns:
            New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
        return self.optimizer.param_groups[0]['lr']


def create_scheduler(scheduler_type, optimizer, **kwargs):
    """
    Factory function to create a scheduler based on type.

    Args:
        scheduler_type: Type of scheduler ('cosine', 'cosine_warm_restarts', 'one_cycle', 'linear_decay', 'validation', None)
        optimizer: PyTorch optimizer
        **kwargs: Additional arguments for specific scheduler types

    Returns:
        Scheduler instance
    """
    if scheduler_type == 'cosine':
        # Extract parameters with defaults
        post_warmup_epochs = kwargs.get('post_warmup_epochs', kwargs.get('num_epochs', 100))
        min_lr = kwargs.get('min_lr', 5e-6)

        # Create cosine annealing scheduler
        info(f"Creating CosineAnnealingLR scheduler for {post_warmup_epochs} epochs with min_lr={min_lr}")
        return CosineAnnealingLR(
            optimizer,
            T_max=post_warmup_epochs,
            eta_min=min_lr
        )

    elif scheduler_type == 'cosine_warm_restarts':
        # Extract parameters with defaults
        t0 = kwargs.get('t0', 5)  # First restart after 5 epochs
        t_mult = kwargs.get('t_mult', 2)  # Double the period after each restart
        min_lr = kwargs.get('min_lr', 5e-6)

        # Create cosine annealing with warm restarts scheduler
        info(f"Creating CosineAnnealingWarmRestarts scheduler with T_0={t0}, T_mult={t_mult}, min_lr={min_lr}")
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0,
            T_mult=t_mult,
            eta_min=min_lr
        )

    elif scheduler_type == 'one_cycle':
        # Extract parameters with defaults
        post_warmup_epochs = kwargs.get('post_warmup_epochs', kwargs.get('num_epochs', 100))
        steps_per_epoch = kwargs.get('steps_per_epoch', 100)
        max_lr = kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10)
        div_factor = kwargs.get('div_factor', 25)
        final_div_factor = kwargs.get('final_div_factor', 10000)
        pct_start = kwargs.get('pct_start', 0.3)

        # Create one-cycle learning rate scheduler
        info(f"Creating OneCycleLR scheduler with max_lr={max_lr}, total_steps={steps_per_epoch * post_warmup_epochs}")
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=steps_per_epoch * post_warmup_epochs,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy='cos'
        )

    elif scheduler_type == 'linear_decay':
        # Extract parameters with defaults
        post_warmup_epochs = kwargs.get('post_warmup_epochs', kwargs.get('num_epochs', 100))
        min_lr = kwargs.get('min_lr', 5e-6)
        initial_lr = optimizer.param_groups[0]['lr']

        # Create linear decay scheduler using LambdaLR
        lambda_fn = lambda epoch: 1 - (1 - min_lr / initial_lr) * (epoch / post_warmup_epochs)
        info(f"Creating linear decay scheduler from {initial_lr:.6f} to {min_lr:.6f}")
        return LambdaLR(optimizer, lr_lambda=lambda_fn)

    elif scheduler_type == 'validation' or scheduler_type is None:
        # Extract parameters with defaults
        factor = kwargs.get('lr_decay_factor', 0.95)
        min_lr = kwargs.get('min_lr', 5e-6)
        patience = kwargs.get('patience', 1)

        # Create validation-based scheduler
        info(f"Creating validation-based scheduler with factor={factor}, min_lr={min_lr}, patience={patience}")
        return ValidationBasedScheduler(
            optimizer,
            factor=factor,
            min_lr=min_lr,
            patience=patience
        )

    else:
        warning(f"Unknown scheduler type: {scheduler_type}. Using validation-based adjustment")
        return create_scheduler('validation', optimizer, **kwargs)


def create_warmup_scheduler(optimizer, use_warmup, **kwargs):
    """
    Create a warmup scheduler if warmup is enabled.

    Args:
        optimizer: PyTorch optimizer
        use_warmup: Whether to use warmup
        **kwargs: Additional arguments for warmup scheduler

    Returns:
        WarmupScheduler instance or None if warmup is disabled
    """
    if not use_warmup:
        return None

    # Extract parameters with defaults
    warmup_epochs = kwargs.get('warmup_epochs', 5)
    initial_lr = optimizer.param_groups[0]['lr']
    warmup_start_lr = kwargs.get('warmup_start_lr', initial_lr / 10.0)
    warmup_end_lr = kwargs.get('warmup_end_lr', initial_lr)

    # Verify warmup values are sensible
    if warmup_start_lr >= warmup_end_lr:
        info(f"WARNING: warmup_start_lr ({warmup_start_lr}) >= warmup_end_lr ({warmup_end_lr})")
        info("This would cause LR to decrease during warmup instead of increasing. Swapping values.")
        warmup_start_lr, warmup_end_lr = warmup_end_lr, warmup_start_lr
        info(f"After swap: warmup_start_lr ({warmup_start_lr}) → warmup_end_lr ({warmup_end_lr})")

    # Create warmup scheduler
    info(f"Creating warmup scheduler for {warmup_epochs} epochs: {warmup_start_lr:.6f} → {warmup_end_lr:.6f}")
    return WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        start_lr=warmup_start_lr,
        end_lr=warmup_end_lr
    )


# Test code for each scheduler using Animator.
if __name__ == '__main__':

    # Create a dummy optimizer with one parameter.
    param = [torch.nn.Parameter(torch.tensor(1.0))]
    initial_lr = 0.1
    optimizer = optim.SGD(param, lr=initial_lr)
    num_epochs = 20
    warmup_epochs = 5
    final_lr = 0.001

    def test_scheduler(scheduler_class, scheduler_kwargs, title):
        # Reset learning rate.
        optimizer.param_groups[0]['lr'] = initial_lr
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        lr_history = []
        for epoch in range(num_epochs):
            lr_history.append(optimizer.param_groups[0]['lr'])
            # scheduler.step()
        animator = Animator(xlabel='Epoch', ylabel='Learning Rate', legend=[title],
                            xlim=(0, num_epochs), ylim=(0, initial_lr*1.1), figsize=(5, 3))
        for epoch, lr in enumerate(lr_history):
            animator.add(epoch, lr)
        plt.title(title)
        plt.ioff()
        plt.show()

    test_scheduler(ExponentialDecayScheduler, {'gamma': 0.9}, 'Exponential Decay')
    test_scheduler(PolynomialDecayScheduler, {'max_epochs': num_epochs, 'power': 2.0, 'min_lr': final_lr}, 'Polynomial Decay')
    test_scheduler(CosineWarmRestartScheduler, {'T_0': 5, 'T_mult': 1}, 'Cosine Warm Restarts')
    test_scheduler(CosineScheduler, {'max_update': num_epochs, 'base_lr': initial_lr, 'final_lr': final_lr,
                                     'warmup_steps': warmup_epochs, 'warmup_begin_lr': 0.0}, 'Cosine Scheduler with Warmup')

    # Test factory function
    print("\nTesting scheduler factory function:")
    for scheduler_type in ['cosine', 'cosine_warm_restarts', 'one_cycle', 'linear_decay', 'validation', None, 'unknown']:
        scheduler = create_scheduler(scheduler_type, optimizer, num_epochs=num_epochs, min_lr=final_lr)
        print(f"Created {type(scheduler).__name__} for type '{scheduler_type}'")

    # Test warmup scheduler
    print("\nTesting warmup scheduler:")
    warmup = create_warmup_scheduler(optimizer, True, warmup_epochs=warmup_epochs,
                                    warmup_start_lr=initial_lr/10, warmup_end_lr=initial_lr)
    print(f"Created {type(warmup).__name__}")
    for epoch in range(1, warmup_epochs+1):
        lr = warmup.step(epoch)
        print(f"Epoch {epoch}: LR = {lr:.6f}")
