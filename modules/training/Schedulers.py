import sys
import os
# Add project root to sys.path so that absolute imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts
from modules.utils.Animator import Animator  # now works after sys.path insertion

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
        self.max_steps = self.max_update - self.warmup_steps
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

# Test code for each scheduler using Animator.
if __name__ == '__main__':
    import torch.optim as optim
    import matplotlib.pyplot as plt

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
            scheduler.step()
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

