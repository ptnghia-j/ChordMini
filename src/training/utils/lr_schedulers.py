"""
Learning-rate schedulers used by ChordMini Phase 1 training.

This module intentionally implements only the Phase 1 behaviors needed in the
ChordMini pipeline:

- cosine annealing with a single linear warmup phase
- validation-based LR decay with optional standalone warmup

The API is deliberately small so the training loop stays readable.
"""

from __future__ import annotations

import math


def _expand_group_values(value, reference_values):
    """Expand a scalar or iterable value to match optimizer param groups."""
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) != len(reference_values):
            raise ValueError("Scheduler values must match optimizer param groups")
        return [float(v) for v in values]
    return [float(value) for _ in reference_values]


def _set_optimizer_lrs(optimizer, lrs):
    """Assign a learning rate to each optimizer parameter group."""
    for param_group, lr in zip(optimizer.param_groups, lrs):
        param_group['lr'] = float(lr)


class LinearWarmupScheduler:
    """Linearly increase LR from ``start_lr`` to ``end_lr`` over warmup epochs."""

    def __init__(self, optimizer, warmup_epochs, start_lr, end_lrs):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.end_lrs = [float(lr) for lr in end_lrs]
        self.start_lrs = _expand_group_values(start_lr, self.end_lrs)

        if self.warmup_epochs > 0:
            _set_optimizer_lrs(self.optimizer, self.start_lrs)

    def is_active(self, epoch):
        return self.warmup_epochs > 0 and epoch <= self.warmup_epochs

    def step(self, epoch, batch_idx, num_batches):
        if not self.is_active(epoch):
            return self.optimizer.param_groups[0]['lr']

        num_batches = max(1, int(num_batches))
        total_steps = max(1, self.warmup_epochs * num_batches)
        current_step = (epoch - 1) * num_batches + batch_idx + 1
        progress = min(1.0, current_step / total_steps)
        lrs = [
            start_lr + progress * (end_lr - start_lr)
            for start_lr, end_lr in zip(self.start_lrs, self.end_lrs)
        ]
        _set_optimizer_lrs(self.optimizer, lrs)
        return lrs[0]

    def state_dict(self):
        return {
            'warmup_epochs': self.warmup_epochs,
            'end_lrs': list(self.end_lrs),
            'start_lrs': list(self.start_lrs),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        self.warmup_epochs = max(0, int(state.get('warmup_epochs', self.warmup_epochs)))
        self.end_lrs = [float(lr) for lr in state.get('end_lrs', self.end_lrs)]
        self.start_lrs = [float(lr) for lr in state.get('start_lrs', self.start_lrs)]


class CosineWarmupScheduler:
    """Per-batch cosine decay with an optional single warmup phase."""

    def __init__(
        self,
        optimizer,
        num_epochs,
        min_lr,
        warmup_epochs=0,
        warmup_start_lr=None,
        warmup_end_lr=None,
    ):
        self.optimizer = optimizer
        self.num_epochs = max(1, int(num_epochs))
        initial_lrs = [float(group['lr']) for group in optimizer.param_groups]
        self.base_lrs = _expand_group_values(warmup_end_lr, initial_lrs) if warmup_end_lr is not None else initial_lrs
        self.min_lrs = _expand_group_values(min_lr, self.base_lrs)
        self.warmup_epochs = max(0, int(warmup_epochs))
        default_start = [base_lr / 10.0 for base_lr in self.base_lrs]
        if warmup_start_lr is None:
            self.warmup_start_lrs = default_start
        else:
            self.warmup_start_lrs = _expand_group_values(warmup_start_lr, self.base_lrs)

        if self.warmup_epochs > 0:
            _set_optimizer_lrs(self.optimizer, self.warmup_start_lrs)

    def step(self, epoch, batch_idx, num_batches):
        num_batches = max(1, int(num_batches))
        total_steps = max(1, self.num_epochs * num_batches)
        warmup_steps = self.warmup_epochs * num_batches
        current_step = min(total_steps, (epoch - 1) * num_batches + batch_idx + 1)

        if warmup_steps > 0 and current_step <= warmup_steps:
            progress = current_step / warmup_steps
            lrs = [
                start_lr + progress * (base_lr - start_lr)
                for start_lr, base_lr in zip(self.warmup_start_lrs, self.base_lrs)
            ]
        else:
            decay_steps = max(1, total_steps - warmup_steps)
            decay_progress = min(1.0, max(0.0, (current_step - warmup_steps) / decay_steps))
            lrs = [
                min_lr + (base_lr - min_lr) * (1.0 + math.cos(math.pi * decay_progress)) / 2.0
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
            ]

        _set_optimizer_lrs(self.optimizer, lrs)
        return lrs[0]

    def state_dict(self):
        return {
            'num_epochs': self.num_epochs,
            'base_lrs': list(self.base_lrs),
            'min_lrs': list(self.min_lrs),
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lrs': list(self.warmup_start_lrs),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        self.num_epochs = max(1, int(state.get('num_epochs', self.num_epochs)))
        self.base_lrs = [float(lr) for lr in state.get('base_lrs', self.base_lrs)]
        self.min_lrs = [float(lr) for lr in state.get('min_lrs', self.min_lrs)]
        self.warmup_epochs = max(0, int(state.get('warmup_epochs', self.warmup_epochs)))
        self.warmup_start_lrs = [float(lr) for lr in state.get('warmup_start_lrs', self.warmup_start_lrs)]


class ValidationBasedScheduler:
    """Reduce LR when validation accuracy stops improving."""

    def __init__(self, optimizer, factor=0.95, min_lr=5e-6, patience=1):
        self.optimizer = optimizer
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.patience = max(1, int(patience))
        self.best_val_acc = float('-inf')
        self.consecutive_no_improve = 0

    def step(self, val_acc, in_warmup=False):
        if in_warmup:
            return False

        if val_acc < self.best_val_acc:
            self.consecutive_no_improve += 1
            if self.consecutive_no_improve >= self.patience:
                self._reduce_lr()
                self.consecutive_no_improve = 0
                return True
            return False

        self.consecutive_no_improve = 0
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        return False

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(float(param_group['lr']) * self.factor, self.min_lr)
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'factor': self.factor,
            'min_lr': self.min_lr,
            'patience': self.patience,
            'best_val_acc': self.best_val_acc,
            'consecutive_no_improve': self.consecutive_no_improve,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        self.factor = float(state.get('factor', self.factor))
        self.min_lr = float(state.get('min_lr', self.min_lr))
        self.patience = max(1, int(state.get('patience', self.patience)))
        self.best_val_acc = float(state.get('best_val_acc', self.best_val_acc))
        self.consecutive_no_improve = max(
            0,
            int(state.get('consecutive_no_improve', self.consecutive_no_improve)),
        )
