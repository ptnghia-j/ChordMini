# Learning Rate Scheduler Refactoring

## Overview

This refactoring extracts learning rate scheduling logic from the `StudentTrainer` and `DistributedStudentTrainer` classes into a dedicated `Schedulers.py` module. The goal is to improve code organization, reduce duplication, and make the scheduling logic more maintainable.

## Changes Made

### 1. Enhanced `Schedulers.py` Module

- Added new scheduler classes:
  - `WarmupScheduler`: Handles linear warmup from start_lr to end_lr over warmup_epochs
  - `ValidationBasedScheduler`: Adjusts learning rate based on validation accuracy

- Added factory functions:
  - `create_scheduler(scheduler_type, optimizer, **kwargs)`: Creates appropriate scheduler based on type
  - `create_warmup_scheduler(optimizer, use_warmup, **kwargs)`: Creates warmup scheduler if enabled

- Supported scheduler types:
  - 'cosine': CosineAnnealingLR
  - 'cosine_warm_restarts': CosineAnnealingWarmRestarts
  - 'one_cycle': OneCycleLR
  - 'linear_decay': LambdaLR
  - 'validation': ValidationBasedScheduler
  - None: ValidationBasedScheduler (default)

### 2. Updated `StudentTrainer.py`

- Removed direct scheduler creation in `_create_smooth_scheduler`
  - Now uses `create_scheduler` factory function

- Refactored `_warmup_learning_rate` method
  - Now uses `create_warmup_scheduler` and `WarmupScheduler.step()`

- Refactored `_adjust_learning_rate` method
  - Now uses `create_scheduler` with 'validation' type and `ValidationBasedScheduler.step()`

- Added local imports in methods to avoid circular imports
  - `from torch.optim.lr_scheduler import ...` in methods that need scheduler types

### 3. No Changes to `DistributedStudentTrainer.py`

- Since `DistributedStudentTrainer` inherits from `StudentTrainer` and doesn't override the learning rate scheduling methods, no changes were needed
- The refactored methods in `StudentTrainer` are automatically used by `DistributedStudentTrainer`

## Benefits

1. **Improved Modularity**: Learning rate scheduling logic is now in a dedicated module
2. **Reduced Duplication**: Common scheduling logic is centralized
3. **Better Maintainability**: Changes to scheduling logic only need to be made in one place
4. **Cleaner Code**: Trainer classes are now focused on training logic, not scheduling details
5. **Easier Extension**: New scheduler types can be added to the factory function without modifying trainer classes

## Testing Considerations

The refactoring should maintain full compatibility with the existing pipeline. When testing:

1. Verify that all scheduler types work as before
2. Check that warmup behaves correctly
3. Ensure validation-based learning rate adjustment works properly
4. Confirm that distributed training still functions correctly
5. Test resuming training from checkpoints with different scheduler types
