"""
Trainer for Phase 1: Pseudo-Labeling (PL).

Supports both Phase 1 input modes:
  - pre-extracted spectrograms/labels (+ optional offline teacher logits)
  - online unlabeled audio, where pseudo-labels and optional KD logits are
    generated at runtime from a frozen teacher

For ChordNet (2E1D) only, validation/test accuracy mirrors the non-factored inference
path by using Gaussian smoothing and overlap-aware frame voting when the batch
contains segment metadata.

Design notes on distributed training:
    This trainer runs on a single GPU.  To extend to multi-GPU:
    1. Wrap self.model with torch.nn.parallel.DistributedDataParallel.
    2. Use a DistributedSampler on the DataLoader (not here).
    3. Only save checkpoints on rank 0 (guard _save_checkpoint calls).
    4. Synchronize metrics across ranks before logging.
"""
import os
import time

import torch

from src.utils.logger import info, warning
from src.utils.gradient_utils import safe_clip_grad_norm_
from src.training.utils.checkpointing import TrainerCheckpointMixin
from src.training.utils.losses import TrainerLossMixin
from src.training.utils.lr_schedulers import (
    CosineWarmupScheduler,
    LinearWarmupScheduler,
    ValidationBasedScheduler,
)
from src.training.utils.trainer_common import (
    accumulate_batch_accuracy,
    count_correct_predictions,
    finalize_vote_accuracy,
    normalize_features,
    teacher_logits_from_model,
    to_stat_tensor,
    use_chordnet_eval_defaults,
)


class PseudoLabelingTrainer(TrainerCheckpointMixin, TrainerLossMixin):
    """
    OOP trainer for the pseudo-labeling phase.

    Args:
        model: Student model (BTC or ChordNet).
        optimizer: PyTorch optimizer.
        device: torch.device to train on.
        num_epochs: Maximum training epochs.
        checkpoint_dir: Directory to save checkpoints.
        normalization: Dict with 'mean' and 'std' tensors for input normalization.
        idx_to_chord: Optional dict mapping index -> chord name (for logging).
        use_kd_loss: Whether to use knowledge distillation loss.
        kd_alpha: Mixing coefficient  L = alpha*L_KD + (1-alpha)*L_CE.
        temperature: Softmax temperature for KD.
        max_grad_norm: Maximum gradient norm for clipping.
        early_stopping_patience: Epochs without improvement before stopping.
        use_focal_loss: Use focal loss instead of cross-entropy.
        focal_gamma: Gamma parameter for focal loss.
        lr_schedule_type: One of ``validation``, ``cosine``, or ``none``.
        use_warmup: Enable a single warmup phase before the main schedule.
        warmup_epochs: Number of warmup epochs.
        warmup_start_lr: Warmup starting LR (defaults to base_lr / 10).
        warmup_end_lr: Warmup end LR for standalone warmup.
        lr_decay_factor: Multiplicative decay for validation-based scheduling.
        min_lr: Minimum LR floor used by schedulers.
        lr_decay_factor: Factor to decay LR on validation plateau.
        min_lr: Minimum learning rate.
    """

    def __init__(
        self,
        model,
        optimizer,
        teacher_model=None,
        teacher_mean=None,
        teacher_std=None,
        device=None,
        num_epochs=100,
        checkpoint_dir='./checkpoints/pseudo_labeling',
        normalization=None,
        idx_to_chord=None,
        use_kd_loss=False,
        kd_alpha=0.5,
        temperature=3.0,
        max_grad_norm=1.0,
        early_stopping_patience=10,
        use_focal_loss=False,
        focal_gamma=2.0,
        lr_schedule_type='validation',
        use_warmup=False,
        warmup_epochs=None,
        warmup_start_lr=None,
        warmup_end_lr=None,
        lr_decay_factor=0.95,
        min_lr=1e-6,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device or next(model.parameters()).device
        self.model = self.model.to(self.device)
        self.teacher_model = teacher_model.to(self.device) if teacher_model is not None else None
        if self.teacher_model is not None:
            self.teacher_model.eval()
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.normalization = normalization
        self.idx_to_chord = idx_to_chord
        self.teacher_mean = to_stat_tensor(teacher_mean, self.device)
        self.teacher_std = to_stat_tensor(teacher_std, self.device)

        # Loss configuration
        self.use_kd_loss = use_kd_loss
        self.kd_alpha = kd_alpha
        self.temperature = temperature
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.max_grad_norm = max_grad_norm

        # Early stopping / LR scheduling
        self.early_stopping_patience = early_stopping_patience
        self.lr_schedule_type = (lr_schedule_type or 'validation').lower()
        if self.lr_schedule_type not in {'cosine', 'validation', 'none'}:
            warning(f"Unknown lr_schedule_type={lr_schedule_type}; falling back to validation")
            self.lr_schedule_type = 'validation'
        self.use_warmup = bool(use_warmup)
        self.warmup_epochs = int(warmup_epochs) if warmup_epochs is not None else (5 if self.use_warmup else 0)
        self.warmup_epochs = max(0, self.warmup_epochs)
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.best_val_acc = float('-inf')
        self.early_stop_counter = 0
        self.base_lrs = [float(group['lr']) for group in self.optimizer.param_groups]
        self.batch_scheduler = None
        self.validation_scheduler = None
        self._configure_lr_schedulers()

        # History
        self.train_losses = []
        self.val_losses = []
        self.start_epoch = 1

        self.best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    def _configure_lr_schedulers(self):
        end_lrs = self.base_lrs if self.warmup_end_lr is None else [float(self.warmup_end_lr)] * len(self.base_lrs)

        if self.lr_schedule_type == 'cosine':
            self.batch_scheduler = CosineWarmupScheduler(
                self.optimizer,
                num_epochs=self.num_epochs,
                min_lr=self.min_lr,
                warmup_epochs=self.warmup_epochs if self.use_warmup else 0,
                warmup_start_lr=self.warmup_start_lr,
                warmup_end_lr=self.warmup_end_lr,
            )
        elif self.use_warmup:
            start_lr = self.warmup_start_lr
            if start_lr is None:
                start_lr = self.base_lrs[0] / 10.0
            self.batch_scheduler = LinearWarmupScheduler(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                start_lr=start_lr,
                end_lrs=end_lrs,
            )

        if self.lr_schedule_type == 'validation':
            self.validation_scheduler = ValidationBasedScheduler(
                self.optimizer,
                factor=self.lr_decay_factor,
                min_lr=self.min_lr,
                patience=1,
            )

    def _current_lr(self):
        return float(self.optimizer.param_groups[0]['lr'])

    def _is_in_warmup(self, epoch):
        return self.use_warmup and self.warmup_epochs > 0 and epoch <= self.warmup_epochs

    def _step_batch_scheduler(self, epoch, batch_idx, num_batches):
        if self.batch_scheduler is None:
            return
        if isinstance(self.batch_scheduler, LinearWarmupScheduler) and not self.batch_scheduler.is_active(epoch):
            return
        self.batch_scheduler.step(epoch, batch_idx, num_batches)

    def _step_validation_scheduler(self, val_acc, epoch):
        if self.validation_scheduler is None:
            return False
        old_lr = self._current_lr()
        reduced = self.validation_scheduler.step(val_acc, in_warmup=self._is_in_warmup(epoch))
        new_lr = self._current_lr()
        if reduced and new_lr < old_lr:
            info(f"  LR reduced: {old_lr:.6f} -> {new_lr:.6f}")
        return reduced

    def _use_chordnet_eval_defaults(self, batch):
        return use_chordnet_eval_defaults(self.model, batch)

    def _evaluate_loader(self, loader):
        """Evaluate a loader and apply ChordNet overlap-aware accuracy when possible."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        vote_accumulator = {}
        target_accumulator = {}

        with torch.no_grad():
            for batch in loader:
                raw_features = batch['spectro'].to(self.device, non_blocking=True)
                targets, teacher_logits = self._resolve_targets(batch, raw_features)
                features = raw_features

                if self.normalization is not None:
                    features = normalize_features(
                        features,
                        self.normalization['mean'],
                        self.normalization['std'],
                    )

                outputs = self.model(features)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.compute_loss(logits, targets, teacher_logits)
                total_loss += loss.item()

                if self._use_chordnet_eval_defaults(batch):
                    _, _, uses_votes = accumulate_batch_accuracy(
                        self.model,
                        batch,
                        logits,
                        targets,
                        vote_accumulator,
                        target_accumulator,
                    )
                    if uses_votes:
                        continue

                batch_correct, batch_total, _ = accumulate_batch_accuracy(self.model, batch, logits, targets)
                correct += batch_correct
                total += batch_total

        if vote_accumulator:
            correct, total = finalize_vote_accuracy(vote_accumulator, target_accumulator)

        return {
            'loss': total_loss / max(1, len(loader)),
            'accuracy': correct / total if total > 0 else 0.0,
            'correct': correct,
            'total': total,
        }

    def _get_teacher_logits(self, raw_features):
        return teacher_logits_from_model(
            self.teacher_model,
            raw_features,
            teacher_mean=self.teacher_mean,
            teacher_std=self.teacher_std,
            normalization=self.normalization,
        )

    def _resolve_targets(self, batch, raw_features):
        targets = None
        if 'chord_idx' in batch:
            targets = batch['chord_idx'].to(self.device, non_blocking=True)

        need_teacher_for_targets = targets is None
        need_teacher_for_kd = bool(self.use_kd_loss)
        need_teacher_logits = need_teacher_for_targets or need_teacher_for_kd

        teacher_logits = None
        if need_teacher_logits and 'teacher_logits' in batch:
            teacher_logits = batch['teacher_logits'].to(self.device, non_blocking=True)
        elif need_teacher_logits and self.teacher_model is not None:
            teacher_logits = self._get_teacher_logits(raw_features)

        if targets is None:
            if teacher_logits is None:
                raise ValueError("Pseudo-labeling requires chord_idx targets or a teacher model.")
            targets = teacher_logits.argmax(dim=-1)

        if not need_teacher_for_kd:
            teacher_logits = None

        return targets, teacher_logits

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader=None):
        """Run the full training loop."""
        info(f"Starting PL training: {self.num_epochs} epochs, KD={'ON' if self.use_kd_loss else 'OFF'}")
        info(
            f"  LR schedule: {self.lr_schedule_type} | "
            f"Warmup: {'ON' if self.use_warmup else 'OFF'} | "
            f"Initial LR: {self._current_lr():.6f}"
        )
        if self.lr_schedule_type == 'validation' and val_loader is None:
            warning("Validation-based LR scheduling requested without a validation loader; LR will stay fixed after warmup.")

        if self.start_epoch > self.num_epochs:
            info(f"Checkpoint already reached epoch {self.start_epoch - 1}; nothing to resume.")
            return

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            elapsed = time.time() - t0

            self.train_losses.append(train_loss)
            info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"LR: {self._current_lr():.6f} | Time: {elapsed:.1f}s"
            )

            save_periodic_checkpoint = (epoch % 10 == 0 or epoch == self.num_epochs)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.val_losses.append(val_loss)
                info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                improved = self._save_best(val_acc, val_loss, epoch)
                if self.lr_schedule_type == 'validation':
                    self._step_validation_scheduler(val_acc, epoch)
                if save_periodic_checkpoint:
                    self._save_checkpoint(epoch, train_loss)
                if self._should_stop():
                    info(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpoint when there is no validation loop
            if val_loader is None and save_periodic_checkpoint:
                self._save_checkpoint(epoch, train_loss)

        info("PL training complete.")

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = max(1, len(loader))

        for batch_idx, batch in enumerate(loader):
            raw_features = batch['spectro'].to(self.device, non_blocking=True)
            targets, teacher_logits = self._resolve_targets(batch, raw_features)
            features = raw_features

            # Normalize
            if self.normalization is not None:
                features = normalize_features(
                    features,
                    self.normalization['mean'],
                    self.normalization['std'],
                )

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = self.compute_loss(logits, targets, teacher_logits)

            if torch.isfinite(loss):
                loss.backward()
                safe_clip_grad_norm_(self.model.parameters(), self.max_grad_norm, verbose=False)
                self.optimizer.step()
                self._step_batch_scheduler(epoch, batch_idx, num_batches)

                total_loss += loss.item()

                with torch.no_grad():
                    batch_correct, batch_total = count_correct_predictions(logits, targets)
                    correct += batch_correct
                    total += batch_total

            if batch_idx % 50 == 0:
                info(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, len(loader))
        acc = correct / max(1, total)
        return avg_loss, acc

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, loader):
        metrics = self._evaluate_loader(loader)
        return metrics['loss'], metrics['accuracy']

    def evaluate_loader(self, loader):
        """Public loader-level evaluation helper used by test loops."""
        return self._evaluate_loader(loader)

    # ------------------------------------------------------------------
    # Checkpointing & early stopping
    # ------------------------------------------------------------------

    def _should_stop(self):
        return self.early_stop_counter >= self.early_stopping_patience

    def _get_additional_checkpoint_state(self):
        state = {}
        if self.batch_scheduler is not None and hasattr(self.batch_scheduler, 'state_dict'):
            state['batch_scheduler_state'] = self.batch_scheduler.state_dict()
        if self.validation_scheduler is not None and hasattr(self.validation_scheduler, 'state_dict'):
            state['validation_scheduler_state'] = self.validation_scheduler.state_dict()
        return state

    def _restore_additional_checkpoint_state(self, checkpoint):
        batch_state = checkpoint.get('batch_scheduler_state')
        if self.batch_scheduler is not None and batch_state and hasattr(self.batch_scheduler, 'load_state_dict'):
            self.batch_scheduler.load_state_dict(batch_state)

        validation_state = checkpoint.get('validation_scheduler_state')
        if self.validation_scheduler is not None and validation_state and hasattr(self.validation_scheduler, 'load_state_dict'):
            self.validation_scheduler.load_state_dict(validation_state)

    def _adjust_lr(self):
        for pg in self.optimizer.param_groups:
            old_lr = pg['lr']
            new_lr = max(old_lr * self.lr_decay_factor, self.min_lr)
            if new_lr < old_lr:
                pg['lr'] = new_lr
                info(f"  LR reduced: {old_lr:.6f} -> {new_lr:.6f}")
