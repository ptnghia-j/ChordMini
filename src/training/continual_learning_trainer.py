"""
Trainer for Phase 2: Continual Learning (CL).

Standalone implementation that merges the functionality of BaseTrainer,
StudentTrainer, and the original ContinualLearningTrainer into a single
clean class.

Supports:
  - Online KD: teacher logits generated on-the-fly from a frozen teacher
  - Focal loss as an alternative to cross-entropy
  - Early stopping with validation-based LR decay
  - Checkpoint saving/loading

For ChordNet (2E1D) only, validation/test accuracy uses the same default post-
processing configuration as ``test_labeled_audio.py``: Gaussian smoothing plus
overlap-aware frame voting when per-segment metadata is available.

Design notes on distributed training:
    This trainer is single-GPU. To extend to DDP:
    1. Wrap self.model with DistributedDataParallel.
    2. Guard checkpoint saves with rank == 0.
    3. Synchronize val metrics with dist.all_reduce before LR/early-stop decisions.
"""
import os
import time

import torch

from src.utils.logger import info
from src.utils.gradient_utils import safe_clip_grad_norm_
from src.training.utils.checkpointing import TrainerCheckpointMixin
from src.training.utils.losses import TrainerLossMixin
from src.training.utils.trainer_common import (
    accumulate_batch_accuracy,
    count_correct_predictions,
    finalize_vote_accuracy,
    normalize_features,
    teacher_logits_from_model,
    use_chordnet_eval_defaults,
)


class ContinualLearningTrainer(TrainerCheckpointMixin, TrainerLossMixin):
    """
    Continual Learning trainer with online Knowledge Distillation.

    Args:
        model: Student model (BTC or ChordNet).
        optimizer: PyTorch optimizer.
        teacher_model: Frozen teacher model for online KD (or None).
        teacher_mean / teacher_std: Normalization stats for the teacher's input.
        kd_alpha: Mixing weight  L = alpha*L_KD + (1-alpha)*L_CE.
        temperature: Softmax temperature for KD.
        device: torch.device.
        num_epochs: Max training epochs.
        checkpoint_dir: Directory for saving checkpoints.
        idx_to_chord: Optional index-to-chord mapping for logging.
        normalization: Dict {'mean': tensor, 'std': tensor} for student input.
        early_stopping_patience: Epochs w/o improvement before stopping.
        use_focal_loss / focal_gamma / focal_alpha: Focal loss settings.
        class_weights: Optional per-class weight tensor for focal loss.
        lr_decay_factor / min_lr: LR reduction on plateau.
        selective_kd / kd_confidence_threshold / kd_min_confidence_threshold:
            Asymmetric KD weighting (ignore near-random, reduce overconfident).
    """

    def __init__(
        self,
        model,
        optimizer,
        teacher_model=None,
        teacher_mean=0.0,
        teacher_std=1.0,
        kd_alpha=0.3,
        temperature=2.0,
        device=None,
        num_epochs=50,
        checkpoint_dir='./checkpoints/continual_learning',
        idx_to_chord=None,
        normalization=None,
        early_stopping_patience=10,
        use_focal_loss=False,
        focal_gamma=2.0,
        focal_alpha=None,
        class_weights=None,
        lr_decay_factor=0.9,
        min_lr=1e-6,
        max_grad_norm=1.0,
        selective_kd=True,
        kd_confidence_threshold=0.9,
        kd_min_confidence_threshold=0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device or next(model.parameters()).device
        self.model = self.model.to(self.device)
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Teacher for online KD
        self.teacher_model = teacher_model
        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        # Loss settings
        self.use_kd = teacher_model is not None
        self.kd_alpha = kd_alpha
        self.temperature = temperature
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.selective_kd = selective_kd
        self.kd_confidence_threshold = kd_confidence_threshold
        self.kd_min_confidence_threshold = kd_min_confidence_threshold

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.weight_tensor = None
        if class_weights is not None:
            self.weight_tensor = torch.tensor(class_weights, device=self.device, dtype=torch.float32)

        # Normalization, metrics
        self.normalization = normalization
        self.idx_to_chord = idx_to_chord
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr

        self.best_val_acc = float('-inf')
        self.early_stop_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.start_epoch = 1
        self.best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

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
                features = batch['spectro'].to(self.device, non_blocking=True)
                targets = batch['chord_idx'].to(self.device, non_blocking=True)
                if features.dim() == 4 and features.shape[1] == 1:
                    features = features.squeeze(1)
                if self.normalization is not None:
                    features = normalize_features(
                        features,
                        self.normalization['mean'],
                        self.normalization['std'],
                    )

                outputs = self.model(features)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                fl = logits.reshape(-1, logits.size(-1)) if logits.dim() == 3 else logits
                ft = targets.reshape(-1) if targets.dim() == 2 else targets
                total_loss += self.loss_fn(fl, ft).item()

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
            'accuracy': correct / total if total else 0.0,
            'correct': correct,
            'total': total,
        }

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _get_teacher_logits(self, raw_features):
        return teacher_logits_from_model(
            self.teacher_model,
            raw_features,
            teacher_mean=self.teacher_mean,
            teacher_std=self.teacher_std,
        )

    def compute_loss(self, logits, targets, teacher_logits=None):
        """Compute combined CE/focal + KD loss."""
        loss = super().compute_loss(logits, targets, teacher_logits=teacher_logits)

        return torch.clamp(loss, min=0.0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader=None):
        info(f"Starting CL training: {self.num_epochs} epochs, KD={'ON' if self.use_kd else 'OFF'}")

        if self.start_epoch > self.num_epochs:
            info(f"Checkpoint already reached epoch {self.start_epoch - 1}; nothing to resume.")
            return

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            elapsed = time.time() - t0
            self.train_losses.append(train_loss)
            info(f"Epoch {epoch}/{self.num_epochs} | Train Loss: {train_loss:.4f} | "
                 f"Acc: {train_acc:.4f} | Time: {elapsed:.1f}s")

            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.val_losses.append(val_loss)
                info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                improved = self._save_best(val_acc, val_loss, epoch)
                if not improved:
                    self._adjust_lr()
                if self._should_stop():
                    info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 5 == 0 or epoch == self.num_epochs:
                self._save_checkpoint(epoch, train_loss)

        info("CL training complete.")

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, batch in enumerate(loader):
            features = batch['spectro'].to(self.device, non_blocking=True)
            targets = batch['chord_idx'].to(self.device, non_blocking=True)
            if features.dim() == 4 and features.shape[1] == 1:
                features = features.squeeze(1)

            raw_features = features  # keep unnormalized for teacher

            if self.normalization is not None:
                features = normalize_features(
                    features,
                    self.normalization['mean'],
                    self.normalization['std'],
                )

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Online KD
            teacher_logits = self._get_teacher_logits(raw_features) if self.use_kd else None

            loss = self.compute_loss(logits, targets, teacher_logits)

            if torch.isfinite(loss):
                loss.backward()
                safe_clip_grad_norm_(self.model.parameters(), self.max_grad_norm, verbose=False)
                self.optimizer.step()
                total_loss += loss.item()

                with torch.no_grad():
                    batch_correct, batch_total = count_correct_predictions(logits, targets)
                    correct += batch_correct
                    total += batch_total

            if batch_idx % 10 == 0:
                info(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

        return total_loss / max(1, len(loader)), correct / max(1, total)

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

    def _adjust_lr(self):
        for pg in self.optimizer.param_groups:
            old = pg['lr']
            new = max(old * self.lr_decay_factor, self.min_lr)
            if new < old:
                pg['lr'] = new
                info(f"  LR: {old:.6f} -> {new:.6f}")
