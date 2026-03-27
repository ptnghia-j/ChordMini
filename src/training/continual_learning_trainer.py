"""
Trainer for Phase 2: Continual Learning (CL).

Standalone implementation that merges the functionality of BaseTrainer,
StudentTrainer, and the original ContinualLearningTrainer into a single
clean class.

Supports:
  - Online KD: teacher logits generated on-the-fly from a frozen teacher
  - Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
  - Focal loss as an alternative to cross-entropy
  - Early stopping with validation-based LR decay
  - Checkpoint saving/loading

For ChordNet only, validation/test accuracy uses the same default post-
processing configuration as ``test_labeled_audio.py``: Gaussian smoothing plus
overlap-aware frame voting when per-segment metadata is available.

Design notes on distributed training:
    This trainer is single-GPU. To extend to DDP:
    1. Wrap self.model with DistributedDataParallel.
    2. Guard checkpoint saves with rank == 0.
    3. Synchronize val metrics with dist.all_reduce before LR/early-stop decisions.
"""
import os
import copy
import time

import torch
import torch.nn.functional as F

from src.utils.logger import info, warning
from src.utils.gradient_utils import safe_clip_grad_norm_
from src.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from src.training.utils.trainer_common import (
    accumulate_batch_accuracy,
    finalize_vote_accuracy,
    normalize_features,
    teacher_logits_from_model,
    use_chordnet_eval_defaults,
)


class ContinualLearningTrainer:
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
        ewc_lambda / original_params: Elastic Weight Consolidation.
        use_pod_loss / pod_alpha: Pooled Outputs Distillation (feature matching).
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
        ewc_lambda=0.0,
        original_params=None,
        use_pod_loss=False,
        pod_alpha=0.1,
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

        # EWC
        self.ewc_lambda = ewc_lambda
        self.original_params = original_params

        # POD loss (feature distillation)
        self.use_pod_loss = use_pod_loss
        self.pod_alpha = pod_alpha
        self.original_model = None
        if use_pod_loss:
            self.original_model = copy.deepcopy(model)
            self.original_model.eval()
            for p in self.original_model.parameters():
                p.requires_grad = False

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

    def _kd_loss(self, student_logits, teacher_logits):
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        if self.selective_kd:
            confidence, _ = soft_targets.max(dim=-1)
            weights = torch.ones_like(confidence)
            weights[confidence < self.kd_min_confidence_threshold] = 0.0
            high = confidence > self.kd_confidence_threshold
            if high.any():
                excess = (confidence[high] - self.kd_confidence_threshold) / (1.0 - self.kd_confidence_threshold + 1e-8)
                weights[high] = 1.0 - 0.8 * excess
            weights = weights.clamp(0.0, 1.0)
            per_sample = F.kl_div(log_probs, soft_targets, reduction='none').sum(dim=-1)
            return (per_sample * weights).mean() * (self.temperature ** 2)

        return F.kl_div(log_probs, soft_targets, reduction='batchmean') * (self.temperature ** 2)

    def _focal_loss(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        p_t = probs[torch.arange(logits.size(0), device=logits.device), targets]
        focal_weight = (1 - p_t) ** self.focal_gamma
        ce = F.cross_entropy(logits, targets, reduction='none')
        loss = focal_weight * ce
        if self.focal_alpha == 'auto' and self.weight_tensor is not None:
            loss = self.weight_tensor[targets] * loss
        return loss.mean()

    def _get_teacher_logits(self, features):
        teacher_logits = teacher_logits_from_model(
            self.teacher_model,
            features,
            teacher_mean=self.teacher_mean,
            teacher_std=self.teacher_std,
        )
        if teacher_logits is not None and teacher_logits.dim() == 3:
            teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))
        return teacher_logits

    def compute_loss(self, logits, targets, teacher_logits=None):
        """Compute combined CE/focal + KD + EWC loss."""
        if logits.dim() == 3 and targets.dim() == 2:
            B, T, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
            if teacher_logits is not None and teacher_logits.dim() == 3:
                teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))

        # Standard loss
        if self.use_focal_loss:
            std_loss = self._focal_loss(logits, targets)
        else:
            std_loss = self.loss_fn(logits, targets)

        # KD loss
        if self.use_kd and teacher_logits is not None and logits.shape == teacher_logits.shape:
            kd = self._kd_loss(logits, teacher_logits)
            loss = self.kd_alpha * kd + (1 - self.kd_alpha) * std_loss
        else:
            loss = std_loss

        # EWC regularization
        if self.ewc_lambda > 0 and self.original_params is not None and self.model.training:
            ewc = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.original_params:
                    ewc = ewc + ((param - self.original_params[name]) ** 2).sum()
            loss = loss + self.ewc_lambda * ewc

        return torch.clamp(loss, min=0.0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader=None):
        info(f"Starting CL training: {self.num_epochs} epochs, KD={'ON' if self.use_kd else 'OFF'}")

        for epoch in range(1, self.num_epochs + 1):
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
                    fl = logits.reshape(-1, logits.size(-1)) if logits.dim() == 3 else logits
                    ft = targets.reshape(-1) if targets.dim() == 2 else targets
                    correct += (fl.argmax(dim=-1) == ft).sum().item()
                    total += ft.numel()

            if batch_idx % 10 == 0:
                info(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

        return total_loss / max(1, len(loader)), correct / max(1, total)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, loader):
        metrics = self._evaluate_loader(loader)
        return metrics['loss'], metrics['accuracy']

    def validate_step(self, batch):
        """Single validation batch step.

        Loader-level validation/testing should prefer ``_evaluate_loader`` so
        ChordNet can aggregate votes across all overlapping windows of a song.
        """
        self.model.eval()
        with torch.no_grad():
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
            loss = self.loss_fn(fl, ft)

            if self._use_chordnet_eval_defaults(batch):
                vote_accumulator = {}
                target_accumulator = {}
                _, _, _ = accumulate_batch_accuracy(
                    self.model,
                    batch,
                    logits,
                    targets,
                    vote_accumulator,
                    target_accumulator,
                )
                correct, total = finalize_vote_accuracy(vote_accumulator, target_accumulator)
            else:
                correct = (fl.argmax(dim=-1) == ft).sum().item()
                total = ft.numel()
        return {'loss': loss.item(), 'accuracy': correct / total if total else 0.0,
                'correct': correct, 'total': total}

    def evaluate_loader(self, loader):
        """Public loader-level evaluation helper used by test loops."""
        return self._evaluate_loader(loader)

    # ------------------------------------------------------------------
    # Checkpointing & early stopping
    # ------------------------------------------------------------------

    def _save_best(self, val_acc, val_loss, epoch):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.early_stop_counter = 0
            save_checkpoint(self.best_model_path, epoch=epoch,
                            model_state_dict=self.model.state_dict(),
                            optimizer_state_dict=self.optimizer.state_dict(),
                            accuracy=val_acc, loss=val_loss,
                            normalization=self.normalization,
                            idx_to_chord=self.idx_to_chord)
            info(f"  Saved best model (acc={val_acc:.4f})")
            return True
        self.early_stop_counter += 1
        return False

    def _should_stop(self):
        return self.early_stop_counter >= self.early_stopping_patience

    def _adjust_lr(self):
        for pg in self.optimizer.param_groups:
            old = pg['lr']
            new = max(old * self.lr_decay_factor, self.min_lr)
            if new < old:
                pg['lr'] = new
                info(f"  LR: {old:.6f} -> {new:.6f}")

    def _save_checkpoint(self, epoch, loss):
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        save_checkpoint(path, epoch=epoch,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        loss=loss, normalization=self.normalization)

    def load_best_model(self):
        ckpt = load_checkpoint(self.best_model_path, device=self.device)
        if ckpt and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            info(f"Loaded best model (acc={ckpt.get('accuracy', '?')})")
            return True
        warning(f"Could not load best model from {self.best_model_path}")
        return False
