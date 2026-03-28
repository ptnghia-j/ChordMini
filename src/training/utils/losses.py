from __future__ import annotations

import torch
import torch.nn.functional as F

from src.training.utils.trainer_common import flatten_logits_and_targets


class TrainerLossMixin:
    """Shared CE/focal/KD loss helpers for training classes."""

    def _flatten_loss_inputs(self, logits, targets, teacher_logits=None):
        return flatten_logits_and_targets(logits, targets, teacher_logits)

    def _kd_loss(self, student_logits, teacher_logits):
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        if getattr(self, 'selective_kd', False):
            confidence, _ = soft_targets.max(dim=-1)
            weights = torch.ones_like(confidence)
            min_conf = float(getattr(self, 'kd_min_confidence_threshold', 0.1))
            max_conf = float(getattr(self, 'kd_confidence_threshold', 0.9))
            weights[confidence < min_conf] = 0.0
            high = confidence > max_conf
            if high.any():
                excess = (confidence[high] - max_conf) / (1.0 - max_conf + 1e-8)
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
        if getattr(self, 'focal_alpha', None) == 'auto' and getattr(self, 'weight_tensor', None) is not None:
            loss = self.weight_tensor[targets] * loss
        return loss.mean()

    def compute_loss(self, logits, targets, teacher_logits=None):
        logits, targets, teacher_logits = self._flatten_loss_inputs(logits, targets, teacher_logits)

        if self.use_focal_loss:
            standard_loss = self._focal_loss(logits, targets)
        else:
            standard_loss = self.loss_fn(logits, targets)

        use_kd = bool(getattr(self, 'use_kd_loss', getattr(self, 'use_kd', False)))
        if use_kd and teacher_logits is not None and logits.shape == teacher_logits.shape:
            kd = self._kd_loss(logits, teacher_logits)
            loss = self.kd_alpha * kd + (1 - self.kd_alpha) * standard_loss
        else:
            loss = standard_loss

        return torch.clamp(loss, min=0.0)


__all__ = ['TrainerLossMixin']
