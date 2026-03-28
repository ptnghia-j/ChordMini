from __future__ import annotations

import os

from src.utils.checkpoint_utils import (
    apply_model_state,
    apply_optimizer_state,
    load_checkpoint,
    save_checkpoint,
)
from src.utils.logger import info, warning


class TrainerCheckpointMixin:
    """Shared checkpoint save/load helpers for trainer classes."""

    def _build_trainer_state_payload(self):
        payload = {
            'best_val_acc': self.best_val_acc,
            'early_stop_counter': self.early_stop_counter,
            'train_losses': list(self.train_losses),
            'val_losses': list(self.val_losses),
        }
        payload.update(self._get_additional_checkpoint_state())
        return payload

    def _get_additional_checkpoint_state(self):
        return {}

    def _restore_additional_checkpoint_state(self, checkpoint):
        return None

    def _build_best_checkpoint_payload(self, epoch, val_acc, val_loss):
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': val_acc,
            'loss': val_loss,
            'normalization': self.normalization,
            'idx_to_chord': self.idx_to_chord,
        }
        payload.update(self._build_trainer_state_payload())
        return payload

    def _build_periodic_checkpoint_payload(self, epoch, loss):
        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'normalization': self.normalization,
        }
        payload.update(self._build_trainer_state_payload())
        return payload

    def _save_best(self, val_acc, val_loss, epoch):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.early_stop_counter = 0
            save_checkpoint(
                self.best_model_path,
                **self._build_best_checkpoint_payload(epoch, val_acc, val_loss),
            )
            info(f"  Saved best model (acc={val_acc:.4f})")
            return True
        self.early_stop_counter += 1
        return False

    def _save_checkpoint(self, epoch, loss):
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        save_checkpoint(path, **self._build_periodic_checkpoint_payload(epoch, loss))

    def load_best_model(self):
        ckpt = load_checkpoint(self.best_model_path, device=self.device)
        if ckpt and 'model_state_dict' in ckpt:
            apply_model_state(self.model, ckpt['model_state_dict'])
            info(f"Loaded best model from {self.best_model_path} (acc={ckpt.get('accuracy', '?')})")
            return True
        warning(f"Could not load best model from {self.best_model_path}")
        return False

    def resume_from_checkpoint(self, checkpoint_path):
        ckpt = load_checkpoint(checkpoint_path, device=self.device)
        if not ckpt or 'model_state_dict' not in ckpt:
            warning(f"Could not resume training from {checkpoint_path}")
            return False

        apply_model_state(self.model, ckpt['model_state_dict'])
        apply_optimizer_state(self.optimizer, ckpt.get('optimizer_state_dict'), device=self.device)

        if isinstance(ckpt.get('normalization'), dict):
            self.normalization = ckpt['normalization']
        if ckpt.get('idx_to_chord') is not None:
            self.idx_to_chord = ckpt['idx_to_chord']

        self.best_val_acc = float(ckpt.get('best_val_acc', ckpt.get('accuracy', self.best_val_acc)))
        self.early_stop_counter = int(ckpt.get('early_stop_counter', 0))
        self.train_losses = list(ckpt.get('train_losses', self.train_losses))
        self.val_losses = list(ckpt.get('val_losses', self.val_losses))
        self.start_epoch = max(1, int(ckpt.get('epoch', 0)) + 1)

        self._restore_additional_checkpoint_state(ckpt)

        info(
            f"Resumed training state from {checkpoint_path} "
            f"(next epoch: {self.start_epoch}, best_val_acc={self.best_val_acc:.4f})"
        )
        return True


__all__ = ['TrainerCheckpointMixin']
