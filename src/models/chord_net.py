"""ChordMini ChordNet model.

The architecture matches the non-factored ChordNet implementation used outside
``ChordMini``. Validation/test-time prediction also supports the same temporal
smoothing interface, including Gaussian smoothing with ``kernel_size=9``.
"""
import torch
import torch.nn as nn
import warnings

from src.models.common.base_transformer import BaseTransformer
from src.models.common.config import ModelConfig
from src.models.common.temporal_smoothing import apply_temporal_smoothing


class ChordNet(nn.Module):
    def __init__(self, n_freq=144, n_classes=170, n_group=12,
                 f_layer=5, f_head=8, t_layer=5, t_head=8,
                 d_layer=5, d_head=8, dropout=0.2, ignore_index=None, **kwargs):
        super().__init__()

        if n_freq % n_group != 0:
            warnings.warn(f"n_freq={n_freq} not divisible by n_group={n_group}")
        feature_dim = n_freq // n_group
        if feature_dim % f_head != 0:
            for h in range(f_head, 0, -1):
                if feature_dim % h == 0:
                    f_head = h
                    break

        self.transformer = BaseTransformer(
            n_channel=1, n_freq=n_freq, n_group=n_group,
            f_layer=f_layer, f_head=f_head, f_dropout=dropout,
            t_layer=t_layer, t_head=t_head, t_dropout=dropout,
            d_layer=d_layer, d_head=d_head, d_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.n_classes = n_classes
        self.fc = nn.Linear(n_freq, n_classes)
        self.ignore_index = ignore_index
        self.idx_to_chord = kwargs.get('idx_to_chord', None)

    def forward(self, x, targets=None, weight=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            if self.training and torch.isnan(x).any():
                warnings.warn("Input tensor contains NaN values! Replacing with zeros.")
                x = torch.nan_to_num(x, nan=0.0)
        elif x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(1)

        _, features = self.transformer(x, weight)
        features = self.dropout(features)
        logits = self.fc(features)

        if targets is not None:
            criterion = (
                nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                if self.ignore_index is not None else
                nn.CrossEntropyLoss()
            )
            if logits.ndim == 3 and targets.ndim == 2:
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            elif logits.ndim == 3 and targets.ndim == 1:
                loss = criterion(logits.mean(dim=1), targets)
            else:
                loss = criterion(logits, targets)
            return logits, loss

        return logits, features

    def predict(self, x, per_frame=False, smooth=True, kernel_size=9, use_gaussian=False):
        """Make a prediction with optional temporal smoothing.

        Args:
            x: Input tensor.
            per_frame: Return frame-level predictions when True.
            smooth: Apply temporal smoothing before decoding.
            kernel_size: Odd smoothing kernel size.
            use_gaussian: Use Gaussian-weighted smoothing instead of uniform.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            logits = output[0] if isinstance(output, tuple) else output
            if smooth and logits.dim() >= 2:
                logits = self._apply_temporal_smoothing(logits, kernel_size, use_gaussian)
            if per_frame:
                return logits.argmax(dim=-1)
            if logits.dim() > 2:
                return logits.mean(dim=1).argmax(dim=-1)
            return logits.argmax(dim=-1)

    def _apply_temporal_smoothing(self, logits, kernel_size, use_gaussian=False):
        """Apply uniform or Gaussian temporal smoothing to frame logits."""
        return apply_temporal_smoothing(logits, kernel_size, use_gaussian)

    def load_state_dict(self, state_dict, strict=True):
        if 'fc.weight' in state_dict and hasattr(self, 'fc'):
            pre = state_dict['fc.weight'].size(0)
            cur = self.fc.weight.size(0)
            if pre != cur:
                warnings.warn(f"Output layer mismatch: checkpoint={pre}, model={cur}")
        return super().load_state_dict(state_dict, strict)

    def predict_per_frame(self, x):
        """Alias for ``predict(x, per_frame=True)``."""
        return self.predict(x, per_frame=True)

    def predict_frames(self, x):
        """Alias for ``predict(x, per_frame=True)``."""
        return self.predict(x, per_frame=True)


def create_chordnet_model(config: ModelConfig):
    """Factory: instantiate ChordNet from ModelConfig."""
    return ChordNet(**config.to_chordnet_kwargs())
