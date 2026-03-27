"""
Shared temporal smoothing helpers for frame-wise chord logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temporal_smoothing(logits, kernel_size, use_gaussian=False):
    """Apply uniform or Gaussian temporal smoothing to 2-D or 3-D logits."""
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    def _smooth_1d(logits_t):
        padding = kernel_size // 2
        if use_gaussian:
            sigma = kernel_size / 6.0
            x = torch.arange(kernel_size, dtype=logits_t.dtype, device=logits_t.device)
            x = x - kernel_size // 2
            gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            gaussian_kernel = gaussian_kernel.view(1, 1, -1)
            n_classes = logits_t.shape[1]
            gaussian_kernel = gaussian_kernel.expand(n_classes, 1, -1)
            return F.conv1d(
                F.pad(logits_t, (padding, padding), mode='replicate'),
                gaussian_kernel,
                groups=n_classes,
            )
        return F.avg_pool1d(logits_t, kernel_size=kernel_size, stride=1, padding=padding)

    if logits.dim() == 3:
        logits_t = logits.transpose(1, 2)
        return _smooth_1d(logits_t).transpose(1, 2)
    if logits.dim() == 2:
        logits_t = logits.T.unsqueeze(0)
        return _smooth_1d(logits_t).squeeze(0).T
    return logits
