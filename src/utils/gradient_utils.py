"""
Standalone gradient clipping utilities for the ChordMini pipeline.
"""

import torch
import warnings


def safe_clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False, verbose=True):
    """Clip gradient norm with NaN/Inf handling."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if not parameters:
        return torch.tensor(0.0)

    has_nonfinite = False
    for p in parameters:
        if not torch.isfinite(p.grad).all():
            has_nonfinite = True
            break

    if has_nonfinite:
        for p in parameters:
            if p.grad is not None:
                mask = ~torch.isfinite(p.grad)
                if mask.any():
                    p.grad[mask] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=error_if_nonfinite
        )
    return total_norm
