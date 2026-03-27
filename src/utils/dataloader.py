"""
Shared DataLoader configuration helpers for ChordMini training scripts.
"""

from __future__ import annotations


def build_dataloader_kwargs(device, num_workers=0, prefetch_factor=2, persistent_workers=True):
    """Build safe DataLoader kwargs for the current device and worker count.

    Prefetching and persistent workers are only enabled when ``num_workers > 0``
    because PyTorch rejects those options for the single-process loading path.
    """
    num_workers = max(0, int(num_workers))
    device_type = getattr(device, 'type', str(device))

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': device_type == 'cuda',
    }

    if num_workers > 0:
        kwargs['persistent_workers'] = bool(persistent_workers)
        kwargs['prefetch_factor'] = max(1, int(prefetch_factor))

    return kwargs
