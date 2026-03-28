"""
Shared DataLoader configuration helpers for ChordMini training scripts.
"""

from __future__ import annotations


def build_dataloader_kwargs(device, num_workers=0):
    """Build safe DataLoader kwargs for the current device and worker count.

    The current default keeps worker processes persistent and uses a small
    prefetch window when multiprocessing is enabled.
    """
    num_workers = max(0, int(num_workers))
    device_type = getattr(device, 'type', str(device))

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': device_type == 'cuda',
    }

    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2

    return kwargs
