"""
Standalone device detection for the ChordMini pipeline.
"""

import torch

_device = None

def get_device() -> torch.device:
    """Auto-detect the best available device (CUDA > MPS > CPU)."""
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _device = torch.device('mps')
    else:
        _device = torch.device('cpu')
    return _device
