"""
Standalone device detection for the ChordMini pipeline.
"""

import torch
import platform
import os

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

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def to_device(tensor):
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.to(get_device(), non_blocking=True)
