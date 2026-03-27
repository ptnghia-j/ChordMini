"""
Standalone checkpoint save/load utilities for the ChordMini pipeline.
"""
import os
import torch
from .logger import info, warning, error


def save_checkpoint(filepath, **kwargs):
    try:
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(kwargs, filepath)
        info(f"Checkpoint saved to {filepath}")
        return True
    except Exception as e:
        error(f"Error saving checkpoint to {filepath}: {e}")
        return False


def load_checkpoint(filepath, device='cpu'):
    if not os.path.exists(filepath):
        info(f"Checkpoint not found at {filepath}")
        return None
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        error(f"Error loading checkpoint from {filepath}: {e}")
        return None


def extract_model_state_dict(checkpoint):
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    if isinstance(state_dict, dict) and state_dict and next(iter(state_dict)).startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return state_dict


def extract_normalization_stats(checkpoint, default_mean=0.0, default_std=1.0):
    mean = checkpoint.get('mean', default_mean)
    std = checkpoint.get('std', default_std)
    if 'normalization' in checkpoint and isinstance(checkpoint['normalization'], dict):
        mean = checkpoint['normalization'].get('mean', mean)
        std = checkpoint['normalization'].get('std', std)
    if hasattr(mean, 'item'):
        mean = float(mean.item())
    if hasattr(std, 'item'):
        std = float(std.item())
    return float(mean), max(float(std), 1e-8)


def extract_state_dict_and_stats(checkpoint, default_mean=0.0, default_std=1.0):
    return (
        extract_model_state_dict(checkpoint),
        *extract_normalization_stats(checkpoint, default_mean=default_mean, default_std=default_std),
    )


def apply_model_state(model, state_dict):
    if model is None or state_dict is None:
        return
    is_ddp = hasattr(model, 'module')
    has_prefix = next(iter(state_dict), '').startswith('module.')
    if is_ddp and not has_prefix:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif not is_ddp and has_prefix:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    info("Model state loaded successfully")


def apply_optimizer_state(optimizer, state_dict, device='cpu'):
    if optimizer and state_dict:
        try:
            optimizer.load_state_dict(state_dict)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception as e:
            warning(f"Could not load optimizer state: {e}")
