import os
import torch
from modules.utils.logger import info, warning, error

def save_checkpoint(filepath, **kwargs):
    """
    Saves a checkpoint. All data to be saved should be passed as keyword arguments.

    Args:
        filepath (str): Path to save the checkpoint.
        **kwargs: Arbitrary data to save in the checkpoint file.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(kwargs, filepath)
        info(f"Checkpoint saved to {filepath}")
        return True
    except Exception as e:
        error(f"Error saving checkpoint to {filepath}: {str(e)}")
        import traceback
        error(traceback.format_exc())
        return False

def load_checkpoint(filepath, device='cpu'):
    """
    Loads a checkpoint from a file.

    Args:
        filepath (str): Path to the checkpoint file.
        device (str or torch.device): Device to map the loaded tensors to.

    Returns:
        dict or None: The loaded checkpoint data as a dictionary, or None if loading failed.
    """
    if not os.path.exists(filepath):
        info(f"Checkpoint not found at {filepath}")
        return None
    try:
        checkpoint = torch.load(filepath, map_location=device)
        info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        error(f"Error loading checkpoint from {filepath}: {str(e)}")
        import traceback
        error(traceback.format_exc())
        return None

def apply_model_state(model, state_dict):
    """
    Applies a state dictionary to a model, handling distributed vs non-distributed model differences.

    Args:
        model (torch.nn.Module): The model to apply the state to.
        state_dict (dict): The state dictionary to load.
    """
    if model and state_dict:
        try:
            # Check if we need to handle distributed model prefix mismatch
            is_distributed_model = hasattr(model, 'module')
            has_module_prefix = list(state_dict.keys())[0].startswith('module.') if state_dict else False

            # Handle prefix mismatch between checkpoint and model
            if is_distributed_model and not has_module_prefix:
                # Model is distributed but checkpoint isn't - add 'module.' prefix
                info("Adding 'module.' prefix to state dict keys for distributed model")
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif not is_distributed_model and has_module_prefix:
                # Model is not distributed but checkpoint is - remove 'module.' prefix
                info("Removing 'module.' prefix from state dict keys for non-distributed model")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Now load the state dict
            model.load_state_dict(state_dict)
            info("Model state loaded successfully")
        except Exception as e:
            error(f"Error applying model state: {str(e)}. Ensure model architecture matches checkpoint.")

def apply_optimizer_state(optimizer, state_dict, device='cpu'):
    """
    Applies a state dictionary to an optimizer and moves its state to the specified device.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to apply the state to.
        state_dict (dict): The state dictionary to load.
        device (str or torch.device): Device to move optimizer state tensors to.
    """
    if optimizer and state_dict:
        try:
            optimizer.load_state_dict(state_dict)
            # Move optimizer state to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception as e:
            warning(f"Could not load optimizer state: {e}. Optimizer may be reinitialized.")


def apply_scheduler_state(scheduler, state_dict):
    """
    Applies a state dictionary to a learning rate scheduler.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to apply the state to.
        state_dict (dict): The state dictionary to load.
    """
    if scheduler and state_dict:
        try:
            scheduler.load_state_dict(state_dict)
        except Exception as e:
            warning(f"Could not load scheduler state: {e}. Scheduler may be reinitialized or use its default state.")

