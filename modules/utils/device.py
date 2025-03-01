import torch
import logging

def get_device() -> torch.device:
    """
    Returns a torch.device based on the available hardware.
    Prioritizes CUDA, then MPS (Apple Silicon), and defaults to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Silicon).")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device.")
    return device