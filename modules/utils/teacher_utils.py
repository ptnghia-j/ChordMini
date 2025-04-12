import os
import sys
import torch
import numpy as np
import importlib.util
from pathlib import Path
from modules.utils import logger

def load_btc_model(teacher_model_path, device, use_voca=False):
    """
    Load a BTC model from an external path.
    
    Args:
        teacher_model_path: Path to the teacher model file
        device: Device to load the model on
        use_voca: Whether to use large vocabulary
        
    Returns:
        model: Loaded teacher model
        mean: Mean for normalization
        std: Standard deviation for normalization
    """
    # Get the BTC directory (parent of the teacher model file)
    btc_dir = os.path.dirname(os.path.abspath(teacher_model_path))
    
    logger.info(f"Looking for BTC model files in: {btc_dir}")
    
    # First check if BTC directory exists
    if not os.path.exists(btc_dir):
        raise FileNotFoundError(f"BTC directory not found: {btc_dir}")
    
    # Add BTC directory to sys.path temporarily to be able to import modules
    sys.path.insert(0, btc_dir)
    
    try:
        # Try to import BTC model dynamically
        btc_model_path = os.path.join(btc_dir, "btc_model.py")
        if not os.path.exists(btc_model_path):
            # Look in parent directory or models subdirectory
            parent_dir = os.path.dirname(btc_dir)
            btc_model_path = os.path.join(parent_dir, "btc_model.py")
            if not os.path.exists(btc_model_path):
                # Try models subdirectory
                btc_model_path = os.path.join(btc_dir, "models", "btc_model.py")
                if not os.path.exists(btc_model_path):
                    raise FileNotFoundError(f"Could not find btc_model.py in {btc_dir} or parent directory")
        
        logger.info(f"Loading BTC model module from: {btc_model_path}")
        
        # Dynamically import the module
        spec = importlib.util.spec_from_file_location("btc_model", btc_model_path)
        btc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(btc_module)
        
        # Now try to find the config file
        config_path = os.path.join(btc_dir, "run_config.yaml")
        if not os.path.exists(config_path):
            # Look in parent directory or configs subdirectory
            parent_dir = os.path.dirname(btc_dir)
            config_path = os.path.join(parent_dir, "run_config.yaml")
            if not os.path.exists(config_path):
                # Try configs subdirectory
                config_path = os.path.join(btc_dir, "configs", "run_config.yaml")
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Could not find run_config.yaml in {btc_dir} or parent directory")
        
        logger.info(f"Loading BTC config from: {config_path}")
        
        # Import HParams from utils
        utils_path = os.path.join(btc_dir, "utils")
        if os.path.exists(utils_path):
            sys.path.insert(0, utils_path)
            try:
                from hparams import HParams
            except ImportError:
                # Try different structure
                sys.path.pop(0)  # Remove utils_path
                try:
                    from utils.hparams import HParams
                except ImportError:
                    raise ImportError("Could not import HParams from either utils.hparams or hparams")
        else:
            # Try importing directly
            try:
                from utils.hparams import HParams
            except ImportError:
                raise ImportError("Could not import HParams from utils.hparams")
        
        # Load config
        config = HParams.load(config_path)
        
        # Configure for vocabulary size
        if use_voca:
            config.feature['large_voca'] = True
            config.model['num_chords'] = 170
            logger.info("Using large vocabulary (170 chords)")
        else:
            config.feature['large_voca'] = False
            config.model['num_chords'] = 25
            logger.info("Using standard vocabulary (25 chords)")
        
        # Create model
        model = btc_module.BTC_model(config=config.model).to(device)
        
        # Load weights
        logger.info(f"Loading teacher model weights from: {teacher_model_path}")
        checkpoint = torch.load(teacher_model_path, map_location=device)
        
        # Extract mean and std
        if 'mean' in checkpoint:
            mean = checkpoint['mean']
        else:
            mean = 0.0
        
        if 'std' in checkpoint:
            std = checkpoint['std']
        else:
            std = 1.0
        
        # Load state dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to eval mode
        model.eval()
        logger.info("Teacher model loaded successfully")
        
        return model, mean, std
        
    except Exception as e:
        logger.error(f"Error loading BTC model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clean up sys.path
        if btc_dir in sys.path:
            sys.path.remove(btc_dir)
        if utils_path in sys.path:
            sys.path.remove(utils_path)

def extract_logits_from_teacher(teacher_model, spectrograms, mean, std, device, timestep=None):
    """
    Extract logits from a teacher model for knowledge distillation.
    
    Args:
        teacher_model: The teacher model
        spectrograms: Input spectrograms [batch_size, seq_len, n_bins]
        mean: Mean for normalization
        std: Standard deviation for normalization
        device: Device to run inference on
        timestep: Model timestep (if None, will process entire sequence at once)
        
    Returns:
        logits: Teacher logits [batch_size, seq_len, n_classes]
    """
    if teacher_model is None:
        return None
    
    # Normalize input
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        normalized_specs = (spectrograms - mean) / std
    else:
        # Handle tensor means/stds
        normalized_specs = (spectrograms - mean.to(device)) / std.to(device)
    
    # Process with teacher model
    with torch.no_grad():
        teacher_model.eval()
        
        # Extract teacher model features
        if hasattr(teacher_model, 'self_attn_layers') and hasattr(teacher_model, 'output_layer'):
            # BTC model architecture
            if timestep is None:
                # Process entire sequence at once
                self_attn_out, _ = teacher_model.self_attn_layers(normalized_specs)
                
                # Get logits directly from the output projection
                if hasattr(teacher_model.output_layer, 'output_projection'):
                    logits = teacher_model.output_layer.output_projection(self_attn_out)
                else:
                    # Fallback - use the full output layer but capture the logits before softmax
                    orig_probs_out = teacher_model.output_layer.probs_out
                    teacher_model.output_layer.probs_out = False
                    logits, _ = teacher_model.output_layer(self_attn_out)
                    teacher_model.output_layer.probs_out = orig_probs_out
            else:
                # Process in chunks using timestep
                n_timestep = timestep
                all_logits = []
                
                for t in range(0, normalized_specs.size(1), n_timestep):
                    end_t = min(t + n_timestep, normalized_specs.size(1))
                    chunk = normalized_specs[:, t:end_t, :]
                    
                    # Process this chunk
                    self_attn_out, _ = teacher_model.self_attn_layers(chunk)
                    
                    # Get logits directly from the output projection
                    if hasattr(teacher_model.output_layer, 'output_projection'):
                        chunk_logits = teacher_model.output_layer.output_projection(self_attn_out)
                    else:
                        # Fallback
                        orig_probs_out = teacher_model.output_layer.probs_out
                        teacher_model.output_layer.probs_out = False
                        chunk_logits, _ = teacher_model.output_layer(self_attn_out)
                        teacher_model.output_layer.probs_out = orig_probs_out
                    
                    all_logits.append(chunk_logits)
                
                # Concatenate all chunks
                logits = torch.cat(all_logits, dim=1)
        else:
            # Assume ChordNet or similar architecture with a standard forward pass
            # that returns logits as the first output
            output = teacher_model(normalized_specs)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
    
    return logits

def generate_teacher_predictions(teacher_model, data_loader, mean, std, device, save_dir=None):
    """
    Generate and optionally save teacher model predictions for a dataset.
    
    Args:
        teacher_model: The teacher model
        data_loader: DataLoader for the dataset
        mean: Mean for normalization
        std: Standard deviation for normalization
        device: Device to run inference on
        save_dir: Directory to save predictions (if None, won't save)
        
    Returns:
        predictions: Dictionary mapping sample IDs to logits
    """
    if teacher_model is None:
        return None
    
    predictions = {}
    
    with torch.no_grad():
        teacher_model.eval()
        
        for batch_idx, batch in enumerate(data_loader):
            # Get input spectrograms
            spectrograms = batch['spectro'].to(device)
            
            # Get sample IDs
            if 'song_id' in batch:
                sample_ids = batch['song_id']
            else:
                # Use batch index as sample ID
                sample_ids = [f"batch_{batch_idx}_{i}" for i in range(len(spectrograms))]
            
            # Extract logits
            logits = extract_logits_from_teacher(
                teacher_model, spectrograms, mean, std, device
            )
            
            # Store logits
            for i, sample_id in enumerate(sample_ids):
                predictions[sample_id] = logits[i].cpu()
            
            # Save logits if requested
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                for i, sample_id in enumerate(sample_ids):
                    save_path = os.path.join(save_dir, f"{sample_id}_logits.pt")
                    torch.save(logits[i].cpu(), save_path)
    
    return predictions
