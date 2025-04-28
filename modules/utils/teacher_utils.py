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
    # Check if the path is absolute or relative
    if not os.path.isabs(teacher_model_path):
        # Try to find the model in the external storage path first
        external_path = f"/mnt/storage/checkpoints/btc/{os.path.basename(teacher_model_path)}"
        if os.path.exists(external_path):
            logger.info(f"Found teacher model in external storage: {external_path}")
            teacher_model_path = external_path
        else:
            # Check for btc_model_large_voca.pt specifically
            if "btc_model_large_voca.pt" in teacher_model_path:
                external_path = "/mnt/storage/checkpoints/btc/btc_model_large_voca.pt"
                if os.path.exists(external_path):
                    logger.info(f"Found btc_model_large_voca.pt in external storage: {external_path}")
                    teacher_model_path = external_path

            # If still not found, use the original path
            if not os.path.exists(teacher_model_path):
                logger.warning(f"Teacher model not found at {teacher_model_path} or in external storage")
                logger.info(f"Will try to use the original path: {teacher_model_path}")

    # Get the BTC directory (parent of the teacher model file)
    btc_dir = os.path.dirname(os.path.abspath(teacher_model_path))

    logger.info(f"Looking for BTC model files in: {btc_dir}")

    # First check if BTC directory exists
    if not os.path.exists(btc_dir):
        raise FileNotFoundError(f"BTC directory not found: {btc_dir}")

    # Check if the model file exists
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Teacher model file not found: {teacher_model_path}")

    # Try to use the local BTC model implementation first
    try:
        # Import the local BTC model implementation
        from modules.models.Transformer.btc_model import BTC_model
        from modules.utils.hparams import HParams

        logger.info("Using local BTC model implementation")

        # Load config from local config file
        config_path = "./config/btc_config.yaml"
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "btc_config.yaml")

        logger.info(f"Loading BTC config from: {config_path}")
        config = HParams.load(config_path)

        # Configure for vocabulary size
        if use_voca:
            config.model['num_chords'] = 170
            logger.info("Using large vocabulary (170 chords)")
        else:
            config.model['num_chords'] = 25
            logger.info("Using standard vocabulary (25 chords)")

        # Create model
        model = BTC_model(config=config.model).to(device)

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
        logger.info("Teacher model loaded successfully using local implementation")

        return model, mean, std

    except Exception as e:
        logger.warning(f"Failed to load model using local implementation: {e}")
        logger.info("Falling back to dynamic import method")

    # Add BTC directory to sys.path temporarily to be able to import modules
    sys.path.insert(0, btc_dir)
    utils_path = None

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
                    # Try using the local implementation from modules
                    btc_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                "modules", "models", "Transformer", "btc_model.py")
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
                    # Try local config
                    config_path = "./config/btc_config.yaml"
                    if not os.path.exists(config_path):
                        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                "config", "btc_config.yaml")
                        if not os.path.exists(config_path):
                            raise FileNotFoundError(f"Could not find config file in {btc_dir} or parent directory")

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
                    # Try local HParams
                    from modules.utils.hparams import HParams
        else:
            # Try importing directly
            try:
                from utils.hparams import HParams
            except ImportError:
                # Try local HParams
                from modules.utils.hparams import HParams

        # Load config
        config = HParams.load(config_path)

        # Configure for vocabulary size
        if use_voca:
            if hasattr(config, 'feature'):
                config.feature['large_voca'] = True
            config.model['num_chords'] = 170
            logger.info("Using large vocabulary (170 chords)")
        else:
            if hasattr(config, 'feature'):
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
        logger.info("Teacher model loaded successfully using dynamic import")

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
        if utils_path in sys.path and utils_path is not None:
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

        # Check if this is a BTC model with the expected structure
        if hasattr(teacher_model, 'self_attn_layers') and hasattr(teacher_model, 'output_layer'):
            # BTC model architecture
            if timestep is None or timestep >= normalized_specs.size(1):
                # Process entire sequence at once
                try:
                    # First get the self-attention outputs
                    self_attn_out, _ = teacher_model.self_attn_layers(normalized_specs)

                    # Then get logits directly from the output projection
                    if hasattr(teacher_model.output_layer, 'output_projection'):
                        logits = teacher_model.output_layer.output_projection(self_attn_out)
                        logger.info(f"Successfully extracted logits from BTC model with shape: {logits.shape}")
                    else:
                        # Fallback - use the full output layer
                        logits = teacher_model.output_layer(self_attn_out)
                        logger.info(f"Used fallback method to extract logits with shape: {logits.shape}")
                except Exception as e:
                    logger.error(f"Error extracting logits from BTC model: {e}")
                    # Try using the forward method directly as a last resort
                    logits = teacher_model(normalized_specs)
                    logger.info(f"Used forward method to extract logits with shape: {logits.shape}")
            else:
                # Process in chunks using timestep
                n_timestep = timestep
                all_logits = []

                for t in range(0, normalized_specs.size(1), n_timestep):
                    end_t = min(t + n_timestep, normalized_specs.size(1))
                    chunk = normalized_specs[:, t:end_t, :]

                    try:
                        # Process this chunk
                        self_attn_out, _ = teacher_model.self_attn_layers(chunk)

                        # Get logits directly from the output projection
                        if hasattr(teacher_model.output_layer, 'output_projection'):
                            chunk_logits = teacher_model.output_layer.output_projection(self_attn_out)
                        else:
                            # Fallback
                            chunk_logits = teacher_model.output_layer(self_attn_out)

                        all_logits.append(chunk_logits)
                    except Exception as e:
                        logger.error(f"Error processing chunk {t}-{end_t}: {e}")
                        # Try using the forward method directly as a last resort
                        chunk_logits = teacher_model(chunk)
                        all_logits.append(chunk_logits)

                # Concatenate all chunks
                logits = torch.cat(all_logits, dim=1)
                logger.info(f"Processed in chunks, final logits shape: {logits.shape}")
        else:
            # Not a BTC model or doesn't have the expected structure
            # Try using the forward method directly
            logger.info("Model doesn't have expected BTC structure, using forward method")
            output = teacher_model(normalized_specs)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            logger.info(f"Used forward method, logits shape: {logits.shape}")

    # Ensure consistent dimensionality - always return 3D tensors [batch, time, chords]
    if logits.dim() == 2:
        # If we got a 2D tensor [batch*time, chords], try to reshape to 3D
        # This requires knowing the batch size and sequence length
        batch_size = spectrograms.size(0)
        seq_len = spectrograms.size(1)
        try:
            logits = logits.view(batch_size, seq_len, -1)
            logger.info(f"Reshaped 2D logits to 3D with shape: {logits.shape}")
        except Exception as e:
            logger.error(f"Failed to reshape 2D logits to 3D: {e}")
            # If reshaping fails, unsqueeze to add a dimension
            logits = logits.unsqueeze(0)
            logger.info(f"Added dimension to logits, new shape: {logits.shape}")

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
