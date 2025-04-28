import os
import sys
import torch
import numpy as np
import importlib.util
from pathlib import Path
from modules.utils import logger

def validate_teacher_model(model):
    """
    Validate that a loaded teacher model has the expected structure for logit extraction.

    Args:
        model: The model to validate

    Returns:
        bool: True if the model has the expected structure, False otherwise
        str: Description of validation result
    """
    # Check for BTC model structure (self_attn_layers and output_layer)
    has_self_attn = hasattr(model, 'self_attn_layers')
    has_output_layer = hasattr(model, 'output_layer')

    if has_self_attn and has_output_layer:
        # Check if output_layer has output_projection
        has_output_projection = hasattr(model.output_layer, 'output_projection')
        if has_output_projection:
            return True, "Model has complete BTC structure with output_projection"
        else:
            return True, "Model has BTC structure but no output_projection"

    # Check if model has a forward method (minimum requirement)
    has_forward = hasattr(model, 'forward') and callable(getattr(model, 'forward'))
    if has_forward:
        return True, "Model has forward method but not BTC structure"

    return False, "Model lacks required structure for logit extraction"

def find_teacher_model_path(teacher_model_path):
    """
    Find the actual path to the teacher model by checking multiple standard locations.

    Args:
        teacher_model_path: The provided path to the teacher model

    Returns:
        str: The resolved path to the teacher model
        bool: True if the model was found, False otherwise
    """
    # List of standard locations to check
    standard_locations = [
        # External storage paths
        "/mnt/storage/checkpoints/btc",
        "/mnt/storage/checkpoints/teacher",
        "/mnt/storage/checkpoints",
        # Local paths
        "./checkpoints/btc",
        "./checkpoints/teacher",
        "./checkpoints"
    ]

    # If path is already absolute and exists, return it
    if os.path.isabs(teacher_model_path) and os.path.exists(teacher_model_path):
        return teacher_model_path, True

    # If path is relative, check if it exists relative to current directory
    if os.path.exists(teacher_model_path):
        return os.path.abspath(teacher_model_path), True

    # Get the basename of the model file
    model_basename = os.path.basename(teacher_model_path)

    # Check standard locations
    for location in standard_locations:
        candidate_path = os.path.join(location, model_basename)
        if os.path.exists(candidate_path):
            logger.info(f"Found teacher model at: {candidate_path}")
            return candidate_path, True

    # Check for standard model names if the provided path doesn't exist
    standard_model_names = [
        "btc_model_large_voca.pt",
        "btc_model_best.pt",
        "teacher_model_best.pt",
        "teacher_model_large_voca.pt"
    ]

    # If the provided path doesn't match any standard names, check for standard names
    if model_basename not in standard_model_names:
        for model_name in standard_model_names:
            for location in standard_locations:
                candidate_path = os.path.join(location, model_name)
                if os.path.exists(candidate_path):
                    logger.info(f"Found standard teacher model at: {candidate_path}")
                    return candidate_path, True

    # If we get here, we couldn't find the model
    return teacher_model_path, False

def load_btc_model(teacher_model_path, device, use_voca=False):
    """
    Load a BTC model from an external path with enhanced error handling and validation.

    Args:
        teacher_model_path: Path to the teacher model file
        device: Device to load the model on
        use_voca: Whether to use large vocabulary

    Returns:
        model: Loaded teacher model (None if loading failed)
        mean: Mean for normalization (0.0 if loading failed)
        std: Standard deviation for normalization (1.0 if loading failed)
        status: Dictionary with loading status information
    """
    status = {
        "success": False,
        "message": "",
        "model_found": False,
        "model_loaded": False,
        "model_validated": False,
        "model_path": "",
        "implementation": "",
        "has_mean_std": False
    }

    # Find the actual path to the teacher model
    resolved_path, model_found = find_teacher_model_path(teacher_model_path)
    teacher_model_path = resolved_path
    status["model_path"] = teacher_model_path
    status["model_found"] = model_found

    if not model_found:
        logger.error(f"Teacher model not found at {teacher_model_path} or in any standard location")
        status["message"] = f"Teacher model not found at {teacher_model_path} or in any standard location"
        return None, 0.0, 1.0, status

    # Get the BTC directory (parent of the teacher model file)
    btc_dir = os.path.dirname(os.path.abspath(teacher_model_path))
    logger.info(f"Looking for BTC model files in: {btc_dir}")

    # First check if BTC directory exists
    if not os.path.exists(btc_dir):
        logger.error(f"BTC directory not found: {btc_dir}")
        status["message"] = f"BTC directory not found: {btc_dir}"
        return None, 0.0, 1.0, status

    # Check if the model file exists
    if not os.path.exists(teacher_model_path):
        logger.error(f"Teacher model file not found: {teacher_model_path}")
        status["message"] = f"Teacher model file not found: {teacher_model_path}"
        return None, 0.0, 1.0, status

    # Try to use the local BTC model implementation first
    try:
        # Import the local BTC model implementation
        from modules.models.Transformer.btc_model import BTC_model
        from modules.utils.hparams import HParams

        logger.info("Using local BTC model implementation")
        status["implementation"] = "local"

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
        mean = 0.0
        std = 1.0
        if 'mean' in checkpoint:
            mean = checkpoint['mean']
            status["has_mean_std"] = True
        else:
            logger.warning("Mean not found in checkpoint, using default value 0.0")

        if 'std' in checkpoint:
            std = checkpoint['std']
            status["has_mean_std"] = True
        else:
            logger.warning("Std not found in checkpoint, using default value 1.0")

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
        status["model_loaded"] = True

        # Validate the model
        is_valid, validation_message = validate_teacher_model(model)
        status["model_validated"] = is_valid
        if is_valid:
            logger.info(f"Teacher model validation: {validation_message}")
        else:
            logger.warning(f"Teacher model validation failed: {validation_message}")

        status["success"] = True
        status["message"] = "Teacher model loaded successfully using local implementation"
        return model, mean, std, status

    except Exception as e:
        logger.warning(f"Failed to load model using local implementation: {e}")
        logger.info("Falling back to dynamic import method")
        # Continue to dynamic import method

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
                        logger.error(f"Could not find btc_model.py in {btc_dir} or parent directory")
                        status["message"] = f"Could not find btc_model.py in {btc_dir} or parent directory"
                        return None, 0.0, 1.0, status

        logger.info(f"Loading BTC model module from: {btc_model_path}")
        status["implementation"] = "dynamic"

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
                            logger.error(f"Could not find config file in {btc_dir} or parent directory")
                            status["message"] = f"Could not find config file in {btc_dir} or parent directory"
                            return None, 0.0, 1.0, status

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
        mean = 0.0
        std = 1.0
        if 'mean' in checkpoint:
            mean = checkpoint['mean']
            status["has_mean_std"] = True
        else:
            logger.warning("Mean not found in checkpoint, using default value 0.0")

        if 'std' in checkpoint:
            std = checkpoint['std']
            status["has_mean_std"] = True
        else:
            logger.warning("Std not found in checkpoint, using default value 1.0")

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
        status["model_loaded"] = True

        # Validate the model
        is_valid, validation_message = validate_teacher_model(model)
        status["model_validated"] = is_valid
        if is_valid:
            logger.info(f"Teacher model validation: {validation_message}")
        else:
            logger.warning(f"Teacher model validation failed: {validation_message}")

        status["success"] = True
        status["message"] = "Teacher model loaded successfully using dynamic import"
        return model, mean, std, status

    except Exception as e:
        logger.error(f"Error loading BTC model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        status["message"] = f"Error loading BTC model: {str(e)}"
        return None, 0.0, 1.0, status
    finally:
        # Clean up sys.path
        if btc_dir in sys.path:
            sys.path.remove(btc_dir)
        if utils_path in sys.path and utils_path is not None:
            sys.path.remove(utils_path)

def extract_logits_from_teacher(teacher_model, spectrograms, mean, std, device, timestep=None, debug_save_path=None):
    """
    Extract logits from a teacher model for knowledge distillation with enhanced validation and error handling.

    Args:
        teacher_model: The teacher model
        spectrograms: Input spectrograms [batch_size, seq_len, n_bins]
        mean: Mean for normalization
        std: Standard deviation for normalization
        device: Device to run inference on
        timestep: Model timestep (if None, will process entire sequence at once)
        debug_save_path: Optional path to save intermediate tensors for debugging

    Returns:
        logits: Teacher logits [batch_size, seq_len, n_classes]
        status: Dictionary with extraction status information
    """
    status = {
        "success": False,
        "message": "",
        "method_used": "",
        "input_shape": None,
        "output_shape": None,
        "needed_reshape": False
    }

    # Validate inputs
    if teacher_model is None:
        status["message"] = "Teacher model is None"
        return None, status

    # Record input shape for debugging
    status["input_shape"] = list(spectrograms.shape)

    # Validate input dimensions
    if spectrograms.dim() != 3:
        error_msg = f"Expected 3D input tensor [batch, time, features], got shape: {spectrograms.shape}"
        logger.error(error_msg)
        status["message"] = error_msg
        return None, status

    # Save input tensor for debugging if requested
    if debug_save_path:
        os.makedirs(debug_save_path, exist_ok=True)
        try:
            torch.save(spectrograms.cpu(), os.path.join(debug_save_path, "input_spectrograms.pt"))
            logger.info(f"Saved input spectrograms to {debug_save_path}")
        except Exception as e:
            logger.warning(f"Failed to save input spectrograms for debugging: {e}")

    # Normalize input with better error handling
    try:
        if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
            normalized_specs = (spectrograms - mean) / (std if std != 0 else 1.0)
        else:
            # Handle tensor means/stds
            mean_tensor = mean.to(device) if hasattr(mean, 'to') else torch.tensor(mean, device=device)
            std_tensor = std.to(device) if hasattr(std, 'to') else torch.tensor(std, device=device)
            # Avoid division by zero
            std_tensor = torch.clamp(std_tensor, min=1e-6)
            normalized_specs = (spectrograms - mean_tensor) / std_tensor

        # Save normalized input for debugging if requested
        if debug_save_path:
            try:
                torch.save(normalized_specs.cpu(), os.path.join(debug_save_path, "normalized_specs.pt"))
            except Exception as e:
                logger.warning(f"Failed to save normalized spectrograms for debugging: {e}")
    except Exception as e:
        error_msg = f"Error normalizing input: {e}"
        logger.error(error_msg)
        status["message"] = error_msg
        return None, status

    # Process with teacher model
    with torch.no_grad():
        teacher_model.eval()

        # Determine extraction method based on model structure
        is_btc_model = hasattr(teacher_model, 'self_attn_layers') and hasattr(teacher_model, 'output_layer')
        has_output_projection = is_btc_model and hasattr(teacher_model.output_layer, 'output_projection')

        # Try different extraction methods in order of preference
        extraction_methods = []

        if is_btc_model:
            if has_output_projection:
                extraction_methods.append(("btc_with_projection", lambda x: teacher_model.output_layer.output_projection(teacher_model.self_attn_layers(x)[0])))
            extraction_methods.append(("btc_output_layer", lambda x: teacher_model.output_layer(teacher_model.self_attn_layers(x)[0])))

        # Always add forward method as fallback
        extraction_methods.append(("forward", lambda x: teacher_model(x)))

        # Try each method until one succeeds
        logits = None
        for method_name, extraction_func in extraction_methods:
            try:
                if method_name == "forward":
                    output = extraction_func(normalized_specs)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                else:
                    logits = extraction_func(normalized_specs)

                status["method_used"] = method_name
                logger.info(f"Successfully extracted logits using {method_name} method")
                break
            except Exception as e:
                logger.warning(f"Failed to extract logits using {method_name} method: {e}")
                continue

        # If all methods failed, return error
        if logits is None:
            error_msg = "All extraction methods failed"
            logger.error(error_msg)
            status["message"] = error_msg
            return None, status

        # Save raw logits for debugging if requested
        if debug_save_path:
            try:
                torch.save(logits.cpu(), os.path.join(debug_save_path, "raw_logits.pt"))
            except Exception as e:
                logger.warning(f"Failed to save raw logits for debugging: {e}")

        # Record output shape
        status["output_shape"] = list(logits.shape)

        # Ensure consistent dimensionality - always return 3D tensors [batch, time, chords]
        if logits.dim() == 2:
            # If we got a 2D tensor [batch*time, chords], try to reshape to 3D
            batch_size = spectrograms.size(0)
            seq_len = spectrograms.size(1)
            try:
                # Check if the first dimension is batch_size * seq_len
                if logits.size(0) == batch_size * seq_len:
                    logits = logits.view(batch_size, seq_len, -1)
                    logger.info(f"Reshaped 2D logits to 3D with shape: {logits.shape}")
                    status["needed_reshape"] = True
                else:
                    # If dimensions don't match, try to infer the correct reshape
                    n_classes = logits.size(1)
                    if batch_size == 1:
                        # If batch size is 1, reshape to [1, time, classes]
                        logits = logits.unsqueeze(0)
                        logger.info(f"Added batch dimension to logits, new shape: {logits.shape}")
                        status["needed_reshape"] = True
                    else:
                        # Otherwise, this is likely a flattened batch, try to reshape
                        logger.warning(f"Logits shape {logits.shape} doesn't match expected dimensions. Attempting to reshape.")
                        logits = logits.unsqueeze(0)
                        logger.info(f"Added dimension to logits, new shape: {logits.shape}")
                        status["needed_reshape"] = True
            except Exception as e:
                logger.error(f"Failed to reshape 2D logits to 3D: {e}")
                # If reshaping fails, unsqueeze to add a dimension
                logits = logits.unsqueeze(0)
                logger.info(f"Added dimension to logits, new shape: {logits.shape}")
                status["needed_reshape"] = True

        # Final validation of output shape
        if logits.dim() != 3:
            error_msg = f"Expected 3D output tensor [batch, time, classes], got shape: {logits.shape}"
            logger.error(error_msg)
            status["message"] = error_msg
            return None, status

        # Save final logits for debugging if requested
        if debug_save_path:
            try:
                torch.save(logits.cpu(), os.path.join(debug_save_path, "final_logits.pt"))
            except Exception as e:
                logger.warning(f"Failed to save final logits for debugging: {e}")

        # Update final status
        status["success"] = True
        status["message"] = f"Successfully extracted logits with shape {logits.shape} using {status['method_used']} method"
        status["output_shape"] = list(logits.shape)

        return logits, status

def generate_teacher_predictions(teacher_model, data_loader, mean, std, device, save_dir=None, debug_mode=False):
    """
    Generate and optionally save teacher model predictions for a dataset with enhanced error handling.

    Args:
        teacher_model: The teacher model
        data_loader: DataLoader for the dataset
        mean: Mean for normalization
        std: Standard deviation for normalization
        device: Device to run inference on
        save_dir: Directory to save predictions (if None, won't save)
        debug_mode: Whether to save debug information

    Returns:
        predictions: Dictionary mapping sample IDs to logits
        status: Dictionary with generation status information
    """
    status = {
        "success": False,
        "message": "",
        "total_samples": 0,
        "successful_samples": 0,
        "failed_samples": 0,
        "extraction_methods_used": {},
        "sample_shapes": {}
    }

    if teacher_model is None:
        status["message"] = "Teacher model is None"
        return None, status

    predictions = {}

    # Create save directory if needed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving teacher predictions to: {save_dir}")

    # Create debug directory if needed
    debug_save_path = None
    if debug_mode and save_dir is not None:
        debug_save_path = os.path.join(save_dir, "debug")
        os.makedirs(debug_save_path, exist_ok=True)
        logger.info(f"Saving debug information to: {debug_save_path}")

    with torch.no_grad():
        teacher_model.eval()

        # Process each batch
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Get input spectrograms
                if 'spectro' in batch:
                    spectrograms = batch['spectro'].to(device)
                elif 'spectrogram' in batch:
                    spectrograms = batch['spectrogram'].to(device)
                else:
                    # Try to find a tensor in the batch that might be the spectrogram
                    found = False
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor) and value.dim() == 3:
                            spectrograms = value.to(device)
                            logger.warning(f"Using '{key}' as spectrogram input (guessed from tensor shape)")
                            found = True
                            break

                    if not found:
                        logger.error(f"Could not find spectrogram tensor in batch keys: {list(batch.keys())}")
                        status["failed_samples"] += len(batch)
                        continue

                # Get sample IDs
                if 'song_id' in batch:
                    sample_ids = batch['song_id']
                elif 'id' in batch:
                    sample_ids = batch['id']
                else:
                    # Use batch index as sample ID
                    sample_ids = [f"batch_{batch_idx}_{i}" for i in range(len(spectrograms))]

                # Create batch-specific debug directory if needed
                batch_debug_path = None
                if debug_mode and debug_save_path is not None:
                    batch_debug_path = os.path.join(debug_save_path, f"batch_{batch_idx}")
                    os.makedirs(batch_debug_path, exist_ok=True)

                # Extract logits with enhanced error handling
                logits, extraction_status = extract_logits_from_teacher(
                    teacher_model, spectrograms, mean, std, device,
                    debug_save_path=batch_debug_path
                )

                # Update extraction method statistics
                method = extraction_status.get("method_used", "unknown")
                if method in status["extraction_methods_used"]:
                    status["extraction_methods_used"][method] += 1
                else:
                    status["extraction_methods_used"][method] = 1

                # Check if extraction was successful
                if not extraction_status["success"] or logits is None:
                    logger.error(f"Failed to extract logits for batch {batch_idx}: {extraction_status['message']}")
                    status["failed_samples"] += len(spectrograms)
                    continue

                # Store logits and update statistics
                for i, sample_id in enumerate(sample_ids):
                    try:
                        # Store the logits
                        predictions[sample_id] = logits[i].cpu()

                        # Record the shape for debugging
                        status["sample_shapes"][sample_id] = list(logits[i].shape)

                        # Save logits if requested
                        if save_dir is not None:
                            save_path = os.path.join(save_dir, f"{sample_id}_logits.pt")
                            torch.save(logits[i].cpu(), save_path)

                        status["successful_samples"] += 1
                    except Exception as e:
                        logger.error(f"Error processing sample {sample_id}: {e}")
                        status["failed_samples"] += 1

                # Log progress
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches, "
                                f"{status['successful_samples']} successful, "
                                f"{status['failed_samples']} failed")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                status["failed_samples"] += len(batch) if hasattr(batch, "__len__") else 1

    # Update final status
    status["total_samples"] = status["successful_samples"] + status["failed_samples"]
    if status["total_samples"] > 0:
        success_rate = (status["successful_samples"] / status["total_samples"]) * 100
        status["message"] = f"Generated predictions for {status['successful_samples']}/{status['total_samples']} samples ({success_rate:.2f}%)"
        status["success"] = status["successful_samples"] > 0
    else:
        status["message"] = "No samples were processed"
        status["success"] = False

    logger.info(status["message"])
    logger.info(f"Extraction methods used: {status['extraction_methods_used']}")

    return predictions, status
