import os
import numpy as np
import torch
import mir_eval
import pretty_midi as pm
from utils import logger
from btc_model import *
from utils.mir_eval_modules import idx2chord, idx2voca_chord
import argparse
import warnings
import sys

warnings.filterwarnings('ignore')
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main():
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--npz_file', type=str, default='./gtzan_data.npz')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--feature_per_second', type=float, default=43.1,
                       help='Time resolution for spectrograms (frames per second)')
    parser.add_argument('--debug', action='store_true', help='Enable extensive debugging output')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Process only this many samples (for debugging)')
    parser.add_argument('--print_tensor', action='store_true', 
                        help='Print tensor content for debugging')
    parser.add_argument('--tensor_max_items', type=int, default=10,
                        help='Maximum number of tensor items to print')
    args = parser.parse_args()
    
    # Load configuration
    config = HParams.load("run_config.yaml")
    
    if args.voca is True:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        model_file = './test/btc_model_large_voca.pt'
        idx_to_chord = idx2voca_chord()
        logger.info("label type: large voca")
    else:
        model_file = './test/btc_model.pt'
        idx_to_chord = idx2chord
        logger.info("label type: Major and minor")
    
    model = BTC_model(config=config.model).to(device)
    
    # Load model
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        mean = checkpoint['mean']
        std = checkpoint['std']
        model.load_state_dict(checkpoint['model'])
        logger.info("Model restored")
        # Print model configuration - safely access keys
        logger.info(f"Model configuration:")
        logger.info(f"  Available config keys: {list(config.model.keys())}")
        
        # Get input dimension - the feature_size is likely the input dimension
        input_dim = None
        if 'feature_size' in config.model:
            input_dim = config.model['feature_size']
            logger.info(f"  Feature size: {input_dim}")
        elif 'input_dims' in config.model:
            input_dim = config.model['input_dims']
            logger.info(f"  Input dims: {input_dim}")
        elif 'input_dim' in config.model:
            input_dim = config.model['input_dim']
            logger.info(f"  Input dim: {input_dim}")
        else:
            logger.info("  Could not find input dimension in config, will try to infer from model")

        # Try to infer input dimension from model parameters if not found in config
        if input_dim is None:
            # Check if we can infer from the first layer weights
            first_layer_params = None
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    first_layer_params = param
                    logger.info(f"  Inferring input dim from {name}: {param.shape}")
                    # Linear layer weights have shape (out_features, in_features)
                    input_dim = param.shape[1]
                    logger.info(f"  Inferred input dimension: {input_dim}")
                    break
        
        logger.info(f"  Timestep: {config.model.get('timestep', 'not found')}")
        logger.info(f"  Num chords: {config.model.get('num_chords', 'not found')}")
        
        # Print mean and std information
        if hasattr(mean, 'shape'):
            logger.info(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
        else:
            logger.info(f"Mean: {mean} (scalar), Std: {std} (scalar)")
    else:
        logger.error(f"Model file not found: {model_file}")
        return
    
    # Load npz data
    try:
        data = np.load(args.npz_file, allow_pickle=True)
        logger.info(f"Successfully loaded {args.npz_file}")
        
        # Check if the loaded data is a numpy array (.npy file) or a npz archive (.npz file)
        is_npy_file = isinstance(data, np.ndarray)
        
        if is_npy_file:
            logger.info(f"Loaded a .npy file with array shape: {data.shape}")
            
            # Print content of the loaded tensor if requested
            if args.print_tensor:
                logger.info("Raw tensor content (first few elements):")
                # Flatten for easier viewing if multi-dimensional
                flat_data = data.reshape(-1)
                max_items = min(args.tensor_max_items, len(flat_data))
                logger.info(f"  Values: {flat_data[:max_items]}")
                logger.info(f"  Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}")
                logger.info(f"  Data type: {data.dtype}")
        else:
            logger.info(f"Loaded a .npz file with keys: {data.files}")
    except Exception as e:
        logger.error(f"Failed to load {args.npz_file}: {str(e)}")
        return
    
    # Examine the structure of the loaded data
    spectrograms = []
    filenames = []
    
    if is_npy_file:
        # For .npy files, use the array directly
        if len(data.shape) == 2:
            # Single spectrogram
            spectrograms = [data]
            filenames = ["sample_0"]
            logger.info(f"Using single spectrogram with shape {data.shape}")
        elif len(data.shape) == 3:
            # Multiple spectrograms or multi-channel spectrogram
            if data.shape[0] <= 10:  # Likely channels, not samples
                spectrograms = [data]  # Keep as one multi-channel spectrogram
                filenames = ["sample_0"]
                logger.info(f"Treating as one multi-channel spectrogram with shape {data.shape}")
            else:
                # Multiple spectrograms
                spectrograms = [data[i] for i in range(data.shape[0])]
                filenames = [f"sample_{i}" for i in range(data.shape[0])]
                logger.info(f"Using {len(spectrograms)} spectrograms from array")
        else:
            logger.error(f"Unexpected array shape: {data.shape}")
            return
    else:
        # Handle the case where 'gtzan' is an array of objects (.npz file)
        if 'gtzan' in data.files and isinstance(data['gtzan'], np.ndarray):
            gtzan_data = data['gtzan']
            logger.info(f"Found 'gtzan' array with {len(gtzan_data)} items")
            
            # Check the first item to understand its structure
            if len(gtzan_data) > 0:
                # Detailed inspection of sample items
                for idx in range(min(3, len(gtzan_data))):
                    sample_item = gtzan_data[idx]
                    logger.info(f"Sample {idx} type: {type(sample_item)}")
                    
                    if hasattr(sample_item, 'shape'):
                        logger.info(f"Sample {idx} shape: {sample_item.shape}")
                        logger.info(f"Sample {idx} data type: {sample_item.dtype}")
                        logger.info(f"Sample {idx} min/max values: {np.min(sample_item):.6f}/{np.max(sample_item):.6f}")
                        
                        # Check for NaN or Inf values
                        has_nan = np.isnan(sample_item).any()
                        has_inf = np.isinf(sample_item).any()
                        logger.info(f"Sample {idx} has NaN: {has_nan}, has Inf: {has_inf}")
                        
                        # If 3D, might be multi-channel spectrogram
                        if len(sample_item.shape) == 3:
                            logger.info(f"Sample {idx} appears to be multi-channel with {sample_item.shape[0]} channels")
                            
                            # Show info for each channel
                            for ch in range(min(sample_item.shape[0], 3)):
                                logger.info(f"  Channel {ch} shape: {sample_item[ch].shape}")
                                logger.info(f"  Channel {ch} min/max: {np.min(sample_item[ch])::.6f}/{np.max(sample_item[ch]):.6f}")
                
                # If items are dictionaries
                if isinstance(sample_item, dict):
                    logger.info(f"Dictionary keys: {sample_item.keys()}")
                    
                    # Look for spectrogram data in each item
                    for i, item in enumerate(gtzan_data):
                        spec_key = None
                        # Try different possible keys for spectrograms
                        for possible_key in ['spectrogram', 'spec', 'feature', 'data', 'melspectrogram', 'mel_spectrogram']:
                            if possible_key in item:
                                spec_key = possible_key
                                break
                        
                        if spec_key is not None:
                            spec_data = item[spec_key]
                            spectrograms.append(spec_data)
                            
                            # Try to get filename
                            if 'filename' in item:
                                filenames.append(item['filename'])
                            elif 'name' in item:
                                filenames.append(item['name'])
                            elif 'id' in item:
                                filenames.append(f"sample_{item['id']}")
                            else:
                                filenames.append(f"sample_{i}")
                        else:
                            logger.info(f"Could not find spectrogram data in item {i}")
                
                # If items contain arrays directly
                elif hasattr(sample_item, 'shape'):
                    logger.info(f"Sample item shape: {sample_item.shape}")
                    for i, item in enumerate(gtzan_data):
                        spectrograms.append(item)
                        filenames.append(f"sample_{i}")
            
            if not spectrograms:
                logger.error("No spectrograms found in gtzan data")
                return
    
    # If no spectrograms were found, try the original approach
    if not spectrograms:
        # Try to identify spectrograms and filenames in the data
        for key in data.files:
            item = data[key]
            shape_str = getattr(item, 'shape', 'N/A')
            logger.info(f"Key: {key}, Type: {type(item)}, Shape: {shape_str}")
            
            # Look for likely spectrogram data (2D or 3D arrays)
            if hasattr(item, 'shape') and len(item.shape) >= 2:
                if key.lower() in ['spectrograms', 'features', 'specs']:
                    spectrograms = item
                elif spectrograms is None and len(item.shape) >= 2:
                    # Use the largest array as spectrograms if no specific key matches
                    spectrograms = item
                    
            # Look for filenames or file identifiers
            if key.lower() in ['filenames', 'files', 'names', 'ids']:
                filenames = item
        
        # If we couldn't find filenames, create synthetic ones
        if spectrograms is not None and filenames is None:
            num_samples = spectrograms.shape[0] if len(spectrograms.shape) >= 3 else 1
            filenames = [f"sample_{i}" for i in range(num_samples)]
            logger.info("Using synthetic filenames")
        
        # Handle single spectrogram case
        if spectrograms is not None and len(spectrograms.shape) == 2:
            spectrograms = [spectrograms]
            filenames = [filenames[0] if isinstance(filenames, list) else "sample_0"]
    
    if not spectrograms:
        logger.error("Could not identify spectrogram data in the npz file")
        return
    
    logger.info(f"Found {len(spectrograms)} spectrograms to process")
    
    # Limit samples for debugging if requested
    if args.max_samples is not None and args.max_samples > 0:
        orig_count = len(spectrograms)
        spectrograms = spectrograms[:args.max_samples]
        filenames = filenames[:args.max_samples]
        logger.info(f"Limited processing to {len(spectrograms)} samples out of {orig_count}")
    
    # Process each spectrogram
    for i, (spectrogram, filename) in enumerate(zip(spectrograms, filenames)):
        logger.info(f"======== {i + 1} of {len(spectrograms)} in progress ========")
        
        # Enhanced spectrogram shape analysis
        logger.info(f"Original spectrogram shape: {spectrogram.shape}")
        logger.info(f"Spectrogram dtype: {spectrogram.dtype}")
        logger.info(f"Spectrogram min/max values: {np.min(spectrogram):.6f}/{np.max(spectrogram):.6f}")
        
        # Handle multi-channel spectrograms if detected
        if len(spectrogram.shape) == 3:
            logger.info(f"Multi-channel spectrogram detected with {spectrogram.shape[0]} channels")
            
            # Average all channels instead of using just the first one
            logger.info("Averaging all channels of the spectrogram")
            spectrogram = np.mean(spectrogram, axis=0)
            
            # Important: The data appears to be in (features, time) format
            # We need to transpose to get (time, features) which is what the model expects
            logger.info(f"Before transpose - shape: {spectrogram.shape}")
            spectrogram = spectrogram.T  # Transpose to convert from (features, time) to (time, features)
            logger.info(f"After transpose - shape: {spectrogram.shape}")
            
            logger.info(f"After channel averaging and transpose, shape: {spectrogram.shape}")
        
        # The model expects input with feature dimension matching the inferred dimensions
        expected_feature_dim = input_dim if input_dim is not None else spectrogram.shape[1]
        
        # More detailed analysis of dimensions
        logger.info(f"Model expects {expected_feature_dim} features, spectrogram has {spectrogram.shape[1]} features")
        
        # Handle different spectrogram shapes
        if spectrogram.shape[1] != expected_feature_dim:
            logger.info(f"Spectrogram feature dimension {spectrogram.shape[1]} doesn't match expected {expected_feature_dim}")
            
            # Option 1: If spectrogram is larger, truncate
            if spectrogram.shape[1] > expected_feature_dim:
                logger.info(f"Truncating spectrogram from {spectrogram.shape[1]} to {expected_feature_dim} features")
                spectrogram = spectrogram[:, :expected_feature_dim]
            
            # Option 2: If spectrogram is smaller, pad with zeros
            else:
                logger.info(f"Padding spectrogram from {spectrogram.shape[1]} to {expected_feature_dim} features")
                padding = np.zeros((spectrogram.shape[0], expected_feature_dim - spectrogram.shape[1]))
                spectrogram = np.hstack((spectrogram, padding))
        
        # We don't need to transpose if the model expects (time, feature) format
        # Check the model's expected input format and adjust accordingly
        feature = spectrogram  # No transpose
        
        # Normalize with mean and std
        logger.info(f"Before normalization - min: {np.min(feature):.6f}, max: {np.max(feature):.6f}")
        
        # Handle different types of mean and std (scalar vs array)
        if hasattr(mean, 'shape') and mean.shape == feature.shape[1:]:
            # Mean and std are arrays matching feature dimensions
            feature = (feature - mean) / std
        else:
            # Mean and std are scalars
            feature = (feature - mean) / std
            
        logger.info(f"After normalization - min: {np.min(feature):.6f}, max: {np.max(feature):.6f}")
        
        logger.info(f"Processed feature shape: {feature.shape}")
        
        # Print processed feature tensor if requested
        if args.print_tensor:
            logger.info("Processed feature tensor (first few rows and columns):")
            max_rows = min(5, feature.shape[0])
            max_cols = min(10, feature.shape[1])
            logger.info(f"  Values (subset {max_rows}x{max_cols}):")
            for r in range(max_rows):
                logger.info(f"    Row {r}: {feature[r, :max_cols]}")
            logger.info(f"  Statistics - Min: {np.min(feature)}, Max: {np.max(feature)}, Mean: {np.mean(feature)}")
        
        # First assign feature_per_second, then calculate audio_duration
        n_timestep = config.model['timestep']
        feature_per_second = args.feature_per_second
        
        # Calculate and log the audio duration
        audio_duration = feature.shape[0] / feature_per_second
        logger.info(f"Estimated audio duration: {audio_duration:.2f} seconds")
        
        # Reshape feature to match the model's expected input format
        # The model expects batches of sequences of shape [batch, seq_len, features]
        
        # Pad the feature to be a multiple of n_timestep
        num_pad = 0
        if feature.shape[0] % n_timestep != 0:
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep
        
        logger.info(f"Padded feature shape: {feature.shape}, num_instances: {num_instance}")
        
        start_time = 0.0
        lines = []
        with torch.no_grad():
            model.eval()
            predictions = []
            
            # Detailed network analysis
            if args.debug and i == 0:  # Only for first sample when debugging
                logger.info("Model layer information:")
                for name, param in model.named_parameters():
                    logger.info(f"  Layer: {name}, Shape: {param.shape}")
            
            # Process in smaller batches if needed
            for t in range(num_instance):
                # Get the current timestep chunk
                chunk = feature[n_timestep * t:n_timestep * (t + 1), :]
                
                # Print detailed info for first few chunks
                if t < 3:
                    logger.info(f"Chunk {t} shape: {chunk.shape}")
                
                # Add batch dimension
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
                
                if t == 0:  # Print tensor info for first chunk
                    logger.info(f"First chunk tensor shape: {chunk_tensor.shape}")
                    logger.info(f"Expected by model.self_attn_layers: [batch_size, sequence_length, features]")
                
                try:
                    # Try reshaping if necessary based on model architecture
                    # Detect if we need to reshape based on model's expected input
                    if hasattr(model.self_attn_layers, 'embedding_proj'):
                        proj_weight = model.self_attn_layers.embedding_proj.weight
                        logger.info(f"Embedding projection weight shape: {proj_weight.shape}")
                        expected_in_features = proj_weight.shape[1]
                        
                        if chunk_tensor.shape[-1] != expected_in_features:
                            logger.info(f"Chunk tensor feature dimension {chunk_tensor.shape[-1]} doesn't match "
                                     f"model's expected input {expected_in_features}. Attempting reshape.")
                            
                            # Try to reshape or modify the tensor to match expected dimensions
                            # This depends on model architecture - we might need to transpose
                            if t == 0:  # Only log detailed info for first chunk
                                logger.info(f"Model expects input shape compatible with {expected_in_features} features")
                                logger.info(f"Current shape: {chunk_tensor.shape}")
                    
                    # Forward pass with detailed error catching
                    self_attn_output, _ = model.self_attn_layers(chunk_tensor)
                    if t == 0:  # Print output shape for first chunk
                        logger.info(f"Self attention output shape: {self_attn_output.shape}")
                    
                    prediction, _ = model.output_layer(self_attn_output)
                    
                    # Log the raw prediction shape before squeezing
                    if t == 0:
                        logger.info(f"Raw prediction shape: {prediction.shape}")
                    
                    prediction = prediction.squeeze()
                    
                    # Log the prediction shape after squeezing
                    if t == 0:
                        logger.info(f"Squeezed prediction shape: {prediction.shape}")
                        
                        # Print prediction values if requested
                        if args.print_tensor:
                            if prediction.dim() == 1:
                                # For 1D predictions
                                logger.info("Prediction values (first few):")
                                max_items = min(args.tensor_max_items, prediction.shape[0])
                                logger.info(f"  {prediction[:max_items].cpu().numpy()}")
                                # Print top 3 predicted classes
                                values, indices = torch.topk(prediction, min(3, prediction.shape[0]))
                                logger.info(f"  Top predictions: {[(idx_to_chord[idx.item()], val.item()) for idx, val in zip(indices, values)]}")
                            else:
                                # For 2D predictions, print first few rows
                                logger.info("Prediction values (first few rows):")
                                max_rows = min(3, prediction.shape[0])
                                max_cols = min(5, prediction.shape[1])
                                for r in range(max_rows):
                                    logger.info(f"  Row {r}: {prediction[r, :max_cols].cpu().numpy()}")
                                    # Print top class for this row
                                    idx = prediction[r].argmax().item()
                                    logger.info(f"  Top prediction for row {r}: {idx_to_chord[idx]}")
                    
                    predictions.append(prediction)
                    
                    # Log prediction statistics for debugging
                    if t < 3 or t == num_instance - 1:
                        # Handle both 1D and 2D tensor cases
                        if prediction.dim() == 1:
                            # For 1D tensor (single prediction), get the max index
                            unique_chords = torch.unique(prediction.argmax().unsqueeze(0))
                        else:
                            # For 2D tensor (multiple predictions), get argmax along class dimension
                            unique_chords = torch.unique(torch.argmax(prediction, dim=1))
                            
                        logger.info(f"Prediction tensor has {prediction.dim()} dimensions")
                        logger.info(f"Chunk {t} unique chord predictions: {len(unique_chords)}")
                        if len(unique_chords) <= 5:  # If few unique predictions, show them
                            chord_ids = [idx.item() for idx in unique_chords]
                            
                            # Add safety check for out-of-range indices
                            valid_chord_ids = []
                            for idx in chord_ids:
                                if idx >= 0 and idx < len(idx_to_chord):
                                    valid_chord_ids.append(idx)
                                else:
                                    logger.info(f"Invalid chord index detected: {idx} (max valid index: {len(idx_to_chord)-1})")
                                    # Replace with safe default (0 = "N" typically)
                                    valid_chord_ids.append(0)
                            
                            chord_names = [idx_to_chord[idx] for idx in valid_chord_ids]
                            logger.info(f"  Chord predictions: {chord_names}")
                            
                            # Check for prevalence of "N" predictions
                            n_chord_idx = [i for i, name in enumerate(chord_names) if name == "N"]
                            if n_chord_idx:
                                if prediction.dim() == 1:
                                    # For 1D tensor, just check if the argmax equals the N chord index
                                    n_count = 1 if prediction.argmax().item() == valid_chord_ids[n_chord_idx[0]] else 0
                                    n_percent = 100.0 if n_count == 1 else 0.0
                                else:
                                    # For 2D tensor
                                    n_count = (torch.argmax(prediction, dim=1) == valid_chord_ids[n_chord_idx[0]]).sum().item()
                                    n_percent = n_count / prediction.shape[0] * 100
                                logger.info(f"  'N' chord prediction: {n_count}/{prediction.shape[0] if prediction.dim() > 1 else 1} frames ({n_percent:.1f}%)")
                
                except RuntimeError as e:
                    logger.error(f"Error during model inference: {e}")
                    logger.error(f"Chunk shape: {chunk.shape}, tensor shape: {chunk_tensor.shape}")
                    
                    # Print model's layer output shapes to diagnose where the error occurs
                    if t == 0:
                        try:
                            # Get first layer's output shape
                            if hasattr(model.self_attn_layers, 'embedding_proj'):
                                with torch.no_grad():
                                    first_layer_output = model.self_attn_layers.embedding_proj(chunk_tensor)
                                    logger.info(f"First layer output shape: {first_layer_output.shape}")
                        except Exception as inner_e:
                            logger.error(f"Error getting first layer output: {inner_e}")
                    
                    # Exit after error if debugging
                    if args.debug:
                        logger.error("Exiting due to error in debug mode")
                        sys.exit(1)
                    continue
            
            # Create chord annotations from predictions
            prev_chord = None
            chord_counts = {}  # For debugging
            
            # Check if we have any predictions
            if not predictions:
                logger.error("No predictions generated!")
                continue
                
            for t, prediction_batch in enumerate(predictions):
                # Handle both 1D and multi-dim predictions
                if prediction_batch.dim() == 0:  # scalar
                    chord_idx = prediction_batch.item()
                    # Safety check for chord index
                    if chord_idx < 0 or chord_idx >= len(idx_to_chord):
                        logger.info(f"Invalid chord index {chord_idx} detected (max valid: {len(idx_to_chord)-1}). Using 0 instead.")
                        chord_idx = 0  # Use "N" (no chord) as fallback
                    chord_name = idx_to_chord[chord_idx]
                    chord_counts[chord_name] = chord_counts.get(chord_name, 0) + 1
                    if t == 0:
                        prev_chord = chord_idx
                elif prediction_batch.dim() == 1:  # vector (single prediction)
                    chord_idx = prediction_batch.argmax().item()
                    # Safety check for chord index
                    if chord_idx < 0 or chord_idx >= len(idx_to_chord):
                        logger.info(f"Invalid chord index {chord_idx} detected (max valid: {len(idx_to_chord)-1}). Using 0 instead.")
                        chord_idx = 0  # Use "N" (no chord) as fallback
                    chord_name = idx_to_chord[chord_idx]
                    chord_counts[chord_name] = chord_counts.get(chord_name, 0) + 1
                    if t == 0:
                        prev_chord = chord_idx
                    else:
                        if chord_idx != prev_chord:
                            # Safety check for prev_chord index
                            if prev_chord < 0 or prev_chord >= len(idx_to_chord):
                                logger.info(f"Invalid previous chord index {prev_chord} detected (max valid: {len(idx_to_chord)-1}). Using 0 instead.")
                                prev_chord = 0
                            lines.append(
                                '%.3f %.3f %s\n' % (start_time, t / feature_per_second, idx_to_chord[prev_chord]))
                            start_time = t / feature_per_second
                            prev_chord = chord_idx
                else:  # 2D or higher
                    for j in range(min(n_timestep, prediction_batch.shape[0])):
                        # Get chord prediction
                        if prediction_batch.dim() == 2:
                            chord_idx = prediction_batch[j].argmax().item()
                        else:
                            # Handle unexpected tensor dimensions
                            chord_idx = prediction_batch[j].flatten().argmax().item()
                        
                        # Safety check for out-of-range indices
                        if chord_idx < 0 or chord_idx >= len(idx_to_chord):
                            logger.info(f"Invalid chord index {chord_idx} detected (max valid: {len(idx_to_chord)-1}). Using 0 instead.")
                            chord_idx = 0  # Use "N" (no chord) as fallback
                            
                        chord_name = idx_to_chord[chord_idx]
                        chord_counts[chord_name] = chord_counts.get(chord_name, 0) + 1
                        
                        if t == 0 and j == 0:
                            prev_chord = chord_idx
                            continue
                        
                        if chord_idx != prev_chord:
                            lines.append(
                                '%.3f %.3f %s\n' % (start_time, (n_timestep * t + j) / feature_per_second, idx_to_chord[prev_chord]))
                            start_time = (n_timestep * t + j) / feature_per_second
                            prev_chord = chord_idx

                # Handle the last segment
                if t == len(predictions) - 1:
                    # Safety check for prev_chord before writing last segment
                    if prev_chord < 0 or prev_chord >= len(idx_to_chord):
                        logger.info(f"Invalid last chord index {prev_chord} detected (max valid: {len(idx_to_chord)-1}). Using 0 instead.")
                        prev_chord = 0
                        
                    if prediction_batch.dim() <= 1:
                        # For scalar or vector predictions
                        last_time = (t + 1) / feature_per_second
                    else:
                        # For 2D predictions
                        last_frame = min(n_timestep, prediction_batch.shape[0]) - 1
                        last_time = (n_timestep * t + last_frame + 1) / feature_per_second
                    
                    lines.append('%.3f %.3f %s\n' % (start_time, last_time, idx_to_chord[prev_chord]))

            # Log chord distribution for debugging
            logger.info("Chord distribution in predictions:")
            for chord, count in sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                percent = count / sum(chord_counts.values()) * 100
                logger.info(f"  {chord}: {count} frames ({percent:.1f}%)")
        
        # If we ended up with no lines (all frames were the same chord), add a single entry
        if not lines and predictions:
            try:
                first_pred = predictions[0][0].argmax().item()
                # Safety check for out-of-range indices
                if first_pred < 0 or first_pred >= len(idx_to_chord):
                    logger.info(f"Invalid chord index {first_pred} detected (max valid: {len(idx_to_chord)-1}). Using 0 instead.")
                    first_pred = 0  # Use "N" (no chord) as fallback
                chord = idx_to_chord[first_pred]
                lines.append(f"0.000 {audio_duration:.3f} {chord}\n")
                logger.info(f"Single chord detected for entire file: {chord}")
            except Exception as e:
                logger.error(f"Error creating default chord annotation: {str(e)}")
                lines.append(f"0.000 {audio_duration:.3f} N\n")
                logger.info("Using N (no chord) as fallback for entire file")

        # Create save directory if it doesn't exist
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        # Save as lab file
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        filename = str(filename).replace('/', '_')
        save_path = os.path.join(args.save_dir, f"{filename}.lab")
        
        with open(save_path, 'w') as f:
            for line in lines:
                f.write(line)
        
        logger.info(f"Label file saved: {save_path}")
        
        # Convert lab file to midi file
        try:
            starts, ends, pitchs = list(), list(), list()
            
            intervals, chords = mir_eval.io.load_labeled_intervals(save_path)
            for p in range(12):
                for i, (interval, chord) in enumerate(zip(intervals, chords)):
                    root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
                    tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
                    if i == 0:
                        start_time = interval[0]
                        label = tmp_label
                        continue
                    if tmp_label != label:
                        if label == 1.0:
                            starts.append(start_time), ends.append(interval[0]), pitchs.append(p + 48)
                        start_time = interval[0]
                        label = tmp_label
                    if i == (len(intervals) - 1): 
                        if label == 1.0:
                            starts.append(start_time), ends.append(interval[1]), pitchs.append(p + 48)
            
            midi = pm.PrettyMIDI()
            instrument = pm.Instrument(program=0)
            
            for start, end, pitch in zip(starts, ends, pitchs):
                pm_note = pm.Note(velocity=120, pitch=pitch, start=start, end=end)
                instrument.notes.append(pm_note)
            
            midi.instruments.append(instrument)
            midi_path = save_path.replace('.lab', '.midi')
            midi.write(midi_path)
            
            logger.info(f"MIDI file saved: {midi_path}")
        except Exception as e:
            logger.error(f"Error processing lab file to MIDI: {str(e)}")

if __name__ == "__main__":
    main()
