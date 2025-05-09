"""
This utility computes the model parameter footprint.
Note that the total parameter count includes extra components such as:
• Feed-forward (MLP) layers inside each attention block,
• Learned positional encodings,
• Layer normalization layers and associated bias terms.
Our ChordNet model (via BaseTransformer) incorporates all these parts,
so the footprint computed by summing model.parameters() already accounts for them.
"""
import os
import sys
import torch
import argparse
from tabulate import tabulate

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import after fixing the path
from modules.models.Transformer.ChordNet import ChordNet
# Import BTC model
from modules.models.Transformer.btc_model import BTC_model
from modules.utils.hparams import HParams
# Import chord mapping utility if needed for n_classes default
# from modules.utils.chords import idx2voca_chord

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_in_bytes(model):
    """Get model size in bytes (assuming 32-bit float parameters)"""
    return count_parameters(model) * 4  # 4 bytes per parameter

def print_model_parameters(model):
    """Print detailed parameter counts per layer"""
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")
            total_params += param.numel()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {count_parameters(model):,}")

def scale_config(config, scale_factor, n_freq, n_group):
    """
    Scale model config parameters by given factor, matching train_student.py logic.

    Args:
        config: HParams object containing model configuration.
        scale_factor: The scaling factor (e.g., 0.5, 1.0, 2.0).
        n_freq: Number of frequency bins.
        n_group: Number of groups (fixed at 12 in ChordNet).

    Returns:
        dict: Scaled model parameters.
    """
    # Get base configuration for the model
    base_config = config.model.get('base_config', {})

    # If base_config is not specified, fall back to direct model parameters
    if not base_config:
        base_config = {
            'f_layer': config.model.get('f_layer', 3),
            # Use base f_head from config or default, train_student now uses 4
            'f_head': config.model.get('f_head', 4),
            't_layer': config.model.get('t_layer', 3),
            # Use base t_head from config or default, train_student now uses 4
            't_head': config.model.get('t_head', 4),
            'd_layer': config.model.get('d_layer', 3),
            # Use base d_head from config or default, train_student now uses 4
            'd_head': config.model.get('d_head', 4)
        }

    # Apply scale to layer parameters
    f_layer = max(1, int(round(base_config.get('f_layer', 3) * scale_factor)))
    t_layer = max(1, int(round(base_config.get('t_layer', 3) * scale_factor)))
    d_layer = max(1, int(round(base_config.get('d_layer', 3) * scale_factor)))

    # Fix t_head and d_head similar to train_student.py (now using 4)
    t_head = 4
    d_head = 4
    f_head = 2

    feature_dim = n_freq // n_group

    if feature_dim % f_head != 0:
        # Find the largest divisor of feature_dim that's <= f_head
        for h in range(f_head, 0, -1):
            if feature_dim % h == 0:
                print(f"Note: Adjusted f_head from {f_head} to {h} for scale {scale_factor}x "
                      f"to ensure compatibility with feature_dim={feature_dim}")
                f_head = h
                break

    return {
        'f_layer': f_layer,
        'f_head': f_head,
        't_layer': t_layer,
        't_head': t_head,
        'd_layer': d_layer,
        'd_head': d_head
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate footprint of the model with different scaling factors")
    parser.add_argument('--config', default=f'{project_root}/config/student_config.yaml', help='Path to student config file')
    # Add argument for BTC config
    parser.add_argument('--btc_config', default=f'{project_root}/config/btc_config.yaml', help='Path to BTC config file')
    parser.add_argument('--scales', nargs='+', type=float, default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                        help='Scaling factors to evaluate for ChordNet')
    parser.add_argument('--n_freq', type=int, default=None, help='Override n_freq value for ChordNet')
    # Default n_classes to 170, matching common usage derived from idx2voca_chord
    parser.add_argument('--n_classes', type=int, default=170,
                        help='Number of output classes (default: 170, common for idx2voca_chord)')
    parser.add_argument('--cqt', action='store_true', help='Use CQT configuration (smaller n_freq) for ChordNet')
    parser.add_argument('--stft', action='store_true', help='Use STFT configuration (larger n_freq) for ChordNet')
    # Add flag to calculate BTC footprint
    parser.add_argument('--calculate_btc', action='store_true', help='Calculate and print BTC model footprint')

    args = parser.parse_args()

    # --- ChordNet Calculation ---
    # Load student configuration
    student_config = HParams.load(args.config)
    print(f"Loaded student configuration from: {args.config}")

    # Get n_group from config
    # n_group_config = student_config.model.get('n_group', 4) # Keep for reference if needed elsewhere
    n_group_fixed = 4 # Hardcode to match ChordNet.py

    # Choose appropriate n_freq based on CQT/STFT flag or config
    if args.n_freq:
        n_freq = args.n_freq
    elif args.cqt:
        n_freq = 144  # Typical CQT size
    elif args.stft:
        n_freq = 1024 # Reasonable STFT size for parameter counting
    else:
        # Fallback to config value if available, otherwise default
        n_freq = student_config.feature.get('freq_bins', 144) # Default to 144 if not in config

    # Get n_classes from config or args
    n_classes = args.n_classes
    print(f"Note: Using n_classes = {n_classes}. Ensure this matches your training setup.")

    print("\nChordNet Parameters:")
    print(f"  n_freq: {n_freq}")
    print(f"  n_classes: {n_classes}")
    print(f"  n_group: {n_group_fixed} (fixed internally in ChordNet)") # Updated print statement

    # Prepare table data
    table_data = []

    # Generate table rows for each scale
    for scale in sorted(args.scales):
        # Scale model config using the updated function
        scaled_params = scale_config(student_config, scale, n_freq, n_group_fixed)

        # Get dropout from config, default to 0.3 like train_student.py if missing
        dropout_rate = student_config.model.get('dropout', 0.3)

        # Create model with scaled parameters, using fixed n_group
        chordnet_model = ChordNet(
            n_freq=n_freq,
            n_classes=n_classes,
            n_group=n_group_fixed, # Use the fixed n_group
            f_layer=scaled_params['f_layer'],
            f_head=scaled_params['f_head'], # Pass scaled f_head
            t_layer=scaled_params['t_layer'],
            t_head=scaled_params['t_head'], # Pass fixed t_head
            d_layer=scaled_params['d_layer'],
            d_head=scaled_params['d_head'], # Pass fixed d_head
            dropout=dropout_rate # Use dropout from config
        )

        # Count parameters
        param_count = count_parameters(chordnet_model)
        size_mb = model_size_in_bytes(chordnet_model) / (1024 * 1024)

        # Add row to table
        table_data.append([
            f"{scale}x",
            scaled_params['f_layer'],
            scaled_params['f_head'],
            scaled_params['t_layer'],
            scaled_params['t_head'],
            scaled_params['d_layer'],
            scaled_params['d_head'],
            f"{param_count:,}",
            f"{size_mb:.2f}"
        ])

    # Print table
    print("\nChordNet Scaling Comparison:")
    print("=" * 80)
    print(tabulate(table_data, headers=[
        "Scale", "F-Layers", "F-Heads", "T-Layers", "T-Heads",
        "D-Layers", "D-Heads", "Parameters", "Size (MB)"
    ], tablefmt="grid"))

    # If we're only checking a single scale, print detailed breakdown
    if len(args.scales) == 1:
        print("\nDetailed ChordNet parameter breakdown:")
        print_model_parameters(chordnet_model)

    # --- BTC Model Calculation ---
    if args.calculate_btc:
        print("\n" + "=" * 80)
        print("Calculating BTC Model Footprint...")
        try:
            # Load BTC configuration
            btc_config = HParams.load(args.btc_config)
            print(f"Loaded BTC configuration from: {args.btc_config}")

            # Instantiate BTC model using its config
            # Ensure necessary parameters are present in the config or provide defaults
            btc_model_config = btc_config.model
            # Override n_classes if provided via args, otherwise use config
            btc_model_config['num_chords'] = args.n_classes or btc_model_config.get('num_chords', 170)
            # Ensure feature_size is set (important for input layer)
            if 'feature_size' not in btc_model_config:
                 # Try to infer from ChordNet n_freq if not in btc_config
                 btc_model_config['feature_size'] = n_freq
                 print(f"Warning: 'feature_size' not found in BTC config, using n_freq: {n_freq}")

            btc_model = BTC_model(config=btc_model_config)

            # Count parameters
            btc_param_count = count_parameters(btc_model)
            btc_size_mb = model_size_in_bytes(btc_model) / (1024 * 1024)

            print("\nBTC Model Footprint:")
            print(f"  Parameters: {btc_param_count:,}")
            print(f"  Size (MB): {btc_size_mb:.2f}")

            print("\nDetailed BTC parameter breakdown:")
            print_model_parameters(btc_model)

        except FileNotFoundError:
            print(f"Error: BTC config file not found at {args.btc_config}")
        except Exception as e:
            print(f"Error calculating BTC footprint: {e}")

if __name__ == "__main__":
    main()