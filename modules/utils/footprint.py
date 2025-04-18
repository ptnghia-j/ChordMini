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
from modules.utils.hparams import HParams

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

def get_valid_n_freq(n_group, suggested_n_freq=None):
    """Get a valid n_freq that's divisible by n_group"""
    if suggested_n_freq is not None and suggested_n_freq % n_group == 0:
        return suggested_n_freq
    
    # Common CQT values
    if n_group <= 12:
        return 144  # Common for CQT (12 bins per octave * 12 octaves)
    
    # For STFT, choose a value divisible by n_group
    # but don't go extremely high as we're just counting parameters
    return 1024  # A more reasonable STFT size for parameter counting

def scale_config(config, scale_factor):
    """Scale model config parameters by given factor"""
    # Get base configuration for the model
    base_config = config.model.get('base_config', {})
    
    # If base_config is not specified, fall back to direct model parameters
    if not base_config:
        base_config = {
            'f_layer': config.model.get('f_layer', 3),
            'f_head': config.model.get('f_head', 6),
            't_layer': config.model.get('t_layer', 3),
            't_head': config.model.get('t_head', 6),
            'd_layer': config.model.get('d_layer', 3),
            'd_head': config.model.get('d_head', 6)
        }
    
    # Apply scale to model parameters
    f_layer = max(1, int(round(base_config['f_layer'] * scale_factor)))
    f_head = max(1, int(round(base_config['f_head'] * scale_factor)))
    t_layer = max(1, int(round(base_config['t_layer'] * scale_factor)))
    t_head = max(1, int(round(base_config['t_head'] * scale_factor)))
    d_layer = max(1, int(round(base_config['d_layer'] * scale_factor)))
    d_head = max(1, int(round(base_config['d_head'] * scale_factor)))
    
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
    parser.add_argument('--config', default=f'{project_root}/config/student_config.yaml', help='Path to config file')
    parser.add_argument('--scales', nargs='+', type=float, default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0], 
                        help='Scaling factors to evaluate')
    parser.add_argument('--n_freq', type=int, default=None, help='Override n_freq value')
    parser.add_argument('--n_classes', type=int, default=170, help='Number of output classes')
    parser.add_argument('--cqt', action='store_true', help='Use CQT configuration (smaller n_freq)')
    parser.add_argument('--stft', action='store_true', help='Use STFT configuration (larger n_freq)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = HParams.load(args.config)
    print(f"Loaded student configuration from: {args.config}")
    
    # Get n_group from config
    n_group = config.model.get('n_group', 4)
    
    # Choose appropriate n_freq based on CQT/STFT flag
    if args.n_freq:
        n_freq = args.n_freq
    elif args.cqt:
        n_freq = 144  # Typical CQT size
    elif args.stft:
        n_freq = 1024  # Reasonable STFT size for parameter counting
    else:
        n_freq = get_valid_n_freq(n_group)
    
    # Get n_classes from config or args
    n_classes = args.n_classes
    
    print("Using parameters:")
    print(f"  n_freq: {n_freq} (must be divisible by n_group)")
    print(f"  n_classes: {n_classes}")
    print(f"  n_group: {n_group}")
    
    # Prepare table data
    table_data = []
    
    # Generate table rows for each scale
    for scale in sorted(args.scales):
        # Scale model config
        scaled_params = scale_config(config, scale)
        
        # Create model with scaled parameters
        model = ChordNet(
            n_freq=n_freq,
            n_classes=n_classes,
            n_group=n_group,
            f_layer=scaled_params['f_layer'],
            f_head=scaled_params['f_head'],
            t_layer=scaled_params['t_layer'],
            t_head=scaled_params['t_head'],
            d_layer=scaled_params['d_layer'],
            d_head=scaled_params['d_head'],
            dropout=config.model.get('dropout', 0.1)
        )
        
        # Count parameters
        param_count = count_parameters(model)
        size_mb = model_size_in_bytes(model) / (1024 * 1024)
        
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
    print("\nModel Scaling Comparison:")
    print("=" * 80)
    print(tabulate(table_data, headers=[
        "Scale", "F-Layers", "F-Heads", "T-Layers", "T-Heads", 
        "D-Layers", "D-Heads", "Parameters", "Size (MB)"
    ], tablefmt="grid"))
    
    # If we're only checking a single scale, print detailed breakdown
    if len(args.scales) == 1:
        print("\nDetailed parameter breakdown:")
        print_model_parameters(model)

if __name__ == "__main__":
    main()