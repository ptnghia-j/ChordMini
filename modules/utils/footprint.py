"""
This utility computes the model parameter footprint.
Note that the total parameter count includes extra components such as:
• Feed-forward (MLP) layers inside each attention block,
• Learned positional encodings,
• Layer normalization layers and associated bias terms.
Our ChordNet model (via BaseTransformer) incorporates all these parts,
so the footprint computed by summing model.parameters() already accounts for them.
"""
import torch
import sys
import os
import yaml
import argparse
import copy
from tabulate import tabulate

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import after fixing the path
from modules.models.Transformer.ChordNet import ChordNet
from modules.utils.hparams import HParams

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_in_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model):
    """Print detailed parameters per layer of the model"""
    print("\nDetailed Model Parameters:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Shape':<20} {'Parameters':<10}")
    print("-" * 80)
    
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            total_params += param_count
            print(f"{name:<40} {str(list(parameter.shape)):<20} {param_count:<10,}")
    
    print("-" * 80)
    print(f"{'Total':<40} {'':<20} {total_params:<10,}")
    print("-" * 80)

def get_valid_n_freq(n_group, suggested_n_freq=None):
    """Find a valid n_freq that is divisible by n_group.
    If suggested_n_freq is provided, return the closest valid value to it.
    Otherwise, return a reasonable default that's divisible by n_group.
    """
    if suggested_n_freq is not None:
        # Find the closest multiple of n_group
        return ((suggested_n_freq + n_group // 2) // n_group) * n_group
    else:
        # Default to 144 for matching teacher model
        # Check if it's divisible by n_group, if not find closest valid value
        if 144 % n_group == 0:
            return 144
        else:
            return ((144 + n_group // 2) // n_group) * n_group

def scale_config(config, scale_factor):
    """Scale model configuration by a given factor."""
    scaled_config = copy.deepcopy(config)
    
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
    scaled_config.model['f_layer'] = max(1, int(round(base_config['f_layer'] * scale_factor)))
    scaled_config.model['f_head'] = max(1, int(round(base_config['f_head'] * scale_factor)))
    scaled_config.model['t_layer'] = max(1, int(round(base_config['t_layer'] * scale_factor)))
    scaled_config.model['t_head'] = max(1, int(round(base_config['t_head'] * scale_factor)))
    scaled_config.model['d_layer'] = max(1, int(round(base_config['d_layer'] * scale_factor)))
    scaled_config.model['d_head'] = max(1, int(round(base_config['d_head'] * scale_factor)))
    
    return scaled_config

def create_model_with_scale(config, scale_factor, n_freq, n_classes, verbose=False):
    """Create a model with the given scale factor and return its details."""
    scaled_config = scale_config(config, scale_factor)
    
    if verbose:
        print(f"\nCreating model with scale factor: {scale_factor}x")
        print(f"  f_layer: {scaled_config.model['f_layer']}")
        print(f"  f_head: {scaled_config.model['f_head']}")
        print(f"  t_layer: {scaled_config.model['t_layer']}")
        print(f"  t_head: {scaled_config.model['t_head']}")
        print(f"  d_layer: {scaled_config.model['d_layer']}")
        print(f"  d_head: {scaled_config.model['d_head']}")
    
    # Create the model
    model = ChordNet(
        n_freq=n_freq, 
        n_classes=n_classes, 
        n_group=scaled_config.model['n_group'],
        f_layer=scaled_config.model['f_layer'], 
        f_head=scaled_config.model['f_head'], 
        t_layer=scaled_config.model['t_layer'], 
        t_head=scaled_config.model['t_head'], 
        d_layer=scaled_config.model['d_layer'], 
        d_head=scaled_config.model['d_head'], 
        dropout=scaled_config.model['dropout']
    )
    
    total_params = count_parameters(model)
    size_bytes = model_size_in_bytes(model)
    size_mb = size_bytes / (1024 ** 2)
    
    return {
        'scale': scale_factor,
        'f_layer': scaled_config.model['f_layer'],
        'f_head': scaled_config.model['f_head'],
        't_layer': scaled_config.model['t_layer'],
        't_head': scaled_config.model['t_head'],
        'd_layer': scaled_config.model['d_layer'],
        'd_head': scaled_config.model['d_head'],
        'params': total_params,
        'size_mb': size_mb,
        'model': model
    }

def compare_scaled_models(model_results):
    """Print a comparison table of models with different scaling factors."""
    print("\nModel Scaling Comparison:")
    print("=" * 80)
    
    # Prepare table rows
    headers = ["Scale", "F-Layers", "F-Heads", "T-Layers", "T-Heads", "D-Layers", "D-Heads", "Parameters", "Size (MB)"]
    rows = []
    
    for result in model_results:
        # Format parameter count with commas
        params_formatted = f"{result['params']:,}"
        rows.append([
            f"{result['scale']}x",
            result['f_layer'],
            result['f_head'],
            result['t_layer'],
            result['t_head'],
            result['d_layer'],
            result['d_head'],
            params_formatted,
            f"{result['size_mb']:.2f}"
        ])
    
    # Print the table with tabulate
    try:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        # If tabulate is not available, fall back to simple formatting
        print("\t".join(headers))
        print("-" * 80)
        for row in rows:
            print("\t".join(str(cell) for cell in row))

def compute_model_footprint(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    parser = argparse.ArgumentParser(description='Calculate model footprint for different scaling factors')
    parser.add_argument('--n_freq', type=int, help='Number of frequency bins', default=144)
    parser.add_argument('--n_classes', type=int, help='Number of chord classes', default=170)
    parser.add_argument('--scales', type=float, nargs='+', default=[0.5, 1.0, 2.0], 
                       help='Scaling factors for model capacity (default: 0.5, 1.0, 2.0)')
    parser.add_argument('--detail', type=float, default=None,
                       help='Show detailed parameters for a specific scale (e.g., 1.0)')
    args = parser.parse_args()
    
    # Load student config
    config_path = os.path.join(project_root, 'config', 'student_config.yaml')
    try:
        config = HParams.load(config_path)
        print(f"Loaded student configuration from: {config_path}")
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        config = HParams({
            'model': {
                'n_group': 4,
                'f_layer': 3,
                'f_head': 6,
                't_layer': 3,
                't_head': 6,
                'd_layer': 3,
                'd_head': 6,
                'dropout': 0.3
            }
        })
        print("Using default configuration")
    
    # Get a valid n_freq value based on n_group
    n_freq = get_valid_n_freq(config.model['n_group'], args.n_freq)
    n_classes = args.n_classes
    
    print(f"Using parameters:")
    print(f"  n_freq: {n_freq} (must be divisible by n_group)")
    print(f"  n_classes: {n_classes}")
    print(f"  n_group: {config.model['n_group']}")
    
    # Create models for each scaling factor
    model_results = []
    for scale in args.scales:
        result = create_model_with_scale(config, scale, n_freq, n_classes, verbose=(scale == args.detail))
        model_results.append(result)
        
        # If this is the detail scale, print detailed parameters
        if args.detail is not None and scale == args.detail:
            print(f"\nDetailed analysis for {scale}x scale model:")
            print(f"Total trainable parameters: {result['params']:,}")
            print(f"Model size: {result['size_mb']:.2f} MB")
            print_model_parameters(result['model'])
    
    # Compare all scaled models
    compare_scaled_models(model_results)
    
    # Print recommendations based on available parameters
    baseline_params = next((r['params'] for r in model_results if r['scale'] == 1.0), None)
    if baseline_params:
        print("\nDeployment Recommendations:")
        print("=" * 80)
        print(f"- Baseline model (1.0x): {baseline_params:,} parameters, suitable for standard deployment")
        
        half_params = next((r['params'] for r in model_results if r['scale'] == 0.5), None)
        if half_params:
            print(f"- Half-scale model (0.5x): {half_params:,} parameters ({half_params/baseline_params:.1%} of baseline)")
            print("  Recommended for memory-constrained environments")
        
        double_params = next((r['params'] for r in model_results if r['scale'] == 2.0), None)
        if double_params:
            print(f"- Double-scale model (2.0x): {double_params:,} parameters ({double_params/baseline_params:.1%} of baseline)")
            print("  Recommended for high-performance environments with sufficient resources")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Instantiate our model (default configuration)
    model = ChordNet()
    total, trainable = compute_model_footprint(model)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")
    # The printed footprint includes parameters from all components:
    # feed-forward layers, positional encodings, layer normalization, and biases.