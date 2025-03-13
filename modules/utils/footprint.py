import torch
import sys
import os
import yaml
import argparse

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

def compare_models(student_model, teacher_params):
    """Compare student model parameters with teacher model parameters"""
    print("\nModel Comparison - Student vs Teacher:")
    print("-" * 80)
    print(f"{'Parameter':<30} {'Student':<20} {'Teacher':<20}")
    print("-" * 80)
    
    # Get properties directly from the student model
    student_params = {
        'n_freq': student_model.transformer.encoder_f[0].n_freq if hasattr(student_model, 'transformer') else 'N/A',
        'n_group': student_model.transformer.encoder_f[0].n_group if hasattr(student_model, 'transformer') else 'N/A',
        'f_layer': len(student_model.transformer.encoder_f[0].attn_layer) if hasattr(student_model, 'transformer') else 'N/A',
        'f_head': student_model.transformer.encoder_f[0].attn_layer[0].num_heads if hasattr(student_model, 'transformer') else 'N/A',
        't_layer': len(student_model.transformer.encoder_t[0].attn_layer) if hasattr(student_model, 'transformer') else 'N/A', 
        't_head': student_model.transformer.encoder_t[0].attn_layer[0].num_heads if hasattr(student_model, 'transformer') else 'N/A',
        'd_layer': len(student_model.transformer.decoder.attn_layer1) if hasattr(student_model, 'transformer') else 'N/A',
        'd_head': student_model.transformer.decoder.attn_layer1[0].num_heads if hasattr(student_model, 'transformer') else 'N/A',
        'n_classes': student_model.fc.out_features if hasattr(student_model, 'fc') else 'N/A',
        'Total Params': f"{count_parameters(student_model):,}"
    }
    
    # Print comparison
    for param in student_params:
        print(f"{param:<30} {str(student_params[param]):<20} {str(teacher_params.get(param, 'N/A')):<20}")
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description='Calculate model footprint')
    parser.add_argument('--n_freq', type=int, help='Number of frequency bins', default=144)
    parser.add_argument('--n_classes', type=int, help='Number of chord classes', default=170)
    args = parser.parse_args()
    
    # Load student config
    config_path = os.path.join(project_root, 'config', 'student_config.yaml')
    config = HParams.load(config_path)
    
    print(f"Loaded student configuration from: {config_path}")
    
    # Instantiate student model with parameters from config
    print("Creating student model with configuration:")
    print(f"  n_group: {config.model['n_group']}")
    print(f"  f_layer: {config.model['f_layer']}")
    print(f"  f_head: {config.model['f_head']}")
    print(f"  t_layer: {config.model['t_layer']}")
    print(f"  t_head: {config.model['t_head']}")
    print(f"  d_layer: {config.model['d_layer']}")
    print(f"  d_head: {config.model['d_head']}")
    print(f"  dropout: {config.model['dropout']}")

    # Get a valid n_freq value based on n_group
    n_freq = get_valid_n_freq(config.model['n_group'], args.n_freq)
    n_classes = args.n_classes
    
    print(f"  n_freq: {n_freq} (must be divisible by n_group)")
    print(f"  n_classes: {n_classes}")
    
    # Create the model
    model = ChordNet(n_freq=n_freq, 
                     n_classes=n_classes, 
                     n_group=config.model['n_group'],
                     f_layer=config.model['f_layer'], 
                     f_head=config.model['f_head'], 
                     t_layer=config.model['t_layer'], 
                     t_head=config.model['t_head'], 
                     d_layer=config.model['d_layer'], 
                     d_head=config.model['d_head'], 
                     dropout=config.model['dropout'])
    
    total_params = count_parameters(model)
    size_bytes = model_size_in_bytes(model)
    size_mb = size_bytes / (1024 ** 2)
    
    print(f"\nStudent Model Summary")
    print(f"===================")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size: {size_mb:.2f} MB")
    
    # Print detailed parameter breakdown
    print_model_parameters(model)
    
    # Define teacher model parameters for comparison
    teacher_params = {
        'n_freq': 144,
        'n_group': 'N/A',  # Teacher doesn't use groups as described
        'f_layer': 'N/A',  # Teacher uses bidirectional attention
        'f_head': 'N/A',   # Different architecture
        't_layer': 'N/A',  # Different architecture 
        't_head': 'N/A',   # Different architecture
        'd_layer': 'N/A',  # Different architecture
        'd_head': 'N/A',   # Different architecture
        'n_classes': 170,
        'Total Params': 'N/A'  # We don't have exact count
    }
    
    # Compare student model with teacher model
    compare_models(model, teacher_params)

if __name__ == "__main__":
    main()