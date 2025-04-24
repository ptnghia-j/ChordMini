"""
Test script to verify model scaling implementation.
"""
import os
import sys
import torch
from modules.models.Transformer.ChordNet import ChordNet
from modules.utils.hparams import HParams

def test_model_scaling(scale_factor):
    """Test model scaling with the given scale factor."""
    print(f"\n=== Testing model with scale_factor={scale_factor} ===")
    
    # Load config
    config = HParams.load('./config/student_config.yaml')
    
    # Get base configuration
    base_config = config.model.get('base_config', {})
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
    f_layer = max(1, int(round(base_config.get('f_layer', 3) * scale_factor)))
    f_head = max(1, int(round(base_config.get('f_head', 6) * scale_factor)))
    t_layer = max(1, int(round(base_config.get('t_layer', 3) * scale_factor)))
    t_head = max(1, int(round(base_config.get('t_head', 6) * scale_factor)))
    d_layer = max(1, int(round(base_config.get('d_layer', 3) * scale_factor)))
    d_head = max(1, int(round(base_config.get('d_head', 6) * scale_factor)))
    
    # Fixed parameters
    n_freq = 144
    n_classes = 170
    n_group = 12
    feature_dim = n_freq // n_group
    
    # Ensure f_head is compatible with feature_dim
    if feature_dim % f_head != 0:
        original_f_head = f_head
        for h in range(f_head, 0, -1):
            if feature_dim % h == 0:
                f_head = h
                print(f"Adjusted f_head from {original_f_head} to {f_head} to ensure compatibility with feature_dim={feature_dim}")
                break
    
    print(f"Scaled parameters:")
    print(f"  Frequency encoder: {f_layer} layers, {f_head} heads")
    print(f"  Time encoder: {t_layer} layers, {t_head} heads")
    print(f"  Decoder: {d_layer} layers, {d_head} heads")
    
    # Create model
    try:
        model = ChordNet(
            n_freq=n_freq,
            n_classes=n_classes,
            n_group=n_group,
            f_layer=f_layer,
            f_head=f_head,
            t_layer=t_layer,
            t_head=t_head,
            d_layer=d_layer,
            d_head=d_head,
            dropout=0.3
        )
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per parameter
        
        print(f"Model created successfully!")
        print(f"Total parameters: {param_count:,}")
        print(f"Model size: {size_mb:.2f} MB")
        
        return True
    except Exception as e:
        print(f"Error creating model: {e}")
        return False

if __name__ == "__main__":
    # Test with different scaling factors
    scales = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    success_count = 0
    for scale in scales:
        if test_model_scaling(scale):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(scales)} tests passed")
