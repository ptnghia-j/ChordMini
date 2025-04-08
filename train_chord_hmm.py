#!/usr/bin/env python3

"""
Train HMM for chord progression modeling using real-world labeled data.
Uses a pretrained chord recognition model for emission probabilities
and trains the transition probabilities between chords.
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR
from collections import Counter

from modules.models.HMM.ChordHMM import ChordHMM, ChordHMMTrainer, visualize_transitions
from modules.models.Transformer.ChordNet import ChordNet
from modules.utils.hparams import HParams
from modules.utils import logger
from modules.utils.chords import idx2voca_chord
from modules.data.LabeledDataset import LabeledDataset
from modules.utils.device import get_device

def load_pretrained_model(model_path, config, device):
    """Load pretrained ChordNet model"""
    # Get model parameters
    n_freq = config.feature.get('n_bins', 144)
    n_classes = 170  # Large vocabulary size
    
    model = ChordNet(
        n_freq=n_freq,
        n_classes=n_classes,
        n_group=config.model.get('n_group', 4),
        f_layer=config.model.get('f_layer', 3),
        f_head=config.model.get('f_head', 6),
        t_layer=config.model.get('t_layer', 3),
        t_head=config.model.get('t_head', 6),
        d_layer=config.model.get('d_layer', 3),
        d_head=config.model.get('d_head', 6),
        dropout=config.model.get('dropout', 0.3)
    ).to(device)

    # Load weights with robust error handling
    try:
        # First try safe loading with weights_only=True
        try:
            # For PyTorch 2.6+, whitelist numpy scalar type
            import numpy as np
            from numpy.core.multiarray import scalar
            import torch.serialization
            torch.serialization.add_safe_globals([scalar])
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e1:
            logger.info(f"Safe loading failed: {e1}")
            logger.info("Trying to load with weights_only=False (only do this for trusted checkpoints)")
            # Fallback to weights_only=False for older checkpoints
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if model state dict is directly available or nested
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # Try to load the state dict directly
            model.load_state_dict(checkpoint)
        
        # Get normalization parameters
        mean = checkpoint.get('mean', 0.0)
        std = checkpoint.get('std', 1.0)
        
        # Attach chord mapping
        idx_to_chord = idx2voca_chord()
        model.idx_to_chord = idx_to_chord
        
        logger.info("Pretrained model loaded successfully")
        return model, mean, std, idx_to_chord
    
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        raise

def get_chord_distribution_from_dataset(dataset):
    """
    Calculate distribution of chords in the dataset.
    
    Args:
        dataset: LabeledDataset with chord indices
    
    Returns:
        Normalized distribution array
    """
    # Get unique chord indices
    chord_mapping = dataset.chord_mapping
    num_chords = max(chord_mapping.values()) + 1
    
    # Count chord occurrences
    chord_counts = np.zeros(num_chords)
    
    for sample in dataset.samples:
        chord_indices = sample['chord_idx']
        for idx in chord_indices:
            if isinstance(idx, (int, np.integer)):
                if 0 <= idx < num_chords:
                    chord_counts[idx] += 1
    
    # Add a small value to prevent zeros
    chord_counts += 1e-6
    
    # Normalize to get probability distribution
    distribution = chord_counts / chord_counts.sum()
    
    return distribution

def calculate_transition_statistics(dataset, num_states):
    """
    Calculate transition statistics from dataset.
    
    Args:
        dataset: LabeledDataset with chord indices
        num_states: Number of chord states
    
    Returns:
        Transition statistics matrix
    """
    # Initialize transition count matrix
    transitions = torch.zeros((num_states, num_states))
    
    # Count transitions in each sequence
    for sample in dataset.samples:
        chord_indices = sample['chord_idx']
        if len(chord_indices) > 1:
            for i in range(len(chord_indices) - 1):
                from_chord = chord_indices[i]
                to_chord = chord_indices[i + 1]
                
                # Make sure indices are valid
                if 0 <= from_chord < num_states and 0 <= to_chord < num_states:
                    transitions[from_chord, to_chord] += 1.0
    
    # Add small value to prevent zeros
    transitions += 0.1
    
    # Normalize rows to get probability distribution
    row_sums = transitions.sum(dim=1, keepdim=True)
    transition_probs = transitions / row_sums
    
    logger.info(f"Calculated transition statistics from dataset: shape={transition_probs.shape}")
    
    return transition_probs

def create_scheduler(scheduler_type, optimizer, args, train_loader=None):
    """
    Create learning rate scheduler based on specified type.
    
    Args:
        scheduler_type: Type of scheduler to use
        optimizer: Optimizer to schedule
        args: Command line arguments
        train_loader: Training data loader for OneCycleLR
        
    Returns:
        Learning rate scheduler
    """
    # Number of batches per epoch for OneCycleLR
    steps_per_epoch = len(train_loader) if train_loader else 100
    
    if scheduler_type == 'cosine':
        # Cosine annealing from initial LR to min LR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )
        logger.info(f"Using CosineAnnealingLR scheduler from {args.lr} to {args.min_lr} over {args.epochs} epochs")
        
    elif scheduler_type == 'step':
        # Step LR: decay by gamma every step_size epochs
        scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
        logger.info(f"Using StepLR scheduler with step size {args.step_size} and gamma {args.gamma}")
        
    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau: reduce LR when validation loss plateaus
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.gamma,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=True
        )
        logger.info(f"Using ReduceLROnPlateau scheduler with patience {args.lr_patience}, " 
                   f"factor {args.gamma}, and min_lr {args.min_lr}")
        
    elif scheduler_type == 'onecycle':
        # OneCycleLR: One cycle learning rate policy from fast.ai
        total_steps = steps_per_epoch * args.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr * args.cycle_mult,
            total_steps=total_steps,
            pct_start=0.3,  # Spend 30% ramping up, 70% ramping down
            anneal_strategy='cos',
            div_factor=25,  # Initial LR = max_lr/25
            final_div_factor=10000,  # Final LR = max_lr/10000
        )
        logger.info(f"Using OneCycleLR scheduler with max_lr={args.lr*args.cycle_mult} for {total_steps} total steps")
        
    else:
        # No scheduler
        scheduler = None
        logger.info("Using constant learning rate (no scheduler)")
        
    return scheduler

def main():
    parser = argparse.ArgumentParser(description="Train HMM for chord progression using real-world data")
    parser.add_argument('--config', type=str, default='./config/student_config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pretrained ChordNet model')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/hmm_real',
                        help='Directory to save HMM model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--audio_dirs', type=str, nargs='+', default=None,
                        help='Optional: Override default audio directories')
    parser.add_argument('--label_dirs', type=str, nargs='+', default=None,
                        help='Optional: Override default label directories')
    parser.add_argument('--cache_dir', type=str, default='./cache/features',
                        help='Directory to cache extracted features')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Sequence length for training')
    parser.add_argument('--stride', type=int, default=5,
                        help='Stride between consecutive sequences')
    
    # New arguments for LR scheduling
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'plateau', 'onecycle', 'none'],
                       default='cosine', help='Learning rate scheduler type')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate for schedulers')
    parser.add_argument('--step_size', type=int, default=10,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Gamma for StepLR and ReduceLROnPlateau schedulers')
    parser.add_argument('--lr_patience', type=int, default=2,
                       help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--cycle_mult', type=float, default=10.0,
                       help='Multiplier for max_lr in OneCycleLR (default: 10x base_lr)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load configuration
    config = HParams.load(args.config)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load pretrained model
    pretrained_model, mean, std, idx_to_chord = load_pretrained_model(args.model, config, device)
    
    # Create chord mapping
    chord_mapping = {chord: idx for idx, chord in idx_to_chord.items()}
    
    # Log some sample chord mappings to help debug
    sample_keys = ['C', 'C#', 'D', 'D#', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    sample_qualities = ['', ':min', ':maj', ':min7', ':maj7']
    
    logger.info("Sample chord mappings:")
    for key in sample_keys:
        for quality in sample_qualities:
            full_key = f"{key}{quality}"
            if full_key in chord_mapping:
                logger.info(f"  {full_key} -> {chord_mapping[full_key]}")
            else:
                logger.info(f"  {full_key} -> NOT FOUND")
    
    # Verify major chord formatting
    logger.info("Checking how major chords are represented in the mapping:")
    for root in ['C', 'G', 'D', 'A']:
        with_quality = f"{root}:maj"
        without_quality = root
        
        logger.info(f"  '{without_quality}' in mapping: {without_quality in chord_mapping}")
        logger.info(f"  '{with_quality}' in mapping: {with_quality in chord_mapping}")
        
        if without_quality in chord_mapping:
            logger.info(f"  Value for '{without_quality}': {chord_mapping[without_quality]}")
            
        if with_quality in chord_mapping:
            logger.info(f"  Value for '{with_quality}': {chord_mapping[with_quality]}")
    
    # Get a helper to analyze chord handling
    from modules.utils.chords import Chords
    chord_helper = Chords()
    # Set the chord mapping explicitly
    chord_helper.set_chord_mapping(chord_mapping)
    # Check if the mapping is complete
    chord_helper.check_chord_mapping_completeness()
    
    # Initialize the dataset
    logger.info("Initializing dataset...")
    dataset_args = {
        'chord_mapping': chord_mapping,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'cache_features': True,
        'cache_dir': args.cache_dir,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'feature_config': config.feature,
        'device': device
    }
    
    # Only add these arguments if explicitly provided by the user
    if args.audio_dirs:
        dataset_args['audio_dirs'] = args.audio_dirs
        logger.info(f"Using custom audio directories: {args.audio_dirs}")
    else:
        logger.info("Using default audio directories from LabeledDataset")
        
    if args.label_dirs:
        dataset_args['label_dirs'] = args.label_dirs
        logger.info(f"Using custom label directories: {args.label_dirs}")
    else:
        logger.info("Using default label directories from LabeledDataset")
    
    # Create the dataset with the appropriate arguments
    dataset = LabeledDataset(**dataset_args)
    
    # Get chord distribution from dataset
    logger.info("Calculating chord distribution from dataset...")
    chord_distribution = get_chord_distribution_from_dataset(dataset)
    
    # Calculate transition statistics from dataset
    transition_stats = calculate_transition_statistics(dataset, len(idx_to_chord))
    
    # Create data loaders
    train_loader = dataset.get_train_iterator(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    
    val_loader = dataset.get_val_iterator(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = dataset.get_test_iterator(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create HMM model
    logger.info("Initializing HMM model...")
    num_chords = len(idx_to_chord)
    
    hmm_model = ChordHMM(
        pretrained_model=pretrained_model,
        num_states=num_chords,
        init_distribution=chord_distribution,
        transition_stats=transition_stats,
        device=device
    ).to(device)
    
    # Create optimizer for HMM model (only train transition params)
    optimizer = Adam([hmm_model.raw_transitions, hmm_model.start_probs], lr=args.lr)
    
    # Prepare scheduler parameters
    scheduler_params = {
        'min_lr': args.min_lr,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'lr_patience': args.lr_patience,
        'cycle_mult': args.cycle_mult,
        'steps_per_epoch': len(train_loader)  # Pass actual steps per epoch
    }
    
    # Create trainer with scheduler parameters
    trainer = ChordHMMTrainer(
        hmm_model=hmm_model,
        optimizer=optimizer,
        device=device,
        max_epochs=args.epochs,
        patience=args.patience,
        scheduler_type=None if args.scheduler == 'none' else args.scheduler,
        scheduler_params=scheduler_params
    )
    
    logger.info("HMM model initialized with:")
    logger.info(f"- Number of chord states: {num_chords}")
    logger.info(f"- Learning rate: {args.lr}")
    logger.info(f"- Maximum epochs: {args.epochs}")
    logger.info(f"- Batch size: {args.batch_size}")
    logger.info(f"- Scheduler: {args.scheduler}")
    
    # Train the model
    logger.info("Starting HMM training...")
    trainer.train(train_loader, val_loader)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'hmm_model_final.pth')
    torch.save({
        'model_state_dict': hmm_model.state_dict(),
        'config': {
            'num_states': num_chords,
            'pretrained_model_path': args.model,
        },
        'chord_mapping': chord_mapping,
        'idx_to_chord': idx_to_chord,
        'mean': mean,
        'std': std,
    }, final_model_path)
    
    logger.info(f"Final HMM model saved to {final_model_path}")
    
    # Test the model
    logger.info("Evaluating HMM model...")
    preds, true, accuracy = trainer.decode(test_loader)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Visualize transition probabilities
    try:
        logger.info("Visualizing transition matrix...")
        visualize_transitions(
            hmm_model=hmm_model,
            idx_to_chord=idx_to_chord,
            top_k=20,
            save_path=os.path.join(args.save_dir, 'chord_transitions.png')
        )
        logger.info(f"Transition visualization saved to {os.path.join(args.save_dir, 'chord_transitions.png')}")
    except Exception as e:
        logger.error(f"Error visualizing transitions: {e}")

if __name__ == "__main__":
    main()
