#!/usr/bin/env python
import os
import sys
import argparse
import torch
import numpy as np
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import yaml
import logging

# Project imports
from modules.utils.mir_eval_modules import large_voca_score_calculation
from modules.utils.device import get_device, is_cuda_available
from modules.data.SynthDataset import SynthDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.StudentTrainer import StudentTrainer
from modules.training.RayDistributedTrainer import RayDistributedTrainer
from modules.utils.hparams import HParams
from modules.utils.chords import idx2voca_chord
from modules.training.Tester import Tester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ray_train_student")

def train_func(config):
    """
    Training function that will be executed on each Ray worker.
    
    Args:
        config: Dictionary containing training configuration
    """
    # Set up distributed training environment
    train.torch.setup()
    
    # Get distributed training info from Ray
    world_size = train.get_context().get_world_size()
    rank = train.get_context().get_rank()
    local_rank = train.get_context().get_local_rank()
    
    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Log basic information
    logger.info(f"Worker initialized: rank={rank}, world_size={world_size}, device={device}")
    logger.info(f"Training with config: {config}")
    
    # Load chord mapping
    master_mapping = idx2voca_chord()
    chord_mapping = {chord: idx for idx, chord in master_mapping.items()}
    
    # Create dataset
    dataset_args = {
        'spec_dir': config.get("spec_dir"),
        'label_dir': config.get("label_dir"),
        'logits_dir': config.get("logits_dir"),
        'chord_mapping': chord_mapping,
        'seq_len': config.get("seq_len", 10),
        'stride': config.get("seq_stride", 5),
        'frame_duration': config.get("hop_duration", 0.1),
        'verbose': True,
        'device': device,
        'pin_memory': False,
        'prefetch_factor': config.get("prefetch_factor", 2),
        'num_workers': 0,  # Use 0 for Ray to avoid conflicts
        'require_teacher_logits': config.get("use_kd_loss", False),
        'use_cache': not config.get("disable_cache", False),
        'metadata_only': config.get("metadata_cache", False),
        'cache_fraction': config.get("cache_fraction", 0.1),
        'lazy_init': config.get("lazy_init", False),
        'batch_gpu_cache': config.get("batch_gpu_cache", False),
        'dataset_type': config.get("dataset_type", "fma"),
        'small_dataset_percentage': config.get("small_dataset", None),
    }
    
    logger.info("Creating dataset...")
    synth_dataset = SynthDataset(**dataset_args)
    
    # Create data loaders with Ray-aware distributed sampling
    batch_size = config.get("batch_size", 16)
    
    # Create samplers for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        synth_dataset.train_indices,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        synth_dataset.eval_indices,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = synth_dataset.get_train_iterator(
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle here, the sampler will do it
        sampler=train_sampler,
        num_workers=0,  # Force single worker for Ray
        pin_memory=False
    )
    
    val_loader = synth_dataset.get_eval_iterator(
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,  # Force single worker for Ray
        pin_memory=False
    )
    
    # Calculate dataset statistics
    logger.info("Calculating dataset statistics...")
    mean, std = 0.0, 1.0
    try:
        # Create stats loader
        stats_batch_size = min(16, batch_size)
        stats_loader = torch.utils.data.DataLoader(
            synth_dataset,
            batch_size=stats_batch_size,
            sampler=torch.utils.data.RandomSampler(
                synth_dataset,
                replacement=True,
                num_samples=min(1000, len(synth_dataset))
            ),
            num_workers=0,
            pin_memory=False
        )
        
        # Process in chunks with progress
        mean_sum = 0.0
        square_sum = 0.0
        sample_count = 0
        
        for i, batch in enumerate(stats_loader):
            if i >= 10:  # Limit batches processed
                break
                
            # Move to CPU explicitly
            features = batch['spectro'].to('cpu')
            
            # Calculate stats
            batch_mean = torch.mean(features).item()
            batch_square_mean = torch.mean(features.pow(2)).item()
            
            # Update running sums
            mean_sum += batch_mean
            square_sum += batch_square_mean
            sample_count += 1
            
            # Free memory
            del features
            
            # Clear GPU cache periodically
            if i % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final statistics
        if sample_count > 0:
            mean = mean_sum / sample_count
            square_mean = square_sum / sample_count
            std = np.sqrt(square_mean - mean**2)
            logger.info(f"Dataset stats: mean={mean:.4f}, std={std:.4f}")
        else:
            logger.warning("No samples processed for statistics")
    except Exception as e:
        logger.error(f"Error in stats calculation: {e}")
    
    # Create normalized tensors on device
    mean = torch.tensor(mean, device=device)
    std = torch.tensor(std, device=device)
    normalization = {'mean': mean, 'std': std}
    
    # Create model
    n_freq = config.get("freq_bins", 144)
    n_classes = len(chord_mapping)
    model_scale = config.get("model_scale", 1.0)
    n_group = max(1, int(32 * model_scale))
    dropout_rate = config.get("dropout", 0.3)
    
    logger.info(f"Creating model with n_freq={n_freq}, n_classes={n_classes}, n_group={n_group}")
    model = ChordNet(
        n_freq=n_freq,
        n_classes=n_classes,
        n_group=n_group,
        f_layer=config.get("f_layer", 3),
        f_head=config.get("f_head", 6),
        t_layer=config.get("t_layer", 3),
        t_head=config.get("t_head", 6),
        d_layer=config.get("d_layer", 3),
        d_head=config.get("d_head", 6),
        dropout=dropout_rate
    ).to(device)
    
    # Wrap model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=config.get("find_unused_parameters", False)
    )
    
    # Attach chord mapping to model
    model.module.idx_to_chord = master_mapping
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.0001),
        weight_decay=config.get("weight_decay", 0.0)
    )
    
    # Create trainer
    logger.info("Creating RayDistributedTrainer...")
    trainer = RayDistributedTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=config.get("num_epochs", 100),
        logger=logger,
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
        class_weights=None,
        idx_to_chord=master_mapping,
        normalization=normalization,
        early_stopping_patience=config.get("early_stopping_patience", 5),
        lr_decay_factor=config.get("lr_decay_factor", 0.95),
        min_lr=config.get("min_learning_rate", 5e-6),
        use_warmup=config.get("use_warmup", False),
        warmup_epochs=config.get("warmup_epochs", 5),
        warmup_start_lr=config.get("warmup_start_lr", None),
        warmup_end_lr=config.get("warmup_end_lr", None),
        lr_schedule_type=config.get("lr_schedule", None),
        use_focal_loss=config.get("use_focal_loss", False),
        focal_gamma=config.get("focal_gamma", 2.0),
        focal_alpha=config.get("focal_alpha", None),
        use_kd_loss=config.get("use_kd_loss", False),
        kd_alpha=config.get("kd_alpha", 0.5),
        temperature=config.get("temperature", 1.0),
        rank=rank,
        world_size=world_size
    )
    
    # Train the model
    logger.info("Starting training...")
    for epoch in range(1, config.get("num_epochs", 100) + 1):
        # Set epoch for samplers to ensure proper shuffling
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Train for one epoch
        early_stop = trainer.train_epoch(train_loader, val_loader, epoch)
        
        # Report metrics to Ray
        train.report({
            "epoch": epoch,
            "train_loss": trainer.train_loss,
            "train_accuracy": trainer.train_accuracy,
            "val_loss": trainer.val_loss,
            "val_accuracy": trainer.val_accuracy,
            "best_val_accuracy": trainer.best_val_acc,
        })
        
        # Check for early stopping
        if early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    if rank == 0:
        logger.info("Saving final model...")
        trainer.save_checkpoint(epoch, trainer.val_accuracy)
        
        # Also save in Ray's expected format
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': trainer.val_accuracy,
            'best_val_accuracy': trainer.best_val_acc,
            'chord_mapping': chord_mapping,
            'normalization': {'mean': mean.item(), 'std': std.item()},
        }
        
        # Report final checkpoint to Ray
        train.report({"checkpoint": checkpoint})
    
    logger.info("Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Train a chord recognition model using Ray distributed training")
    
    # Ray-specific arguments
    parser.add_argument('--num_workers', type=int, default=3,
                      help='Number of Ray workers (nodes) to use')
    parser.add_argument('--cpus_per_worker', type=int, default=2,
                      help='Number of CPUs to allocate per worker')
    parser.add_argument('--gpus_per_worker', type=int, default=1,
                      help='Number of GPUs to allocate per worker')
    parser.add_argument('--address', type=str, default=None,
                      help='Address of existing Ray cluster (leave empty to start a new one)')
    
    # Model and training arguments
    parser.add_argument('--config', type=str, default='./config/student_config.yaml',
                      help='Path to the configuration file')
    parser.add_argument('--dataset_type', type=str, choices=['fma', 'maestro', 'combined'], default='fma',
                      help='Dataset format type')
    parser.add_argument('--spec_dir', type=str, default=None,
                      help='Directory containing spectrograms')
    parser.add_argument('--label_dir', type=str, default=None,
                      help='Directory containing labels')
    parser.add_argument('--logits_dir', type=str, default=None,
                      help='Directory containing teacher logits')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                      help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int, default=None,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='Learning rate')
    parser.add_argument('--small_dataset', type=float, default=None,
                      help='Use only a small percentage of dataset for testing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration from YAML
    config = HParams.load(args.config)
    
    # Convert config to dictionary for Ray
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    
    # Override config with command line arguments
    if args.dataset_type:
        config_dict['dataset_type'] = args.dataset_type
    if args.spec_dir:
        config_dict['spec_dir'] = args.spec_dir
    if args.label_dir:
        config_dict['label_dir'] = args.label_dir
    if args.logits_dir:
        config_dict['logits_dir'] = args.logits_dir
    if args.checkpoint_dir:
        config_dict['checkpoint_dir'] = args.checkpoint_dir
    if args.num_epochs:
        config_dict['num_epochs'] = args.num_epochs
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    if args.learning_rate:
        config_dict['learning_rate'] = args.learning_rate
    if args.small_dataset is not None:
        config_dict['small_dataset'] = args.small_dataset
    
    # Initialize Ray
    if args.address:
        ray.init(address=args.address)
        logger.info(f"Connected to existing Ray cluster at {args.address}")
    else:
        ray.init()
        logger.info("Started new Ray cluster")
    
    # Configure training
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.gpus_per_worker > 0,
        resources_per_worker={
            "CPU": args.cpus_per_worker,
            "GPU": args.gpus_per_worker
        }
    )
    
    # Create the trainer
    trainer = TorchTrainer(
        train_func,
        train_loop_config=config_dict,
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="chord_mini_training",
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute="val_accuracy",
                checkpoint_score_order="max",
            )
        )
    )
    
    # Run training
    logger.info("Starting Ray distributed training...")
    result = trainer.fit()
    
    # Print results
    logger.info(f"Training completed. Best result: {result.metrics}")
    logger.info(f"Checkpoint saved at: {result.checkpoint}")
    
    # Shutdown Ray
    ray.shutdown()
    logger.info("Ray cluster shutdown complete")

if __name__ == "__main__":
    main()
