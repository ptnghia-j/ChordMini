#!/usr/bin/env python
import os
import sys
import argparse
import torch
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("simple_ray_distributed")

def train_func(config):
    """
    Simple distributed training function to test Ray setup.
    
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
    
    # Log basic information
    logger.info(f"Worker initialized: rank={rank}, world_size={world_size}, device={device}")
    
    # Create a simple model
    model = torch.nn.Linear(10, 10).to(device)
    
    # Wrap the model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None
    )
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy data
    data = torch.randn(20, 10).to(device)
    target = torch.randn(20, 10).to(device)
    
    # Run a few training steps
    for step in range(10):
        # Forward pass
        output = model(data)
        loss = torch.nn.MSELoss()(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log progress
        logger.info(f"Rank {rank}, Step {step}, Loss: {loss.item()}")
        
        # Add a barrier to ensure all processes are in sync
        torch.distributed.barrier()
        
        # Report metrics to Ray
        train.report({
            "step": step,
            "loss": loss.item(),
            "rank": rank
        })
    
    logger.info(f"Rank {rank} completed training successfully")
    
    # Save final model (only on rank 0)
    if rank == 0:
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        train.report({"checkpoint": checkpoint})

def main():
    parser = argparse.ArgumentParser(description="Simple Ray distributed training test")
    
    # Ray-specific arguments
    parser.add_argument('--num_workers', type=int, default=3,
                      help='Number of Ray workers (nodes) to use')
    parser.add_argument('--cpus_per_worker', type=int, default=1,
                      help='Number of CPUs to allocate per worker')
    parser.add_argument('--gpus_per_worker', type=int, default=1,
                      help='Number of GPUs to allocate per worker')
    parser.add_argument('--address', type=str, default=None,
                      help='Address of existing Ray cluster (leave empty to start a new one)')
    
    # Parse arguments
    args = parser.parse_args()
    
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
        train_loop_config={},  # No specific config needed for this simple test
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="simple_distributed_test"
        )
    )
    
    # Run training
    logger.info("Starting Ray distributed training test...")
    result = trainer.fit()
    
    # Print results
    logger.info(f"Training completed. Final metrics: {result.metrics}")
    
    # Shutdown Ray
    ray.shutdown()
    logger.info("Ray cluster shutdown complete")

if __name__ == "__main__":
    main()
