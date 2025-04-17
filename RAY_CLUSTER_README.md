# ChordMini with Ray Cluster

This guide explains how to use Ray Cluster for distributed training of the ChordMini model.

## Overview

Ray is an open-source unified framework for scaling AI and Python applications. It provides a simple API for distributed computing without requiring deep knowledge of distributed systems. This implementation adapts ChordMini to work with Ray Cluster for multi-node, multi-GPU training.

## Files

- `ray_train_student.py`: Main training script adapted for Ray
- `modules/training/RayDistributedTrainer.py`: Ray-specific distributed trainer
- `simple_ray_distributed.py`: Simple test script for Ray distributed setup
- `ray_cluster_k8s.yaml`: Kubernetes configuration for Ray cluster
- `run_ray_training.sh`: Script to deploy and run training on Ray cluster

## Prerequisites

- Kubernetes cluster with GPU support
- `kubectl` configured to access your cluster
- Ray Operator installed in your Kubernetes cluster

## Setting Up Ray Operator

If you don't have the Ray Operator installed, you can install it using Helm:

```bash
# Add the Ray Helm repository
helm repo add kuberay https://ray-project.github.io/kuberay-helm/

# Update Helm repositories
helm repo update

# Install the KubeRay operator
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0
```

## Deploying a Ray Cluster

You can deploy a Ray cluster using the provided YAML file:

```bash
kubectl apply -f ray_cluster_k8s.yaml -n csuf-titans
```

This will create a Ray cluster with:
- 1 head node
- 3 worker nodes with 1 GPU each

## Running the Training Script

The easiest way to run the training is using the provided script:

```bash
chmod +x run_ray_training.sh
./run_ray_training.sh
```

You can customize the training parameters:

```bash
./run_ray_training.sh \
  --num-workers 3 \
  --gpus-per-worker 1 \
  --dataset-type combined \
  --spec-dir /mnt/storage/data/logits/synth/spectrograms \
  --label-dir /mnt/storage/data/logits/synth/labels \
  --logits-dir /mnt/storage/data/logits/synth/logits \
  --checkpoint-dir /mnt/storage/checkpoints/ray_distributed \
  --batch-size 16 \
  --num-epochs 100 \
  --learning-rate 0.001
```

## Accessing the Ray Dashboard

The Ray dashboard provides insights into your distributed training job. To access it:

```bash
kubectl port-forward -n csuf-titans service/chord-mini-ray-cluster-head-svc 8265:8265
```

Then visit http://localhost:8265 in your browser.

## Manual Execution

If you prefer to run commands manually:

1. Get the head pod name:
   ```bash
   HEAD_POD=$(kubectl get pods -n csuf-titans --selector=ray.io/node-type=head,ray.io/cluster=chord-mini-ray-cluster -o jsonpath='{.items[0].metadata.name}')
   ```

2. Copy the training script:
   ```bash
   kubectl cp ray_train_student.py csuf-titans/$HEAD_POD:/tmp/ray_train_student.py
   kubectl cp modules/training/RayDistributedTrainer.py csuf-titans/$HEAD_POD:/mnt/storage/ChordMini/modules/training/RayDistributedTrainer.py
   ```

3. Run the training:
   ```bash
   kubectl exec -n csuf-titans $HEAD_POD -- python /tmp/ray_train_student.py \
     --num_workers 3 \
     --gpus_per_worker 1 \
     --dataset_type combined \
     --spec_dir /mnt/storage/data/logits/synth/spectrograms \
     --label_dir /mnt/storage/data/logits/synth/labels \
     --logits_dir /mnt/storage/data/logits/synth/logits \
     --checkpoint_dir /mnt/storage/checkpoints/ray_distributed
   ```

## Cleaning Up

To delete the Ray cluster:

```bash
kubectl delete raycluster -n csuf-titans chord-mini-ray-cluster
```

## Troubleshooting

### Worker Nodes Not Starting

If worker nodes are not starting, check:
- GPU availability in your cluster
- Resource quotas in your namespace
- Pod events: `kubectl describe pod -n csuf-titans <worker-pod-name>`

### Training Not Progressing

If training starts but doesn't progress:
- Check the logs of the head node: `kubectl logs -n csuf-titans $HEAD_POD`
- Check the Ray dashboard for task failures
- Verify that all worker nodes can communicate with the head node

### Out of Memory Errors

If you encounter OOM errors:
- Reduce batch size
- Reduce model size
- Increase memory allocation in the Ray cluster YAML

## Comparison with PyTorch DDP

Ray provides several advantages over direct PyTorch DDP:

1. **Simplified Deployment**: Ray handles node discovery and communication setup
2. **Dynamic Scaling**: Ray can scale workers up/down based on workload
3. **Fault Tolerance**: Ray can recover from worker failures
4. **Resource Management**: Ray integrates with Kubernetes for efficient resource allocation
5. **Unified Framework**: Ray supports both data processing and model training in one framework

However, PyTorch DDP might offer more fine-grained control over the distributed training process.

## Performance Considerations

For optimal performance:

1. **Batch Size**: Adjust batch size based on GPU memory and model size
2. **Data Loading**: Use `num_workers=0` with Ray to avoid conflicts
3. **GPU Utilization**: Monitor GPU utilization and adjust batch size accordingly
4. **Network Bandwidth**: Ensure high-speed networking between nodes
5. **Storage**: Use fast storage for data access
