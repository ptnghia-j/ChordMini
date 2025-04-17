#!/bin/bash
# Script to deploy and run Ray training on Kubernetes

# Set default values
NAMESPACE="csuf-titans"
CLUSTER_NAME="chord-mini-ray-cluster"
NUM_WORKERS=3
GPUS_PER_WORKER=1
CPUS_PER_WORKER=2
DATASET_TYPE="combined"
SPEC_DIR="/mnt/storage/data/logits/synth/spectrograms"
LABEL_DIR="/mnt/storage/data/logits/synth/labels"
LOGITS_DIR="/mnt/storage/data/logits/synth/logits"
CHECKPOINT_DIR="/mnt/storage/checkpoints/ray_distributed"
BATCH_SIZE=16
NUM_EPOCHS=100
LEARNING_RATE=0.001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --cluster-name)
      CLUSTER_NAME="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --gpus-per-worker)
      GPUS_PER_WORKER="$2"
      shift 2
      ;;
    --cpus-per-worker)
      CPUS_PER_WORKER="$2"
      shift 2
      ;;
    --dataset-type)
      DATASET_TYPE="$2"
      shift 2
      ;;
    --spec-dir)
      SPEC_DIR="$2"
      shift 2
      ;;
    --label-dir)
      LABEL_DIR="$2"
      shift 2
      ;;
    --logits-dir)
      LOGITS_DIR="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --namespace NAMESPACE          Kubernetes namespace (default: csuf-titans)"
      echo "  --cluster-name NAME            Ray cluster name (default: chord-mini-ray-cluster)"
      echo "  --num-workers N                Number of worker nodes (default: 3)"
      echo "  --gpus-per-worker N            GPUs per worker (default: 1)"
      echo "  --cpus-per-worker N            CPUs per worker (default: 2)"
      echo "  --dataset-type TYPE            Dataset type: fma, maestro, combined (default: combined)"
      echo "  --spec-dir DIR                 Directory containing spectrograms"
      echo "  --label-dir DIR                Directory containing labels"
      echo "  --logits-dir DIR               Directory containing teacher logits"
      echo "  --checkpoint-dir DIR           Directory to save checkpoints"
      echo "  --batch-size N                 Batch size (default: 16)"
      echo "  --num-epochs N                 Number of epochs (default: 100)"
      echo "  --learning-rate LR             Learning rate (default: 0.001)"
      echo "  --help                         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if Ray cluster is already running
if kubectl get raycluster -n $NAMESPACE $CLUSTER_NAME &> /dev/null; then
  echo "Ray cluster $CLUSTER_NAME already exists in namespace $NAMESPACE"
else
  echo "Creating Ray cluster $CLUSTER_NAME in namespace $NAMESPACE"
  kubectl apply -f ray_cluster_k8s.yaml -n $NAMESPACE
  
  # Wait for the cluster to be ready
  echo "Waiting for Ray cluster to be ready..."
  while [[ $(kubectl get raycluster -n $NAMESPACE $CLUSTER_NAME -o jsonpath='{.status.state}') != "ready" ]]; do
    echo "Waiting for Ray cluster to be ready..."
    sleep 10
  done
  echo "Ray cluster is ready!"
fi

# Get the head pod name
HEAD_POD=$(kubectl get pods -n $NAMESPACE --selector=ray.io/node-type=head,ray.io/cluster=$CLUSTER_NAME -o jsonpath='{.items[0].metadata.name}')
echo "Ray head pod: $HEAD_POD"

# Copy the necessary files to the head pod
echo "Copying files to the head pod..."
kubectl cp ray_train_student.py $NAMESPACE/$HEAD_POD:/tmp/ray_train_student.py
kubectl cp simple_ray_distributed.py $NAMESPACE/$HEAD_POD:/tmp/simple_ray_distributed.py
kubectl cp modules/training/RayDistributedTrainer.py $NAMESPACE/$HEAD_POD:/mnt/storage/ChordMini/modules/training/RayDistributedTrainer.py

# Install required packages on the head pod
echo "Installing required packages on the head pod..."
kubectl exec -n $NAMESPACE $HEAD_POD -- pip install torch torchvision torchaudio librosa scikit-learn pyyaml matplotlib tqdm

# Run the simple distributed test first
echo "Running simple distributed test..."
kubectl exec -n $NAMESPACE $HEAD_POD -- python /tmp/simple_ray_distributed.py \
  --num_workers $NUM_WORKERS \
  --cpus_per_worker $CPUS_PER_WORKER \
  --gpus_per_worker $GPUS_PER_WORKER

# If the simple test succeeds, run the actual training
if [ $? -eq 0 ]; then
  echo "Simple distributed test succeeded. Starting actual training..."
  kubectl exec -n $NAMESPACE $HEAD_POD -- python /tmp/ray_train_student.py \
    --num_workers $NUM_WORKERS \
    --cpus_per_worker $CPUS_PER_WORKER \
    --gpus_per_worker $GPUS_PER_WORKER \
    --dataset_type $DATASET_TYPE \
    --spec_dir $SPEC_DIR \
    --label_dir $LABEL_DIR \
    --logits_dir $LOGITS_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE
else
  echo "Simple distributed test failed. Please check the logs."
  exit 1
fi

# Create a port-forward to the Ray dashboard
echo "Creating port-forward to Ray dashboard. Visit http://localhost:8265 in your browser."
kubectl port-forward -n $NAMESPACE service/$CLUSTER_NAME-head-svc 8265:8265
