import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Sampler   # <-- Added Sampler import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # <-- For testing
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record  # NEW: add error recorder
import argparse  # NEW: import argparse for argument parsing

from modules.utils.device import get_device
from modules.data.CrossDataset import CrossDataset, get_unified_mapping
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.Trainer import BaseTrainer
from modules.training.Schedulers import CosineScheduler
# NEW: import AudioDatasetLoader from our new dataset module
from modules.data.AudioProcessingDataset import AudioDatasetLoader

# New testing helper: partition_test_set using 90-10 split.
def partition_test_set(concat_dataset):
    test_indices = []
    offset = 0
    for ds in concat_dataset.datasets:
        test_start = int(len(ds) * 0.9)
        test_indices.extend(range(offset + test_start, offset + len(ds)))
        offset += len(ds)
    return test_indices

# New testing helper: ListSampler.
class ListSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

# New testing helper: Tester.
class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in self.test_loader:
                inputs = batch['chroma'].to(self.device)
                targets = batch['chord_idx'].to(self.device)
                preds = self.model.predict(inputs)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")

@record  # NEW: decorate the main entrypoint for error recording
def main():
    # NEW: Parse command-line arguments to consume extra parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    args, unknown = parser.parse_known_args()

    # Distributed training setup
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['NCCL_DEBUG'] = 'INFO'
        dist.init_process_group(backend='nccl')
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {local_rank}] Process group initialized. Using device: {device}")
    else:
        device = get_device()
        local_rank = None
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    chroma_dir = os.path.join(project_root, "data", "cross-era_chroma-nnls")
    label_dir  = os.path.join(project_root, "data", "cross-era_chords-chordino")
    chroma_dir2 = os.path.join(project_root, "data", "cross-composer_chroma-nnls")
    label_dir2 = os.path.join(project_root, "data", "cross-composer_chords-chordino")
    label_dirs = [label_dir, label_dir2]
    unified_mapping = get_unified_mapping(label_dirs)
    print("Unified chord mapping (total labels):", len(unified_mapping))
    
    dataset1 = CrossDataset(chroma_dir, label_dir, chord_mapping=unified_mapping, seq_len=10, stride=3)
    dataset2 = CrossDataset(chroma_dir2, label_dir2, chord_mapping=unified_mapping, seq_len=10, stride=3)
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print("Total combined samples:", len(combined_dataset))
    
    # Explicitly partition the combined dataset into 80% training, 10% validation, and 10% test.
    total_samples = len(combined_dataset)
    train_end = int(total_samples * 0.8)
    val_end = int(total_samples * 0.9)
    indices_train = list(range(0, train_end))
    indices_val = list(range(train_end, val_end))
    indices_test = list(range(val_end, total_samples))
    
    from torch.utils.data import Subset
    train_subset = Subset(combined_dataset, indices_train)
    val_subset = Subset(combined_dataset, indices_val)
    test_subset = Subset(combined_dataset, indices_test)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

    # NEW: Debug prints for each partition.
    print("=== Debug: Training Partition ===")
    for batch in train_loader:
         print("Train Batch - Input shape:", batch['chroma'].shape,
               "Chord indices:", batch['chord_idx'],
               "Chord labels:", batch['chord_label'])
         break

    print("=== Debug: Validation Partition ===")
    for batch in val_loader:
         print("Validation Batch - Input shape:", batch['chroma'].shape,
               "Chord indices:", batch['chord_idx'],
               "Chord labels:", batch['chord_label'])
         break

    print("=== Debug: Test Partition ===")
    for batch in test_loader:
         print("Test Batch - Input shape:", batch['chroma'].shape,
               "Chord indices:", batch['chord_idx'],
               "Chord labels:", batch['chord_label'])
         break

    # NEW: Debug - check the first batch in val_loader for correct label mapping.
    for batch in val_loader:
        targets = batch['chord_idx']
        print("Debug: First validation batch target indices:", targets)
        break
    
    model = ChordNet(n_freq=12, n_classes=274, n_group=3,
                     f_layer=2, f_head=4, 
                     t_layer=2, t_head=4, 
                     d_layer=2, d_head=4, 
                     dropout=0.3).to(device)
    
    # Wrap model with DistributedDataParallel if in distributed mode
    if local_rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    warmup_steps = 1
    num_epochs = 5
    scheduler = CosineScheduler(optimizer, max_update=num_epochs, base_lr=0.0001,
                                final_lr=0.000001, warmup_steps=warmup_steps, warmup_begin_lr=0.00001)
   
    trainer = BaseTrainer(model, optimizer, scheduler=scheduler,
                          num_epochs=num_epochs, device=device,
                          ignore_index=unified_mapping["N"])
    # Utilize more GPUs if available.
    if torch.cuda.device_count() > 1:
        trainer.train_multi_gpu(train_loader, val_loader=val_loader)
    else:
        trainer.train(train_loader, val_loader=val_loader)
    
    # Automatically run the testing phase after training.
    print("Starting testing phase.")
    test_indices = partition_test_set(combined_dataset)
    test_loader = DataLoader(combined_dataset, batch_size=128, sampler=ListSampler(test_indices))
    tester = Tester(model, test_loader, device)
    tester.evaluate()

    # === New dataset pipeline ===
    # data_to_load = ['smc']          # adjust as needed
    # test_only_data = ['gtzan']       # adjust as needed
    # data_path = os.path.join(project_root, "data", "demix_spectrogram_data.npz")
    # annotation_path = os.path.join(project_root, "data", "full_beat_annotation.npz")
    # dataset_loader = AudioDatasetLoader(
    #     data_to_load=data_to_load,
    #     test_only_data=test_only_data,
    #     data_path=data_path,
    #     annotation_path=annotation_path,
    #     fps=44100/1024,
    #     seed=0,
    #     num_folds=8,
    #     mask_value=-1,
    #     sample_size=512
    # )
    # train_set, val_set, test_set = dataset_loader.get_fold(fold=0)
    # train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # # Initialize chord net and training components.
    # model = ChordNet(n_freq=12, n_classes=274, n_group=3,
    #                  f_layer=2, f_head=4, 
    #                  t_layer=2, t_head=4, 
    #                  d_layer=2, d_head=4, 
    #                  dropout=0.3).to(device)
    
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # num_epochs = 5
    # scheduler = CosineScheduler(optimizer, max_update=num_epochs, base_lr=0.001,
    #                             final_lr=0.000001, warmup_steps=1, warmup_begin_lr=0.000001)
   
    # trainer = BaseTrainer(model, optimizer, scheduler=scheduler,
    #                       num_epochs=num_epochs, device=device,
    #                       ignore_index=0)  # adjust ignore_index as needed
    # trainer.train(train_loader, val_loader=val_loader)
    
    # # Optionally, testing phase can be added here.
    # print("Training complete for chord net using the new dataset.")

    
if __name__ == '__main__':
    main()
