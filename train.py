import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Sampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from modules.utils.device import get_device
from modules.data.CrossDataset import CrossDataset, get_unified_mapping
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.Trainer import BaseTrainer
from modules.training.Schedulers import CosineScheduler

def partition_test_set(concat_dataset):
    test_indices = []
    offset = 0
    for ds in concat_dataset.datasets:
        if hasattr(ds, 'test_indices'):
            test_indices.extend([offset + i for i in ds.test_indices])
        else:
            test_start = int(len(ds) * 0.9)
            test_indices.extend(range(offset + test_start, offset + len(ds)))
        offset += len(ds)
    return test_indices

class ListSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

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

def main():
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
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
    print("Unified chord mapping (example):", unified_mapping)
    
    dataset1 = CrossDataset(chroma_dir, label_dir, chord_mapping=unified_mapping, seq_len=10, stride=3)
    dataset2 = CrossDataset(chroma_dir2, label_dir2, chord_mapping=unified_mapping, seq_len=10, stride=3)
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print("Total combined samples:", len(combined_dataset))
    
    train_loader = DataLoader(ConcatDataset([dataset1.get_train_iterator(batch_size=128, shuffle=True).dataset,
                                              dataset2.get_train_iterator(batch_size=128, shuffle=True).dataset]),
                              batch_size=128, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(ConcatDataset([dataset1.get_eval_iterator(batch_size=128, shuffle=False).dataset,
                                              dataset2.get_eval_iterator(batch_size=128, shuffle=False).dataset]),
                              batch_size=128, shuffle=False, pin_memory=True)
    
    print("=== Debug: Training set sample ===")
    train_batch = next(iter(train_loader))
    print("Training Chroma tensor:", train_batch['chroma'])
    print("Training labels:", train_batch['chord_idx'])
    
    print("=== Debug: Evaluation set sample ===")
    eval_batch = next(iter(val_loader))
    print("Evaluation Chroma tensor:", eval_batch['chroma'])
    print("Evaluation labels:", eval_batch['chord_idx'])
    
    model = ChordNet(n_freq=12, n_classes=len(unified_mapping), n_group=3,
                     f_layer=2, f_head=4, 
                     t_layer=2, t_head=4, 
                     d_layer=2, d_head=4, 
                     dropout=0.3).to(device)
    
    if local_rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6)
    warmup_steps = 10  # increased warmup steps for a gentler start
    num_epochs = 20
    scheduler = CosineScheduler(optimizer, max_update=num_epochs, base_lr=1e-5,
                                final_lr=1e-6, warmup_steps=warmup_steps, warmup_begin_lr=1e-6)
   
    from collections import Counter
    import matplotlib.pyplot as plt

    dist_counter = Counter()
    for ds in [dataset1, dataset2]:
        dist_counter.update([s['chord_label'] for s in ds.samples])
    # Use the keys from unified_mapping (ensuring order) rather than dist_counter keys.
    unified_keys = sorted(unified_mapping, key=lambda k: unified_mapping[k])
    total_samples = sum(dist_counter.values())
    num_classes = len(unified_keys)
    class_weights = [total_samples / (num_classes * dist_counter.get(ch, 1)) for ch in unified_keys]
    print("Computed class weights:", class_weights)
    
    # Replace chord_set with unified_keys for plotting.
    indices = list(range(len(unified_keys)))
    counts = [dist_counter[ch] for ch in unified_keys]
    plt.figure(figsize=(12, 6))
    plt.bar(indices, counts, align='center')
    plt.xlabel("Chord Index")
    plt.ylabel("Count")
    plt.title("Chord Label Distribution")
    plt.xticks(indices)
    plt.tight_layout()
    plt.savefig("chord_distribution.png")
    plt.close()

    trainer = BaseTrainer(model, optimizer, scheduler=scheduler,
                          num_epochs=num_epochs, device=device,
                          ignore_index=unified_mapping["N"],
                          class_weights=class_weights)
    trainer.train(train_loader, val_loader=val_loader)
    
    print("Starting testing phase.")
    test_indices = partition_test_set(combined_dataset)
    test_loader = DataLoader(combined_dataset, batch_size=128, sampler=ListSampler(test_indices), pin_memory=True)
    
    print("=== Debug: Test set sample ===")
    test_batch = next(iter(test_loader))
    print("Test Chroma tensor:", test_batch['chroma'])
    print("Test labels:", test_batch['chord_idx'])
    
    tester = Tester(model, test_loader, device)
    tester.evaluate()
    
if __name__ == '__main__':
    main()