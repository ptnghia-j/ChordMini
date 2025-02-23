import sys
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, Sampler   # <-- Added Sampler import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # <-- For testing

from modules.utils.device import get_device
from modules.data.CrossDataset import CrossDataset, get_unified_mapping
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.Trainer import BaseTrainer
from modules.training.Schedulers import CosineScheduler

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

def main():
    device = get_device()
    
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
    
    dataset1 = CrossDataset(chroma_dir, label_dir, chord_mapping=unified_mapping, seq_len=10)
    dataset2 = CrossDataset(chroma_dir2, label_dir2, chord_mapping=unified_mapping, seq_len=10)
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print("Total combined samples:", len(combined_dataset))
    
    # Use default DataLoader ordering.
    train_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True)
    val_loader   = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    
    model = ChordNet(n_freq=12, n_classes=274, n_group=3,
                     f_layer=2, f_head=4, 
                     t_layer=2, t_head=4, 
                     d_layer=2, d_head=4, 
                     dropout=0.3).to(device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    warmup_steps = 5
    num_epochs = 20
    scheduler = CosineScheduler(optimizer, max_update=num_epochs, base_lr=0.0001,
                                final_lr=0.000001, warmup_steps=warmup_steps, warmup_begin_lr=0.0001)
   
    trainer = BaseTrainer(model, optimizer, scheduler=scheduler,
                          num_epochs=num_epochs, device=device,
                          ignore_index=unified_mapping["N"])
    trainer.train(train_loader, val_loader=val_loader)
    
    # Automatically run the testing phase after training.
    print("Starting testing phase.")
    test_indices = partition_test_set(combined_dataset)
    test_loader = DataLoader(combined_dataset, batch_size=128, sampler=ListSampler(test_indices))
    tester = Tester(model, test_loader, device)
    tester.evaluate()
    
    # Removed external call to test.py.
    
if __name__ == '__main__':
    main()
