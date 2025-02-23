import os
import sys
import random
import torch
from torch.utils.data import DataLoader, ConcatDataset, Sampler
from modules.models.Transformer.ChordNet import ChordNet
from modules.utils.device import get_device
from modules.data.CrossDataset import CrossDataset, get_unified_mapping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Remove previous CSV testing functions...
# ...existing code removed...

# Sampler helper.
class ListSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

# Updated partition_test_set helper using a 90-10 split.
def partition_test_set(concat_dataset):
    test_indices = []
    offset = 0
    for ds in concat_dataset.datasets:
        # Use last 10% for evaluation.
        test_start = int(len(ds) * 0.9)
        test_indices.extend(range(offset + test_start, offset + len(ds)))
        offset += len(ds)
    return test_indices

def load_model(checkpoint_path, device):
    # Use the same parameters as training.
    model = ChordNet(n_freq=12, n_classes=274, n_group=3,
                     f_layer=2, f_head=4,
                     t_layer=2, t_head=4,
                     d_layer=2, d_head=4,
                     dropout=0.3)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

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
    # ...existing dataset and checkpoint initialization removed...
    print("Test phase is integrated into training. Please run train.py to perform evaluation.")

if __name__ == '__main__':
    main()