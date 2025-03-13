import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Sampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt

from modules.utils.device import get_device
from modules.data.CrossDataset import CrossDataset, get_unified_mapping
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.Trainer import BaseTrainer
from modules.training.Schedulers import CosineScheduler
from modules.utils.mir_eval_modules import root_majmin_score_calculation, large_voca_score_calculation

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
    def __init__(self, model, test_loader, device, unified_mapping=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        # Store mapping to inspect predictions
        self.unified_mapping = unified_mapping
        self.idx_to_chord = {v: k for k, v in unified_mapping.items()} if unified_mapping else None

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        # Debug counters
        pred_counter = Counter()
        target_counter = Counter()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                inputs = batch['chroma'].to(self.device)
                targets = batch['chord_idx'].to(self.device)
                
                # Debug first few batches
                if batch_idx == 0:
                    print(f"DEBUG: Input shape: {inputs.shape}, target shape: {targets.shape}")
                    print(f"DEBUG: First few targets: {targets[:10]}")
                
                # Get raw logits before prediction
                logits, _ = self.model(inputs)
                
                # Check if logits have reasonable values
                if batch_idx == 0:
                    print(f"DEBUG: Logits shape: {logits.shape}")
                    print(f"DEBUG: Logits mean: {logits.mean().item()}, std: {logits.std().item()}")
                    print(f"DEBUG: First batch sample logits (max 5 values): {logits[0, :5]}")
                
                # Get predictions
                preds = self.model.predict(inputs)
                
                # Convert and store
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                # Count distribution
                pred_counter.update(preds_np)
                target_counter.update(targets_np)
                
                all_preds.extend(preds_np)
                all_targets.extend(targets_np)
                
                # Debug first batch predictions vs targets
                if batch_idx == 0:
                    print(f"DEBUG: First batch - Predictions: {preds_np[:10]}")
                    print(f"DEBUG: First batch - Targets: {targets_np[:10]}")
        
        # Print distribution statistics
        print("\nDEBUG: Target Distribution (top 10):")
        for idx, count in target_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") if self.idx_to_chord else str(idx)
            print(f"Target {idx} ({chord_name}): {count} occurrences ({count/len(all_targets)*100:.2f}%)")
            
        print("\nDEBUG: Prediction Distribution (top 10):")
        for idx, count in pred_counter.most_common(10):
            chord_name = self.idx_to_chord.get(idx, "Unknown") if self.idx_to_chord else str(idx)
            print(f"Prediction {idx} ({chord_name}): {count} occurrences ({count/len(all_preds)*100:.2f}%)")
        
        # NEW: Print complete test set chord distribution
        print("\nComplete Test Set Chord Distribution:")
        full_distribution = Counter(all_targets)
        for chord_idx in sorted(full_distribution):
            chord_name = self.idx_to_chord.get(chord_idx, str(chord_idx))
            print(f"Chord {chord_idx} ({chord_name}): {full_distribution[chord_idx]} occurrences")
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # Calculate WCSR if we have the mapping
        wcsr = 0.0
        if self.idx_to_chord:
            wcsr = weighted_chord_symbol_recall(all_targets, all_preds, self.idx_to_chord)
            print(f"\nWeighted Chord Symbol Recall (WCSR): {wcsr:.4f}")
        
        # Print confusion matrix for most common chords (top 10)
        if len(target_counter) > 1:
            print("\nAnalyzing most common predictions vs targets:")
            top_chords = [idx for idx, _ in target_counter.most_common(10)]
            for true_idx in top_chords:
                true_chord = self.idx_to_chord.get(true_idx, str(true_idx))
                pred_indices = [p for t, p in zip(all_targets, all_preds) if t == true_idx]
                if pred_indices:
                    pred_counts = Counter(pred_indices)
                    most_common_pred = pred_counts.most_common(1)[0][0]
                    most_common_pred_chord = self.idx_to_chord.get(most_common_pred, str(most_common_pred))
                    accuracy_for_chord = pred_counts.get(true_idx, 0) / len(pred_indices)
                    print(f"True: {true_chord} -> Most common prediction: {most_common_pred_chord} (Accuracy: {accuracy_for_chord:.2f})")
        
        print(f"\nTest Metrics:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Weighted Chord Symbol Recall: {wcsr:.4f}")

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
                     dropout=0.3,
                     ignore_index=unified_mapping.get("N")).to(device)  # Pass ignore_index directly
    
    if local_rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6)
    warmup_steps = 3  # increased warmup steps for a gentler start
    num_epochs = 20
    scheduler = CosineScheduler(optimizer, max_update=num_epochs, base_lr=1e-3,
                                final_lr=1e-6, warmup_steps=warmup_steps, warmup_begin_lr=1e-5)

    dist_counter = Counter()
    for ds in [dataset1, dataset2]:
        dist_counter.update([s['chord_label'] for s in ds.samples])
    # Use the keys from unified_mapping (ensuring order) rather than dist_counter keys.
    unified_keys = sorted(unified_mapping, key=lambda k: unified_mapping[k])
    total_samples = sum(dist_counter.values())
    
    # NEW FEATURE: Print total chord instances and each chord's distribution.
    print("Total chord instances:", total_samples)
    for ch in unified_keys:
        ratio = dist_counter[ch] / total_samples * 100
        print(f"Chord: {ch}, Count: {dist_counter[ch]}, Percentage: {ratio:.4f}%")
    
    # FIXED: Include N in dropped chords if it has 0 count
    # dropped = [ch for ch in unified_keys if ((dist_counter[ch] / total_samples) < 0.0005 or dist_counter[ch] == 0)]
    # if dropped:
    #     print("Dropping chords due to low distribution (<0.05%) or zero count:", dropped)
    dropped = []
    # Apply log scaling to class weights and set to zero for dropped chords; ensure float32 dtype
    # class_weights = np.array([0.0 if ch in dropped else np.log1p(total_samples / max(dist_counter.get(ch, 1), 1))
    #                          for ch in unified_keys], dtype=np.float32)
    class_weights = np.array([0.0 if ch in dropped else total_samples / max(dist_counter.get(ch, 1), 1)
                             for ch in unified_keys], dtype=np.float32)
    
    # Set extra high penalty for N class
    # n_index = unified_mapping.get("N")
    # if n_index is not None and n_index < len(class_weights):
    #     class_weights[n_index] = -100.0  # Strong negative weight to heavily discourage N prediction
    #     print(f"Setting special penalty for N class at index {n_index}")
        
    print("Computed class weights:", class_weights)
    
    # Create idx_to_chord mapping for the loss function
    idx_to_chord = {v: k for k, v in unified_mapping.items()}
    
    # Create the trainer with chord-aware loss
    trainer = BaseTrainer(model, optimizer, scheduler=scheduler,
                          num_epochs=num_epochs, device=device,
                          ignore_index=unified_mapping.get("N", len(unified_mapping)),  # Use the correct index for "N"
                          class_weights=class_weights,
                          idx_to_chord=idx_to_chord,
                          use_chord_aware_loss=False)  # Use chord-aware loss
    
    # NEW: Print complete chord distribution before training starts.
    print("Final Chord Distribution in Training Set:")
    for ch in unified_keys:
        ratio = dist_counter[ch] / total_samples * 100
        print(f"Chord: {ch}, Count: {dist_counter[ch]}, Percentage: {ratio:.2f}%")
    
    trainer.train(train_loader, val_loader=val_loader)
    
    print("Starting testing phase.")
    test_indices = partition_test_set(combined_dataset)
    test_loader = DataLoader(combined_dataset, batch_size=128, sampler=ListSampler(test_indices), pin_memory=True)
    
    print("=== Debug: Test set sample ===")
    test_batch = next(iter(test_loader))
    print("Test Chroma tensor:", test_batch['chroma'])
    print("Test labels:", test_batch['chord_idx'])
    
    tester = Tester(model, test_loader, device, unified_mapping=unified_mapping)
    tester.evaluate()

if __name__ == '__main__':
    main()