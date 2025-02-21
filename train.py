import sys
import os
import pandas as pd  # Newly added import
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader, random_split
from functools import partial
from modules.utils.device import get_device

from modules.data.CrossEraDataset import CrossEraDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.Trainer import BaseTrainer
from modules.training.Schedulers import CosineScheduler  # new import

# -------------------------
# Utility: build chord mapping from dataset
def build_chord_mapping(dataset):
    chords = set()
    # Iterate over each unique match_key (each file pair) instead of every segment
    for match_key in dataset.keys:
        label_key = dataset.label_keys[match_key]
        # Read label file only once per file pair
        label_df = pd.read_csv(dataset.label_f[label_key], header=None)
        segments = dataset.get_piece(label_df)
        for seg in segments:
            # Extract unique chord labels from each segment (assuming column 2 holds the chord label)
            chord_segment = seg.iloc[:, 2].astype(str).unique().tolist()
            chords.update(chord_segment)
    chord_to_idx = {chord: idx for idx, chord in enumerate(sorted(chords))}
    return chord_to_idx

# -------------------------
# Custom collate function: each sample is a single chroma vector; add a time dimension then duplicate channel.
def custom_collate(batch, chord_to_idx):
    inputs, targets = [], []
    for sample in batch:
        # sample["chroma"] has shape [12], unsqueeze to [1,12] then duplicate channel -> [2,1,12]
        chroma_tensor = sample["chroma"].unsqueeze(0).repeat(2, 1, 1)
        inputs.append(chroma_tensor)
        tgt = torch.tensor(chord_to_idx.get(sample["chord_label"], 0), dtype=torch.long, device=chroma_tensor.device)
        targets.append(tgt)
    batch_inputs = torch.stack(inputs, dim=0)  # shape: [B, 2, 1, 12]
    # Unsqueeze targets to have shape [B, 1] so that loss function gets matching dimensions
    batch_targets = torch.stack(targets, dim=0).unsqueeze(1)  # shape: [B, 1]
    return batch_inputs, batch_targets

# -------------------------
# Main training function
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    chroma_dir = os.path.join(project_root, "data", "cross-era_chroma-nnls")
    label_dir  = os.path.join(project_root, "data", "cross-era_chords-chordino")
    
    # Create the dataset. (Ensure directories exist.)
    dataset = CrossEraDataset(chroma_dir, label_dir)
    # Build chord mapping from dataset samples.
    chord_to_idx = build_chord_mapping(dataset)
    print("Number of chord classes:", len(chord_to_idx))
    
    # Split dataset (e.g., 80% train, 20% validation)
    total = len(dataset)
    train_len = int(0.8 * total)
    val_len = total - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    
    # Create DataLoaders with our custom collate function.
    collate_fn = partial(custom_collate, chord_to_idx=chord_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)  # lowered batch size to 4
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)      # lowered batch size to 4

    # Instantiate model: note it now expects input shape [B, 2, T, 12]
    model = ChordNet(n_freq=12, n_group=4, f_head=1, t_head=1, d_head=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Use CosineScheduler for learning rate scheduling.
    warmup_steps = 3
    num_epochs = 10
    scheduler = CosineScheduler(
        optimizer,
        max_update=num_epochs,
        base_lr=0.001,
        final_lr=0.0001,
        warmup_steps=warmup_steps,
        warmup_begin_lr=0.0
    )
    
    # Use GPU if available.
    device = get_device()
    
    # Initialize BaseTrainer with the selected device.
    trainer = BaseTrainer(model, optimizer, scheduler=scheduler, num_epochs=num_epochs, device=device)

    # Start training
    trainer.train(train_loader, val_loader=val_loader)

if __name__ == '__main__':
    main()
