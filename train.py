import os
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
    for sample in dataset:
        for seg in sample:
            chords.update(seg["chord_labels"])
    chord_to_idx = {chord: idx for idx, chord in enumerate(sorted(chords))}
    return chord_to_idx

# -------------------------
# Custom collate function: convert each segment to a tensor 
def custom_collate(batch, chord_to_idx):
    inputs, targets = [], []
    max_T = 0
    for item in batch:  # each item is a list of segments
        for seg in item:
            # seg["chroma"]: numpy array of shape [T, 12]
            chroma_tensor = seg["chroma"].clone().detach().float()  # [T, 12]
            T = chroma_tensor.size(0)
            if T > max_T:
                max_T = T
            # Duplicate channel: now shape becomes [2, T, 12]
            chroma_tensor = chroma_tensor.unsqueeze(0).repeat(2, 1, 1)
            inputs.append(chroma_tensor)
            # Convert chord labels to indices (unknown labels get 0) and create tgt on same device as chroma_tensor
            tgt = torch.tensor([chord_to_idx.get(lbl, 0) for lbl in seg["chord_labels"]],
                               dtype=torch.long, device=chroma_tensor.device)
            targets.append(tgt)
    # Pad all samples along time dimension to max_T
    padded_inputs, padded_targets = [], []
    for inp, tgt in zip(inputs, targets):
        T = inp.size(1)
        if T < max_T:
            pad_size = max_T - T
            # Pad with zeros for input and -100 (ignore index) for targets using inp.device
            pad_inp = torch.zeros(inp.size(0), pad_size, inp.size(2), device=inp.device)
            inp = torch.cat([inp, pad_inp], dim=1)
            tgt = torch.cat([tgt, torch.full((pad_size,), -100, dtype=torch.long, device=inp.device)])
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    batch_inputs = torch.stack(padded_inputs, dim=0)   # shape: [B, 2, max_T, 12]
    batch_targets = torch.stack(padded_targets, dim=0)   # shape: [B, max_T]
    return batch_inputs, batch_targets

# -------------------------
# Main training function
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

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
