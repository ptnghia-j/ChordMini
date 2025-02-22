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
# Remove or comment out the custom_collate if not used.
# def custom_collate(batch):
#     inputs, targets = [], []
#     for sample in batch:
#         # sample["chroma"] has shape [12], unsqueeze to [1,12] then duplicate channel -> [2,1,12]
#         chroma_tensor = sample["chroma"].unsqueeze(0).repeat(2, 1, 1)
#         inputs.append(chroma_tensor)
#         # Use chord_idx already built in the dataset
#         tgt = torch.tensor(sample["chord_idx"], dtype=torch.long, device=chroma_tensor.device)
#         targets.append(tgt)
#     batch_inputs = torch.stack(inputs, dim=0)  # shape: [B, 2, 1, 12]
#     # Unsqueeze targets to have shape [B, 1] so that loss function gets matching dimensions
#     batch_targets = torch.stack(targets, dim=0).unsqueeze(1)  # shape: [B, 1]
#     return batch_inputs, batch_targets

# -------------------------
# Main training function
def main():
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    chroma_dir = os.path.join(project_root, "data", "cross-era_chroma-nnls")
    label_dir  = os.path.join(project_root, "data", "cross-era_chords-chordino")
    
    # Create the dataset. (Ensure directories exist.)
    dataset = CrossEraDataset(chroma_dir, label_dir)
    print("Number of chord classes:", len(dataset.chord_to_idx))
    
    # Use the dataset's iterator methods.
    train_loader = dataset.get_train_iterator(batch_size=128)
    val_loader = dataset.get_eval_iterator(batch_size=128)  # added evaluation iterator
    
    model = ChordNet(n_freq=12, n_group=4, f_head=1, t_head=1, d_head=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    warmup_steps = 3
    num_epochs = 5
    scheduler = CosineScheduler(
        optimizer,
        max_update=num_epochs,
        base_lr=0.001,
        final_lr=0.0001,
        warmup_steps=warmup_steps,
        warmup_begin_lr=0.0
    )
    
    device = get_device()
    model = model.to(device)  # move model to device
    
    trainer = BaseTrainer(model, optimizer, scheduler=scheduler, num_epochs=num_epochs, device=device)
    trainer.train(train_loader, val_loader=val_loader)

if __name__ == '__main__':
    main()
