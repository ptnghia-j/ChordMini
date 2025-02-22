import sys
import os
import pandas as pd  # Newly added import
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import csv  # Needed for quoting options in pd.read_csv
import torch
from torch.utils.data import DataLoader, ConcatDataset
from modules.utils.device import get_device

from modules.data.CrossDataset import CrossDataset  # updated import
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.Trainer import BaseTrainer
from modules.training.Schedulers import CosineScheduler  # new import
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def get_unified_mapping(label_dirs):
    import pandas as pd, csv
    chord_set = set()
    for label_dir in label_dirs:
        for fname in os.listdir(label_dir):
            if fname.endswith('.csv'):
                fpath = os.path.join(label_dir, fname)
                df = pd.read_csv(fpath, header=None, sep=',', engine='python',
                                 quoting=csv.QUOTE_NONE, escapechar='\\')
                df[0] = df[0].replace("Blank", pd.NA).ffill()
                df.columns = ['piece', 'timestamp', 'chord']
                # Strip extraneous quotes from chord labels.
                chords = [str(c).strip('"') for c in df['chord'].fillna("N").unique()]
                chord_set.update(chords)
    mapping = {chord: idx+1 for idx, chord in enumerate(sorted(chord_set))}
    return mapping

def main():
    import os
    # Check if distributed training is enabled (i.e. environment variable RANK is set)
    if "RANK" in os.environ:
        print("Initializing distributed process group...")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        print("Distributed training not enabled; running on single process.")
        device = torch.device("cpu")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Original dataset directories.
    chroma_dir = os.path.join(project_root, "data", "cross-era_chroma-nnls")
    label_dir  = os.path.join(project_root, "data", "cross-era_chords-chordino")
    # New dataset directories.
    chroma_dir2 = os.path.join(project_root, "data", "cross-composer_chroma-nnls")
    label_dir2 = os.path.join(project_root, "data", "cross-composer_chords-chordino")
    
    # Compute unified chord mapping by scanning both label directories
    label_dirs = [label_dir, label_dir2]
    unified_mapping = get_unified_mapping(label_dirs)
    print("Unified chord mapping (total labels):", len(unified_mapping))
    
    # Create individual datasets using the unified mapping.
    dataset1 = CrossDataset(chroma_dir, label_dir, chord_mapping=unified_mapping)
    dataset2 = CrossDataset(chroma_dir2, label_dir2, chord_mapping=unified_mapping)
    
    # Combine them vertically.
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print("Total combined samples:", len(combined_dataset))
    
    # Create DataLoader using DistributedSampler only if the process group is initialized.
    if dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(combined_dataset)
        val_sampler = DistributedSampler(combined_dataset, shuffle=False)
        train_loader = DataLoader(combined_dataset, batch_size=128, sampler=train_sampler)
        val_loader = DataLoader(combined_dataset, batch_size=128, sampler=val_sampler)
    else:
        train_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
        val_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    
    # Use a lighter transformer suited for a 12-bin chromagram:
    model = ChordNet(n_freq=12, n_classes=274, n_group=3,   # changed n_group from 4 to 3
                     f_layer=2, f_head=4, 
                     t_layer=2, t_head=4, 
                     d_layer=2, d_head=4, 
                     dropout=0.2)
    model = model.to(device)
    # Wrap model with DistributedDataParallel.
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    warmup_steps = 1
    num_epochs = 5
    scheduler = CosineScheduler(
        optimizer,
        max_update=num_epochs,
        base_lr=0.01,
        final_lr=0.00001,
        warmup_steps=warmup_steps,
        warmup_begin_lr=0.001
    )
    
    # Pass the unified mapping value for "N" as ignore_index.
    trainer = BaseTrainer(model, optimizer, scheduler=scheduler,
                          num_epochs=num_epochs, device=device,
                          ignore_index=unified_mapping["N"])
    trainer.train(train_loader, val_loader=val_loader)
    
    # Clean up distributed resources.
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
