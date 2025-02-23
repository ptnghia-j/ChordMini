import sys
import os
import platform
import csv
import torch
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from modules.utils.device import get_device
from modules.data.CrossDataset import CrossDataset
from modules.models.Transformer.ChordNet import ChordNet
from modules.training.Trainer import BaseTrainer
from modules.training.Schedulers import CosineScheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def get_unified_mapping(label_dirs):
    chord_set = set()
    for label_dir in label_dirs:
        for fname in os.listdir(label_dir):
            if fname.endswith('.csv'):
                fpath = os.path.join(label_dir, fname)
                try:
                    import cudf
                    df = cudf.read_csv(fpath, header=None, sep=',', quoting=csv.QUOTE_NONE, escapechar='\\')
                    df[0] = df[0].replace("Blank", None).ffill()
                    df.columns = ['piece', 'timestamp', 'chord']
                    chords = df['chord'].fillna("N").unique().to_pandas().tolist()
                except ImportError:
                    df = pd.read_csv(fpath, header=None, sep=',', engine='c',
                                     quoting=csv.QUOTE_NONE, escapechar='\\')
                    df[0] = df[0].replace("Blank", pd.NA).ffill()
                    df.columns = ['piece', 'timestamp', 'chord']
                    chords = [str(c).strip('"') for c in df['chord'].fillna("N").unique()]
                chord_set.update(chords)
    return {chord: idx+1 for idx, chord in enumerate(sorted(chord_set))}

def main():
    if "RANK" in os.environ:
        print("Initializing distributed process group...")
        backend = "nccl" if (torch.cuda.is_available() and platform.system() == "Linux") else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        print("Distributed training not enabled; running on single process.")
        local_rank = 0
        device = torch.device("cpu")
    
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
    dataset1 = CrossDataset(chroma_dir, label_dir, chord_mapping=unified_mapping)
    dataset2 = CrossDataset(chroma_dir2, label_dir2, chord_mapping=unified_mapping)
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print("Total combined samples:", len(combined_dataset))
    if dist.is_initialized():
        train_sampler = DistributedSampler(combined_dataset)
        val_sampler = DistributedSampler(combined_dataset, shuffle=False)
        train_loader = DataLoader(combined_dataset, batch_size=128, sampler=train_sampler)
        val_loader = DataLoader(combined_dataset, batch_size=128, sampler=val_sampler)
    else:
        train_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
        val_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    model = ChordNet(n_freq=12, n_classes=274, n_group=3,
                     f_layer=2, f_head=4, 
                     t_layer=2, t_head=4, 
                     d_layer=2, d_head=4, 
                     dropout=0.5)
    model = model.to(device)
    if dist.is_initialized():
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    warmup_steps = 1
    num_epochs = 5
    scheduler = CosineScheduler(optimizer, max_update=num_epochs, base_lr=0.01,
                                final_lr=0.00001, warmup_steps=warmup_steps, warmup_begin_lr=0.001)
    trainer = BaseTrainer(model, optimizer, scheduler=scheduler,
                          num_epochs=num_epochs, device=device,
                          ignore_index=unified_mapping["N"])
    trainer.train(train_loader, val_loader=val_loader)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
