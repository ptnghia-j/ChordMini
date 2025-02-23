import os
import sys
import csv
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

from modules.utils.device import get_device

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Helper to load CSV with options and common processing.
def read_csv_file(fpath: Path, columns, extra_process=None):
    # Always use pandas for CSV reading.
    df = pd.read_csv(str(fpath), header=None, sep=',', engine='c',
                     quoting=csv.QUOTE_NONE, escapechar='\\',
                     dtype={0: str}, low_memory=False)
    df[0] = df[0].replace("", pd.NA).ffill()
    df.columns = columns
    if extra_process is not None:
        df = extra_process(df)
    return df

def load_chroma(fpath: Path) -> pd.DataFrame:
    # Load chroma CSV file; columns: piece, timestamp, and 12 pitch values.
    columns = ['piece', 'timestamp'] + [f'pitch_{i}' for i in range(12)]
    return read_csv_file(fpath, columns)

def load_chord(fpath: Path) -> pd.DataFrame:
    # Load chord CSV file; shift timestamp by +0.1 sec.
    def shift_timestamp(df):
        df['timestamp'] = df['timestamp'] + 0.1
        return df
    columns = ['piece', 'timestamp', 'chord']
    return read_csv_file(fpath, columns, extra_process=shift_timestamp)

class CrossDataset(Dataset):
    """
    Custom dataset that aggregates chroma and chord CSV files.
    It merges chroma rows with chord events and builds a unified chord mapping.
    """
    def __init__(self, chroma_dir: str, label_dir: str, transform=None,
                 seq_len: int = 4, chord_mapping: dict = None):
        self.chroma_dir = Path(chroma_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.seq_len = seq_len
        self.chord_mapping = chord_mapping
        self.samples = []
        self.chord_to_idx = {}
        self._build_file_dicts()
        self.aggregate_data()

    def _build_file_dicts(self):
        self.chroma_f = {f: self.chroma_dir / f for f in os.listdir(self.chroma_dir)
                         if f.endswith('.csv')}
        self.label_f = {f: self.label_dir / f for f in os.listdir(self.label_dir)
                        if f.endswith('.csv')}

    def aggregate_data(self):
        """
        Aggregates and merges data from chroma and chord files.
        1. Reads all chroma files and chord files concurrently.
        2. Merges chroma with chord events for each piece.
        3. Builds a chord mapping from the collected chord labels.
        4. Splits data into training (80%) and evaluation (20%).
        """
        # Reads files concurrently, merges chroma and chord events per piece.
        with ThreadPoolExecutor() as executor:
            chroma_dfs = list(executor.map(load_chroma, self.chroma_f.values()))
            chord_dfs = list(executor.map(load_chord, self.label_f.values()))
        chroma_df = pd.concat(chroma_dfs, ignore_index=True).sort_values(['piece', 'timestamp'])
        chord_df = pd.concat(chord_dfs, ignore_index=True).sort_values(['piece', 'timestamp'])
        samples = []
        for piece, chroma_group in chroma_df.groupby('piece'):
            chroma_group = chroma_group.sort_values('timestamp')
            chords_piece = chord_df[chord_df['piece'] == piece].sort_values('timestamp')
            merged = pd.merge_asof(chroma_group, chords_piece, on='timestamp', direction='backward')
            merged['chord'] = merged['chord'].fillna("N")
            for row in merged.itertuples(index=False):
                samples.append({
                    'piece': piece,
                    'timestamp': row.timestamp,
                    'chroma': [getattr(row, f'pitch_{i}') for i in range(12)],
                    'chord_label': row.chord
                })
        self.samples = samples
        if self.chord_mapping is not None:
            self.chord_to_idx = self.chord_mapping
        else:
            chords_set = {s['chord_label'] for s in self.samples}
            self.chord_to_idx = {chord: idx + 1 for idx, chord in enumerate(sorted(chords_set))}
        eval_ratio = 0.2
        self.split_index = int(len(self.samples) * (1 - eval_ratio))

    def get_train_iterator(self, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
        """
        Returns a DataLoader for the training set grouped by piece.
        """
        groups = defaultdict(list)
        for i in range(self.split_index):
            groups[self.samples[i]['piece']].append(i)
        pieces = list(groups.keys())
        if shuffle:
            import random
            random.shuffle(pieces)
        indices = [i for piece in pieces for i in groups[piece]]
        train_subset = Subset(self, indices)
        return DataLoader(train_subset, batch_size=batch_size, shuffle=False)

    def get_eval_iterator(self, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
        """
        Returns a DataLoader for the evaluation set.
        """
        eval_subset = Subset(self, range(self.split_index, len(self.samples)))
        return DataLoader(eval_subset, batch_size=batch_size, shuffle=shuffle)

    def get_batch_scheduler(self, batch_size: int):
        """
        Yields batches in order, grouping samples by piece.
        Each yielded batch contains only samples from a single piece.
        """
        groups = defaultdict(list)
        for sample in self.samples:
            groups[sample['piece']].append(sample)
        for piece, samples in groups.items():
            samples = sorted(samples, key=lambda s: s['timestamp'])
            for i in range(0, len(samples), batch_size):
                yield samples[i: i + batch_size]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        # Modified to always assign the chord of the first sample
        # and remove unnecessary variables.
        first_sample = self.samples[idx]
        first_chord = first_sample['chord_label'].strip('"')
        sequence = []
        i = 0
        while (
            i < self.seq_len and 
            (idx + i) < len(self.samples) and 
            self.samples[idx + i]['piece'] == first_sample['piece'] and 
            self.samples[idx + i]['chord_label'].strip('"') == first_chord
        ):
            sequence.append(torch.tensor(self.samples[idx + i]['chroma'], dtype=torch.float))
            i += 1
        while i < self.seq_len:
            sequence.append(torch.zeros(12, dtype=torch.float))
            i += 1
        chroma_seq = torch.stack(sequence, dim=0).to(get_device())
        
        sample_out = {
            'chroma': chroma_seq,
            'chord_idx': self.chord_to_idx[first_chord],
            'chord_label': first_chord
        }
        if self.transform:
            sample_out = self.transform(sample_out)
        return sample_out

def get_unified_mapping(label_dirs: list) -> dict:
    """
    Creates a unified chord mapping from a list of label directories.
    """
    chord_set = set()
    for label_dir in label_dirs:
        label_dir = Path(label_dir)
        for fname in os.listdir(label_dir):
            if fname.endswith('.csv'):
                fpath = label_dir / fname
                df = pd.read_csv(str(fpath), header=None, sep=',', engine='c',
                                 quoting=csv.QUOTE_NONE, escapechar='\\')
                df[0] = df[0].replace("", pd.NA).ffill()
                df.columns = ['piece', 'timestamp', 'chord']
                chord_set.update({str(chord).strip('"') for chord in df['chord'].fillna("N").unique()})
    return {chord: idx + 1 for idx, chord in enumerate(sorted(chord_set))}

def main():
    # Removed distributed initialization.
    device = get_device()
    # Calculate the project root (three levels up from this file)
    project_root = Path(__file__).resolve().parents[2]

    # Define dataset directories
    chroma_dir = project_root / 'data' / 'cross-era_chroma-nnls'
    label_dir = project_root / 'data' / 'cross-era_chords-chordino'
    chroma_dir2 = project_root / 'data' / 'cross-composer_chroma-nnls'
    label_dir2 = project_root / 'data' / 'cross-composer_chords-chordino'

    # Compute unified chord mapping from both label directories.
    unified_mapping = get_unified_mapping([label_dir, label_dir2])
    print("Unified chord mapping (total labels):", len(unified_mapping))
    print("Unified chord mapping:", unified_mapping)

    # Create datasets using the unified mapping.
    dataset1 = CrossDataset(str(chroma_dir), str(label_dir), chord_mapping=unified_mapping)
    dataset2 = CrossDataset(str(chroma_dir2), str(label_dir2), chord_mapping=unified_mapping)

    # Concatenate datasets vertically.
    combined_dataset = ConcatDataset([dataset1, dataset2])
    print("Total combined samples:", len(combined_dataset))

    # Always use standard DataLoader.
    loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)

    # Debug: print first few samples.
    print("-- Combined Dataset first 10 samples --")
    for i in range(min(10, len(loader.dataset))):
        sample = loader.dataset[i]
        print(f"Instance {i}: Label: {sample['chord_label']}, Chroma: {sample['chroma']}")

if __name__ == '__main__':
    main()
