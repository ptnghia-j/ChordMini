import os
import sys  # added import
import csv  # added import

# Insert project root (two levels up) into sys.path BEFORE any module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from modules.utils.device import get_device  # added import

class CrossEraDataset(Dataset):
    def __init__(self, chroma_dir, label_dir, transform=None, seq_len=5):
        # Build file dictionaries from directories using all CSV files
        self.chroma_f = {f: os.path.join(chroma_dir, f)
                         for f in os.listdir(chroma_dir) if f.endswith('.csv')}
        self.label_f = {f: os.path.join(label_dir, f)
                        for f in os.listdir(label_dir) if f.endswith('.csv')}
        self.transform = transform
        self.seq_len = seq_len  # added sequence length hyperparameter
        self.samples = []
        self.chord_to_idx = {}
        self.aggregate_data()
    
    def aggregate_data(self):
        """
        1. Read all chroma files (format: no header;
           col0: piece (only provided when piece changes),
           col1: timestamp,
           col2-13: 12 musical pitch values).
           Fill missing piece names by forward filling.
        2. Read all chord files (format: no header;
           col0: piece,
           col1: timestamp,
           col2: chord label). Fill missing piece names,
           shift timestamp by +0.1 sec.
        3. For each piece, merge chroma rows (which are at 0.1s intervals)
           with chord events by merging on timestamp (using merge_asof) and forward filling chord.
        4. Build chord mapping from the collected chord labels.
        """
        # ----- Process chroma files -----
        chroma_list = []
        for fname, fpath in self.chroma_f.items():
            print("Processing chroma file:", fpath, flush=True)  # updated debug print with flush=True
            df = pd.read_csv(fpath, header=None, sep=',', engine='python', quoting=csv.QUOTE_NONE, escapechar='\\')  # changed separator to comma
            # Forward-fill piece names; assume 'Blank' indicates no new piece.
            df[0] = df[0].replace("Blank", pd.NA).ffill()
            # Rename columns: col0->'piece', col1->'timestamp', rest are pitch features.
            df.columns = ['piece', 'timestamp'] + [f'pitch_{i}' for i in range(12)]
            chroma_list.append(df)
        chroma_df = pd.concat(chroma_list, ignore_index=True)
        chroma_df = chroma_df.sort_values(['piece', 'timestamp'])
        
        # ----- Process chord files -----
        chord_list = []
        for fname, fpath in self.label_f.items():
            print("Processing chord file:", fpath, flush=True)  # updated debug print with flush=True
            df = pd.read_csv(fpath, header=None, sep=',', engine='python', quoting=csv.QUOTE_NONE, escapechar='\\')  # changed separator to comma
            df[0] = df[0].replace("Blank", pd.NA).ffill()
            df.columns = ['piece', 'timestamp', 'chord']
            # Shift the timestamp by +0.1 seconds as required
            df['timestamp'] = df['timestamp'] + 0.1
            chord_list.append(df)
        chord_df = pd.concat(chord_list, ignore_index=True)
        chord_df = chord_df.sort_values(['piece', 'timestamp'])
        
        # ----- Join chroma and chord data per piece -----
        samples = []
        for piece, chroma_group in chroma_df.groupby('piece'):
            chroma_group = chroma_group.sort_values('timestamp')
            chords_piece = chord_df[chord_df['piece'] == piece].sort_values('timestamp')
            merged = pd.merge_asof(chroma_group, chords_piece, on='timestamp', direction='backward')
            merged['chord'] = merged['chord'].fillna("N")
            for row in merged.itertuples(index=False):
                sample = {
                    'piece': piece,
                    'timestamp': row.timestamp,
                    'chroma': [getattr(row, f'pitch_{i}') for i in range(12)],
                    'chord_label': row.chord
                }
                samples.append(sample)
        self.samples = samples
        
        # ----- Build chord mapping -----
        chords_set = set(s['chord_label'] for s in self.samples)
        # Map chords to numbers from 1 to 92 (if fewer than 92 chords are present,
        # the mapping will cover the available set; missing ones can be integrated later)
        self.chord_to_idx = {chord: idx+1 for idx, chord in enumerate(sorted(chords_set))}
        
        # Define evaluation split: reserve 20% for evaluation, 80% for training.
        eval_ratio = 0.2
        self.split_index = int(len(self.samples) * (1 - eval_ratio))
        # Removed erroneous DataLoader return preserving sample order

    def get_train_iterator(self, batch_size=128, shuffle=True):
        # Use the dataset's training subset without shuffling (order is important)
        train_subset = torch.utils.data.Subset(self, range(self.split_index))
        return DataLoader(train_subset, batch_size=batch_size, shuffle=False)  # fixed shuffle
    
    def get_eval_iterator(self, batch_size=128, shuffle=False):
        eval_subset = torch.utils.data.Subset(self, range(self.split_index, len(self.samples)))
        return DataLoader(eval_subset, batch_size=batch_size, shuffle=shuffle)
    
    def get_batch_scheduler(self, batch_size):
        """
        Yield batches in order, grouping samples by piece.
        Each yielded batch contains only samples from a single piece.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for sample in self.samples:
            groups[sample['piece']].append(sample)
        # Yield ordered batches per piece
        for piece, samples in groups.items():
            samples = sorted(samples, key=lambda s: s['timestamp'])
            for i in range(0, len(samples), batch_size):
                yield samples[i: i+batch_size]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Build sequence of chroma vectors with length = self.seq_len and ensure same-piece
        sequence = []
        first_piece = self.samples[idx]['piece']
        for i in range(self.seq_len):
            if idx + i < len(self.samples) and self.samples[idx+i]['piece'] == first_piece:
                sample_i = self.samples[idx+i]
                tensor_i = torch.tensor(sample_i['chroma'], dtype=torch.float)
                sequence.append(tensor_i)
            else:
                # Pad with zeros if sequence shorter than desired length
                sequence.append(torch.zeros(12, dtype=torch.float))
        chroma_seq = torch.stack(sequence, dim=0)  # shape: [T, 12]
        chroma_seq = chroma_seq.to(get_device())
        # Use the chord index from the last valid sample as target
        last_sample = self.samples[min(idx+self.seq_len-1, len(self.samples)-1)]
        chord_idx = self.chord_to_idx.get(last_sample['chord_label'], 0)
        sample_out = {'chroma': chroma_seq, 'chord_idx': chord_idx, 'chord_label': last_sample['chord_label']}
        if self.transform:
            sample_out = self.transform(sample_out)
        return sample_out

# For testing purposes:
def main():
    import os
    # Calculate project root: three levels up from this file.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chroma_dir = os.path.join(project_root, 'data', 'cross-era_chroma-nnls')
    label_dir = os.path.join(project_root, 'data', 'cross-era_chords-chordino')
    
    dataset = CrossEraDataset(chroma_dir, label_dir)
    print("Total samples:", len(dataset))
    count = min(20000, len(dataset))
    for i in range(count):
        sample = dataset[i]
        print(f"Instance {i}: Label: {sample['chord_label']}, Chroma: {sample['chroma']}")
    
if __name__ == '__main__':
    main()
