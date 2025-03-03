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
from modules.utils.chord_normalization import normalize_chord  # NEW: import normalization function

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
    def shift_timestamp(df):
        df['timestamp'] = df['timestamp'] + 0.1
        return df
    columns = ['piece', 'timestamp', 'chord']
    df = read_csv_file(fpath, columns, extra_process=shift_timestamp)
    df['chord'] = df['chord'].fillna("N").apply(normalize_chord)
    # NEW: Check for 'nan' labels originating from the CSV file.
    # print(f"[DEBUG] {fpath}: Checking for 'nan' labels in chord column.")
    n_nan = (df['chord'] == 'nan').sum()
    if n_nan > 0:
        print(f"[DEBUG] {fpath}: Found {n_nan} occurrences of 'nan' in chord labels.")
    return df

class CrossDataset(Dataset):
    """
    CrossEraDataset for cross-era data.
    This dataset combines chroma vectors (0.1-second instances with 12 pitch values)
    with chord labels expanded from chord change files.
    Each sequence is a contiguous set of chroma vectors with their corresponding chord labels.
    The parameter "stride" controls the step between segments:
      - stride == seq_len: non-overlapping segments.
      - stride < seq_len: overlapping segments.
    """
    def __init__(self, chroma_dir: str, label_dir: str, transform=None,
                 seq_len: int = 4, chord_mapping: dict = None, stride: int = None):
        self.chroma_dir = Path(chroma_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.seq_len = seq_len
        # NEW: Use provided stride or default to seq_len (non-overlapping)
        self.stride = stride if stride is not None else seq_len
        self.chord_mapping = chord_mapping
        self.samples = []
        self.chord_to_idx = {}
        self._build_file_dicts()
        self.aggregate_data()
        # Modify how ignore_index is set
        if self.chord_mapping is not None:
            self.chord_to_idx = self.chord_mapping
            # Set ignore_index to the value for "N" or to the maximum index + 1
            self.ignore_index = self.chord_mapping.get("N", len(self.chord_mapping))
        else:
            chords_set = {s['chord_label'] for s in self.samples}
            self.chord_to_idx = {chord: idx for idx, chord in enumerate(sorted(chords_set))}
            # When building our own mapping, put "N" at the end
            if "N" not in self.chord_to_idx:
                self.chord_to_idx["N"] = len(self.chord_to_idx)
            self.ignore_index = self.chord_to_idx["N"]
        # NEW: Compute mutually exclusive segment indices split for 8-1-1 using segment_indices length.
        total_segs = len(self.segment_indices)
        self.train_indices = list(range(0, int(total_segs * 0.8)))
        self.eval_indices  = list(range(int(total_segs * 0.8), int(total_segs * 0.9)))
        self.test_indices  = list(range(int(total_segs * 0.9), total_segs))

    def _build_file_dicts(self):
        self.chroma_f = {f: self.chroma_dir / f for f in os.listdir(self.chroma_dir)
                         if f.endswith('.csv')}
        self.label_f = {f: self.label_dir / f for f in os.listdir(self.label_dir)
                        if f.endswith('.csv')}

    def aggregate_data(self):
        with ThreadPoolExecutor() as executor:
            chroma_dfs = list(executor.map(load_chroma, self.chroma_f.values()))
            chord_dfs = list(executor.map(load_chord, self.label_f.values()))
        chroma_df = pd.concat(chroma_dfs, ignore_index=True).sort_values(['piece', 'timestamp'])
        chord_df = pd.concat(chord_dfs, ignore_index=True).sort_values(['piece', 'timestamp'])
        samples = []
        for piece, chroma_group in chroma_df.groupby('piece'):
            chroma_group = chroma_group.sort_values('timestamp')
            chords_piece = chord_df[chord_df['piece'] == piece].sort_values('timestamp')
            merged = pd.merge_asof(chroma_group, chords_piece,
                                   on='timestamp',
                                   direction='backward')
            merged['chord'] = merged['chord'].astype(str).ffill().fillna("N")
            # NEW: Normalize labels using the normalize_chord function
            merged['chord'] = merged['chord'].apply(normalize_chord)
            for row in merged.itertuples(index=False):
                chroma_vector = [getattr(row, f'pitch_{i}') for i in range(12)]
                samples.append({
                    'piece': piece,
                    'timestamp': row.timestamp,
                    'chroma': chroma_vector,
                    'chord_label': row.chord
                })
    
        samples = [s for s in samples if s['chord_label'] not in ["N", "nan"] and not all(float(val)==0 for val in s['chroma'])]
        self.samples = samples

        # Compute segment indices for each contiguous block using stride.
        # Instead of just storing the start index, store a (start, end) tuple, where "end" is the index at which the piece ends.
        self.segment_indices = []
        current_piece = None
        current_start = 0
        for i, sample in enumerate(self.samples):
            if sample['piece'] != current_piece:
                if current_piece is not None:
                    block_length = i - current_start
                    for j in range(0, block_length, self.stride):
                        self.segment_indices.append((current_start + j, i))
                current_piece = sample['piece']
                current_start = i
        if current_piece is not None:
            block_length = len(self.samples) - current_start
            for j in range(0, block_length, self.stride):
                self.segment_indices.append((current_start + j, len(self.samples)))

    def get_train_iterator(self, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
        # Use precomputed train_indices.
        return DataLoader(Subset(self, self.train_indices), batch_size=batch_size, shuffle=shuffle)

    def get_eval_iterator(self, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
        return DataLoader(Subset(self, self.eval_indices), batch_size=batch_size, shuffle=shuffle)

    def get_batch_scheduler(self, batch_size: int):
        groups = defaultdict(list)
        for sample in self.samples:
            groups[sample['piece']].append(sample)
        for piece, samples in groups.items():
            samples = sorted(samples, key=lambda s: s['timestamp'])
            for i in range(0, len(samples), batch_size):
                yield samples[i: i + batch_size]

    def __len__(self) -> int:
        return len(self.segment_indices)

    def __getitem__(self, idx: int) -> dict:
        seg_start, seg_end = self.segment_indices[idx]
        sequence = []
        label_seq = []
        for i in range(self.seq_len):
            pos = seg_start + i
            if pos < seg_end:
                sample_i = self.samples[pos]
                ch_vec = torch.tensor(sample_i['chroma'], dtype=torch.float)
                chord_label = normalize_chord(sample_i['chord_label'])
                # if chord_label == 'nan' or torch.all(ch_vec == 0):
                #     # Use proper ignore_index instead of assuming "N" is at index 0
                #     label_seq.append(self.ignore_index)
                # else:
                try:
                    label_seq.append(self.chord_to_idx[chord_label])
                except KeyError:
                    # Only warn once per chord
                    if not hasattr(self, '_warned_chords'):
                        self._warned_chords = set()
                    if chord_label not in self._warned_chords:
                        print(f"Warning: Unknown chord label '{chord_label}', using ignore_index")
                        self._warned_chords.add(chord_label)
                    label_seq.append(self.ignore_index)
                sequence.append(ch_vec)
            else:
                sequence.append(torch.zeros(12, dtype=torch.float))
                label_seq.append(self.ignore_index)
        
        from collections import Counter
        target = Counter(label_seq).most_common(1)[0][0]
        sample_out = {
            'chroma': torch.stack(sequence, dim=0),
            'chord_idx': target,
            'chord_label': self.samples[seg_start]['chord_label']
        }
        if self.transform:
            sample_out = self.transform(sample_out)
        return sample_out

def get_unified_mapping(label_dirs: list) -> dict:
    """
    Creates a unified chord mapping from a list of label directories.
    The keys are normalized by:
      - Removing surrounding quotes and extra whitespace.
      - Converting to lower-case.
      - Converting "nan" or "n" to "N" (the no-chord indicator).
      - For simple majors (e.g., "c_maj"), removing the "_maj" suffix.
      - For simple minors (e.g., "c_min"), converting to the more compact "cm".
    """
    chord_set = set()
    for label_dir in label_dirs:
        label_dir = Path(label_dir)
        for fname in os.listdir(label_dir):
            if fname.endswith('.csv'):
                fpath = label_dir / fname
                df = pd.read_csv(
                    str(fpath), header=None, sep=',', engine='c',
                    quoting=csv.QUOTE_NONE, escapechar='\\'
                )
                df[0] = df[0].replace("", pd.NA).ffill()
                df.columns = ['piece', 'timestamp', 'chord']
                # Apply normalization to each chord label.
                chord_set.update({normalize_chord(chord) for chord in df['chord'].fillna("N").unique()})
    # Add "N" but ensure it's not given index 0 by sorting the set without "N" first
    mapping = {}
    chords_without_n = sorted([c for c in chord_set if c != "N"])
    idx = 0
    for chord in chords_without_n:
        mapping[chord] = idx
        idx += 1
    # Now add "N" at the end with the highest index
    mapping["N"] = idx
    return mapping

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

    # Debug: print first 100 samples.
    print("-- Combined Dataset first 100 samples --")
    for i in range(min(100, len(loader.dataset))):
        sample = loader.dataset[i]
        print(f"Instance {i}: Chroma tensor: {sample['chroma']}, Chord Mapping: {sample['chord_idx']}")

if __name__ == '__main__':
    main()
