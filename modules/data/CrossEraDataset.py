import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from modules.utils.device import get_device  # added import

class CrossEraDataset(Dataset):
    def __init__(self, chroma_dir, label_dir, transform=None, debug=False):
        chroma_f = {os.path.splitext(f)[0]: os.path.join(chroma_dir, f)
                        for f in os.listdir(chroma_dir) if f.endswith('.csv')}
        label_f = {os.path.splitext(f)[0]: os.path.join(label_dir, f)
                       for f in os.listdir(label_dir) if f.endswith('.csv')}
        self.chroma_f = chroma_f
        self.label_f = label_f

        self.chroma_keys = {k.split('_', 1)[1]: k for k in chroma_f if '_' in k}
        self.label_keys = {k.split('_', 1)[1]: k for k in label_f if '_' in k}

        common = sorted(list(set(self.chroma_keys.keys()) & set(self.label_keys.keys())))
        if not common:
            raise ValueError("No matching file pairs found between chroma and label directories.")
        self.keys = common
        self.transform = transform
        self.debug = debug

        # Build an index mapping: each element is (match_key, segment_index)
        self.index_map = []
        for match_key in self.keys:
            label_key = self.label_keys[match_key]
            label_df = pd.read_csv(self.label_f[label_key], header=None)
            segments = self.get_piece(label_df)
            for seg_idx in range(len(segments)):
                self.index_map.append((match_key, seg_idx))

    def __len__(self):
        return len(self.index_map)

    def get_piece(self, df):
        # Debug: print shape of DataFrame being processed
        if self.debug:
            print(f"get_piece: input df shape: {df.shape}")
        # Ensure df is a DataFrame (if read_csv returns a Series, convert it)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        marker = df.iloc[:, 0].notnull() & (df.iloc[:, 0].astype(str).str.strip() != "")
        indices = marker[marker].index.tolist()
        if self.debug:
            print(f"get_piece: found indices: {indices}")
        segments = []
        if indices:
            if indices[0] > 0:
                segments.append(df.iloc[:indices[0]])
            for i, idx in enumerate(indices):
                start = idx
                end = indices[i + 1] if (i + 1) < len(indices) else len(df)
                segments.append(df.iloc[start:end])
        else:
            segments.append(df)
        if self.debug:
            print(f"get_piece: generated {len(segments)} segments")
        return segments
  
    def __getitem__(self, idx):
        match_key, seg_idx = self.index_map[idx]
        if self.debug:
            print(f"__getitem__: index {idx}, match_key {match_key}, segment index {seg_idx}")
        chroma_key = self.chroma_keys[match_key]
        label_key = self.label_keys[match_key]
        chroma_df = pd.read_csv(self.chroma_f[chroma_key], header=None, low_memory=False)
        label_df = pd.read_csv(self.label_f[label_key], header=None)
        if self.debug:
            print(f"__getitem__: chroma_df shape: {chroma_df.shape}, label_df shape: {label_df.shape}")

        chroma_seg = self.get_piece(chroma_df)
        label_seg = self.get_piece(label_df)
        if self.debug:
            print(f"__getitem__: chroma segments: {len(chroma_seg)}, label segments: {len(label_seg)}")
        if len(chroma_seg) != len(label_seg):
            raise ValueError("Number of segments do not match between chroma and label files.")

        # Use only the first chroma vector from the selected segment
        seg_chroma = chroma_seg[seg_idx]
        seg_label = label_seg[seg_idx]
        # Take the first row
        row = seg_chroma.iloc[0]
        piece_name = row[0]
        time_val = pd.to_numeric(seg_chroma.iloc[0, 1], errors='coerce')
        chroma_data = seg_chroma.iloc[0, 2:].apply(pd.to_numeric, errors='coerce').fillna(0).values
        # Assume chord label in label segment is from column 2 (first data column)
        chord_label = seg_label.iloc[0, 2]
        
        device = get_device()
        chroma_tensor = torch.tensor(chroma_data, device=device, dtype=torch.float32)
        if self.transform:
            chroma_tensor = self.transform(chroma_tensor)
        if self.debug:
            print(f"Loaded piece '{piece_name}' with chroma shape: {chroma_tensor.shape}")

        return {
            "piece_name": piece_name,
            "chroma": chroma_tensor,  # shape: [12]
            "time": time_val,
            "chord_label": chord_label
        }

def find_project_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("Project root not found")
        current = parent

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_dir)
    chroma_dir = os.path.join(project_root, "data", "cross-era_chroma-nnls")
    label_dir = os.path.join(project_root, "data", "cross-era_chords-chordino")

    for d in (chroma_dir, label_dir):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    dataset = CrossEraDataset(chroma_dir, label_dir, debug=True)
    print(len(dataset))
    sample = dataset[0]
    print(sample["piece_name"], sample["chroma"].shape, sample["time"], sample["chord_label"])