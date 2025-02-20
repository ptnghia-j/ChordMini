import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from modules.utils.device import get_device  # added import

class CrossEraDataset(Dataset):
    def __init__(self, chroma_dir, label_dir, transform=None):
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

    def __len__(self):
        return len(self.keys)

    def get_piece(self, df):
        marker = df.iloc[:, 0].notnull() & (df.iloc[:, 0].astype(str).str.strip() != "")
        indices = marker[marker].index.tolist()
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
        return segments
  
    def __getitem__(self, idx):
        match_key = self.keys[idx]
        chroma_key = self.chroma_keys[match_key]
        label_key = self.label_keys[match_key]
        chroma_df = pd.read_csv(self.chroma_f[chroma_key], header=None, low_memory=False)
        label_df = pd.read_csv(self.label_f[label_key], header=None)

        chroma_seg = self.get_piece(chroma_df)
        label_seg = self.get_piece(label_df)
        if len(chroma_seg) != len(label_seg):
            raise ValueError("Number of segments do not match between chroma and label files.")
        
        pieces = []
        for chroma, label in zip(chroma_seg, label_seg):
            piece_name = chroma.iloc[0, 0]
            chroma_times = pd.to_numeric(chroma.iloc[:, 1], errors='coerce').values
            chroma_data = chroma.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').fillna(0).values
            label_times = pd.to_numeric(label.iloc[:, 1], errors='coerce').values
            label_data = label.iloc[:, 2].astype(str).values

            chord_labels = []
            i, n = 0, len(label_times)
            for t in chroma_times:
                while i < n - 1 and label_times[i + 1] < t:
                    i += 1
                chord_labels.append(label_data[i])
            
            device = get_device()  # use utils device check
            chroma_tensor = torch.tensor(chroma_data, device=device, dtype=torch.float32)
            times_tensor = torch.tensor(chroma_times, device=device, dtype=torch.float32)

            pieces.append({
                "piece_name": piece_name,
                "chroma": chroma_tensor,
                "times": times_tensor,
                "chord_labels": chord_labels
            })
        return pieces

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

    dataset = CrossEraDataset(chroma_dir, label_dir)
    print(len(dataset))
    pieces = dataset[0]
    print(pieces[0]["piece_name"])
    print(pieces[0]["chroma"].shape)
    print(pieces[0]["times"].shape)
    print(pieces[0]["chord_labels"])