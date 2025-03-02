import os
import re
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

def extract_key_mode_from_filename(fname):
    # Example heuristic: look for pattern "_in_(.*)_(major|minor)_"
    m = re.search(r"_in_([a-zA-Z_]+)_(major|minor)_", fname)
    if m:
        key_str = m.group(1).replace("_", " ").lower()  # normalize to lowercase
        mode = m.group(2).lower()  # normalize to lowercase
        return key_str, mode
    return "", ""

def normalize_key_label(key_label):
    """
    Normalize key label to sharp representation.
    For example, converts "b flat major" to "a# major", "d flat minor" to "c# minor", etc.
    """
    key_label = key_label.strip().lower()
    import re
    m = re.match(r'^([a-g])\s*flat\s*(major|minor)$', key_label)
    if m:
        note, mode = m.groups()
        flat_to_sharp = {
            "db": "c#",
            "eb": "d#",
            "gb": "f#",
            "ab": "g#",
            "bb": "a#"
        }
        # Construct the flat note as letter + "b"
        flat_note = f"{note}b"
        if flat_note in flat_to_sharp:
            return f"{flat_to_sharp[flat_note]} {mode}"
    return key_label

def normalize_join_key(key):
    key = key.strip().lower()
    # Remove surrounding quotes if any
    if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
        key = key[1:-1].strip()
    
    # Extract the file name part (after any directory)
    parts = key.split('/')
    if len(parts) > 1:
        # Get just the filename (last part)
        filename = parts[-1]
        # Standardize to a common prefix
        key = "orchestra_baroque/" + filename
    else:
        # If no path separator, assume it's just the filename
        key = "orchestra_baroque/" + key
        
    # Try to normalize any "crossera-XXXX" pattern to "CrossEra-XXXX" format
    import re
    key = re.sub(r'crossera-(\d+)', lambda m: f"CrossEra-{int(m.group(1)):04d}", key)
    
    return key

class CrossKeyDataset(Dataset):
    def __init__(self, base_dataset, annotation_csv_path, columns=None):
        """
        Args:
            base_dataset (Dataset): A constructed CrossDataset instance.
            annotation_csv_path (str): Path to the annotations CSV.
            columns (dict, optional): A dict defining column mappings with keys:
                'class', 'filename', 'key', 'mode'. Defaults to:
                {'class': 'Class', 'filename': 'Filename', 'key', 'mode': 'Mode'}.
        """
        # First, apply the chord label fixer to fix the base dataset's chord mapping
        from modules.utils.chord_label_fixer import fix_chord_labels
        if hasattr(base_dataset, 'chord_to_idx'):
            # Create a new fixed mapping
            fixed_mapping = fix_chord_labels(base_dataset.chord_to_idx)
            # Set the fixed mapping in the base dataset
            base_dataset.chord_to_idx = fixed_mapping
            # Also fix the chord_mapping attribute if it exists
            if hasattr(base_dataset, 'chord_mapping'):
                base_dataset.chord_mapping = fixed_mapping
            print(f"Fixed chord mapping applied to base dataset.")
            if 'f#' in fixed_mapping:
                print(f"f# is now mapped to index {fixed_mapping['f#']}")
        
        self.base_dataset = base_dataset
        
        # Factor out annotation columns
        if columns is None:
            self.columns = {'class': 'Class', 'filename': 'Filename', 'key': 'Key', 'mode': 'Mode'}
        else:
            self.columns = columns

        # Define fixed 24 key labels (12 major and 12 minor)
        self.KEY_LABELS = [
            "c major", "c minor", "c# major", "c# minor", "d major", "d minor", 
            "d# major", "d# minor", "e major", "e minor", "f major", "f minor", 
            "f# major", "f# minor", "g major", "g minor", "g# major", "g# minor", 
            "a major", "a minor", "a# major", "a# minor", "b major", "b minor"
        ]
        
        # Get unified chord mapping using get_unified_mapping from CrossDataset.py using only cross-era data.
        from modules.data.CrossDataset import get_unified_mapping
        project_root = Path(__file__).resolve().parents[2]
        label_dir  = project_root / "data" / "cross-era_chords-chordino"
        self.unified_mapping = get_unified_mapping([label_dir])
        
        # Read the CSV using the C engine and only required columns.
        self.annotations = pd.read_csv(annotation_csv_path, sep=",", engine="c", header=0,
                                         usecols=[self.columns['class'], self.columns['filename'],
                                                  self.columns['key'], self.columns['mode']], dtype=str)
        # Fill blanks with empty strings and normalize to lowercase.
        self.annotations[self.columns['key']] = self.annotations[self.columns['key']].fillna("").str.lower()
        self.annotations[self.columns['mode']] = self.annotations[self.columns['mode']].fillna("").str.lower()
        # For rows missing key/mode, try to extract from filename.
        mask = (self.annotations[self.columns['key']] == "") & (self.annotations[self.columns['mode']] == "")
        if mask.any():
            extracted = self.annotations.loc[mask, self.columns['filename']].apply(extract_key_mode_from_filename)
            self.annotations.loc[mask, self.columns['key']] = extracted.apply(lambda x: x[0])
            self.annotations.loc[mask, self.columns['mode']] = extracted.apply(lambda x: x[1])
        # Create a new column 'key_label' by joining key and mode, then normalize and filter out rows not in KEY_LABELS.
        self.annotations["key_label"] = (self.annotations[self.columns['key']].str.strip() + " " +
                                         self.annotations[self.columns['mode']].str.strip()).str.lower().str.strip()
        self.annotations["key_label"] = self.annotations["key_label"].apply(normalize_key_label)
        self.annotations = self.annotations[self.annotations["key_label"].isin(self.KEY_LABELS)]
        
        # Build a mapping: normalize both sides of join keys
        self.annotation_map = {}
        self.normalized_annotation_map = {}  # Add a normalized version
        for idx, row in self.annotations.iterrows():
            original_key = f"{row[self.columns['class']]}/{row[self.columns['filename']]}"
            normalized_key = normalize_join_key(original_key)
            self.annotation_map[original_key] = row["key_label"]
            self.normalized_annotation_map[normalized_key] = row["key_label"]
        
        # Print some debug info about the keys
        print("\nDebug: Some annotation keys and their normalized versions:")
        for i, key in enumerate(list(self.annotation_map.keys())[:3]):
            print(f"Original: {key} -> Normalized: {normalize_join_key(key)}")
        
        # Filter base_dataset samples with better matching
        valid_indices = []
        for i in range(len(self.base_dataset)):
            seg_start = self.base_dataset.segment_indices[i][0]
            sample_piece = self.base_dataset.samples[seg_start].get("piece", "")
            
            # Try different normalization approaches
            norm_key1 = normalize_join_key(sample_piece)
            
            # Look for matches in normalized annotation map
            if norm_key1 in self.normalized_annotation_map:
                valid_indices.append(i)
                
            # Debug a few samples to see what's happening
            if i < 5 or (len(valid_indices) < 5 and i < 100):
                print(f"Debug: Sample {i} piece: '{sample_piece}'")
                print(f"       normalized: '{norm_key1}'")
                print(f"       match found: {norm_key1 in self.normalized_annotation_map}")
                
        self.valid_indices = valid_indices
        print(f"Debug: {len(self.valid_indices)} valid indices found out of {len(self.base_dataset)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sample = self.base_dataset[real_idx]
        seg_start = self.base_dataset.segment_indices[real_idx][0]
        join_key = self.base_dataset.samples[seg_start].get("piece", "")
        piece_name = join_key  # Store the original piece name
        norm_key = normalize_join_key(join_key)
        
        # Try to find in both original and normalized maps
        if norm_key in self.normalized_annotation_map:
            sample["key_annotation"] = self.normalized_annotation_map[norm_key]
        else:
            sample["key_annotation"] = None
        
        # Add piece name to the returned sample for better debugging
        sample["piece_name"] = piece_name
        
        return sample

if __name__ == '__main__':
    from modules.data.CrossDataset import CrossDataset
    chroma_dir = "/Users/nghiaphan/Desktop/ChordMini/data/cross-era_chroma-nnls"
    label_dir = "/Users/nghiaphan/Desktop/ChordMini/data/cross-era_chords-chordino"
    unified_mapping = {}  # Provided unified mapping (can be ignored since CrossKeyDataset computes its own)
    base_ds = CrossDataset(chroma_dir, label_dir, chord_mapping=unified_mapping, seq_len=10, stride=3)
    annot_csv = os.path.join("/Users/nghiaphan/Desktop/ChordMini/data", "cross-era_annotations.csv")
    key_dataset = CrossKeyDataset(base_ds, annot_csv)
    print(f"Filtered Dataset length: {len(key_dataset)}")
    print("First 20 rows of the annotation join table:")
    join_keys = list(key_dataset.annotation_map.keys())[:20]
    for key in join_keys:
        print(f"{key} -> {key_dataset.annotation_map[key]}")
    
    # Get the first sample
    sample = key_dataset[0]
    print("\nSample details:")
    print(f"Piece name: {sample['piece_name']}")
    print(f"Chord label: {sample['chord_label']}")
    print(f"Key annotation: {sample['key_annotation']}")
    print(f"Chroma shape: {sample['chroma'].shape}")
    
    # Print first few samples with complete information
    print("\nFirst 5 samples with complete information:")
    for i in range(min(100, len(key_dataset))):
        sample = key_dataset[i]
        print(f"\nSample {i}:")
        print(f"  Piece: {sample['piece_name']}")
        print(f"  Chord: {sample['chord_label']} (index: {sample['chord_idx']})")
        print(f"  Key: {sample['key_annotation']}")
