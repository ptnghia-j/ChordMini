import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import glob
from pathlib import Path
import matplotlib.pyplot as plt

class SynthDataset(Dataset):
    """
    Dataset for loading preprocessed spectrograms and chord labels.
    This dataset is designed to work with pre-generated data from a teacher model.
    
    Args:
        spec_dir: Directory containing spectrogram files (.npy files)
        label_dir: Directory containing label files (.lab files)
        chord_mapping: Dictionary mapping chord names to indices
        seq_len: Length of each sequence in frames
        stride: Step between consecutive sequences (if None, use seq_len for non-overlapping)
    """
    def __init__(self, spec_dir, label_dir, chord_mapping=None, seq_len=10, stride=None):
        self.spec_dir = Path(spec_dir)
        self.label_dir = Path(label_dir)
        self.chord_mapping = chord_mapping
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.samples = []
        self.segment_indices = []
        
        # Map from chord name to index
        if self.chord_mapping is not None:
            self.chord_to_idx = self.chord_mapping
        else:
            self.chord_to_idx = {}
            
        # Load all data
        self._load_data()
        
        # Generate sequence segments
        self._generate_segments()
        
        # Split data for train/eval/test
        total_segs = len(self.segment_indices)
        self.train_indices = list(range(0, int(total_segs * 0.8)))
        self.eval_indices = list(range(int(total_segs * 0.8), int(total_segs * 0.9)))
        self.test_indices = list(range(int(total_segs * 0.9), total_segs))
        
    def _load_data(self):
        """Load all spectrogram and label files"""
        # Find all subdirectories (like '000', '001', etc.)
        subdirs = [d for d in self.spec_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            # If no subdirectories, search directly in the main directory
            spec_files = list(self.spec_dir.glob("*_spec.npy"))
            for spec_file in spec_files:
                base_name = spec_file.stem.replace("_spec", "")
                label_file = self.label_dir / f"{base_name}.lab"
                
                if label_file.exists():
                    self._process_file_pair(spec_file, label_file)
        else:
            # Process each subdirectory
            for subdir in subdirs:
                spec_subdir = self.spec_dir / subdir.name
                label_subdir = self.label_dir / subdir.name
                
                if not spec_subdir.exists() or not label_subdir.exists():
                    continue
                    
                # Find all spectrogram files in this subdirectory
                spec_files = list(spec_subdir.glob("*_spec.npy"))
                for spec_file in spec_files:
                    base_name = spec_file.stem.replace("_spec", "")
                    label_file = label_subdir / f"{base_name}.lab"
                    
                    if label_file.exists():
                        self._process_file_pair(spec_file, label_file)
        
        # Report on spectrogram dimensions to help identify CQT vs STFT
        if self.samples:
            # Analyze the first sample to get frequency dimension
            first_spec = self.samples[0]['spectro']
            freq_dim = first_spec.shape[-1] if len(first_spec.shape) > 0 else 0
            spec_type = "CQT (Constant-Q Transform)" if freq_dim <= 256 else "STFT"
            print(f"Loaded {len(self.samples)} valid samples")
            print(f"Spectrogram frequency dimension: {freq_dim} (likely {spec_type})")
        else:
            print("No samples loaded. Check your data paths.")
        
    def _process_file_pair(self, spec_file, label_file):
        """Process a pair of spectrogram and label files"""
        # Load spectrogram
        try:
            spec = np.load(spec_file)
            
            # Check for NaN values and log warning if found
            if np.isnan(spec).any():
                print(f"Warning: NaN values found in {spec_file}")
                # Replace NaNs with zeros for stability
                spec = np.nan_to_num(spec, nan=0.0)
            
            # Check for extreme values
            if np.abs(spec).max() > 1000:
                print(f"Warning: Extreme values found in {spec_file}. Max abs value: {np.abs(spec).max()}")
            
            # Load label file
            chord_labels = self._parse_label_file(label_file)
            
            # Ensure the spectrogram and labels have matching length
            time_frames = spec.shape[0]  # Assuming shape is (time, features)
            
            # Add to samples
            for t in range(time_frames):
                # Find the chord label for this time frame
                frame_time = t * 0.1  # Assuming 0.1s per frame
                chord_label = self._find_chord_at_time(chord_labels, frame_time)
                
                self.samples.append({
                    'spectro': spec[t],
                    'chord_label': chord_label
                })
                
            # Log the first file's dimensions in detail
            if len(self.samples) <= time_frames:  # This must be the first file processed
                print(f"First spectrogram shape: {spec.shape}")
                print(f"Example frame dimensions: {self.samples[0]['spectro'].shape}")
                print(f"First sample chord: {self.samples[0]['chord_label']}")
                
        except Exception as e:
            print(f"Error processing {spec_file} and {label_file}: {e}")
    
    def _parse_label_file(self, label_file):
        """Parse a label file into a list of (start_time, end_time, chord) tuples"""
        result = []
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        chord = parts[2]
                        result.append((start_time, end_time, chord))
        return result
    
    def _find_chord_at_time(self, chord_labels, time):
        """Find the chord label at a specific time point"""
        for start, end, chord in chord_labels:
            if start <= time < end:
                return chord
        return "N"  # No chord found
    
    def _generate_segments(self):
        """Generate sequence segments from the loaded samples"""
        if len(self.samples) <= self.seq_len:
            return
            
        # Generate segments with stride
        num_samples = len(self.samples)
        segment_start = 0
        
        while segment_start + self.seq_len <= num_samples:
            segment_end = segment_start + self.seq_len
            self.segment_indices.append((segment_start, segment_end))
            segment_start += self.stride
            
        print(f"Generated {len(self.segment_indices)} segments")
    
    def __len__(self):
        return len(self.segment_indices)
    
    def __getitem__(self, idx):
        seg_start, seg_end = self.segment_indices[idx]
        sequence = []
        label_seq = []
        
        for i in range(self.seq_len):
            pos = seg_start + i
            if pos < seg_end:
                sample_i = self.samples[pos]
                spec_vec = torch.tensor(sample_i['spectro'], dtype=torch.float)
                chord_label = sample_i['chord_label']
                label_seq.append(self.chord_to_idx[chord_label])
                sequence.append(spec_vec)
            else:
                sequence.append(torch.zeros_like(spec_vec))
                label_seq.append(0)  # or another default value if needed
        
        # Use the most common chord as target
        target = Counter(label_seq).most_common(1)[0][0]
        
        sample_out = {
            'spectro': torch.stack(sequence, dim=0),
            'chord_idx': target,
            'chord_label': self.samples[seg_start]['chord_label']
        }
        
        return sample_out
    
    def get_train_iterator(self, batch_size=128, shuffle=True):
        """Get a DataLoader for the training set"""
        return DataLoader(
            SynthSegmentSubset(self, self.train_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True
        )
    
    def get_eval_iterator(self, batch_size=128, shuffle=False):
        """Get a DataLoader for the evaluation set"""
        return DataLoader(
            SynthSegmentSubset(self, self.eval_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True
        )
    
    def get_test_iterator(self, batch_size=128, shuffle=False):
        """Get a DataLoader for the test set"""
        return DataLoader(
            SynthSegmentSubset(self, self.test_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True
        )


class SynthSegmentSubset(Dataset):
    """Subset of the SynthDataset based on specified indices"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def visualize_spectrogram(spec):
    """Visualize a spectrogram sample"""
    plt.figure(figsize=(10, 6))
    plt.imshow(spec.T, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()


def analyze_chord_distribution(dataset):
    """Analyze and print chord distribution from dataset"""
    chord_counter = Counter()
    
    for i in range(len(dataset)):
        sample = dataset[i]
        chord_counter[sample['chord_label']] += 1
    
    total = sum(chord_counter.values())
    print(f"\nChord distribution (total: {total}):")
    for chord, count in chord_counter.most_common(20):
        print(f"{chord}: {count} ({count/total*100:.2f}%)")


if __name__ == "__main__":
    # Path to test data
    import sys
    from pathlib import Path
    
    # Get the project root directory
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = str(Path(__file__).resolve().parents[2])  # Go up 2 levels from this file
    
    spec_dir = os.path.join(project_root, "data", "synth", "spectrograms")
    label_dir = os.path.join(project_root, "data", "synth", "labels")
    
    print(f"Testing SynthDataset with data from:")
    print(f"  Spectrograms: {spec_dir}")
    print(f"  Labels: {label_dir}")
    
    # Build a chord mapping
    temp_dataset = SynthDataset(spec_dir, label_dir, chord_mapping=None, seq_len=1, stride=1)
    unique_chords = set(sample['chord_label'] for sample in temp_dataset.samples)
    chord_mapping = {chord: idx for idx, chord in enumerate(sorted(unique_chords))}
    
    # Make sure 'N' is included for no-chord label
    if "N" not in chord_mapping:
        chord_mapping["N"] = len(chord_mapping)
    
    print(f"\nGenerated chord mapping with {len(chord_mapping)} unique chords")
    print(f"First 5 mappings: {dict(list(chord_mapping.items())[:5])}")
    
    # Create dataset with reasonable sequence length
    dataset = SynthDataset(spec_dir, label_dir, chord_mapping=chord_mapping, seq_len=10, stride=5)
    
    # Print dataset stats
    print(f"\nDataset Statistics:")
    print(f"  Total frames: {len(dataset.samples)}")
    print(f"  Total segments: {len(dataset)}")
    print(f"  Train segments: {len(dataset.train_indices)}")
    print(f"  Eval segments: {len(dataset.eval_indices)}")
    print(f"  Test segments: {len(dataset.test_indices)}")
    
    # Analyze chord distribution
    analyze_chord_distribution(dataset)
    
    # Test loaders
    train_loader = dataset.get_train_iterator(batch_size=16, shuffle=True)
    val_loader = dataset.get_eval_iterator(batch_size=16)
    test_loader = dataset.get_test_iterator(batch_size=16)
    
    # Display sample batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Spectrogram shape: {sample_batch['spectro'].shape}")
    print(f"  Target chord indices: {sample_batch['chord_idx']}")
    
    # Visualize a spectrogram if possible
    try:
        print("\nVisualizing first spectrogram in batch...")
        visualize_spectrogram(sample_batch['spectro'][0])
    except Exception as e:
        print(f"Could not visualize spectrogram: {e}")
        
    print("\nTest complete!")