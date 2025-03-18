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
    This dataset is designed to work with pre-generated data from a teacher model,
    with support for nested directory structures.
    
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
        """Load all spectrogram and label files, handling nested directory structure"""
        # Check if directories exist
        if not self.spec_dir.exists():
            print(f"WARNING: Spectrogram directory does not exist: {self.spec_dir}")
        if not self.label_dir.exists():
            print(f"WARNING: Label directory does not exist: {self.label_dir}")
        
        # Find all subdirectories (like '000', '001', etc.)
        subdirs = []
        spec_files_total = 0
        label_files_total = 0
        
        # Check if we have a nested directory structure or flat structure
        try:
            if self.spec_dir.exists():
                subdirs = [d for d in self.spec_dir.iterdir() if d.is_dir()]
                print(f"Found {len(subdirs)} subdirectories in spectrogram directory")
                
                # If very few subdirs, check if they match the expected pattern (numeric folders)
                if 1 <= len(subdirs) <= 5:
                    subdir_names = [d.name for d in subdirs]
                    print(f"Subdirectory names: {subdir_names}")
        except Exception as e:
            print(f"Error checking subdirectories: {e}")
            subdirs = []
        
        if not subdirs:
            # No subdirectories - try flat structure first
            print("No subdirectories found. Searching for files directly...")
            spec_files = list(self.spec_dir.glob("*.npy")) if self.spec_dir.exists() else []
            print(f"Found {len(spec_files)} .npy files directly in spectrogram directory")
            
            if spec_files:
                # We found files directly in the directory
                for spec_file in spec_files:
                    base_name = spec_file.stem
                    if base_name.endswith("_spec"):
                        base_name = base_name.replace("_spec", "")
                        
                    # Try several label file naming patterns
                    potential_label_files = [
                        self.label_dir / f"{base_name}.lab",
                        self.label_dir / f"{base_name}_lab.lab"
                    ]
                    
                    for label_file in potential_label_files:
                        if label_file.exists():
                            self._process_file_pair(spec_file, label_file)
                            break
            else:
                # Try searching recursively with glob pattern for nested files
                print("Searching recursively for .npy files...")
                try:
                    spec_files = list(self.spec_dir.glob("**/*.npy")) if self.spec_dir.exists() else []
                    print(f"Found {len(spec_files)} .npy files in recursive search")
                    
                    # Process found files
                    for spec_file in spec_files:
                        # Extract the relative path components
                        rel_path = spec_file.relative_to(self.spec_dir)
                        base_name = rel_path.stem
                        if base_name.endswith("_spec"):
                            base_name = base_name.replace("_spec", "")
                        
                        # Check if spectrogram is in a subdirectory
                        if len(rel_path.parts) > 1:
                            # File is in a subdirectory
                            subdir_name = rel_path.parts[0]
                            # Look for label in the same subdirectory structure
                            potential_label_files = [
                                self.label_dir / subdir_name / f"{base_name}.lab",
                                self.label_dir / subdir_name / f"{base_name}_lab.lab"
                            ]
                        else:
                            # File is directly in the directory
                            potential_label_files = [
                                self.label_dir / f"{base_name}.lab",
                                self.label_dir / f"{base_name}_lab.lab"
                            ]
                        
                        # Try to find matching label file
                        label_found = False
                        for label_file in potential_label_files:
                            if label_file.exists():
                                self._process_file_pair(spec_file, label_file)
                                label_found = True
                                break
                        
                        if not label_found:
                            # Try searching for the label file anywhere in label_dir
                            label_search = list(self.label_dir.glob(f"**/{base_name}.lab")) + \
                                         list(self.label_dir.glob(f"**/{base_name}_lab.lab"))
                            if label_search:
                                self._process_file_pair(spec_file, label_search[0])
                            else:
                                print(f"WARNING: No matching label file found for {spec_file}")
                except Exception as e:
                    print(f"Error during recursive search: {e}")
        else:
            # We have a nested directory structure - process each subdirectory
            for subdir in subdirs:
                print(f"Processing subdirectory: {subdir.name}")
                spec_subdir = self.spec_dir / subdir.name
                label_subdir = self.label_dir / subdir.name
                
                if not spec_subdir.exists():
                    print(f"WARNING: Spectrogram subdirectory does not exist: {spec_subdir}")
                    continue
                
                # Check if corresponding label subdirectory exists
                if not label_subdir.exists():
                    print(f"WARNING: Label subdirectory does not exist: {label_subdir}")
                
                # Find all spectrogram files in this subdirectory
                spec_files = list(spec_subdir.glob("*.npy"))
                print(f"  Found {len(spec_files)} spectrogram files in {subdir.name}")
                spec_files_total += len(spec_files)
                
                # Get corresponding label files count
                if label_subdir.exists():
                    label_files = list(label_subdir.glob("*.lab"))
                    label_files_total += len(label_files)
                    print(f"  Found {len(label_files)} label files in {subdir.name}")
                
                for spec_file in spec_files:
                    # Try multiple patterns for base name extraction
                    base_name = spec_file.stem
                    if base_name.endswith("_spec"):
                        base_name = base_name.replace("_spec", "")
                    
                    # Try different label file naming patterns
                    potential_label_files = []
                    
                    # First, try the label in the same subdirectory
                    if label_subdir.exists():
                        potential_label_files.extend([
                            label_subdir / f"{base_name}.lab",
                            label_subdir / f"{base_name}_lab.lab",
                        ])
                    
                    # Then try looking for the label file directly in the main label directory
                    potential_label_files.extend([
                        self.label_dir / f"{base_name}.lab",
                        self.label_dir / f"{base_name}_lab.lab",
                    ])
                    
                    # Try searching for the label file in any other subdirectory
                    for other_subdir in self.label_dir.glob("*"):
                        if other_subdir.is_dir() and other_subdir.name != subdir.name:
                            potential_label_files.extend([
                                other_subdir / f"{base_name}.lab",
                                other_subdir / f"{base_name}_lab.lab",
                            ])
                    
                    # Try to find a matching label file from the potential candidates
                    label_found = False
                    for label_file in potential_label_files:
                        if label_file.exists():
                            self._process_file_pair(spec_file, label_file)
                            label_found = True
                            break
                    
                    if not label_found:
                        print(f"WARNING: No matching label file found for {spec_file}")
        
        # Report on spectrogram dimensions to help identify CQT vs STFT
        if self.samples:
            # Analyze the first sample to get frequency dimension
            first_spec = self.samples[0]['spectro']
            freq_dim = first_spec.shape[-1] if len(first_spec.shape) > 0 else 0
            spec_type = "CQT (Constant-Q Transform)" if freq_dim <= 256 else "STFT"
            print(f"Loaded {len(self.samples)} valid samples")
            print(f"Total found: {spec_files_total} spectrogram files, {label_files_total} label files")
            print(f"Spectrogram frequency dimension: {freq_dim} (likely {spec_type})")
            
            # Report on class distribution
            chord_counter = Counter(sample['chord_label'] for sample in self.samples)
            print(f"Found {len(chord_counter)} unique chord classes")
        else:
            print("No samples loaded. Check your data paths and structure.")
        
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
            time_frames = spec.shape[0] if len(spec.shape) > 1 else 1  # Assuming shape is (time, features)
            
            # Add to samples
            if len(spec.shape) <= 1:
                # Handle single frame spectrograms
                chord_label = self._find_chord_at_time(chord_labels, 0.0)
                
                # Make sure the chord label exists in the mapping
                if self.chord_mapping is None:
                    if chord_label not in self.chord_to_idx:
                        self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                elif chord_label not in self.chord_mapping:
                    print(f"Warning: Unknown chord label {chord_label} found in {label_file}")
                    # Skip this sample if we can't map it
                    return
                    
                self.samples.append({
                    'spectro': spec,
                    'chord_label': chord_label
                })
            else:
                # Handle multi-frame spectrograms
                for t in range(time_frames):
                    # Find the chord label for this time frame
                    frame_time = t * 0.1  # Assuming 0.1s per frame
                    chord_label = self._find_chord_at_time(chord_labels, frame_time)
                    
                    # Make sure the chord label exists in the mapping
                    if self.chord_mapping is None:
                        if chord_label not in self.chord_to_idx:
                            self.chord_to_idx[chord_label] = len(self.chord_to_idx)
                    elif chord_label not in self.chord_mapping:
                        print(f"Warning: Unknown chord label {chord_label} found in {label_file}")
                        chord_label = "N"  # Default to no-chord
                        
                    self.samples.append({
                        'spectro': spec[t],
                        'chord_label': chord_label
                    })
                
            # Log the first file's dimensions in detail
            if len(self.samples) <= time_frames:  # This must be the first file processed
                print(f"First spectrogram shape: {spec.shape}")
                print(f"Example frame dimensions: {self.samples[0]['spectro'].shape if len(spec.shape) > 1 else spec.shape}")
                print(f"First sample chord: {self.samples[0]['chord_label']}")
                
        except Exception as e:
            print(f"Error processing {spec_file} and {label_file}: {e}")
    
    def _parse_label_file(self, label_file):
        """Parse a label file into a list of (start_time, end_time, chord) tuples"""
        result = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            chord = parts[2]
                            result.append((start_time, end_time, chord))
        except Exception as e:
            print(f"Error parsing label file {label_file}: {e}")
            
        return result
    
    def _find_chord_at_time(self, chord_labels, time):
        """Find the chord label at a specific time point"""
        if not chord_labels:
            return "N"  # Return no-chord for empty label files
            
        for start, end, chord in chord_labels:
            if start <= time < end:
                return chord
                
        # If we have chord labels but time is out of bounds
        if chord_labels and time >= chord_labels[-1][1]:
            # Use the last chord if the time exceeds the end
            return chord_labels[-1][2]
            
        return "N"  # No chord found
    
    def _generate_segments(self):
        """Generate sequence segments from the loaded samples"""
        if not self.samples:
            print("WARNING: No samples to generate segments from")
            return
            
        if len(self.samples) <= self.seq_len:
            print(f"WARNING: Not enough samples ({len(self.samples)}) to create segments of length {self.seq_len}")
            # Create at least one segment using the available samples
            self.segment_indices.append((0, len(self.samples)))
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
        if not self.segment_indices:
            raise IndexError("Dataset is empty - no segments available")
            
        seg_start, seg_end = self.segment_indices[idx]
        sequence = []
        label_seq = []
        
        # Get the shape of a sample spectrogram to use for padding
        first_spec = self.samples[0]['spectro']
        
        for i in range(seg_start, seg_end):
            if i < len(self.samples):
                sample_i = self.samples[i]
                spec_vec = torch.tensor(sample_i['spectro'], dtype=torch.float)
                chord_label = sample_i['chord_label']
                
                # Handle case where chord_label might not be in mapping
                chord_idx = self.chord_to_idx.get(chord_label, self.chord_to_idx.get("N", 0))
                label_seq.append(chord_idx)
                sequence.append(spec_vec)
            else:
                # Pad with zeros and a default label (here, using 0)
                if not sequence:
                    padding_shape = first_spec.shape
                else:
                    padding_shape = sequence[-1].shape
                sequence.append(torch.zeros(padding_shape, dtype=torch.float))
                label_seq.append(self.chord_to_idx.get("N", 0))
        
        # Return full frame-level label sequence instead of majority vote
        sample_out = {
            'spectro': torch.stack(sequence, dim=0),       # [seq_len, feature_dim]
            'chord_idx': torch.tensor(label_seq, dtype=torch.long)  # [seq_len]
        }
        
        return sample_out
    
    def get_train_iterator(self, batch_size=128, shuffle=True):
        """Get a DataLoader for the training set"""
        if not self.train_indices:
            print("WARNING: No training segments available")
            # Return an empty dataset
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True
            )
            
        return DataLoader(
            SynthSegmentSubset(self, self.train_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True
        )
    
    def get_eval_iterator(self, batch_size=128, shuffle=False):
        """Get a DataLoader for the evaluation set"""
        if not self.eval_indices:
            print("WARNING: No evaluation segments available")
            # Return an empty dataset
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True
            )
            
        return DataLoader(
            SynthSegmentSubset(self, self.eval_indices),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True
        )
    
    def get_test_iterator(self, batch_size=128, shuffle=False):
        """Get a DataLoader for the test set"""
        if not self.test_indices:
            print("WARNING: No test segments available")
            # Return an empty dataset
            return DataLoader(
                SynthSegmentSubset(self, []),
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True
            )
            
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
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.indices)} indices")
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
    
    for sample in dataset.samples:
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