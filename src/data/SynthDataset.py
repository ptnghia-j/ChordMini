"""
Cached Phase 1 dataset for pre-extracted pseudo-labeling artifacts.

ChordMini's default Phase 1 path is now ``UnlabeledAudioDataset``, which extracts
features and teacher pseudo-labels online during training. ``SynthDataset`` is
the compatibility path for already-cached Phase 1 artifacts, such as
``*_spec.npy`` spectrograms, ``.lab`` / ``.txt`` labels, and optional
``*_logits.npy`` teacher logits.

The dataset recursively scans ``spec_dir`` for files matching ``*_spec.npy`` and
expects the corresponding label file to share the same relative stem under
``label_dir``. Optional KD logits follow the same stem with the
``_logits.npy`` suffix. ``__getitem__`` returns segment tensors plus song/frame
metadata used by validation/test-time overlap voting.

Design notes on distributed training:
    The current implementation is single-process / single-GPU.  To add
    distributed training later:
    1. Wrap the dataset with torch.utils.data.distributed.DistributedSampler
       in the DataLoader creation (not inside this class).
    2. Shard the file list in __init__ by rank/world_size if memory is a
       concern during the scan phase.
    3. Replace print() calls with a rank-aware logger.
    No changes to __getitem__ are required.
"""
import os
import re
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.utils import estimate_normalization_from_dataset, song_level_split_indices
from src.utils.chords import Chords, idx2voca_chord


class SynthDataset(Dataset):
    """
    Dataset for cached Phase 1 spectrogram / label / logit artifacts.

    Supported naming convention:
      - spectrogram: ``<stem>_spec.npy``
      - label: ``<stem>.lab`` or ``<stem>.txt``
      - optional logits: ``<stem>_logits.npy``

    The same relative stem is preserved across ``spec_dir``, ``label_dir``, and
    ``logits_dir`` when nested subdirectories are used. This keeps the class
    compatible with flat layouts and prefixed layouts such as
    ``000/000123_spec.npy``.

    ``__getitem__`` returns both segment tensors and song/frame metadata so
    ChordNet validation/test overlap voting can reconstruct full-song frame
    predictions across overlapping windows.
    """

    def __init__(
        self,
        spec_dir: str,
        label_dir: str,
        chord_mapping: dict = None,
        seq_len: int = 10,
        stride: int = None,
        frame_duration: float = 0.09288,
        logits_dir: str = None,
        require_teacher_logits: bool = False,
        verbose: bool = True,
        max_files: int = None,
    ):
        """
        Args:
            spec_dir: Root directory containing cached ``*_spec.npy`` files.
            label_dir: Root directory containing matching ``.lab`` / ``.txt`` files.
            chord_mapping: Dict mapping chord label strings to integer indices.
                           If None, labels are built dynamically (not recommended).
            seq_len: Number of frames per training sequence.
            stride: Stride between consecutive sequences (default: same as seq_len).
            frame_duration: Duration of each spectrogram frame in seconds.
            logits_dir: Optional root directory containing cached ``*_logits.npy`` files.
            require_teacher_logits: If True, skip songs without matching logit files.
            verbose: Print progress information during loading.
            max_files: Optional cap on the number of spectrogram files to load
                       (useful for debugging).

        # --- Distributed training hook ---
        # To support DistributedDataParallel, pass a DistributedSampler to
        # the DataLoader rather than modifying this class.  If you need to
        # shard the *file scan* itself (e.g., 100k+ files), partition
        # `spec_files` by rank after the glob in _scan_files().
        """
        self.spec_dir = spec_dir
        self.label_dir = label_dir
        self.logits_dir = logits_dir
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.frame_duration = frame_duration
        self.require_teacher_logits = require_teacher_logits
        self.verbose = verbose

        # Chord label mapping
        if chord_mapping is not None:
            self.chord_to_idx = dict(chord_mapping)
        else:
            self.chord_to_idx = {}
        self.chord_parser = Chords()

        # Scan directories and build per-song metadata
        self.songs = self._scan_files(max_files)

        # Build segment index: list of (song_idx, start_frame) pairs.
        self.segments = self._build_segments()

        if verbose:
            total_frames = sum(s['num_frames'] for s in self.songs)
            print(
                f"SynthDataset: {len(self.songs)} songs, "
                f"{total_frames} total frames, "
                f"{len(self.segments)} segments (seq_len={seq_len}, stride={self.stride})"
            )

    # ------------------------------------------------------------------
    # File scanning
    # ------------------------------------------------------------------

    def _scan_files(self, max_files=None):
        """Walk spec_dir to find spectrogram files and match labels/logits."""
        spec_pattern = re.compile(r'^(.+?)_spec\.npy$')
        songs = []
        skipped = {'no_label': 0, 'no_logits': 0, 'load_error': 0}

        spec_files = []
        for root, _dirs, files in os.walk(self.spec_dir):
            for fname in sorted(files):
                m = spec_pattern.match(fname)
                if m:
                    spec_files.append((os.path.join(root, fname), m.group(1)))

        if max_files is not None and len(spec_files) > max_files:
            spec_files = spec_files[:max_files]

        for spec_path, base_name in spec_files:
            # Resolve the relative sub-path (e.g., "000/000123")
            rel = os.path.relpath(os.path.dirname(spec_path), self.spec_dir)
            if rel == '.':
                file_id = base_name
            else:
                file_id = os.path.join(rel, base_name)

            # --- Find label file ---
            label_path = self._find_file(
                self.label_dir, file_id, extensions=['.lab', '.txt']
            )
            if label_path is None:
                skipped['no_label'] += 1
                continue

            # --- Find logits file (optional) ---
            logit_path = None
            if self.logits_dir is not None:
                logit_path = self._find_file(
                    self.logits_dir, file_id, suffix='_logits', extensions=['.npy']
                )
                if logit_path is None and self.require_teacher_logits:
                    skipped['no_logits'] += 1
                    continue

            # --- Read spectrogram shape (memory-mapped for speed) ---
            try:
                spec_mmap = np.load(spec_path, mmap_mode='r')
                num_frames = spec_mmap.shape[0] if spec_mmap.ndim >= 2 else 1
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Error reading {spec_path}: {e}")
                skipped['load_error'] += 1
                continue

            songs.append({
                'song_id': file_id,
                'spec_path': spec_path,
                'label_path': label_path,
                'logit_path': logit_path,
                'num_frames': num_frames,
            })

        if self.verbose and any(v > 0 for v in skipped.values()):
            print(f"Skipped files: {skipped}")

        return songs

    @staticmethod
    def _find_file(base_dir, file_id, suffix='', extensions=None):
        """
        Try to locate a file under base_dir matching file_id with given extensions.

        Handles both flat layout (base_dir/file_id.ext) and prefixed layout
        (base_dir/prefix/file_id.ext).
        """
        if extensions is None:
            extensions = ['']
        for ext in extensions:
            candidate = os.path.join(base_dir, f"{file_id}{suffix}{ext}")
            if os.path.isfile(candidate):
                return candidate
        return None

    # ------------------------------------------------------------------
    # Segment generation
    # ------------------------------------------------------------------

    def _build_segments(self):
        """Create ``(song_idx, start_frame)`` pairs for all valid windows."""
        segments = []
        for song_idx, song in enumerate(self.songs):
            n = song['num_frames']
            if n < self.seq_len:
                # Short song: include as a single segment (will be padded in __getitem__)
                segments.append((song_idx, 0))
            else:
                for start in range(0, n - self.seq_len + 1, self.stride):
                    segments.append((song_idx, start))
        return segments

    # ------------------------------------------------------------------
    # Label parsing
    # ------------------------------------------------------------------

    def _parse_label_file(self, label_path):
        """Parse a .lab / .txt file into a list of (start, end, chord_label) tuples."""
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        labels.append((float(parts[0]), float(parts[1]), parts[2]))
        except Exception as e:
            warnings.warn(f"Error parsing {label_path}: {e}")
        return labels

    def _chord_at_time(self, labels, t):
        """Return the chord label active at time *t* seconds."""
        for start, end, chord in labels:
            if start <= t < end:
                return chord
        return 'N'

    def _resolve_chord_idx(self, raw_label):
        """Map a raw chord string to an integer index via the chord vocabulary."""
        normalized = self.chord_parser.label_error_modify(raw_label)
        if normalized in ('N', 'X', ''):
            return self.chord_to_idx.get('N', 169)
        if normalized in self.chord_to_idx:
            return self.chord_to_idx[normalized]
        # Dynamic expansion when no mapping is provided
        if not self.chord_to_idx:
            idx = len(self.chord_to_idx)
            self.chord_to_idx[normalized] = idx
            return idx
        return self.chord_to_idx.get('N', 169)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        song_idx, start_frame = self.segments[idx]
        song = self.songs[song_idx]
        end_frame = min(start_frame + self.seq_len, song['num_frames'])

        # --- Load spectrogram slice ---
        spec_full = np.load(song['spec_path'], mmap_mode='r')
        if spec_full.ndim == 1:
            spec_full = spec_full[np.newaxis, :]
        spec = np.array(spec_full[start_frame:end_frame])  # copy from mmap
        freq_bins = spec.shape[-1]

        # --- Load chord labels for each frame ---
        labels_raw = self._parse_label_file(song['label_path'])
        chord_indices = []
        for fi in range(start_frame, end_frame):
            t = fi * self.frame_duration
            raw_chord = self._chord_at_time(labels_raw, t)
            chord_indices.append(self._resolve_chord_idx(raw_chord))

        # --- Load teacher logits (optional) ---
        teacher_logits = None
        if song['logit_path'] is not None:
            try:
                logits_full = np.load(song['logit_path'], mmap_mode='r')
                if logits_full.ndim == 3 and logits_full.shape[0] == 1:
                    logits_full = logits_full[0]
                teacher_logits = np.array(logits_full[start_frame:end_frame])
            except Exception:
                teacher_logits = None

        # --- Pad if needed ---
        actual_len = end_frame - start_frame
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len
            spec = np.pad(spec, ((0, pad_len), (0, 0)), mode='constant')
            chord_indices.extend([self.chord_to_idx.get('N', 169)] * pad_len)
            if teacher_logits is not None:
                n_classes = teacher_logits.shape[-1]
                teacher_logits = np.pad(
                    teacher_logits, ((0, pad_len), (0, 0)), mode='constant'
                )

        # --- Convert to tensors ---
        out = {
            'spectro': torch.from_numpy(spec).float(),
            'chord_idx': torch.tensor(chord_indices, dtype=torch.long),
            'song_id': song['song_id'],
            'start_frame': torch.tensor(start_frame, dtype=torch.long),
            'end_frame': torch.tensor(end_frame, dtype=torch.long),
        }
        if teacher_logits is not None:
            out['teacher_logits'] = torch.from_numpy(teacher_logits).float()

        return out

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_normalization_params(self, n_samples=200):
        """Estimate dataset mean and std from a random sample of segments."""
        return estimate_normalization_from_dataset(self, sample_count=n_samples)

    def split_indices(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Song-level split into train / val / test segment indices.

        Returns:
            Tuple of (train_indices, val_indices, test_indices) as lists of ints.
        """
        return song_level_split_indices(
            segment_song_indices=[song_idx for song_idx, _ in self.segments],
            num_songs=len(self.songs),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
