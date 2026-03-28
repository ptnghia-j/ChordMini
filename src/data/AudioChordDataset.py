"""
On-the-fly labeled-audio dataset used by ChordMini training and evaluation.

This is the default labeled-data path in ChordMini: audio is read directly during
training/evaluation and CQT features are extracted on demand instead of relying
on a separate pre-extraction step. The dataset expects a flat directory of
audio files whose stems match ``.lab`` files in ``label_dir``.

    Supports optional pitch-shifting augmentation via ``pyrubberband``.
    When enabled, augmented copies span the 12 transposition classes.
"""
import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.utils import (
    estimate_normalization_from_dataset,
    song_level_cv_fold_indices,
    song_level_split_indices,
)
from src.utils.chords import Chords, idx2voca_chord, normalize_enharmonic_label, transpose_chord_label
from src.utils.config_utils import get_config_value
from src.utils.audio_io import suppress_stderr as _suppress_stderr

try:
    import pyrubberband as pyrb
    PYRUBBERBAND_AVAILABLE = True
except ImportError:
    PYRUBBERBAND_AVAILABLE = False

class AudioChordDataset(Dataset):
    """
    On-the-fly CQT extraction + chord-label alignment for labeled audio.

    Expected file matching convention:
      - audio: ``<song_id>.mp3`` / ``.wav`` / ``.flac``
      - label: ``<song_id>.lab``

    Feature extraction happens at access time inside ``__getitem__``. Segment
    metadata is returned alongside tensors so validation/test code can rebuild
    full-song predictions from overlapping windows.

    Each item returns:
        {'spectro': Tensor[seq_len, n_bins], 'chord_idx': Tensor[seq_len],
         'song_id': str, 'start_frame': int, 'end_frame': int,
         'pitch_shift': int}
    """

    def __init__(self, audio_dir, label_dir, config, seq_len=108, stride=54,
                 chord_mapping=None, device='cpu', verbose=True, max_songs=None,
                 random_seed=42, enable_augmentation=False, augmentation_semitones=None):
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.config = config
        self.seq_len = seq_len
        self.stride = stride
        self.device = device
        self.verbose = verbose
        self.random_seed = random_seed
        self.enable_augmentation = enable_augmentation
        self.augmentation_semitones = augmentation_semitones or list(range(-5, 7))

        if self.enable_augmentation and not PYRUBBERBAND_AVAILABLE:
            print("WARNING: pyrubberband not installed, disabling augmentation")
            self.enable_augmentation = False

        # Feature extraction params from config
        self.sample_rate = get_config_value(config, 'mp3', 'song_hz', 22050)
        self.hop_length = get_config_value(config, 'feature', 'hop_length', 2048)
        self.n_bins = get_config_value(config, 'feature', 'n_bins', 144)
        self.bins_per_octave = get_config_value(config, 'feature', 'bins_per_octave', 24)
        self.frame_duration = self.hop_length / self.sample_rate

        # Chord vocabulary
        if chord_mapping is None:
            self.idx_to_chord = idx2voca_chord()
            self.chord_to_idx = {v: k for k, v in self.idx_to_chord.items()}
        else:
            self.chord_to_idx = chord_mapping
            self.idx_to_chord = {v: k for k, v in chord_mapping.items()}

        self.chord_parser = Chords()
        self.chord_parser.set_chord_mapping(self.chord_to_idx)
        self.max_songs = max_songs

        self.samples = []
        self.song_segments = []
        self._load_data()

    def _load_data(self):
        audio_files = sorted([f for f in os.listdir(self.audio_dir)
                              if f.endswith(('.mp3', '.wav', '.flac'))])
        if self.max_songs is not None and self.max_songs < len(audio_files):
            rng = np.random.RandomState(self.random_seed)
            sel = np.sort(rng.choice(len(audio_files), size=self.max_songs, replace=False))
            audio_files = [audio_files[i] for i in sel]

        if self.verbose:
            print(f"Found {len(audio_files)} audio files in {self.audio_dir}")

        for audio_file in audio_files:
            song_id = os.path.splitext(audio_file)[0]
            label_file = os.path.join(self.label_dir, f"{song_id}.lab")
            if not os.path.exists(label_file):
                continue
            labels = self._parse_lab(label_file)
            self.samples.append({
                'song_id': song_id,
                'audio_path': os.path.join(self.audio_dir, audio_file),
                'chord_labels': labels,
            })

        if self.verbose:
            print(f"Loaded {len(self.samples)} songs with matching labels")
        self._create_segments()

    @staticmethod
    def _duration_seconds(audio_path):
        with _suppress_stderr():
            try:
                return float(librosa.get_duration(path=audio_path))
            except TypeError:
                return float(librosa.get_duration(filename=audio_path))

    def _parse_lab(self, path):
        labels = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    labels.append((float(parts[0]), float(parts[1]),
                                   self.chord_parser.label_error_modify(parts[2])))
        return labels

    def _create_segments(self):
        self.song_segments = []
        for song_idx, sample in enumerate(self.samples):
            try:
                duration = self._duration_seconds(sample['audio_path'])
                num_frames = max(1, int(np.ceil(duration * self.sample_rate / self.hop_length)))
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {sample['audio_path']}: {e}")
                continue

            for start in range(0, max(1, num_frames - self.seq_len + 1), self.stride):
                end = min(start + self.seq_len, num_frames)
                if end - start < self.seq_len // 2:
                    continue
                self.song_segments.append({
                    'song_idx': song_idx, 'start_frame': start,
                    'end_frame': end, 'num_frames': num_frames,
                    'pitch_shift': 0,
                })

        if self.verbose:
            print(f"Created {len(self.song_segments)} segments from {len(self.samples)} songs")

    def _extract_features(self, audio_path, start_frame, end_frame, pitch_shift=0):
        start_sample = start_frame * self.hop_length
        end_sample = (end_frame + 1) * self.hop_length
        with _suppress_stderr():
            y, sr = librosa.load(audio_path, sr=self.sample_rate,
                                 offset=start_sample / self.sample_rate,
                                 duration=(end_sample - start_sample) / self.sample_rate)
        if pitch_shift != 0 and PYRUBBERBAND_AVAILABLE:
            y = pyrb.pitch_shift(y, sr, pitch_shift)
        cqt = librosa.cqt(y, sr=sr, n_bins=self.n_bins,
                           bins_per_octave=self.bins_per_octave,
                           hop_length=self.hop_length,
                           fmin=librosa.note_to_hz('C1'))
        return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32, copy=False)

    def _chord_labels_for_segment(self, chord_labels, start_frame, end_frame):
        labels = []
        for fi in range(start_frame, end_frame):
            t = fi * self.frame_duration
            chord = 'N'
            for s, e, lbl in chord_labels:
                if s <= t < e:
                    chord = lbl
                    break
            if chord in self.chord_to_idx:
                labels.append(self.chord_to_idx[chord])
                continue

            normalized = normalize_enharmonic_label(chord)
            if normalized in self.chord_to_idx:
                labels.append(self.chord_to_idx[normalized])
                continue

            idx = self.chord_parser.get_chord_idx(chord)
            labels.append(idx if idx is not None else self.chord_to_idx.get('N', 169))
        return labels

    def __len__(self):
        return len(self.song_segments)

    def __getitem__(self, idx):
        seg = self.song_segments[idx]
        sample = self.samples[seg['song_idx']]
        ps = seg.get('pitch_shift', 0)

        feature = self._extract_features(sample['audio_path'],
                                          seg['start_frame'], seg['end_frame'],
                                          pitch_shift=ps)
        chord_labels = sample['chord_labels']
        if ps != 0:
            chord_labels = [(s, e, transpose_chord_label(l, ps)) for s, e, l in chord_labels]
        labels = self._chord_labels_for_segment(
            chord_labels,
            seg['start_frame'],
            seg['end_frame'],
        )

        ft = torch.from_numpy(feature)
        lt = torch.as_tensor(labels, dtype=torch.long)

        if ft.shape[0] < self.seq_len:
            pad = self.seq_len - ft.shape[0]
            ft = torch.nn.functional.pad(ft, (0, 0, 0, pad))
            lt = torch.nn.functional.pad(lt, (0, pad), value=169)
        elif ft.shape[0] > self.seq_len:
            ft = ft[:self.seq_len]
            lt = lt[:self.seq_len]

        return {
            'spectro': ft,
            'chord_idx': lt,
            'song_id': sample['song_id'],
            'start_frame': torch.tensor(seg['start_frame'], dtype=torch.long),
            'end_frame': torch.tensor(seg['end_frame'], dtype=torch.long),
            'pitch_shift': ps,
        }

    def add_augmented_segments_for_indices(self, train_indices, semitones_range=None):
        """Add pitch-shifted copies of training segments (call after split)."""
        if not PYRUBBERBAND_AVAILABLE:
            return train_indices
        if semitones_range is None:
            semitones_range = self.augmentation_semitones
        semitones_range = [s for s in semitones_range if s != 0]
        if not semitones_range:
            return train_indices
        new_indices = list(train_indices)
        for idx in train_indices:
            seg = self.song_segments[idx]
            for s in semitones_range:
                self.song_segments.append({**seg, 'pitch_shift': s})
                new_indices.append(len(self.song_segments) - 1)
        return new_indices

    def get_normalization_params(self):
        return estimate_normalization_from_dataset(self, sample_count=100)


def create_train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.1, seed=42):
    return song_level_split_indices(
        segment_song_indices=[seg['song_idx'] for seg in dataset.song_segments],
        num_songs=len(dataset.samples),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        ensure_val_if_possible=True,
    )


def create_cv_folds(dataset, n_folds=5, seed=42):
    return song_level_cv_fold_indices(
        segment_song_indices=[seg['song_idx'] for seg in dataset.song_segments],
        num_songs=len(dataset.samples),
        n_folds=n_folds,
        seed=seed,
    )
