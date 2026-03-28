"""
Dataset for Phase 1 online pseudo-labeling from unlabeled audio.

Recursively scans a root directory for audio files, computes CQT features on the
fly, and exposes fixed-length segments suitable for pseudo-label generation at
training time. Returned metadata is also used by ChordNet's overlap-aware
validation/test aggregation.
"""
import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.utils import estimate_normalization_from_dataset, song_level_split_indices
from src.utils.audio_io import suppress_stderr as _suppress_stderr


class UnlabeledAudioDataset(Dataset):
    """On-the-fly audio dataset for online pseudo-labeling."""

    AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')

    def __init__(
        self,
        audio_dir,
        config,
        seq_len=108,
        stride=108,
        verbose=True,
        max_files=None,
        random_seed=42,
    ):
        self.audio_dir = audio_dir
        self.config = config
        self.seq_len = seq_len
        self.stride = stride
        self.verbose = verbose
        self.max_files = max_files
        self.random_seed = random_seed

        self.sample_rate = getattr(config, 'sample_rate', 22050)
        self.hop_length = getattr(config, 'hop_length', 2048)
        self.n_bins = getattr(config, 'n_bins', 144)
        self.bins_per_octave = getattr(config, 'bins_per_octave', 24)
        self.frame_duration = getattr(config, 'frame_duration', self.hop_length / self.sample_rate)

        self.samples = []
        self.songs = self.samples  # Compatibility with existing training script.
        self.song_segments = []

        self._load_data()

    def _scan_audio_files(self):
        audio_files = []
        for dirpath, _, filenames in os.walk(self.audio_dir):
            for filename in filenames:
                if filename.lower().endswith(self.AUDIO_EXTENSIONS):
                    audio_files.append(os.path.join(dirpath, filename))
        audio_files.sort()

        if self.max_files is not None and self.max_files < len(audio_files):
            rng = np.random.RandomState(self.random_seed)
            indices = np.sort(rng.choice(len(audio_files), size=self.max_files, replace=False))
            audio_files = [audio_files[i] for i in indices]
        return audio_files

    @staticmethod
    def _duration_seconds(audio_path):
        with _suppress_stderr():
            try:
                return float(librosa.get_duration(path=audio_path))
            except TypeError:
                return float(librosa.get_duration(filename=audio_path))

    def _load_data(self):
        audio_files = self._scan_audio_files()
        if self.verbose:
            print(f"Found {len(audio_files)} audio files in {self.audio_dir}")

        for audio_path in audio_files:
            try:
                duration = self._duration_seconds(audio_path)
                num_frames = max(1, int(np.ceil(duration * self.sample_rate / self.hop_length)))
            except Exception as exc:
                if self.verbose:
                    print(f"Skipping {audio_path}: {exc}")
                continue

            song_id = os.path.splitext(os.path.relpath(audio_path, self.audio_dir))[0].replace(os.sep, '/')
            self.samples.append({
                'song_id': song_id,
                'audio_path': audio_path,
                'num_frames': num_frames,
            })

        if self.verbose:
            print(f"Loaded {len(self.samples)} unlabeled audio files")

        self._create_segments()

    def _create_segments(self):
        self.song_segments = []
        for song_idx, sample in enumerate(self.samples):
            num_frames = sample['num_frames']
            for start_frame in range(0, max(1, num_frames - self.seq_len + 1), self.stride):
                end_frame = min(start_frame + self.seq_len, num_frames)
                if end_frame - start_frame >= max(1, self.seq_len // 2):
                    self.song_segments.append({
                        'song_idx': song_idx,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                    })

        if self.verbose:
            print(
                f"UnlabeledAudioDataset: {len(self.samples)} songs, "
                f"{len(self.song_segments)} segments (seq_len={self.seq_len}, stride={self.stride})"
            )

    def _extract_features(self, audio_path, start_frame, end_frame):
        start_sample = start_frame * self.hop_length
        end_sample = max(start_sample + self.hop_length, (end_frame + 1) * self.hop_length)

        with _suppress_stderr():
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=start_sample / self.sample_rate,
                duration=(end_sample - start_sample) / self.sample_rate,
            )

        cqt = librosa.cqt(
            y,
            sr=sr,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            hop_length=self.hop_length,
            fmin=librosa.note_to_hz('C1'),
        )
        return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.song_segments)

    def __getitem__(self, idx):
        segment = self.song_segments[idx]
        sample = self.samples[segment['song_idx']]

        feature = self._extract_features(
            sample['audio_path'],
            segment['start_frame'],
            segment['end_frame'],
        )

        feature_tensor = torch.from_numpy(feature)
        if feature_tensor.shape[0] < self.seq_len:
            pad = self.seq_len - feature_tensor.shape[0]
            feature_tensor = torch.nn.functional.pad(feature_tensor, (0, 0, 0, pad))
        elif feature_tensor.shape[0] > self.seq_len:
            feature_tensor = feature_tensor[:self.seq_len]

        return {
            'spectro': feature_tensor,
            'song_id': sample['song_id'],
            'start_frame': segment['start_frame'],
            'end_frame': segment['end_frame'],
            'audio_path': sample['audio_path'],
            'num_frames': sample['num_frames'],
            'frame_duration': self.frame_duration,
        }

    def get_normalization_params(self, num_samples=100):
        return estimate_normalization_from_dataset(self, sample_count=num_samples)

    def split_indices(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        return song_level_split_indices(
            segment_song_indices=[seg['song_idx'] for seg in self.song_segments],
            num_songs=len(self.samples),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            min_train_one=True,
            ensure_val_if_possible=True,
        )
