"""
Dataset for Phase 1 online pseudo-labeling from unlabeled audio.

Recursively scans a root directory for audio files, computes CQT features on the
fly, and exposes fixed-length segments suitable for pseudo-label generation by a
teacher model at training time.  Returned metadata is also used by ChordNet's
overlap-aware validation/test aggregation.
"""
import os
from collections import OrderedDict

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

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
        teacher_model=None,
        teacher_mean=None,
        teacher_std=None,
        teacher_cache_size=8,
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
        self.teacher_model = teacher_model
        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std
        self.teacher_cache_size = max(1, int(teacher_cache_size))
        self.teacher_logits_cache = OrderedDict()

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
        return np.log(np.abs(cqt) + 1e-6).T

    def _extract_full_song_features(self, audio_path):
        with _suppress_stderr():
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

        cqt = librosa.cqt(
            y,
            sr=sr,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            hop_length=self.hop_length,
            fmin=librosa.note_to_hz('C1'),
        )
        return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)

    def _teacher_device(self):
        if self.teacher_model is None:
            return None
        return next(self.teacher_model.parameters()).device

    def _get_full_song_teacher_logits(self, song_idx):
        if self.teacher_model is None:
            return None
        if song_idx in self.teacher_logits_cache:
            cached = self.teacher_logits_cache.pop(song_idx)
            self.teacher_logits_cache[song_idx] = cached
            return cached

        sample = self.samples[song_idx]
        feature_matrix = self._extract_full_song_features(sample['audio_path'])
        device = self._teacher_device()
        feature_tensor = torch.from_numpy(feature_matrix).float().unsqueeze(0).to(device)

        mean = torch.as_tensor(self.teacher_mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(self.teacher_std, dtype=torch.float32, device=device)

        with torch.no_grad():
            self.teacher_model.eval()
            teacher_inputs = (feature_tensor - mean) / (std + 1e-8)
            outputs = self.teacher_model(teacher_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)

        self.teacher_logits_cache[song_idx] = logits
        while len(self.teacher_logits_cache) > self.teacher_cache_size:
            self.teacher_logits_cache.popitem(last=False)
        return logits

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

        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        if feature_tensor.shape[0] < self.seq_len:
            pad = self.seq_len - feature_tensor.shape[0]
            feature_tensor = torch.nn.functional.pad(feature_tensor, (0, 0, 0, pad))
        elif feature_tensor.shape[0] > self.seq_len:
            feature_tensor = feature_tensor[:self.seq_len]

        return {
            'spectro': feature_tensor,
            'song_id': sample['song_id'],
            'audio_path': sample['audio_path'],
            'start_frame': segment['start_frame'],
            'end_frame': segment['end_frame'],
            'num_frames': sample['num_frames'],
            'frame_duration': self.frame_duration,
            **self._teacher_slice(segment),
        }

    def _teacher_slice(self, segment):
        if self.teacher_model is None:
            return {}

        full_logits = self._get_full_song_teacher_logits(segment['song_idx'])
        start_frame = int(segment['start_frame'])
        end_frame = int(segment['end_frame'])
        teacher_logits = np.asarray(full_logits[start_frame:end_frame], dtype=np.float32)

        if teacher_logits.shape[0] < self.seq_len:
            pad = self.seq_len - teacher_logits.shape[0]
            teacher_logits = np.pad(teacher_logits, ((0, pad), (0, 0)), mode='constant')
        elif teacher_logits.shape[0] > self.seq_len:
            teacher_logits = teacher_logits[:self.seq_len]

        return {'teacher_logits': torch.from_numpy(teacher_logits).float()}

    def get_normalization_params(self, num_samples=100):
        if not self.song_segments:
            return 0.0, 1.0

        sample_count = min(num_samples, len(self.song_segments))
        indices = np.random.choice(len(self.song_segments), sample_count, replace=False)
        feats = []
        for idx in indices:
            try:
                feats.append(self[idx]['spectro'].numpy())
            except Exception:
                continue

        if not feats:
            return 0.0, 1.0

        all_features = np.concatenate(feats, axis=0)
        return float(np.mean(all_features)), float(np.std(all_features))

    def split_indices(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        rng = np.random.RandomState(seed)
        song_indices = list(range(len(self.samples)))
        rng.shuffle(song_indices)

        if not song_indices:
            return [], [], []

        n_train = max(1, int(len(song_indices) * train_ratio))
        n_val = int(len(song_indices) * val_ratio)
        if n_val == 0 and len(song_indices) >= 3:
            n_val = 1
            n_train = min(n_train, len(song_indices) - n_val - 1)

        train_songs = set(song_indices[:n_train])
        val_songs = set(song_indices[n_train:n_train + n_val])
        test_songs = set(song_indices[n_train + n_val:])

        train_idx = [
            i for i, seg in enumerate(self.song_segments)
            if seg['song_idx'] in train_songs
        ]
        val_idx = [
            i for i, seg in enumerate(self.song_segments)
            if seg['song_idx'] in val_songs
        ]
        test_idx = [
            i for i, seg in enumerate(self.song_segments)
            if seg['song_idx'] in test_songs
        ]
        return train_idx, val_idx, test_idx
