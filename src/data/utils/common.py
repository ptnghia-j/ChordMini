from __future__ import annotations

import numpy as np


def estimate_normalization_from_dataset(dataset, sample_count=100):
    """Estimate mean/std from a random subset of dataset items."""
    dataset_size = len(dataset)
    if dataset_size == 0:
        return 0.0, 1.0

    sample_count = min(int(sample_count), dataset_size)
    indices = np.random.choice(dataset_size, sample_count, replace=False)
    features = []

    for idx in indices:
        try:
            item = dataset[idx]
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        spectro = item.get('spectro')
        if spectro is None or not hasattr(spectro, 'numpy'):
            continue
        features.append(spectro.numpy())

    if not features:
        return 0.0, 1.0

    all_features = np.concatenate(features, axis=0)
    return float(np.mean(all_features)), float(np.std(all_features))


def song_level_split_indices(
    segment_song_indices,
    num_songs,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    min_train_one=False,
    ensure_val_if_possible=False,
):
    """Split segment indices by song id into train/val/test sets."""
    rng = np.random.RandomState(seed)
    song_ids = list(range(int(num_songs)))
    rng.shuffle(song_ids)

    if not song_ids:
        return [], [], []

    n_train = int(len(song_ids) * train_ratio)
    if min_train_one:
        n_train = max(1, n_train)

    n_val = int(len(song_ids) * val_ratio)
    if ensure_val_if_possible and n_val == 0 and len(song_ids) >= 3:
        n_val = 1
        n_train = min(n_train, len(song_ids) - n_val - 1)

    train_songs = set(song_ids[:n_train])
    val_songs = set(song_ids[n_train:n_train + n_val])
    test_songs = set(song_ids[n_train + n_val:])

    train_idx = [i for i, song_idx in enumerate(segment_song_indices) if song_idx in train_songs]
    val_idx = [i for i, song_idx in enumerate(segment_song_indices) if song_idx in val_songs]
    test_idx = [i for i, song_idx in enumerate(segment_song_indices) if song_idx in test_songs]
    return train_idx, val_idx, test_idx


def song_level_cv_fold_indices(segment_song_indices, num_songs, n_folds=5, seed=42):
    """Create song-level cross-validation folds as (train_idx, val_idx) pairs."""
    rng = np.random.RandomState(seed)
    song_ids = list(range(int(num_songs)))
    rng.shuffle(song_ids)

    if not song_ids:
        return []

    fold_size = len(song_ids) // n_folds
    folds = []

    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        val_songs = set(song_ids[start:] if fold_idx == n_folds - 1 else song_ids[start:start + fold_size])
        train_songs = set(song_ids) - val_songs

        train_idx = [i for i, song_idx in enumerate(segment_song_indices) if song_idx in train_songs]
        val_idx = [i for i, song_idx in enumerate(segment_song_indices) if song_idx in val_songs]
        folds.append((train_idx, val_idx))

    return folds


__all__ = [
    'estimate_normalization_from_dataset',
    'song_level_cv_fold_indices',
    'song_level_split_indices',
]
