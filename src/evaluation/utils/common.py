from __future__ import annotations

import os
import re

import librosa
import mir_eval
import numpy as np

from src.utils import (
    _parse_chord_string,
    extract_normalization_stats,
    get_config_value,
    idx2voca_chord,
    load_checkpoint,
    warning,
    error,
)
from src.utils.audio_io import suppress_stderr as _suppress_stderr


_VOCAB_CHORDS = set(idx2voca_chord().values())
_QUALITY_SIMPLIFICATION_MAP = {
    '5': 'maj',
    '1': 'maj',
    '6': 'maj6',
    '9': '7',
    '11': '7',
    '13': '7',
    'maj9': 'maj7',
    'maj11': 'maj7',
    'maj13': 'maj7',
    'min9': 'min7',
    'min11': 'min7',
    'min13': 'min7',
    'm7b5': 'hdim7',
    'min7b5': 'hdim7',
    '7sus': 'sus4',
    '7sus4': 'sus4',
    'sus': 'sus4',
}
_RAW_QUALITY_OVERRIDE_MAP = {
    '(1,2,4,5)': 'sus4',
    '(1,4,5)': 'sus4',
    '(1,4)': 'sus4',
    'maj(11)': 'maj7',
    'min(11)': 'min7',
    'sus2(4)': 'sus4',
    'sus4(2)': 'sus4',
}


def _extract_raw_quality(label):
    match = re.match(r'^\s*([A-G](?:#|b)?|N|X)(?::([^/]+))?(?:/.*)?\s*$', str(label))
    if not match:
        return ''
    return (match.group(2) or '').strip()


def normalize_chord_label(label, chord_parser):
    root, quality, _ = _parse_chord_string(label)
    if root in ('N', 'X') or root is None:
        return 'N'

    raw_quality = _extract_raw_quality(label)
    if raw_quality in _RAW_QUALITY_OVERRIDE_MAP:
        quality = _RAW_QUALITY_OVERRIDE_MAP[raw_quality]
    elif raw_quality in _QUALITY_SIMPLIFICATION_MAP:
        quality = _QUALITY_SIMPLIFICATION_MAP[raw_quality]
    elif quality in _QUALITY_SIMPLIFICATION_MAP:
        quality = _QUALITY_SIMPLIFICATION_MAP[quality]

    normalized = root if not quality or quality == 'maj' else f'{root}:{quality}'
    if normalized in _VOCAB_CHORDS:
        return normalized

    fallback = chord_parser.label_error_modify(label)
    if fallback in _VOCAB_CHORDS:
        return fallback
    return 'N'


def resolve_audio_label_dirs(args, default_audio_dir, default_label_dir):
    if args.data_dir:
        candidates = [
            (os.path.join(args.data_dir, 'audio'), os.path.join(args.data_dir, 'chordlab')),
            (os.path.join(args.data_dir, 'labeled', 'audio'), os.path.join(args.data_dir, 'labeled', 'chordlab')),
        ]
        for cand_audio, cand_label in candidates:
            if os.path.isdir(cand_audio) and os.path.isdir(cand_label):
                args.audio_dir = cand_audio
                args.label_dir = cand_label
                return
        args.audio_dir = candidates[0][0]
        args.label_dir = candidates[0][1]
        return

    if not os.path.isdir(args.audio_dir) and os.path.isdir(default_audio_dir):
        args.audio_dir = default_audio_dir
    if not os.path.isdir(args.label_dir) and os.path.isdir(default_label_dir):
        args.label_dir = default_label_dir


def extract_norm_stats(checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path, device='cpu') or {}
    return extract_normalization_stats(checkpoint)


def extract_vocab(checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path, device='cpu') or {}
    idx_to_chord = checkpoint.get('idx_to_chord') or idx2voca_chord()
    chord_to_idx = {label: idx for idx, label in idx_to_chord.items()}
    return idx_to_chord, chord_to_idx


def list_song_pairs(audio_dir, label_dir):
    audio_exts = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    label_exts = ('.lab', '.txt')
    pairs = []

    if not os.path.isdir(audio_dir):
        error(f'Audio directory not found: {audio_dir}')
        return pairs
    if not os.path.isdir(label_dir):
        error(f'Label directory not found: {label_dir}')
        return pairs

    labels_by_stem = {}
    for name in os.listdir(label_dir):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in label_exts):
            stem, _ = os.path.splitext(name)
            labels_by_stem[stem] = os.path.join(label_dir, name)

    skipped = 0
    for file_name in sorted(os.listdir(audio_dir)):
        if not file_name.lower().endswith(audio_exts):
            continue
        stem, _ = os.path.splitext(file_name)
        label_path = labels_by_stem.get(stem)
        if not label_path:
            skipped += 1
            continue
        pairs.append({
            'song_id': stem,
            'audio_path': os.path.join(audio_dir, file_name),
            'label_path': label_path,
        })

    if skipped:
        warning(f'Skipped {skipped} audio files without matching labels')
    return pairs


def parse_labels(label_path, chord_parser):
    labels = []
    with open(label_path, 'r') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 3:
                labels.append((
                    float(parts[0]),
                    float(parts[1]),
                    normalize_chord_label(parts[2], chord_parser),
                ))
    return labels


def labels_to_frame_labels(labels, num_frames, frame_duration):
    frame_labels = ['N'] * int(num_frames)
    for start, end, chord in labels:
        start_frame = int(float(start) / float(frame_duration))
        end_frame = min(int(float(end) / float(frame_duration)) + 1, int(num_frames))
        if start_frame < int(num_frames):
            for frame_idx in range(max(0, start_frame), max(0, end_frame)):
                frame_labels[frame_idx] = chord
    return frame_labels


def extract_song_features(audio_path, config):
    sample_rate = get_config_value(config, 'mp3', 'song_hz', 22050)
    hop_length = get_config_value(config, 'feature', 'hop_length', 2048)
    n_bins = get_config_value(config, 'feature', 'n_bins', 144)
    bins_per_octave = get_config_value(config, 'feature', 'bins_per_octave', 24)
    with _suppress_stderr():
        audio, sr = librosa.load(audio_path, sr=sample_rate)
    cqt = librosa.cqt(
        audio,
        sr=sr,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        fmin=librosa.note_to_hz('C1'),
    )
    # Derive frame duration from the actual CQT hop parameters so evaluation
    # stays aligned with extracted frames instead of depending on a rounded
    # config value like 0.09288.
    frame_duration = float(hop_length) / float(sample_rate)
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32), float(frame_duration)


def majority_filter_indices(indices, kernel_size):
    """Apply a categorical majority filter to frame-wise class indices."""
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        return values.copy()

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1 or values.size < kernel_size:
        return values.copy()

    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    filtered = np.empty_like(values)

    for idx in range(values.size):
        window = padded[idx:idx + kernel_size]
        labels, counts = np.unique(window, return_counts=True)
        max_count = counts.max()
        candidates = labels[counts == max_count]
        center = values[idx]
        filtered[idx] = center if center in candidates else int(candidates[0])

    return filtered


def build_intervals(num_frames, frame_duration):
    timestamps = np.arange(num_frames, dtype=np.float64) * float(frame_duration)
    intervals = np.zeros((num_frames, 2), dtype=np.float64)
    intervals[:, 0] = timestamps
    intervals[:-1, 1] = timestamps[1:]
    intervals[-1, 1] = timestamps[-1] + frame_duration

    bad = intervals[:, 0] >= intervals[:, 1]
    if np.any(bad):
        intervals[bad, 1] = intervals[bad, 0] + 1e-6

    return timestamps, intervals


def calculate_chord_scores(frame_duration, reference_labels, prediction_labels):
    min_len = min(len(reference_labels), len(prediction_labels))
    if min_len == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    reference_labels = list(reference_labels[:min_len])
    prediction_labels = list(prediction_labels[:min_len])
    _, intervals = build_intervals(min_len, frame_duration)

    try:
        scores = mir_eval.chord.evaluate(intervals, reference_labels, intervals, prediction_labels)
        return (
            float(scores.get('root', 0.0)),
            float(scores.get('thirds', 0.0)),
            float(scores.get('triads', 0.0)),
            float(scores.get('sevenths', 0.0)),
            float(scores.get('tetrads', 0.0)),
            float(scores.get('majmin', 0.0)),
            float(scores.get('mirex', 0.0)),
        )
    except Exception as ex:
        warning(f'Error calculating MIR chord scores: {ex}')
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def calculate_segmentation_scores(frame_duration, reference_labels, prediction_labels):
    min_len = min(len(reference_labels), len(prediction_labels))
    if min_len == 0:
        return 0.0, 0.0, 0.0

    reference_labels = list(reference_labels[:min_len])
    prediction_labels = list(prediction_labels[:min_len])
    _, intervals = build_intervals(min_len, frame_duration)

    try:
        ref_merged = mir_eval.chord.merge_chord_intervals(intervals, reference_labels)
        pred_merged = mir_eval.chord.merge_chord_intervals(intervals, prediction_labels)
        overseg = float(mir_eval.chord.overseg(ref_merged, pred_merged))
        underseg = float(mir_eval.chord.underseg(ref_merged, pred_merged))
        seg = float(min(overseg, underseg))
        return max(0.0, min(1.0, overseg)), max(0.0, min(1.0, underseg)), max(0.0, min(1.0, seg))
    except Exception as ex:
        warning(f'Error calculating segmentation scores: {ex}')
        return 0.0, 0.0, 0.0


def frame_indices_to_labels(indices, idx_to_chord):
    default_label = idx_to_chord.get(169, 'N')
    labels = []
    for idx in indices:
        labels.append(str(idx_to_chord.get(int(idx), default_label)))
    return labels


def weighted_average(metric_values, durations):
    if not metric_values or not durations:
        return 0.0
    total = float(sum(durations))
    if total <= 0:
        return float(np.mean(metric_values)) if metric_values else 0.0
    return float(sum(v * d for v, d in zip(metric_values, durations)) / total)


def dataset_identifier(args, per_song=None):
    if per_song:
        first_song = per_song[0].get('song_id')
        if first_song:
            return str(first_song).split('/')[0]
    if args.data_dir:
        return os.path.basename(os.path.normpath(args.data_dir)) or 'dataset'
    return os.path.basename(os.path.normpath(args.audio_dir)) or 'dataset'


__all__ = [
    'calculate_chord_scores',
    'calculate_segmentation_scores',
    'dataset_identifier',
    'extract_norm_stats',
    'extract_song_features',
    'extract_vocab',
    'frame_indices_to_labels',
    'labels_to_frame_labels',
    'list_song_pairs',
    'majority_filter_indices',
    'normalize_chord_label',
    'parse_labels',
    'resolve_audio_label_dirs',
    'weighted_average',
]
