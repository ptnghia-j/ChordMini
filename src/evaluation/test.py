"""
Run ChordMini checkpoint inference on audio and write .lab outputs.

This is the ChordMini-local counterpart of the root test.py flow.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import librosa
from tqdm import tqdm

# bootstrap the project root and ensure src is on the path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.cli import bootstrap_cli

bootstrap_cli(__file__)

from src.evaluation.utils.common import (
    extract_norm_stats as _extract_norm_stats,
    extract_song_features as _extract_song_features,
)
from src.evaluation.utils.inference import predict_sliding_windows
from src.models import load_model
from src.utils import HParams, error, get_config_value as _cfg, get_device, idx2voca_chord, info, logging_verbosity, project_path, set_random_seed, warning


DEFAULT_CONFIG = str(project_path('config', 'ChordMini.yaml', start=__file__))
DEFAULT_AUDIO_DIR = str(project_path('data', 'labeled', 'audio', start=__file__))
DEFAULT_SAVE_DIR = str(project_path('outputs', 'chordlab_pred', start=__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate chord .lab files from audio')

    parser.add_argument('--audio_dir', type=str, default=DEFAULT_AUDIO_DIR)
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR)

    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--model_type', type=str, default='ChordNet', choices=['ChordNet', 'BTC'])

    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, default=None)
    parser.add_argument('--model_checkpoint', dest='checkpoint', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--smooth_predictions', action='store_true')
    parser.add_argument('--smooth_logits', action='store_true')
    parser.add_argument('--kernel_size', type=int, default=9)
    parser.add_argument('--use_gaussian', action='store_true')
    parser.add_argument('--use_overlap', action='store_true', default=None)
    parser.add_argument('--overlap_ratio', type=float, default=None)
    parser.add_argument('--vote_aggregation', choices=['hard', 'logit', 'prob'], default='hard')
    parser.add_argument('--min_segment_duration', type=float, default=0.0)

    args = parser.parse_args()

    if args.checkpoint is None and args.model_file is not None:
        args.checkpoint = args.model_file

    if not args.checkpoint:
        parser.error('the following arguments are required: --checkpoint (or --model_checkpoint / --model_file)')

    if args.kernel_size < 1:
        args.kernel_size = 1
    if args.kernel_size % 2 == 0:
        args.kernel_size += 1

    return args


def _list_audio_files(audio_dir):
    exts = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    # Support both a directory of audio files and a single audio file path.
    if os.path.isfile(audio_dir):
        return [audio_dir] if audio_dir.lower().endswith(exts) else []

    files = []
    for root, _, names in os.walk(audio_dir):
        for name in names:
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))
    return sorted(files)


def _prediction_segments(predictions, frame_duration, idx_to_chord, min_segment_duration):
    lines = []
    if predictions.size == 0:
        return lines

    def _label(idx):
        return str(idx_to_chord.get(int(idx), idx_to_chord.get(169, 'N')))

    prev = int(predictions[0])
    start_time = 0.0

    for i in range(1, len(predictions)):
        cur = int(predictions[i])
        if cur != prev:
            end_time = i * frame_duration
            if end_time - start_time >= min_segment_duration:
                lines.append(f'{start_time:.6f} {end_time:.6f} {_label(prev)}\n')
            start_time = end_time
            prev = cur

    final_end = len(predictions) * frame_duration
    if final_end - start_time >= min_segment_duration:
        lines.append(f'{start_time:.6f} {final_end:.6f} {_label(prev)}\n')

    return lines


def _resolve_seq_len(config, model, checkpoint_path):
    seq_len = None

    # Match root test.py priority: checkpoint > model attr > config > default.
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            seq_len = checkpoint.get('timestep')
    except Exception:
        seq_len = None

    if seq_len is None and hasattr(model, 'timestep'):
        seq_len = getattr(model, 'timestep', None)

    if seq_len is None:
        seq_len = _cfg(config, 'model', 'seq_len', _cfg(config, 'model', 'timestep', 108))

    try:
        seq_len = int(seq_len)
    except Exception:
        seq_len = 108

    if seq_len <= 0:
        seq_len = 108

    return seq_len


def _song_duration_seconds(audio_path, sample_rate):
    try:
        return float(librosa.get_duration(path=audio_path))
    except Exception:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        return float(len(audio) / float(sr))


def _extract_song_features_root_compatible(audio_path, config):
    extracted = _extract_song_features(audio_path, config)
    if isinstance(extracted, tuple):
        features = extracted[0]
        frame_duration = float(extracted[1]) if len(extracted) > 1 else float(_cfg(config, 'feature', 'hop_duration', _cfg(config, 'feature', 'hop_length', 2048) / _cfg(config, 'mp3', 'song_hz', 22050)))
    else:
        features = extracted
        frame_duration = float(_cfg(config, 'feature', 'hop_duration', _cfg(config, 'feature', 'hop_length', 2048) / _cfg(config, 'mp3', 'song_hz', 22050)))

    sample_rate = _cfg(config, 'mp3', 'song_hz', 22050)
    song_duration = _song_duration_seconds(audio_path, sample_rate)
    return np.asarray(features, dtype=np.float32), float(frame_duration), float(song_duration)


def main():
    args = parse_args()
    logging_verbosity(2 if args.verbose else 1)

    set_random_seed(args.seed, include_python_random=True)

    if not os.path.exists(args.audio_dir):
        error(f'Audio directory not found: {args.audio_dir}')
        return

    if not os.path.exists(args.config):
        error(f'Config not found: {args.config}')
        return

    if not os.path.exists(args.checkpoint):
        error(f'Checkpoint not found: {args.checkpoint}')
        return

    config = HParams.load(args.config)
    device = get_device()
    info(f'Using device: {device}')

    model, _, _ = load_model(args.checkpoint, args.model_type, config, device, args)
    mean, std = _extract_norm_stats(args.checkpoint)
    # Use root-style canonical vocabulary for .lab outputs.
    idx_to_chord = idx2voca_chord()

    model.idx_to_chord = idx_to_chord
    model.eval()

    if args.model_type == 'BTC' and args.use_gaussian and not args.smooth_logits:
        warning('BTC uses --use_gaussian only when --smooth_logits is enabled')

    seq_len = _resolve_seq_len(config, model, args.checkpoint)

    audio_files = _list_audio_files(args.audio_dir)
    if not audio_files:
        warning(f'No audio files found in {args.audio_dir}')
        return

    os.makedirs(args.save_dir, exist_ok=True)

    info(f'Generating labels for {len(audio_files)} audio files')

    predict_args = argparse.Namespace(
        model_type=args.model_type,
        use_overlap=args.use_overlap,
        overlap_ratio=args.overlap_ratio,
        vote_aggregation=args.vote_aggregation,
        kernel_size=args.kernel_size,
        use_gaussian=args.use_gaussian,
        smooth_logits=args.smooth_logits,
        smooth_predictions=args.smooth_predictions,
    )

    num_classes = 170

    for audio_path in tqdm(audio_files, desc='Generating .lab files'):
        try:
            features, frame_duration, song_duration = _extract_song_features_root_compatible(audio_path, config)

            preds = predict_sliding_windows(
                model=model,
                feature_matrix=features,
                mean=mean,
                std=std,
                seq_len=seq_len,
                batch_size=16,
                model_type=predict_args.model_type,
                n_classes=num_classes,
                vote_aggregation=predict_args.vote_aggregation,
                use_overlap=predict_args.use_overlap,
                overlap_ratio=predict_args.overlap_ratio,
                smooth_logits=predict_args.smooth_logits,
                smooth_predictions=predict_args.smooth_predictions,
                kernel_size=predict_args.kernel_size,
                use_gaussian=predict_args.use_gaussian,
            )

            # Match root timing clamp to avoid extending beyond real song duration.
            max_frames = int(np.floor(song_duration / frame_duration)) if frame_duration > 0 else len(preds)
            if max_frames > 0:
                preds = preds[:max_frames]

            lines = _prediction_segments(
                predictions=preds,
                frame_duration=float(frame_duration),
                idx_to_chord=idx_to_chord,
                min_segment_duration=float(args.min_segment_duration),
            )

            out_name = f"{Path(audio_path).stem}.lab"
            out_path = os.path.join(args.save_dir, out_name)
            with open(out_path, 'w') as handle:
                handle.writelines(lines)

            info(f'Saved chord annotations to {out_path}')
        except Exception as ex:
            error(f'Error processing {audio_path}: {ex}')

    info('Chord recognition complete')


if __name__ == '__main__':
    main()
