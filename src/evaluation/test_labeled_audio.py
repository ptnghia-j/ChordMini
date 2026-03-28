"""
Evaluate ChordMini BTC or ChordNet checkpoints on labeled audio.

This evaluator mirrors the root-level metric coverage:
- MIR metrics: root, thirds, triads, sevenths, tetrads, majmin, mirex
- Segmentation metrics: overseg, underseg, seg
- Chord-quality distribution and per-quality accuracy
- Visualization outputs: distribution/accuracy plot and confusion matrix

It is fully ChordMini-local and does not import from repository root modules.
It evaluates only real audio frames and does not score any sequence padding
past the end of a song.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.cli import bootstrap_cli

# bootstrap the project root and ensure src is on the path
_CHORDMINI_ROOT = bootstrap_cli(__file__)
_CACHE_ROOT = _CHORDMINI_ROOT / '.cache'
# matplotlib cache for the plots
(_CACHE_ROOT / 'matplotlib').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', str(_CACHE_ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(_CACHE_ROOT / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.utils.common import (
    calculate_chord_scores,
    calculate_segmentation_scores,
    dataset_identifier as _dataset_identifier,
    extract_norm_stats as _extract_norm_stats,
    extract_song_features as _extract_song_features,
    extract_vocab as _extract_vocab,
    frame_indices_to_labels as _frame_indices_to_labels,
    labels_to_frame_labels as _labels_to_frame_labels,
    list_song_pairs as _list_song_pairs,
    normalize_chord_label as _normalize_chord_label,
    parse_labels as _parse_labels,
    resolve_audio_label_dirs as _resolve_audio_label_dirs,
    weighted_average as _weighted_average,
)
from src.evaluation.utils.inference import predict_sliding_windows, resolve_effective_overlap_ratio
from src.evaluation.utils.quality_analysis import (
    compute_chord_quality_accuracy,
    compute_macro_frame_metrics,
    compute_paper_quality_metrics,
    generate_chord_distribution_accuracy_plot,
    generate_confusion_matrix_heatmap,
)
from src.training.utils.chordnet_eval import (
    CHORDNET_EVAL_KERNEL_SIZE,
    CHORDNET_EVAL_OVERLAP_RATIO,
)
from src.models import load_model
from src.utils import (
    Chords,
    HParams,
    debug,
    error,
    get_device,
    info,
    get_config_value,
    logging_verbosity,
    project_path,
    set_random_seed,
    warning,
)


DEFAULT_CONFIG = str(project_path('config', 'ChordMini.yaml', start=__file__))
DEFAULT_DATA_DIR = str(project_path('data', 'labeled', start=__file__))
DEFAULT_AUDIO_DIR = str(project_path('data', 'labeled', 'audio', start=__file__))
DEFAULT_LABEL_DIR = str(project_path('data', 'labeled', 'chordlab', start=__file__))
DEFAULT_OUTPUT = 'evaluation_results.json'
_cfg = get_config_value


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BTC or ChordNet on labeled audio')

    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--audio_dir', type=str, default=DEFAULT_AUDIO_DIR)
    parser.add_argument('--label_dir', type=str, default=DEFAULT_LABEL_DIR)

    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--model_type', type=str, choices=['BTC', 'ChordNet'], default='BTC')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, default=None)
    parser.add_argument('--model_checkpoint', dest='checkpoint', type=str, default=None)
    parser.add_argument('--teacher_checkpoint', type=str, default=None)
    parser.add_argument('--teacher_model_type', type=str, choices=['BTC', 'ChordNet'], default='BTC')

    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--window_batch_size', type=int, default=16)

    parser.add_argument('--start_file', type=int, default=0)
    parser.add_argument('--end_file', type=int, default=None)
    parser.add_argument('--max_songs', type=int, default=None)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--random_subset', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--smooth_predictions', action='store_true')
    parser.add_argument('--smooth_logits', action='store_true')
    parser.add_argument('--kernel_size', type=int, default=9)
    parser.add_argument('--use_gaussian', action='store_true')
    parser.add_argument('--use_overlap', action='store_true', default=None)
    parser.add_argument('--overlap_ratio', type=float, default=None)
    parser.add_argument('--vote_aggregation', choices=['hard', 'logit', 'prob'], default='hard')

    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    if args.max_songs is None and args.max_files is not None:
        args.max_songs = args.max_files

    if not args.checkpoint:
        parser.error('the following arguments are required: --checkpoint (or --model_checkpoint)')

    if args.output_json is None:
        args.output_json = args.output

    if args.kernel_size < 1:
        args.kernel_size = 1
    if args.kernel_size % 2 == 0:
        args.kernel_size += 1

    return args


def evaluate_song(model, song_item, config, mean, std, idx_to_chord, chord_to_idx, chord_parser, args, teacher_bundle=None):
    feature_matrix, frame_duration = _extract_song_features(song_item['audio_path'], config)

    preds = predict_sliding_windows(
        model=model,
        feature_matrix=feature_matrix,
        mean=mean,
        std=std,
        seq_len=args.seq_len,
        batch_size=args.window_batch_size,
        model_type=args.model_type,
        n_classes=len(chord_to_idx),
        vote_aggregation=args.vote_aggregation,
        use_overlap=args.use_overlap,
        overlap_ratio=args.overlap_ratio,
        smooth_logits=args.smooth_logits,
        smooth_predictions=args.smooth_predictions,
        kernel_size=args.kernel_size,
        use_gaussian=args.use_gaussian,
        chordnet_default_overlap_ratio=CHORDNET_EVAL_OVERLAP_RATIO,
        use_chordnet_defaults=True,
    )

    labels = _parse_labels(song_item['label_path'], chord_parser)
    ref_labels = _labels_to_frame_labels(labels, len(preds), frame_duration)

    valid_len = min(len(preds), len(ref_labels))
    preds = preds[:valid_len]
    ref_labels = ref_labels[:valid_len]

    pred_labels = _frame_indices_to_labels(preds, idx_to_chord)
    ref_labels = [_normalize_chord_label(lbl, chord_parser) for lbl in ref_labels]
    pred_labels = [_normalize_chord_label(lbl, chord_parser) for lbl in pred_labels]

    teacher_labels = None
    if teacher_bundle is not None:
        teacher_preds = predict_sliding_windows(
            model=teacher_bundle['model'],
            feature_matrix=feature_matrix,
            mean=teacher_bundle['mean'],
            std=teacher_bundle['std'],
            seq_len=args.seq_len,
            batch_size=args.window_batch_size,
            model_type=teacher_bundle['model_type'],
            n_classes=len(chord_to_idx),
            vote_aggregation=args.vote_aggregation,
            use_overlap=args.use_overlap,
            overlap_ratio=args.overlap_ratio,
            smooth_logits=args.smooth_logits,
            smooth_predictions=args.smooth_predictions,
            kernel_size=args.kernel_size,
            use_gaussian=args.use_gaussian,
            chordnet_default_overlap_ratio=CHORDNET_EVAL_OVERLAP_RATIO,
            use_chordnet_defaults=True,
        )
        teacher_preds = teacher_preds[:valid_len]
        teacher_labels = _frame_indices_to_labels(teacher_preds, teacher_bundle['idx_to_chord'])
        teacher_labels = [_normalize_chord_label(lbl, chord_parser) for lbl in teacher_labels]

    root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score = calculate_chord_scores(
        frame_duration=frame_duration,
        reference_labels=ref_labels,
        prediction_labels=pred_labels,
    )
    overseg_score, underseg_score, seg_score = calculate_segmentation_scores(
        frame_duration=frame_duration,
        reference_labels=ref_labels,
        prediction_labels=pred_labels,
    )

    correct = int(sum(1 for r, p in zip(ref_labels, pred_labels) if r == p))
    total = int(valid_len)

    return {
        'song_id': song_item['song_id'],
        'frames': total,
        'correct': correct,
        'accuracy': (correct / total) if total else 0.0,
        'duration': total * float(frame_duration),
        'scores': {
            'root': root_score,
            'thirds': thirds_score,
            'triads': triads_score,
            'sevenths': sevenths_score,
            'tetrads': tetrads_score,
            'majmin': majmin_score,
            'mirex': mirex_score,
            'overseg': overseg_score,
            'underseg': underseg_score,
            'seg': seg_score,
        },
        'ref_labels': ref_labels,
        'pred_labels': pred_labels,
        'teacher_labels': teacher_labels,
    }

def main():
    args = parse_args()
    logging_verbosity(2 if args.verbose else 1)

    _resolve_audio_label_dirs(args, DEFAULT_AUDIO_DIR, DEFAULT_LABEL_DIR)

    set_random_seed(args.seed, include_python_random=True)

    device = get_device()
    info(f'Using device: {device}')

    if not os.path.exists(args.config):
        error(f'Config file not found: {args.config}')
        return

    config = HParams.load(args.config)

    if args.seq_len is None:
        args.seq_len = int(
            _cfg(config, 'model', 'seq_len', _cfg(config, 'model', 'timestep', _cfg(config, 'training', 'seq_len', 108)))
        )

    if not os.path.exists(args.checkpoint):
        error(f'Checkpoint not found: {args.checkpoint}')
        return

    model, _, _ = load_model(args.checkpoint, args.model_type, config, device, args)
    mean, std = _extract_norm_stats(args.checkpoint)
    idx_to_chord, chord_to_idx = _extract_vocab(args.checkpoint)

    chord_parser = Chords()
    chord_parser.set_chord_mapping(chord_to_idx)
    model.idx_to_chord = idx_to_chord
    model.eval()

    teacher_bundle = None
    if args.teacher_checkpoint:
        teacher_model, _, _ = load_model(args.teacher_checkpoint, args.teacher_model_type, config, device, args)
        teacher_mean, teacher_std = _extract_norm_stats(args.teacher_checkpoint)
        teacher_idx_to_chord, _ = _extract_vocab(args.teacher_checkpoint)
        teacher_model.idx_to_chord = teacher_idx_to_chord
        teacher_model.eval()
        teacher_bundle = {
            'model': teacher_model,
            'mean': teacher_mean,
            'std': teacher_std,
            'idx_to_chord': teacher_idx_to_chord,
            'model_type': args.teacher_model_type,
        }

    if args.model_type == 'BTC' and args.use_gaussian and not args.smooth_logits:
        warning('BTC uses --use_gaussian only when --smooth_logits is enabled')

    songs = _list_song_pairs(args.audio_dir, args.label_dir)

    if args.random_subset:
        requested = len(songs) if args.max_songs is None or args.max_songs < 0 else args.max_songs
        requested = min(requested, len(songs))
        if requested < len(songs):
            songs = random.sample(songs, requested)
        else:
            songs = list(songs)
    else:
        if args.end_file is not None:
            songs = songs[args.start_file:args.end_file]
        else:
            songs = songs[args.start_file:]

        if args.max_songs is not None and args.max_songs >= 0:
            songs = songs[:args.max_songs]

    if not songs:
        warning('No labeled audio pairs found for evaluation')
        return

    info(f'Evaluating {len(songs)} songs with {args.model_type}')

    per_song = []
    all_ref_labels = []
    all_pred_labels = []
    all_teacher_labels = []

    with torch.no_grad():
        for song_item in tqdm(songs, desc='Evaluating songs'):
            try:
                metrics = evaluate_song(
                    model,
                    song_item,
                    config,
                    mean,
                    std,
                    idx_to_chord,
                    chord_to_idx,
                    chord_parser,
                    args,
                    teacher_bundle=teacher_bundle,
                )
                per_song.append(metrics)
                all_ref_labels.extend(metrics['ref_labels'])
                all_pred_labels.extend(metrics['pred_labels'])
                if metrics.get('teacher_labels') is not None:
                    all_teacher_labels.extend(metrics['teacher_labels'])
                info(
                    f"Song {metrics['song_id']}: length={metrics['duration']:.1f}s, "
                    f"root={metrics['scores']['root']:.4f}, "
                    f"mirex={metrics['scores']['mirex']:.4f}, "
                    f"seg={metrics['scores']['seg']:.4f}"
                )
            except Exception as ex:
                error(f"Error evaluating {song_item['song_id']}: {ex}")
                debug(traceback.format_exc())

    if not per_song:
        error('No songs were evaluated successfully')
        return

    metric_names = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex', 'overseg', 'underseg', 'seg']
    durations = [song['duration'] for song in per_song]

    average_scores = {}
    segmentation_metrics = {'overseg', 'underseg', 'seg'}
    for metric in metric_names:
        metric_vals = [song['scores'][metric] for song in per_song]
        if metric in segmentation_metrics:
            average_scores[metric] = float(np.mean(metric_vals)) if metric_vals else 0.0
        else:
            average_scores[metric] = _weighted_average(metric_vals, durations)

    total_correct = sum(song['correct'] for song in per_song)
    total_frames = sum(song['frames'] for song in per_song)
    frame_accuracy = (total_correct / total_frames) if total_frames else 0.0
    frame_metrics = compute_macro_frame_metrics(all_ref_labels, all_pred_labels)
    teacher_agreement = compute_macro_frame_metrics(all_teacher_labels, all_pred_labels) if all_teacher_labels else None

    quality_accuracy = {}
    quality_stats = {}
    visualization_paths = {}
    paper_quality_metrics = compute_paper_quality_metrics(per_song)

    if all_ref_labels and all_pred_labels:
        info('\n=== Chord Quality Analysis ===')
        info(f'Collected {len(all_ref_labels)} reference and {len(all_pred_labels)} prediction labels')
        min_len = min(len(all_ref_labels), len(all_pred_labels))
        info(f'Using {min_len} chord pairs for quality analysis')

        ref_labels = all_ref_labels[:min_len]
        pred_labels = all_pred_labels[:min_len]

        quality_accuracy, quality_stats = compute_chord_quality_accuracy(ref_labels, pred_labels)

        try:
            dataset_id = _dataset_identifier(args, per_song=per_song)
            viz_dir = _CHORDMINI_ROOT / 'evaluation_visualizations' / dataset_id
            viz_dir.mkdir(parents=True, exist_ok=True)

            dist_path = str(viz_dir / f'{dataset_id}_chord_distribution_accuracy.png')
            cm_path = str(viz_dir / f'{dataset_id}_chord_confusion_matrix.png')

            visualization_paths['distribution_plot'] = generate_chord_distribution_accuracy_plot(
                quality_stats,
                quality_accuracy,
                dist_path,
                title=f'Chord Quality Distribution and Accuracy - {dataset_id}',
            )
            visualization_paths['confusion_matrix'] = generate_confusion_matrix_heatmap(
                ref_labels,
                pred_labels,
                cm_path,
                title=f'Chord Quality Confusion Matrix - {dataset_id}',
            )

            info(f'Generated visualizations saved to {viz_dir}')
        except Exception as ex:
            error(f'Error generating visualizations: {ex}')
            debug(traceback.format_exc())

    info('\n===== Overall MIR Evaluation Results =====')
    for metric in metric_names:
        info(f'{metric} score: {average_scores[metric]:.4f}')
    info(
        f"frame metrics vs. ground truth: "
        f"acc={frame_metrics['accuracy']:.4f}, "
        f"prec={frame_metrics['precision']:.4f}, "
        f"rec={frame_metrics['recall']:.4f}, "
        f"f1={frame_metrics['f1']:.4f}"
    )
    if teacher_agreement is not None:
        info(
            f"frame metrics vs. teacher: "
            f"acc={teacher_agreement['accuracy']:.4f}, "
            f"prec={teacher_agreement['precision']:.4f}, "
            f"rec={teacher_agreement['recall']:.4f}, "
            f"f1={teacher_agreement['f1']:.4f}"
        )
    info('\n===== Aggregate Chord Quality Metrics =====')
    for chord_quality, accuracy in paper_quality_metrics['quality_accuracy'].items():
        total = paper_quality_metrics['quality_stats'][chord_quality]['total']
        correct = paper_quality_metrics['quality_stats'][chord_quality]['correct']
        if total > 0:
            info(f'{chord_quality}: {accuracy * 100:.2f}% ({correct}/{total})')
    info(f"WCSR: {paper_quality_metrics['wcsr']:.4f}")
    info(f"ACQA: {paper_quality_metrics['acqa']:.4f}")

    if quality_accuracy:
        info('\n===== Individual Chord Quality Accuracy =====')
        meaningful = [
            (q, acc) for q, acc in quality_accuracy.items()
            if quality_stats.get(q, {}).get('total', 0) >= 10 or acc > 0
        ]
        for chord_quality, accuracy in sorted(meaningful, key=lambda x: x[1], reverse=True):
            total = quality_stats.get(chord_quality, {}).get('total', 0)
            correct = quality_stats.get(chord_quality, {}).get('correct', 0)
            if total >= 10:
                info(f'{chord_quality}: {accuracy * 100:.2f}% ({correct}/{total})')

    if visualization_paths:
        info('\n===== Generated Visualizations =====')
        for viz_type, path in visualization_paths.items():
            if path:
                info(f'{viz_type}: {path}')

    effective_overlap_ratio = resolve_effective_overlap_ratio(
        model_type=args.model_type,
        use_overlap=args.use_overlap,
        overlap_ratio=args.overlap_ratio,
        chordnet_default_overlap_ratio=CHORDNET_EVAL_OVERLAP_RATIO,
        use_chordnet_defaults=True,
    )

    result = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir if args.data_dir else None,
        'audio_dir': args.audio_dir,
        'label_dir': args.label_dir,
        'songs_evaluated': len(per_song),
        'correct_frames': total_correct,
        'total_frames': total_frames,
        'frame_accuracy': frame_accuracy,
        'frame_metrics': frame_metrics,
        'teacher_agreement': teacher_agreement,
        'average_scores': average_scores,
        'quality_accuracy': quality_accuracy,
        'quality_stats': {
            k: {'total': int(v['total']), 'correct': int(v['correct'])}
            for k, v in quality_stats.items()
            if v['total'] >= 5
        },
        'paper_quality_accuracy': paper_quality_metrics['quality_accuracy'],
        'paper_quality_stats': paper_quality_metrics['quality_stats'],
        'wcsr_by_quality': paper_quality_metrics['wcsr_by_quality'],
        'wcsr': paper_quality_metrics['wcsr'],
        'acqa': paper_quality_metrics['acqa'],
        'visualization_paths': visualization_paths,
        'post_processing': {
            'kernel_size': args.kernel_size,
            'use_gaussian': bool(args.use_gaussian),
            'use_overlap': effective_overlap_ratio > 0.0,
            'overlap_ratio': effective_overlap_ratio,
            'smooth_predictions': bool(args.smooth_predictions),
            'chordnet_default_kernel_size': CHORDNET_EVAL_KERNEL_SIZE,
        },
        'song_details': [
            {
                'song_id': song['song_id'],
                'frames': song['frames'],
                'correct': song['correct'],
                'accuracy': song['accuracy'],
                'duration': song['duration'],
                'scores': song['scores'],
            }
            for song in per_song
        ],
    }

    output_path = args.output_json or DEFAULT_OUTPUT
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as handle:
        json.dump(result, handle, indent=2)

    info('\n===== Evaluation Complete =====')
    info(f'Results saved to {output_path}')


if __name__ == '__main__':
    main()
