"""
Train BTC or ChordNet (2E1D) from scratch on labeled audio.

This is the supervised-from-scratch counterpart to ``train_continual_learning.py``.
It uses the same labeled-audio dataset and trainer stack, but initializes a fresh
student model instead of requiring a pseudo-label checkpoint.
"""
import os
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.cli import bootstrap_cli

bootstrap_cli(__file__)

from src.models import (
    create_btc_model,
    create_chordnet_model,
    get_btc_config,
    get_chordnet_config,
    load_model,
)
from src.training_scripts.utils import (
    build_continual_learning_trainer,
    build_labeled_audio_dataset,
    build_labeled_dataloaders,
    build_labeled_split_indices,
    build_optional_teacher,
    checkpoint_dir_for_labeled_training,
    evaluate_test_split,
    apply_optional_pitch_shift_augmentation,
)
from src.utils import HParams, get_config_value, get_device, idx2voca_chord, info, project_path, set_random_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train BTC or ChordNet from scratch on labeled audio")
    p.add_argument('--audio_dir', type=str, default=str(project_path('data', 'labeled', 'audio', start=__file__)))
    p.add_argument('--label_dir', type=str, default=str(project_path('data', 'labeled', 'chordlab', start=__file__)))
    p.add_argument('--config', type=str, default=str(project_path('config', 'ChordMini.yaml', start=__file__)))
    p.add_argument('--model_type', type=str, choices=['BTC', 'ChordNet'], default='BTC')
    p.add_argument('--teacher_checkpoint', type=str, default=None,
                   help='Optional BTC teacher checkpoint for KD during supervised training.')
    p.add_argument('--resume_checkpoint', type=str, default=None,
                   help='Resume a previous supervised run from a trainer checkpoint.')
    # ChordNet overrides
    p.add_argument('--n_group', type=int, default=None)
    p.add_argument('--f_layer', type=int, default=None)
    p.add_argument('--f_head', type=int, default=None)
    p.add_argument('--t_layer', type=int, default=None)
    p.add_argument('--t_head', type=int, default=None)
    p.add_argument('--d_layer', type=int, default=None)
    p.add_argument('--d_head', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    # Training
    p.add_argument('--save_dir', type=str, default=str(project_path('checkpoints', 'from_scratch', start=__file__)))
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--early_stopping_patience', type=int, default=10)
    p.add_argument('--seq_len', type=int, default=108)
    p.add_argument('--stride', type=int, default=54)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_songs', type=int, default=None)
    # KD
    p.add_argument('--kd_alpha', type=float, default=0.3)
    p.add_argument('--temperature', type=float, default=3.0)
    p.add_argument('--no_kd', action='store_true')
    # Loss
    p.add_argument('--use_focal_loss', action='store_true')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--focal_alpha', type=str, default=None)
    p.add_argument('--selective_kd', default=True, action=argparse.BooleanOptionalAction)
    p.add_argument('--kd_confidence_threshold', type=float, default=0.9)
    p.add_argument('--kd_min_confidence_threshold', type=float, default=0.1)
    # Split
    p.add_argument('--use_cv', action='store_true')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--run_all_folds', action='store_true',
                   help='When used with --use_cv, train all folds sequentially and save a CV summary.')
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--val_ratio', type=float, default=0.1)
    # Augmentation
    p.add_argument('--enable_augmentation', action='store_true')
    args = p.parse_args()

    if args.resume_checkpoint and not os.path.exists(args.resume_checkpoint):
        p.error(f"Resume checkpoint not found: {args.resume_checkpoint}")

    return args


def build_model_config(args, config):
    if args.model_type == 'BTC':
        model_config = get_btc_config()
        model_config.seq_len = args.seq_len
        model_config.n_freq = get_config_value(config, 'feature', 'n_bins', model_config.n_freq)
        model_config.n_bins = model_config.n_freq
        model_config.bins_per_octave = get_config_value(config, 'feature', 'bins_per_octave', model_config.bins_per_octave)
        model_config.hop_length = get_config_value(config, 'feature', 'hop_length', model_config.hop_length)
        model_config.sample_rate = get_config_value(config, 'mp3', 'song_hz', model_config.sample_rate)
        model_config.frame_duration = get_config_value(config, 'feature', 'hop_duration', model_config.frame_duration)
        model_config.hidden_size = get_config_value(config, 'model', 'hidden_size', model_config.hidden_size)
        model_config.num_layers = get_config_value(config, 'model', 'num_layers', model_config.num_layers)
        model_config.num_heads = get_config_value(config, 'model', 'num_heads', model_config.num_heads)
        model_config.total_key_depth = get_config_value(config, 'model', 'total_key_depth', model_config.total_key_depth)
        model_config.total_value_depth = get_config_value(config, 'model', 'total_value_depth', model_config.total_value_depth)
        model_config.filter_size = get_config_value(config, 'model', 'filter_size', model_config.filter_size)
        model_config.input_dropout = get_config_value(config, 'model', 'input_dropout', model_config.input_dropout)
        model_config.layer_dropout = get_config_value(config, 'model', 'layer_dropout', model_config.layer_dropout)
        model_config.attention_dropout = get_config_value(config, 'model', 'attention_dropout', model_config.attention_dropout)
        model_config.relu_dropout = get_config_value(config, 'model', 'relu_dropout', model_config.relu_dropout)
    else:
        model_config = get_chordnet_config()
        model_config.seq_len = args.seq_len
        model_config.n_freq = get_config_value(config, 'feature', 'n_bins', model_config.n_freq)
        model_config.n_bins = model_config.n_freq
        model_config.bins_per_octave = get_config_value(config, 'feature', 'bins_per_octave', model_config.bins_per_octave)
        model_config.hop_length = get_config_value(config, 'feature', 'hop_length', model_config.hop_length)
        model_config.sample_rate = get_config_value(config, 'mp3', 'song_hz', model_config.sample_rate)
        model_config.frame_duration = get_config_value(config, 'feature', 'hop_duration', model_config.frame_duration)
        for key in ['n_group', 'f_layer', 'f_head', 't_layer', 't_head', 'd_layer', 'd_head', 'dropout']:
            value = getattr(args, key, None)
            if value is not None:
                setattr(model_config, key, value)
            else:
                setattr(model_config, key, get_config_value(config, 'model', key, getattr(model_config, key)))
    return model_config


def create_model(args, config, device):
    model_config = build_model_config(args, config)
    if args.model_type == 'BTC':
        model = create_btc_model(model_config)
    else:
        model = create_chordnet_model(model_config)
    return model.to(device)

def run_training(args, config, device, chord_to_idx, fold_index=None):
    set_random_seed(args.seed + (0 if fold_index is None else fold_index))

    idx_to_chord = idx2voca_chord()
    current_fold = args.fold if fold_index is None else fold_index

    if args.resume_checkpoint:
        student, s_mean, s_std = load_model(args.resume_checkpoint, args.model_type, config, device, args)
        info(
            f"Loaded {args.model_type} resume checkpoint"
            + (f" for fold {current_fold}" if args.use_cv else "")
        )
    else:
        student = create_model(args, config, device)
        s_mean, s_std = 0.0, 1.0
        info(
            f"Initialized {args.model_type} from scratch"
            + (f" for fold {current_fold}" if fold_index is not None else "")
        )

    teacher, t_mean, t_std = build_optional_teacher(args, config, device)

    dataset = build_labeled_audio_dataset(args, config, chord_to_idx)
    if not dataset:
        info("ERROR: No data")
        return None
    info(f"Dataset: {len(dataset)} segments from {len(dataset.samples)} songs")

    train_idx, val_idx, test_idx = build_labeled_split_indices(args, dataset, current_fold)

    train_idx = apply_optional_pitch_shift_augmentation(args, dataset, train_idx)

    train_loader, val_loader, loader_kwargs = build_labeled_dataloaders(
        args=args,
        dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
        device=device,
    )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    normalization = {
        'mean': torch.tensor(s_mean, device=device),
        'std': torch.tensor(s_std, device=device),
    }

    ckpt_dir = checkpoint_dir_for_labeled_training(args.save_dir, args.use_cv, current_fold)

    trainer = build_continual_learning_trainer(
        args=args,
        model=student,
        optimizer=optimizer,
        teacher=teacher,
        teacher_mean=t_mean,
        teacher_std=t_std,
        device=device,
        checkpoint_dir=ckpt_dir,
        idx_to_chord=idx_to_chord,
        normalization=normalization,
    )

    if args.resume_checkpoint:
        trainer.resume_from_checkpoint(args.resume_checkpoint)

    info("=" * 60)
    info(f"Scratch Training | Model: {args.model_type} | KD: {'OFF' if (args.no_kd or teacher is None) else 'ON'}")
    info(f"  Batch: {args.batch_size} | LR: {args.learning_rate} | Epochs: {args.num_epochs}")
    info(f"  DataLoader workers: {loader_kwargs['num_workers']} | Pin memory: {loader_kwargs['pin_memory']}")
    info("=" * 60)

    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        info("Training interrupted")
        return None

    trainer.load_best_model()
    result = {
        'checkpoint_dir': ckpt_dir,
        'best_model_path': trainer.best_model_path,
        'best_val_acc': float(trainer.best_val_acc) if trainer.best_val_acc != float('-inf') else None,
        'test_accuracy': None,
        'fold': current_fold if args.use_cv else None,
    }
    metrics = evaluate_test_split(trainer, dataset, test_idx, args.batch_size, loader_kwargs)
    if metrics is not None:
        result['test_accuracy'] = float(metrics['accuracy']) if metrics['total'] else None

    info(f"Done. Checkpoints: {ckpt_dir}")
    return result


def _write_cv_summary(args, fold_results):
    valid_results = [r for r in fold_results if r is not None]
    if not valid_results:
        return

    summary = {
        'model_type': args.model_type,
        'n_folds': args.n_folds,
        'seed': args.seed,
        'max_songs': args.max_songs,
        'save_dir': args.save_dir,
        'folds': valid_results,
    }

    val_scores = [r['best_val_acc'] for r in valid_results if r.get('best_val_acc') is not None]
    if val_scores:
        summary['mean_best_val_acc'] = float(np.mean(val_scores))
        summary['std_best_val_acc'] = float(np.std(val_scores))

    out_path = os.path.join(args.save_dir, 'cv_results.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(out_path, 'w') as handle:
        json.dump(summary, handle, indent=2)
    info(f"Saved CV summary to {out_path}")


def main():
    args = parse_args()

    if args.run_all_folds and not args.use_cv:
        raise SystemExit("--run_all_folds requires --use_cv")
    if args.run_all_folds and args.resume_checkpoint:
        raise SystemExit("--resume_checkpoint is not supported together with --run_all_folds")
    if args.use_cv and not (0 <= args.fold < args.n_folds):
        raise SystemExit(f"--fold must be in [0, {args.n_folds - 1}]")

    set_random_seed(args.seed)

    device = get_device()
    info(f"Device: {device}")

    config = HParams.load(args.config)
    idx_to_chord = idx2voca_chord()
    chord_to_idx = {v: k for k, v in idx_to_chord.items()}

    if args.use_cv and args.run_all_folds:
        fold_results = []
        for fold_index in range(args.n_folds):
            info("=" * 60)
            info(f"Starting CV fold {fold_index + 1}/{args.n_folds}")
            info("=" * 60)
            fold_results.append(run_training(args, config, device, chord_to_idx, fold_index=fold_index))
        _write_cv_summary(args, fold_results)
        return

    run_training(args, config, device, chord_to_idx, fold_index=None)


if __name__ == "__main__":
    main()
