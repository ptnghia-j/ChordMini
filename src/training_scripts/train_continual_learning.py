#!/usr/bin/env python3
"""
Phase 2: Continual Learning (CL) training script.

Fine-tunes a student model (from Phase 1 PL output) on a small labeled
dataset with real chord annotations, using online knowledge distillation
from a frozen teacher model.

ChordNet validation/test accuracy automatically uses the same default inference
setup as ``test_labeled_audio.py`` (Gaussian smoothing + overlap-aware voting)
when segment metadata is available.
"""
import os
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.cli import bootstrap_cli

_PROJECT_ROOT = bootstrap_cli(__file__)

from src.data import AudioChordDataset, create_cv_folds, create_train_val_test_split
from src.models import load_model
from src.training import ContinualLearningTrainer
from src.utils import (
    HParams,
    get_device,
    idx2voca_chord,
    info,
    project_path,
)
from src.utils.dataloader import build_dataloader_kwargs


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Continual Learning Training")
    p.add_argument('--audio_dir', type=str, default=str(project_path('data', 'labeled', 'audio', start=__file__)))
    p.add_argument('--label_dir', type=str, default=str(project_path('data', 'labeled', 'chordlab', start=__file__)))
    p.add_argument('--config', type=str, default=str(project_path('config', 'ChordMini.yaml', start=__file__)))
    p.add_argument('--model_type', type=str, choices=['BTC', 'ChordNet'], default='BTC')
    p.add_argument('--student_checkpoint', type=str, required=True)
    p.add_argument('--teacher_checkpoint', type=str,
                   default=str(project_path('checkpoints', 'btc_model_large_voca.pt', start=__file__)))
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
    p.add_argument('--save_dir', type=str, default=str(project_path('checkpoints', 'continual_learning', start=__file__)))
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_epochs', type=int, default=50)
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--early_stopping_patience', type=int, default=10)
    p.add_argument('--seq_len', type=int, default=108)
    p.add_argument('--stride', type=int, default=54)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--prefetch_factor', type=int, default=2,
                   help='Batches prefetched per worker when num_workers > 0.')
    p.add_argument('--disable_persistent_workers', action='store_true',
                   help='Disable persistent DataLoader workers when num_workers > 0.')
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
    # Anti-forgetting
    p.add_argument('--freeze_encoder', action='store_true')
    p.add_argument('--ewc_lambda', type=float, default=0.0)
    p.add_argument('--use_pod_loss', action='store_true')
    p.add_argument('--pod_alpha', type=float, default=0.1)
    # Selective KD
    p.add_argument('--selective_kd', default=True, action=argparse.BooleanOptionalAction)
    p.add_argument('--kd_confidence_threshold', type=float, default=0.9)
    p.add_argument('--kd_min_confidence_threshold', type=float, default=0.1)
    # Split
    p.add_argument('--use_cv', action='store_true')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--val_ratio', type=float, default=0.1)
    # Augmentation
    p.add_argument('--enable_augmentation', action='store_true')
    p.add_argument('--augmentation_min_semitones', type=int, default=-5)
    p.add_argument('--augmentation_max_semitones', type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    info(f"Device: {device}")

    config = HParams.load(args.config)
    idx_to_chord = idx2voca_chord()
    chord_to_idx = {v: k for k, v in idx_to_chord.items()}

    # Student
    student, s_mean, s_std = load_model(
        args.student_checkpoint, args.model_type, config, device, args)

    original_params = None
    if args.ewc_lambda > 0:
        original_params = {n: p.clone().detach() for n, p in student.named_parameters()}

    if args.freeze_encoder:
        frozen, trainable = 0, 0
        for name, param in student.named_parameters():
            if 'fc' in name or 'component' in name:
                param.requires_grad = True; trainable += 1
            else:
                param.requires_grad = False; frozen += 1
        info(f"Frozen {frozen}, trainable {trainable}")

    # Teacher
    teacher, t_mean, t_std = None, 0.0, 1.0
    if not args.no_kd and os.path.exists(args.teacher_checkpoint):
        teacher, t_mean, t_std = load_model(
            args.teacher_checkpoint, 'BTC', config, device)
        teacher.eval()

    # Dataset
    dataset = AudioChordDataset(
        audio_dir=args.audio_dir, label_dir=args.label_dir, config=config,
        seq_len=args.seq_len, stride=args.stride, chord_mapping=chord_to_idx,
        device='cpu', verbose=True, max_songs=args.max_songs, random_seed=args.seed)
    if not dataset:
        info("ERROR: No data"); return
    info(f"Dataset: {len(dataset)} segments from {len(dataset.samples)} songs")

    if args.use_cv:
        folds = create_cv_folds(dataset, n_folds=args.n_folds, seed=args.seed)
        train_idx, val_idx = folds[args.fold]; test_idx = []
    else:
        train_idx, val_idx, test_idx = create_train_val_test_split(
            dataset, args.train_ratio, args.val_ratio, args.seed)

    if args.enable_augmentation:
        sem = list(range(args.augmentation_min_semitones, args.augmentation_max_semitones + 1))
        train_idx = dataset.add_augmented_segments_for_indices(train_idx, sem)

    loader_kwargs = build_dataloader_kwargs(
        device=device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.disable_persistent_workers,
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, **loader_kwargs)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    normalization = {'mean': torch.tensor(s_mean, device=device),
                     'std': torch.tensor(s_std, device=device)}

    ckpt_dir = os.path.join(args.save_dir, f"fold_{args.fold}" if args.use_cv else "single_split")

    focal_alpha = args.focal_alpha
    if focal_alpha is not None and focal_alpha != 'auto':
        try: focal_alpha = float(focal_alpha)
        except ValueError: focal_alpha = None

    trainer = ContinualLearningTrainer(
        model=student, optimizer=optimizer,
        teacher_model=teacher, teacher_mean=t_mean, teacher_std=t_std,
        kd_alpha=args.kd_alpha, temperature=args.temperature,
        device=device, num_epochs=args.num_epochs, checkpoint_dir=ckpt_dir,
        idx_to_chord=idx_to_chord, normalization=normalization,
        early_stopping_patience=args.early_stopping_patience,
        use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha, lr_decay_factor=0.9, min_lr=1e-6,
        selective_kd=args.selective_kd,
        kd_confidence_threshold=args.kd_confidence_threshold,
        kd_min_confidence_threshold=args.kd_min_confidence_threshold,
        ewc_lambda=args.ewc_lambda, original_params=original_params,
        use_pod_loss=args.use_pod_loss, pod_alpha=args.pod_alpha)

    info("=" * 60)
    info(f"CL Training | Model: {args.model_type} | KD: {'OFF' if args.no_kd else 'ON'}")
    info(f"  Batch: {args.batch_size} | LR: {args.learning_rate} | Epochs: {args.num_epochs}")
    info(f"  DataLoader workers: {loader_kwargs['num_workers']} | Pin memory: {loader_kwargs['pin_memory']}")
    info("=" * 60)

    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        info("Training interrupted")

    trainer.load_best_model()
    if test_idx:
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                                  shuffle=False, **loader_kwargs)
        metrics = trainer.evaluate_loader(test_loader)
        info(f"Test Accuracy: {metrics['accuracy']:.4f}" if metrics['total'] else "No test data")

    info(f"Done. Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
