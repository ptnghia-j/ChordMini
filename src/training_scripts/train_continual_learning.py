"""
Phase 2: Continual Learning (CL) training script.

Fine-tunes a student model (from Phase 1 PL output) on a small labeled
dataset with real chord annotations, using online knowledge distillation
from a frozen teacher model.

ChordNet (2E1D) validation/test accuracy automatically uses the same default inference
setup as ``test_labeled_audio.py`` (Gaussian smoothing + overlap-aware voting)
when segment metadata is available.
"""
import os
import argparse
import sys
from pathlib import Path

import torch

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.cli import bootstrap_cli

bootstrap_cli(__file__)

from src.models import load_model
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
from src.utils import (
    HParams,
    get_device,
    idx2voca_chord,
    info,
    project_path,
    set_random_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Continual Learning Training")
    p.add_argument('--audio_dir', type=str, default=str(project_path('data', 'labeled', 'audio', start=__file__)))
    p.add_argument('--label_dir', type=str, default=str(project_path('data', 'labeled', 'chordlab', start=__file__)))
    p.add_argument('--config', type=str, default=str(project_path('config', 'ChordMini.yaml', start=__file__)))
    p.add_argument('--model_type', type=str, choices=['BTC', 'ChordNet'], default='BTC')
    p.add_argument('--student_checkpoint', type=str, default=None)
    p.add_argument('--resume_checkpoint', type=str, default=None,
                   help='Resume a previous CL run from a trainer checkpoint.')
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
    args = p.parse_args()

    if not args.student_checkpoint and not args.resume_checkpoint:
        p.error("Provide --student_checkpoint or --resume_checkpoint.")

    if args.resume_checkpoint and not os.path.exists(args.resume_checkpoint):
        p.error(f"Resume checkpoint not found: {args.resume_checkpoint}")

    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = get_device()
    info(f"Device: {device}")

    config = HParams.load(args.config)
    idx_to_chord = idx2voca_chord()
    chord_to_idx = {v: k for k, v in idx_to_chord.items()}

    # Student
    student_checkpoint = args.resume_checkpoint or args.student_checkpoint
    student, s_mean, s_std = load_model(
        student_checkpoint, args.model_type, config, device, args)

    # Teacher
    teacher, t_mean, t_std = build_optional_teacher(args, config, device)

    # Dataset
    dataset = build_labeled_audio_dataset(args, config, chord_to_idx)
    if not dataset:
        info("ERROR: No data"); return
    info(f"Dataset: {len(dataset)} segments from {len(dataset.samples)} songs")

    current_fold = args.fold
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
    normalization = {'mean': torch.tensor(s_mean, device=device),
                     'std': torch.tensor(s_std, device=device)}

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
    info(f"CL Training | Model: {args.model_type} | KD: {'OFF' if args.no_kd else 'ON'}")
    info(f"  Batch: {args.batch_size} | LR: {args.learning_rate} | Epochs: {args.num_epochs}")
    info(f"  DataLoader workers: {loader_kwargs['num_workers']} | Pin memory: {loader_kwargs['pin_memory']}")
    info("=" * 60)

    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        info("Training interrupted")

    trainer.load_best_model()
    evaluate_test_split(trainer, dataset, test_idx, args.batch_size, loader_kwargs)

    info(f"Done. Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
