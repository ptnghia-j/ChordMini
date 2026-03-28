from __future__ import annotations

import os

from torch.utils.data import DataLoader, Subset

from src.data import AudioChordDataset, create_cv_folds, create_train_val_test_split
from src.models import load_model
from src.training import ContinualLearningTrainer
from src.utils import info
from src.utils.dataloader import build_dataloader_kwargs


def build_optional_teacher(args, config, device):
    if args.no_kd or not args.teacher_checkpoint or not os.path.exists(args.teacher_checkpoint):
        return None, 0.0, 1.0

    teacher, teacher_mean, teacher_std = load_model(args.teacher_checkpoint, 'BTC', config, device)
    teacher.eval()
    return teacher, teacher_mean, teacher_std


def build_labeled_audio_dataset(args, config, chord_to_idx):
    return AudioChordDataset(
        audio_dir=args.audio_dir,
        label_dir=args.label_dir,
        config=config,
        seq_len=args.seq_len,
        stride=args.stride,
        chord_mapping=chord_to_idx,
        device='cpu',
        verbose=True,
        max_songs=args.max_songs,
        random_seed=args.seed,
    )


def build_labeled_split_indices(args, dataset, current_fold):
    if args.use_cv:
        folds = create_cv_folds(dataset, n_folds=args.n_folds, seed=args.seed)
        train_idx, val_idx = folds[current_fold]
        return train_idx, val_idx, []

    return create_train_val_test_split(
        dataset,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )


def apply_optional_pitch_shift_augmentation(args, dataset, train_idx):
    if not args.enable_augmentation:
        return train_idx
    return dataset.add_augmented_segments_for_indices(train_idx)


def build_labeled_dataloaders(args, dataset, train_idx, val_idx, batch_size, device):
    loader_kwargs = build_dataloader_kwargs(
        device=device,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, loader_kwargs


def checkpoint_dir_for_labeled_training(save_dir, use_cv, current_fold):
    return os.path.join(save_dir, f"fold_{current_fold}" if use_cv else "single_split")


def parse_optional_focal_alpha(value):
    if value in (None, 'auto'):
        return value
    try:
        return float(value)
    except ValueError:
        return None


def build_continual_learning_trainer(
    args,
    model,
    optimizer,
    teacher,
    teacher_mean,
    teacher_std,
    device,
    checkpoint_dir,
    idx_to_chord,
    normalization,
):
    return ContinualLearningTrainer(
        model=model,
        optimizer=optimizer,
        teacher_model=teacher,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        kd_alpha=args.kd_alpha,
        temperature=args.temperature,
        device=device,
        num_epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir,
        idx_to_chord=idx_to_chord,
        normalization=normalization,
        early_stopping_patience=args.early_stopping_patience,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=parse_optional_focal_alpha(args.focal_alpha),
        lr_decay_factor=0.9,
        min_lr=1e-6,
        selective_kd=args.selective_kd,
        kd_confidence_threshold=args.kd_confidence_threshold,
        kd_min_confidence_threshold=args.kd_min_confidence_threshold,
    )


def evaluate_test_split(trainer, dataset, test_idx, batch_size, loader_kwargs):
    if not test_idx:
        return None

    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, **loader_kwargs)
    metrics = trainer.evaluate_loader(test_loader)
    info(f"Test Accuracy: {metrics['accuracy']:.4f}" if metrics['total'] else "No test data")
    return metrics
