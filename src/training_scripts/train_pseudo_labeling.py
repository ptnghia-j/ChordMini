"""
Phase 1: pseudo-labeling training.

Supports two input modes:
1. Pre-extracted spectrograms/labels (+ optional offline logits).
2. Online unlabeled audio mode, where CQT features, pseudo-labels, and
   optional KD logits are all generated at runtime from a frozen teacher.

ChordNet (2E1D) validation/test accuracy automatically uses the same default inference
setup as ``test_labeled_audio.py`` (Gaussian smoothing + overlap-aware voting)
when segment metadata is available.

Phase 1 supports two LR scheduling modes:
  - ``validation``: reduce LR when validation accuracy plateaus
  - ``cosine``: cosine annealing with an optional single warmup phase
"""
import os
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.cli import bootstrap_cli

bootstrap_cli(__file__)

from src.data import SynthDataset, UnlabeledAudioDataset
from src.models import (
    create_btc_model,
    create_chordnet_model,
    get_btc_config,
    get_chordnet_config,
)
from src.training import PseudoLabelingTrainer
from src.utils import (
    extract_state_dict_and_stats,
    get_device,
    idx2voca_chord,
    info,
    project_path,
    set_random_seed,
)
from src.utils.dataloader import build_dataloader_kwargs


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Pseudo-Labeling Training")
    p.add_argument('--audio_dir', type=str, default=None,
                   help='Root directory of unlabeled audio for online pseudo-labeling.')
    p.add_argument('--spec_dir', type=str, default=None)
    p.add_argument('--label_dir', type=str, default=None)
    p.add_argument('--logits_dir', type=str, default=None)
    p.add_argument('--model_type', type=str, choices=['BTC', 'ChordNet'], default='BTC')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--resume_checkpoint', type=str, default=None,
                   help='Resume a previous PL run from a trainer checkpoint.')
    p.add_argument('--teacher_checkpoint', type=str, default=None,
                   help='Frozen BTC teacher checkpoint for online pseudo-labeling/KD.')
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
    p.add_argument('--save_dir', type=str, default=str(project_path('checkpoints', 'pseudo_labeling', start=__file__)))
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--early_stopping_patience', type=int, default=10)
    p.add_argument('--lr_schedule', type=str,
                   choices=['cosine', 'validation', 'none'], default='cosine',
                   help='Learning rate schedule type (default: cosine).')
    p.add_argument('--lr_decay_factor', type=float, default=0.95,
                   help='LR decay factor for validation-based scheduling.')
    p.add_argument('--min_learning_rate', type=float, default=1e-6,
                   help='Minimum learning rate for scheduler decay.')
    p.add_argument('--use_warmup', default=True, action=argparse.BooleanOptionalAction,
                   help='Use single warmup before the main LR schedule.')
    p.add_argument('--warmup_epochs', type=int, default=10,
                   help='Number of warmup epochs (default: 10 when warmup is enabled).')
    p.add_argument('--warmup_start_lr', type=float, default=1e-4,
                   help='Warmup start LR (paper default: 1e-4).')
    p.add_argument('--warmup_end_lr', type=float, default=3e-4,
                   help='Warmup end LR / cosine peak LR (paper default: 3e-4).')
    p.add_argument('--seq_len', type=int, default=108)
    p.add_argument('--stride', type=int, default=108)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_files', type=int, default=None)
    # KD
    p.add_argument('--use_kd', action='store_true')
    p.add_argument('--kd_alpha', type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=3.0)
    # Loss
    p.add_argument('--use_focal_loss', action='store_true')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    # Split
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--val_ratio', type=float, default=0.1)
    args = p.parse_args()

    if not args.audio_dir and not (args.spec_dir and args.label_dir):
        p.error("Provide either --audio_dir or both --spec_dir and --label_dir.")

    if args.audio_dir and not (args.teacher_checkpoint or args.checkpoint):
        p.error("Online audio mode requires --teacher_checkpoint or --checkpoint.")

    if args.resume_checkpoint and not os.path.exists(args.resume_checkpoint):
        p.error(f"Resume checkpoint not found: {args.resume_checkpoint}")

    return args


def load_model(args, model_config, device, checkpoint_path=None):
    if args.model_type == 'BTC':
        model = create_btc_model(model_config)
    else:
        model = create_chordnet_model(model_config)
    model = model.to(device)

    mean, std = 0.0, 1.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd, mean, std = extract_state_dict_and_stats(ckpt)
        model.load_state_dict(sd, strict=False)
        info(f"Loaded {args.model_type} from {checkpoint_path}")

    return model, float(mean), float(std)


def load_teacher_model(checkpoint_path, seq_len, device):
    info(f"Loading teacher checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    teacher_config = get_btc_config()
    teacher_config.seq_len = seq_len
    teacher = create_btc_model(teacher_config).to(device)
    state_dict, mean, std = extract_state_dict_and_stats(ckpt)
    teacher.load_state_dict(state_dict, strict=False)
    teacher.eval()
    return teacher, mean, std


def build_config(args):
    cfg = get_btc_config() if args.model_type == 'BTC' else get_chordnet_config()
    cfg.seq_len = args.seq_len
    for key in ['n_group', 'f_layer', 'f_head', 't_layer', 't_head', 'd_layer', 'd_head', 'dropout']:
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)
    return cfg


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = get_device()
    info(f"Device: {device}")

    idx_to_chord = idx2voca_chord()
    chord_to_idx = {v: k for k, v in idx_to_chord.items()}

    model_config = build_config(args)
    model_checkpoint = args.resume_checkpoint or args.checkpoint
    model, mean, std = load_model(args, model_config, device, checkpoint_path=model_checkpoint)
    online_mode = args.audio_dir is not None
    teacher_model = None
    teacher_mean = None
    teacher_std = None

    info("Loading dataset...")
    if online_mode:
        teacher_checkpoint = args.teacher_checkpoint or args.checkpoint
        teacher_model, teacher_mean, teacher_std = load_teacher_model(
            teacher_checkpoint, args.seq_len, device)
        dataset = UnlabeledAudioDataset(
            audio_dir=args.audio_dir,
            config=model_config,
            seq_len=args.seq_len,
            stride=args.stride,
            verbose=True,
            max_files=args.max_files,
            random_seed=args.seed,
        )
        mean, std = teacher_mean, teacher_std
    else:
        dataset = SynthDataset(
            spec_dir=args.spec_dir, label_dir=args.label_dir,
            chord_mapping=chord_to_idx, seq_len=args.seq_len, stride=args.stride,
            logits_dir=args.logits_dir,
            require_teacher_logits=args.use_kd and args.logits_dir is not None,
            verbose=True, max_files=args.max_files)

    if len(dataset) == 0:
        info("ERROR: No data loaded.")
        return

    if mean == 0.0 and std == 1.0:
        info("Estimating normalization from data...")
        mean, std = dataset.get_normalization_params()

    train_idx, val_idx, test_idx = dataset.split_indices(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    if not train_idx:
        info("ERROR: Training split is empty.")
        return

    loader_kwargs = build_dataloader_kwargs(
        device=device,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader = None
    if val_idx:
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                                shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    normalization = {'mean': torch.tensor(mean, device=device),
                     'std': torch.tensor(std, device=device)}

    trainer = PseudoLabelingTrainer(
        model=model, optimizer=optimizer, teacher_model=teacher_model,
        teacher_mean=teacher_mean, teacher_std=teacher_std, device=device,
        num_epochs=args.num_epochs, checkpoint_dir=args.save_dir,
        normalization=normalization, idx_to_chord=idx_to_chord,
        use_kd_loss=args.use_kd, kd_alpha=args.kd_alpha, temperature=args.temperature,
        early_stopping_patience=args.early_stopping_patience,
        use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
        lr_schedule_type=args.lr_schedule,
        use_warmup=args.use_warmup,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        warmup_end_lr=args.warmup_end_lr,
        lr_decay_factor=args.lr_decay_factor,
        min_lr=args.min_learning_rate)

    if args.resume_checkpoint:
        trainer.resume_from_checkpoint(args.resume_checkpoint)

    info("=" * 60)
    mode_name = 'online-audio' if online_mode else 'pre-extracted'
    info(f"PL Training | Model: {args.model_type} | Mode: {mode_name} | KD: {'ON' if args.use_kd else 'OFF'}")
    info(f"  Songs: {len(dataset.songs)} | Segments: {len(dataset)} | Batch: {args.batch_size}")
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
        info(f"Test Accuracy: {metrics['accuracy']:.4f}" if metrics['total'] > 0 else "No test data")

    info(f"Done. Best model: {trainer.best_model_path}")


if __name__ == "__main__":
    main()
