from __future__ import annotations

import torch

from src.training.utils.chordnet_eval import (
    accumulate_chordnet_votes,
    chordnet_num_classes,
    finalize_chordnet_votes,
    is_chordnet_model,
    predict_chordnet_frames,
)


def to_stat_tensor(value, device):
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def normalize_features(features, mean=None, std=None):
    if mean is None or std is None:
        return features
    return (features - mean) / (std + 1e-8)


def teacher_logits_from_model(teacher_model, raw_features, teacher_mean=None, teacher_std=None, normalization=None):
    if teacher_model is None:
        return None

    if (teacher_mean is None or teacher_std is None) and normalization is not None:
        teacher_mean = normalization.get('mean')
        teacher_std = normalization.get('std')

    with torch.no_grad():
        teacher_features = normalize_features(raw_features, teacher_mean, teacher_std)
        teacher_outputs = teacher_model(teacher_features)
        teacher_logits = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
    return teacher_logits.detach()


def use_chordnet_eval_defaults(model, batch):
    return (
        is_chordnet_model(model)
        and 'song_id' in batch
        and 'start_frame' in batch
        and 'end_frame' in batch
    )


def accumulate_batch_accuracy(model, batch, logits, targets, vote_accumulator=None, target_accumulator=None):
    if use_chordnet_eval_defaults(model, batch):
        if vote_accumulator is None:
            vote_accumulator = {}
        if target_accumulator is None:
            target_accumulator = {}
        predictions = predict_chordnet_frames(model, logits=logits)
        accumulate_chordnet_votes(
            vote_accumulator,
            target_accumulator,
            predictions,
            targets,
            batch['song_id'],
            batch['start_frame'],
            batch['end_frame'],
            chordnet_num_classes(model),
        )
        return 0, 0, True

    flat_logits = logits.reshape(-1, logits.size(-1)) if logits.dim() == 3 else logits
    flat_targets = targets.reshape(-1) if targets.dim() == 2 else targets
    predictions = flat_logits.argmax(dim=-1)
    return (predictions == flat_targets).sum().item(), flat_targets.numel(), False


def finalize_vote_accuracy(vote_accumulator, target_accumulator):
    if not vote_accumulator:
        return 0, 0
    return finalize_chordnet_votes(vote_accumulator, target_accumulator)


__all__ = [
    'accumulate_batch_accuracy',
    'finalize_vote_accuracy',
    'normalize_features',
    'teacher_logits_from_model',
    'to_stat_tensor',
    'use_chordnet_eval_defaults',
]
