"""
ChordNet-specific validation/test helpers.

This mirrors the default ChordNet inference settings used in
``test_labeled_audio.py``:
  - Gaussian temporal smoothing
  - Overlap-aware frame voting when segment metadata is available
  - ``kernel_size=9`` and ``overlap_ratio=0.5`` semantics

Median filtering remains an optional post-process in the standalone evaluation
script and is not enabled by default inside the training-time validation loops.
"""
from __future__ import annotations

import numpy as np
import torch


CHORDNET_EVAL_KERNEL_SIZE = 9
CHORDNET_EVAL_USE_GAUSSIAN = True
CHORDNET_EVAL_USE_OVERLAP = True
CHORDNET_EVAL_OVERLAP_RATIO = 0.5


def is_chordnet_model(model) -> bool:
    """Return True when the wrapped model is a ChordNet instance."""
    base_model = getattr(model, 'module', model)
    return base_model.__class__.__name__ == 'ChordNet'


def chordnet_num_classes(model, default=170) -> int:
    """Return the ChordNet output class count, including wrapped modules."""
    base_model = getattr(model, 'module', model)
    return int(getattr(base_model, 'n_classes', default))


def smooth_chordnet_logits(model, logits):
    """Apply the default temporal smoothing used by ChordNet eval helpers."""
    base_model = getattr(model, 'module', model)
    if hasattr(base_model, '_apply_temporal_smoothing') and logits.dim() >= 2:
        return base_model._apply_temporal_smoothing(
            logits,
            CHORDNET_EVAL_KERNEL_SIZE,
            CHORDNET_EVAL_USE_GAUSSIAN,
        )
    return logits


def predict_chordnet_frames(model, features=None, logits=None):
    """Decode per-frame ChordNet predictions from features or precomputed logits."""
    if logits is None:
        if features is None:
            raise ValueError("predict_chordnet_frames requires features or logits")
        with torch.no_grad():
            outputs = model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

    logits = smooth_chordnet_logits(model, logits)
    return logits.argmax(dim=-1)


def accumulate_chordnet_votes(vote_accumulator, target_accumulator, predictions, targets,
                              song_ids, start_frames, end_frames, n_classes):
    """Accumulate per-frame class votes across overlapping validation/test windows."""
    if hasattr(start_frames, 'detach'):
        start_frames = start_frames.detach().cpu().tolist()
    else:
        start_frames = list(start_frames)

    if hasattr(end_frames, 'detach'):
        end_frames = end_frames.detach().cpu().tolist()
    else:
        end_frames = list(end_frames)

    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()

    for sample_idx, song_id in enumerate(song_ids):
        song_id = str(song_id)
        start_frame = int(start_frames[sample_idx])
        end_frame = int(end_frames[sample_idx])
        pred_seq = np.asarray(pred_np[sample_idx]).reshape(-1)
        target_seq = np.asarray(target_np[sample_idx]).reshape(-1)
        valid_len = max(0, min(end_frame - start_frame, pred_seq.shape[0], target_seq.shape[0]))

        for offset in range(valid_len):
            key = (song_id, start_frame + offset)
            if key not in vote_accumulator:
                vote_accumulator[key] = np.zeros(n_classes, dtype=np.float32)
            vote_accumulator[key][int(pred_seq[offset])] += 1.0
            target_accumulator[key] = int(target_seq[offset])


def finalize_chordnet_votes(vote_accumulator, target_accumulator):
    """Convert accumulated votes into final correct/total frame counts."""
    correct = 0
    total = 0
    for key, class_votes in vote_accumulator.items():
        if key not in target_accumulator:
            continue
        prediction = int(np.argmax(class_votes))
        target = int(target_accumulator[key])
        correct += int(prediction == target)
        total += 1
    return correct, total
