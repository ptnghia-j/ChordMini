from __future__ import annotations

import numpy as np
import torch

from src.evaluation.utils.common import majority_filter_indices


def resolve_effective_overlap_ratio(
    model_type,
    use_overlap=None,
    overlap_ratio=None,
    chordnet_default_overlap_ratio=0.5,
    use_chordnet_defaults=False,
):
    if use_overlap is False:
        return 0.0
    if overlap_ratio is not None:
        return min(max(float(overlap_ratio), 0.0), 0.99)
    if use_overlap:
        return float(chordnet_default_overlap_ratio)
    if use_chordnet_defaults and model_type == 'ChordNet':
        return float(chordnet_default_overlap_ratio)
    return 0.0


def predict_sliding_windows(
    model,
    feature_matrix,
    mean,
    std,
    seq_len,
    batch_size,
    model_type,
    n_classes,
    *,
    vote_aggregation='hard',
    use_overlap=None,
    overlap_ratio=None,
    smooth_logits=False,
    smooth_predictions=False,
    kernel_size=9,
    use_gaussian=False,
    chordnet_default_overlap_ratio=0.5,
    use_chordnet_defaults=False,
):
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    if feature_matrix.ndim != 2 or feature_matrix.size == 0:
        return np.array([], dtype=np.int64)

    original_num_frames = int(feature_matrix.shape[0])
    seq_len = max(1, int(seq_len))
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    remainder = original_num_frames % seq_len
    num_pad = 0 if remainder == 0 else seq_len - remainder
    padded_feature_matrix = feature_matrix
    if num_pad > 0:
        padded_feature_matrix = np.pad(
            feature_matrix,
            ((0, num_pad), (0, 0)),
            mode='constant',
        )

    effective_overlap_ratio = resolve_effective_overlap_ratio(
        model_type=model_type,
        use_overlap=use_overlap,
        overlap_ratio=overlap_ratio,
        chordnet_default_overlap_ratio=chordnet_default_overlap_ratio,
        use_chordnet_defaults=use_chordnet_defaults,
    )
    stride = max(1, int(seq_len * (1.0 - effective_overlap_ratio))) if effective_overlap_ratio > 0 else seq_len

    padded_frames = int(padded_feature_matrix.shape[0])
    num_instances = max(1, ((padded_frames - seq_len) // stride) + 1)
    starts = [stride * idx for idx in range(num_instances)]

    vote_accumulator = np.zeros((original_num_frames, n_classes), dtype=np.float32)
    vote_counts = np.zeros(original_num_frames, dtype=np.int32)
    device = next(model.parameters()).device
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
    use_model_smoothing = model_type == 'ChordNet' or smooth_logits
    base_model = getattr(model, 'module', model)

    with torch.no_grad():
        model.eval()

        for batch_start in range(0, len(starts), batch_size):
            batch_windows = []
            batch_meta = []

            for start_frame in starts[batch_start:batch_start + batch_size]:
                end_frame = start_frame + seq_len
                if end_frame > padded_frames:
                    continue
                valid_len = min(seq_len, max(0, original_num_frames - start_frame))
                if valid_len <= 0:
                    continue
                batch_windows.append(padded_feature_matrix[start_frame:end_frame])
                batch_meta.append((start_frame, valid_len))

            if not batch_windows:
                continue

            feature_tensor = torch.from_numpy(np.stack(batch_windows)).float().to(device)
            feature_tensor = (feature_tensor - mean_tensor) / (std_tensor + 1e-8)

            if vote_aggregation in ('logit', 'prob'):
                outputs = model(feature_tensor)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                if use_model_smoothing and hasattr(base_model, '_apply_temporal_smoothing'):
                    logits = base_model._apply_temporal_smoothing(
                        logits,
                        kernel_size,
                        use_gaussian,
                    )
                scores = torch.softmax(logits, dim=-1) if vote_aggregation == 'prob' else logits
                score_array = scores.detach().cpu().numpy()

                for local_idx, (start_frame, valid_len) in enumerate(batch_meta):
                    vote_accumulator[start_frame:start_frame + valid_len] += score_array[local_idx, :valid_len]
                    vote_counts[start_frame:start_frame + valid_len] += 1
                continue

            prediction_tensor = model.predict(
                feature_tensor,
                per_frame=True,
                smooth=use_model_smoothing,
                kernel_size=kernel_size,
                use_gaussian=use_gaussian,
            )
            prediction_array = prediction_tensor.detach().cpu().numpy()

            for local_idx, (start_frame, valid_len) in enumerate(batch_meta):
                for offset in range(valid_len):
                    vote_accumulator[start_frame + offset, int(prediction_array[local_idx, offset])] += 1.0
                    vote_counts[start_frame + offset] += 1

    valid_mask = vote_counts > 0
    final_predictions = np.zeros(original_num_frames, dtype=np.int64)
    if np.any(valid_mask):
        final_predictions[valid_mask] = np.argmax(vote_accumulator[valid_mask], axis=1)

    if smooth_predictions and final_predictions.size >= kernel_size:
        final_predictions = majority_filter_indices(final_predictions, kernel_size)

    return np.asarray(final_predictions, dtype=np.int64)


__all__ = ['predict_sliding_windows', 'resolve_effective_overlap_ratio']
