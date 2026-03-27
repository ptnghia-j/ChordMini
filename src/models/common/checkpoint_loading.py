from __future__ import annotations

import torch

from ..btc_model import BTC_model
from ..chord_net import ChordNet
from src.utils import extract_model_state_dict, extract_normalization_stats, info


_CHORDNET_OVERRIDE_KEYS = (
    'n_group',
    'f_layer',
    'f_head',
    't_layer',
    't_head',
    'd_layer',
    'd_head',
    'dropout',
)


def _collect_index_count(state_dict, marker):
    indices = set()
    for key in state_dict:
        parts = key.split('.')
        for idx, part in enumerate(parts[:-1]):
            if part == marker and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                indices.add(int(parts[idx + 1]))
                break
    return max(indices) + 1 if indices else None


def _infer_head_count(state_dict, prefix, key_fragment):
    for key, value in state_dict.items():
        if prefix in key and key_fragment in key:
            embed_dim = value.shape[1]
            for head_count in (8, 6, 4, 2, 1):
                if embed_dim % head_count == 0:
                    return head_count
    return None


def _infer_chordnet(state_dict, config):
    feature_cfg = getattr(config, 'feature', {})
    model_cfg = getattr(config, 'model', {})

    params = {
        'n_freq': feature_cfg.get('n_bins', model_cfg.get('feature_size', 144)),
        'n_classes': model_cfg.get('num_chords', 170),
        'n_group': 2,
        'f_layer': 3,
        'f_head': 2,
        't_layer': 4,
        't_head': 4,
        'd_layer': 3,
        'd_head': 4,
        'dropout': 0.3,
    }

    if 'fc.weight' in state_dict:
        params['n_freq'] = state_dict['fc.weight'].shape[1]
        params['n_classes'] = state_dict['fc.weight'].shape[0]

    for key, value in state_dict.items():
        if 'encoder_f.0.attn_layer.0.out_proj.weight' in key:
            feature_dim = value.shape[0]
            if feature_dim > 0 and params['n_freq'] % feature_dim == 0:
                params['n_group'] = params['n_freq'] // feature_dim
            break

    params['f_layer'] = _collect_index_count(
        {k: v for k, v in state_dict.items() if 'transformer.encoder_f.' in k},
        'attn_layer',
    ) or params['f_layer']
    params['t_layer'] = _collect_index_count(
        {k: v for k, v in state_dict.items() if 'transformer.encoder_t.' in k},
        'attn_layer',
    ) or params['t_layer']
    params['d_layer'] = _collect_index_count(
        {k: v for k, v in state_dict.items() if 'transformer.decoder.' in k},
        'attn_layer1',
    ) or params['d_layer']

    params['f_head'] = (
        _infer_head_count(state_dict, 'transformer.encoder_f.', 'attn_layer.0.in_proj_weight')
        or params['f_head']
    )
    params['t_head'] = (
        _infer_head_count(state_dict, 'transformer.encoder_t.', 'attn_layer.0.in_proj_weight')
        or params['t_head']
    )
    params['d_head'] = (
        _infer_head_count(state_dict, 'transformer.decoder.', 'attn_layer.0.in_proj_weight')
        or params['d_head']
    )

    return params


def load_model(path, model_type, config, device, args=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)
    mean, std = extract_normalization_stats(checkpoint)

    if model_type == 'BTC':
        model = BTC_model(config=config.model).to(device)
    else:
        architecture = _infer_chordnet(state_dict, config)
        checkpoint_config = checkpoint.get('config', {})
        if isinstance(checkpoint_config, dict):
            for key in architecture:
                if key in checkpoint_config:
                    architecture[key] = checkpoint_config[key]
        if args is not None:
            for key in _CHORDNET_OVERRIDE_KEYS:
                value = getattr(args, key, None)
                if value is not None:
                    architecture[key] = value
        model = ChordNet(**architecture).to(device)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        info(f"Strict checkpoint load failed; retrying with strict=False: {exc}")
        model.load_state_dict(state_dict, strict=False)

    info(f"Loaded {model_type} from {path}")
    return model, mean, std


__all__ = ['load_model']
