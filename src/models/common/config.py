"""
Unified model configuration for BTC and ChordNet architectures.

Provides a single ModelConfig dataclass with sensible defaults that both
model architectures can consume, plus factory functions for common configs.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Unified configuration for both BTC and ChordNet models."""

    # ---- Shared parameters ----
    n_freq: int = 144
    n_classes: int = 170
    seq_len: int = 108
    dropout: float = 0.2

    # ---- Audio / feature extraction ----
    sample_rate: int = 22050
    hop_length: int = 2048
    n_bins: int = 144
    bins_per_octave: int = 24
    frame_duration: float = 0.09288  # hop_length / sample_rate

    # ---- BTC-specific ----
    hidden_size: int = 128
    num_layers: int = 8
    num_heads: int = 4
    total_key_depth: int = 128
    total_value_depth: int = 128
    filter_size: int = 128
    input_dropout: float = 0.2
    layer_dropout: float = 0.2
    attention_dropout: float = 0.2
    relu_dropout: float = 0.2

    # ---- ChordNet-specific ----
    n_group: int = 12
    f_layer: int = 5
    f_head: int = 8
    t_layer: int = 5
    t_head: int = 8
    d_layer: int = 5
    d_head: int = 8

    def to_btc_config(self) -> dict:
        """Return a dict suitable for BTC_model(config=...) via HParams-style access."""
        return {
            'feature_size': self.n_freq,
            'num_chords': self.n_classes,
            'seq_len': self.seq_len,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'total_key_depth': self.total_key_depth,
            'total_value_depth': self.total_value_depth,
            'filter_size': self.filter_size,
            'input_dropout': self.input_dropout,
            'layer_dropout': self.layer_dropout,
            'attention_dropout': self.attention_dropout,
            'relu_dropout': self.relu_dropout,
        }

    def to_chordnet_kwargs(self) -> dict:
        """Return keyword arguments for ChordNet(...)."""
        return {
            'n_freq': self.n_freq,
            'n_classes': self.n_classes,
            'n_group': self.n_group,
            'f_layer': self.f_layer,
            'f_head': self.f_head,
            't_layer': self.t_layer,
            't_head': self.t_head,
            'd_layer': self.d_layer,
            'd_head': self.d_head,
            'dropout': self.dropout,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def get_btc_config() -> ModelConfig:
    """Default BTC model config matching config/btc_config.yaml."""
    return ModelConfig(
        n_freq=144,
        n_classes=170,
        seq_len=108,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        total_key_depth=128,
        total_value_depth=128,
        filter_size=128,
        dropout=0.2,
    )


def get_chordnet_config(scale: float = 1.0) -> ModelConfig:
    """
    Default ChordNet model config matching config/student_config.yaml.

    Args:
        scale: Scaling factor for layer/head counts (0.5 = half, 1.0 = base, 2.0 = double).
    """
    base = ModelConfig(
        n_freq=144,
        n_classes=170,
        seq_len=108,
        n_group=12,
        f_layer=3,
        f_head=2,
        t_layer=4,
        t_head=4,
        d_layer=3,
        d_head=4,
        dropout=0.3,
    )
    if scale != 1.0:
        base.f_layer = max(1, int(base.f_layer * scale))
        base.t_layer = max(1, int(base.t_layer * scale))
        base.d_layer = max(1, int(base.d_layer * scale))
    return base
