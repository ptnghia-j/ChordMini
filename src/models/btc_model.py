"""
BTC (Bi-directional Transformer for Chords) model.

Standalone implementation for the ChordMini pipeline.
"""
import torch
import torch.nn as nn

from src.models.common.transformer_modules import (
    _gen_timing_signal, _gen_bias_mask,
    LayerNorm, MultiHeadAttention, PositionwiseFeedForward, SoftmaxOutputLayer,
)
from src.models.common.config import ModelConfig
from src.models.common.temporal_smoothing import apply_temporal_smoothing


class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size,
                 num_heads, bias_mask=None, layer_dropout=0.0, attention_dropout=0.0,
                 relu_dropout=0.0, attention_map=False):
        super().__init__()
        self.attention_map = attention_map
        self.multi_head_attention = MultiHeadAttention(
            hidden_size, total_key_depth, total_value_depth, hidden_size,
            num_heads, bias_mask, attention_dropout, attention_map)
        self.positionwise_convolution = PositionwiseFeedForward(
            hidden_size, filter_size, hidden_size, layer_config='cc',
            padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs
        x_norm = self.layer_norm_mha(x)
        if self.attention_map:
            y, weights = self.multi_head_attention(x_norm, x_norm, x_norm)
        else:
            y = self.multi_head_attention(x_norm, x_norm, x_norm)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_convolution(x_norm)
        y = self.dropout(x + y)
        if self.attention_map:
            return y, weights
        return y


class _BiDirSelfAttention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size,
                 num_heads, max_length, layer_dropout=0.0, attention_dropout=0.0,
                 relu_dropout=0.0):
        super().__init__()
        fwd_params = (hidden_size, total_key_depth or hidden_size,
                      total_value_depth or hidden_size, filter_size, num_heads,
                      _gen_bias_mask(max_length), layer_dropout, attention_dropout,
                      relu_dropout, True)
        self.attn_block = _SelfAttentionBlock(*fwd_params)
        bwd_params = (hidden_size, total_key_depth or hidden_size,
                      total_value_depth or hidden_size, filter_size, num_heads,
                      torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3),
                      layer_dropout, attention_dropout, relu_dropout, True)
        self.backward_attn_block = _SelfAttentionBlock(*bwd_params)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, inputs):
        x, attn_list = inputs
        encoder_outputs, weights = self.attn_block(x)
        reverse_outputs, reverse_weights = self.backward_attn_block(x)
        outputs = torch.cat((encoder_outputs, reverse_outputs), dim=2)
        y = self.linear(outputs)
        attn_list = list(attn_list)
        attn_list.append(weights)
        attn_list.append(reverse_weights)
        return y, attn_list


class _BiDirSelfAttentionLayers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads,
                 total_key_depth, total_value_depth, filter_size, max_length=100,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0,
                 relu_dropout=0.0):
        super().__init__()
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.max_length = max_length
        params = (hidden_size, total_key_depth or hidden_size,
                  total_value_depth or hidden_size, filter_size, num_heads,
                  max_length, layer_dropout, attention_dropout, relu_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(
            *[_BiDirSelfAttention(*params) for _ in range(num_layers)]
        )
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        if x.shape[1] != self.timing_signal.shape[1]:
            x = x + self.timing_signal[:, :x.shape[1], :].type_as(x)
        else:
            x = x + self.timing_signal.type_as(x)
        y, weights_list = self.self_attn_layers((x, []))
        y = self.layer_norm(y)
        return y, weights_list


class BTC_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, ModelConfig):
            config = config.to_btc_config()

        self.timestep = config.get('seq_len', 108)
        params = (config.get('feature_size', 144),
                  config.get('hidden_size', 128),
                  config.get('num_layers', 8),
                  config.get('num_heads', 4),
                  config.get('total_key_depth', 128),
                  config.get('total_value_depth', 128),
                  config.get('filter_size', 128),
                  self.timestep,
                  config.get('input_dropout', 0.2),
                  config.get('layer_dropout', 0.2),
                  config.get('attention_dropout', 0.2),
                  config.get('relu_dropout', 0.2))
        self.self_attn_layers = _BiDirSelfAttentionLayers(*params)
        self.output_layer = SoftmaxOutputLayer(
            hidden_size=config.get('hidden_size', 128),
            output_size=config.get('num_chords', 170))
        self.n_classes = self.output_layer.output_size

    def forward(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)

        max_chunk = self.timestep
        original_len = x.shape[1]
        if original_len == 0:
            nc = self.output_layer.output_size
            return torch.zeros(x.shape[0], 0, nc, device=x.device, dtype=x.dtype)

        chunks = []
        pos = 0
        while pos < original_len:
            chunk = x[:, pos:pos + max_chunk, :]
            if chunk.shape[1] < max_chunk:
                pad = torch.zeros(chunk.shape[0], max_chunk - chunk.shape[1],
                                  chunk.shape[2], device=x.device, dtype=x.dtype)
                chunk = torch.cat((chunk, pad), dim=1)
            attn_out, _ = self.self_attn_layers(chunk)
            chunks.append(attn_out)
            pos += max_chunk

        output = torch.cat(chunks, dim=1)[:, :original_len, :]
        return self.output_layer(output)

    def _apply_temporal_smoothing(self, logits, kernel_size, use_gaussian=False):
        """Apply ChordNet-style temporal smoothing to BTC frame logits."""
        return apply_temporal_smoothing(logits, kernel_size, use_gaussian)

    def predict(self, x, per_frame=True, smooth=False, kernel_size=9, use_gaussian=False):
        with torch.no_grad():
            logits = self(x)
            if smooth and logits.dim() >= 2:
                logits = self._apply_temporal_smoothing(logits, kernel_size, use_gaussian)
            if logits.dim() == 3:
                if per_frame:
                    return logits.argmax(dim=2)
                return logits.mean(dim=1).argmax(dim=1)
            return logits.argmax(dim=1)


def create_btc_model(config: ModelConfig):
    """Factory: instantiate BTC_model from ModelConfig."""
    return BTC_model(config=config)
