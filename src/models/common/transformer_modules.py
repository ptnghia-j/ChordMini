"""
Core transformer building blocks used by BTC_model.

Contains: LayerNorm, MultiHeadAttention, Conv, PositionwiseFeedForward,
SoftmaxOutputLayer, and positional signal generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def _gen_bias_mask(max_length):
    """Generates causal bias mask (-Inf for future timesteps)."""
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).float()
    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(max_length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Sinusoidal positional encoding [1, max_length, channels]."""
    position = np.arange(max_length)
    num_timescales = channels // 2
    log_inc = np.log(float(max_timescale) / float(min_timescale)) / max(float(num_timescales) - 1, 1)
    inv = min_timescale * np.exp(np.arange(num_timescales).astype(np.float64) * -log_inc)
    scaled = np.expand_dims(position, 1) * np.expand_dims(inv, 0)
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant')
    signal = signal.reshape([1, max_length, channels])
    return torch.from_numpy(signal).float()


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SoftmaxOutputLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2,
                            batch_first=True, bidirectional=True)

    @property
    def fc(self):
        """Backward-compatible alias for older BTC head naming."""
        return self.output_projection

    def forward(self, hidden):
        return self.output_projection(hidden)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth,
                 output_depth, num_heads, bias_mask=None, dropout=0.0,
                 attention_map=False):
        super().__init__()
        if total_key_depth % num_heads != 0:
            raise ValueError(f"Key depth ({total_key_depth}) must be divisible by num_heads ({num_heads})")
        if total_value_depth % num_heads != 0:
            raise ValueError(f"Value depth ({total_value_depth}) must be divisible by num_heads ({num_heads})")
        self.attention_map = attention_map
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, L, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * D)

    def forward(self, queries, keys, values):
        queries = self._split_heads(self.query_linear(queries))
        keys = self._split_heads(self.key_linear(keys))
        values = self._split_heads(self.value_linear(values))
        queries = queries * self.query_scale
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits)
        weights = F.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        contexts = self._merge_heads(torch.matmul(weights, values))
        outputs = self.output_linear(contexts)
        if self.attention_map:
            return outputs, weights
        return outputs


class Conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        super().__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        return self.conv(self.pad(inputs.permute(0, 2, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_depth, filter_size, output_depth,
                 layer_config='ll', padding='left', dropout=0.0):
        super().__init__()
        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])
        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError(f"Unknown layer type {lc}")
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)
        return x
