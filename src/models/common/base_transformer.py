"""
BaseTransformer: frequency encoder + temporal encoder + decoder.
Used by ChordNet for chord recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

# Cached positional encoding to avoid recomputing every forward pass
_pe_cache = {}

def positional_encoding(batch_size, n_time, n_feature, zero_pad=False, scale=False, dtype=torch.float32):
    cache_key = (n_time, n_feature, zero_pad, scale, dtype)
    if cache_key not in _pe_cache:
        pos = torch.arange(n_time, dtype=dtype).reshape(-1, 1)
        pos_enc = pos / torch.pow(10000, 2 * torch.arange(0, n_feature, dtype=dtype) / n_feature)
        pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])
        if zero_pad:
            pos_enc = torch.cat([torch.zeros(1, n_feature), pos_enc[1:, :]], 0)
        if scale:
            pos_enc = pos_enc * (n_feature ** 0.5)
        _pe_cache[cache_key] = pos_enc
    return _pe_cache[cache_key].unsqueeze(0).expand(batch_size, -1, -1)


class FeedForward(nn.Module):
    def __init__(self, n_feature=512, dropout=0.2):
        super().__init__()
        n_hidden = n_feature * 4
        self.linear1 = nn.Linear(n_feature, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_feature)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_hidden)
        self.norm_layer = nn.LayerNorm(n_feature)
        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero

    def forward(self, x):
        residual = x
        y = F.relu(self.norm(self.linear1(x)))
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        y = residual + self.alpha * y
        return self.norm_layer(y)


class EncoderF(nn.Module):
    """Frequency-axis encoder: groups frequency bins and applies self-attention."""
    def __init__(self, n_freq, n_group, n_head=8, n_layer=5, dropout=0.2, pr=0.01):
        super().__init__()
        assert n_freq % n_group == 0
        self.d_model = n_freq // n_group
        self.n_freq = n_freq
        self.n_group = n_group
        self.pr = pr
        self.attn_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        self.attn_alphas = nn.ParameterList()
        for _ in range(n_layer):
            self.attn_layer.append(nn.MultiheadAttention(self.d_model, n_head, batch_first=True))
            self.ff_layer.append(FeedForward(n_feature=self.d_model, dropout=dropout))
            self.attn_alphas.append(nn.Parameter(torch.zeros(1)))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_freq)
        self.norm_layer = nn.LayerNorm(n_freq)

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B * T, self.n_group, self.d_model)
        pe = positional_encoding(x.shape[0], x.shape[1], x.shape[2]).to(x.device)
        x = x + pe * self.pr
        for i, (attn, ff) in enumerate(zip(self.attn_layer, self.ff_layer)):
            residual = x
            out, _ = attn(x, x, x, need_weights=False)
            x = ff(residual + self.attn_alphas[i] * out)
        y = self.norm_layer(self.fc(self.dropout(x.reshape(B, T, self.n_freq))))
        return y


class EncoderT(nn.Module):
    """Temporal-axis encoder: self-attention across time steps."""
    def __init__(self, n_freq, n_head=8, n_layer=5, dropout=0.2, pr=0.02):
        super().__init__()
        self.n_freq = n_freq
        self.pr = pr
        self.attn_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        self.attn_alphas = nn.ParameterList()
        for _ in range(n_layer):
            self.attn_layer.append(nn.MultiheadAttention(n_freq, n_head, batch_first=True))
            self.ff_layer.append(FeedForward(n_feature=n_freq, dropout=dropout))
            self.attn_alphas.append(nn.Parameter(torch.zeros(1)))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_freq)
        self.norm_layer = nn.LayerNorm(n_freq)

    def forward(self, x):
        B, T, F = x.shape
        x = x + positional_encoding(B, T, F).to(x.device) * self.pr
        for i, (attn, ff) in enumerate(zip(self.attn_layer, self.ff_layer)):
            residual = x
            out, _ = attn(x, x, x, need_weights=False)
            x = ff(residual + self.attn_alphas[i] * out)
        return self.norm_layer(self.fc(self.dropout(x)))


class Decoder(nn.Module):
    """Cross-attention decoder combining frequency and temporal encoder outputs."""
    def __init__(self, d_model=512, n_head=8, n_layer=5, dropout=0.5,
                 r1=1.0, r2=1.0, wr=1.0, pr=0.01):
        super().__init__()
        self.r1, self.r2, self.wr, self.pr = r1, r2, wr, pr
        self.attn_layer1 = nn.ModuleList()
        self.attn_layer2 = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        self.attn1_alphas = nn.ParameterList()
        self.attn2_alphas = nn.ParameterList()
        for _ in range(n_layer):
            self.attn_layer1.append(nn.MultiheadAttention(d_model, n_head, batch_first=True))
            self.attn_layer2.append(nn.MultiheadAttention(d_model, n_head, batch_first=True))
            self.ff_layer.append(FeedForward(n_feature=d_model, dropout=dropout))
            self.attn1_alphas.append(nn.Parameter(torch.zeros(1)))
            self.attn2_alphas.append(nn.Parameter(torch.zeros(1)))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.norm_layer = nn.LayerNorm(d_model)

    def forward(self, x1, x2, weight=None):
        y = x1 * self.r1 + x2 * self.r2
        if weight is not None:
            while weight.dim() < y.dim():
                weight = weight.unsqueeze(-1)
            if weight.shape[-1] == 1 and y.shape[-1] > 1:
                weight = weight.expand_as(y)
            y = y + weight * self.wr
        y = y + positional_encoding(y.shape[0], y.shape[1], y.shape[2]).to(y.device) * self.pr
        for i in range(len(self.attn_layer1)):
            residual = y
            out1, _ = self.attn_layer1[i](y, y, y, need_weights=False)
            y = self.norm_layer(residual + self.attn1_alphas[i] * self.dropout(out1))
            residual = y
            out2, _ = self.attn_layer2[i](y, x2, x2, need_weights=False)
            y = self.norm_layer(residual + self.attn2_alphas[i] * self.dropout(out2))
            y = self.ff_layer[i](y)
        output = self.fc(self.dropout(y))
        return output, y


class BaseTransformer(nn.Module):
    def __init__(self, n_channel=1, n_freq=2048, n_group=16,
                 f_layer=2, f_head=8, f_dropout=0.2,
                 t_layer=2, t_head=4, t_dropout=0.2,
                 d_layer=2, d_head=4, d_dropout=0.5,
                 r1=1.0, r2=1.0, wr=0.2):
        super().__init__()
        self.n_channel = n_channel
        self.encoder_f = nn.ModuleList()
        self.encoder_t = nn.ModuleList()
        for _ in range(n_channel):
            self.encoder_f.append(EncoderF(n_freq=n_freq, n_group=n_group,
                                           n_head=f_head, n_layer=f_layer, dropout=f_dropout))
            self.encoder_t.append(EncoderT(n_freq=n_freq, n_head=t_head,
                                           n_layer=t_layer, dropout=t_dropout))
        self.decoder = Decoder(d_model=n_freq, n_head=d_head, n_layer=d_layer,
                               dropout=d_dropout, r1=r1, r2=r2, wr=wr)

    def forward(self, x, weight=None):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if self.n_channel == 1:
            y1 = self.encoder_f[0](x[:, 0, :, :])
            y2 = self.encoder_t[0](x[:, 0, :, :])
        else:
            ff, tf = [], []
            for i in range(min(self.n_channel, x.shape[1])):
                ff.append(self.encoder_f[i](x[:, i, :, :]))
                tf.append(self.encoder_t[i](x[:, i, :, :]))
            y1 = torch.sum(torch.stack(ff), dim=0)
            y2 = torch.sum(torch.stack(tf), dim=0)
        y, w = self.decoder(y1, y2, weight)
        return y, w
