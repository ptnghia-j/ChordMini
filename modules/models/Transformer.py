import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(batch_size, n_time, n_feature, zero_pad=False, scale=False, dtype=torch.float32):
  indices = torch.tile(torch.unsqueeze(torch.arange(n_time), 0), [batch_size, 1])

  pos = torch.arange(n_time, dtype=dtype).reshape(-1, 1)
  pos_enc = pos / torch.pow(10000, 2 * torch.arange(0, n_feature, dtype=dtype))
  pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
  pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])

  if zero_pad:
    pos_enc = torch.cat([torch.zeros(size=[1, n_feature]), pos_enc[1:, :]], 0)

  outputs = F.embedding(indices, pos_enc)

  if scale:
    outputs = outputs * (n_feature ** 0.5)

  return outputs

class FeedForward(nn.Module):
  def __init__(self, n_feature=2048, n_hidden=512, dropout=0.2):
    super().__init__()
    self.linear1 = nn.Linear(n_feature, n_hidden)
    self.linear2 = nn.Linear(n_hidden, n_feature)

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x):
    y = self.linear(x)
    y = F.relu(y)
    y = self.dropout1(y)
    y = self.linear2(y)
    y = self.dropout2(y)

    return y
  
class EncodeF(nn.Module):
  def __init__(self, n_freq, n_group, n_head=8, n_layer=5, dropout=0.2, pr=0.01):
    super().__init__()
    assert n_freq % n_group == 0

    self.d_model = d_model = n_freq // n_group
    self.n_freq = n_freq
    self.n_group = n_group
    self.n_layer = n_layer
    self.pr = pr

    self.attn_layer = nn.ModuleList()
    self.ff_layer = nn.ModuleList()

    for _ in range(n_layer):
      self.attn_layer.append(nn.MultiheadAttention(d_model, n_head, batch_first=True))
      self.ff_layer.append(FeedForward(n_feature=d_model, dropout=dropout))

    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(n_freq, n_freq)
    self.norm_layer = nn.LayerNorm(n_freq)

  def forward(self, x):
    B, T, F = x.shape
    x = x.reshape(B * T, self.n_group, self.d_model)
    x += positional_encoding(batch_size=x.shape[0], n_time=x.shape[1], n_feature=x.shape[2]) * self.pr

    for attn, ff in zip(self.attn_layer, self.ff_layer):
      pass
    


  

