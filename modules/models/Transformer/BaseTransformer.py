import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from modules.utils.device import get_device, to_device 

def positional_encoding(batch_size, n_time, n_feature, zero_pad=False, scale=False, dtype=torch.float32):
  indices = torch.unsqueeze(torch.arange(n_time), 0).repeat(batch_size, 1)

  pos = torch.arange(n_time, dtype=dtype).reshape(-1, 1)
  pos_enc = pos / torch.pow(10000, 2 * torch.arange(0, n_feature, dtype=dtype) / n_feature)
  pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
  pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])

  # exclude the first token
  if zero_pad:
    pos_enc = torch.cat([torch.zeros(size=[1, n_feature]), pos_enc[1:, :]], 0)

  outputs = F.embedding(indices, pos_enc)

  if scale:
    outputs = outputs * (n_feature ** 0.5)

  return outputs

class FeedForward(nn.Module):
  def __init__(self, n_feature=2048, n_hidden=512, dropout=0.2):
    n_hidden = n_feature * 4
    super().__init__()
    self.linear1 = nn.Linear(n_feature, n_hidden)
    self.linear2 = nn.Linear(n_hidden, n_feature)

    self.dropout = nn.Dropout(dropout)
    self.norm = nn.LayerNorm(n_hidden)
    # Remove the extra norm inside FeedForward to avoid over-normalization
    # self.batch_norm2 = nn.LayerNorm(n_feature)
    self.norm_layer = nn.LayerNorm(n_feature)

    # Add ReZero parameter (α) initialized to zero
    self.alpha = nn.Parameter(torch.zeros(1))


  def forward(self, x):
    # Store the residual
    residual = x

    # Apply feed-forward operations
    y = self.linear1(x)
    y = self.norm(y)

    # activation and drop out
    y = F.relu(y)
    y = self.dropout(y)

    y = self.linear2(y)

    # Apply dropout but skip the extra normalization
    y = self.dropout(y)

    # Apply ReZero: y = residual + α * F(x)
    y = residual + self.alpha * y

    # Apply final layer normalization
    y = self.norm_layer(y)

    return y

class EncoderF(nn.Module):
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
    # Add ReZero parameters for each attention layer
    self.attn_alphas = nn.ParameterList()

    for _ in range(n_layer):
      self.attn_layer.append(nn.MultiheadAttention(d_model, n_head, batch_first=True))
      # FeedForward uses d_model as n_feature here
      self.ff_layer.append(FeedForward(n_feature=d_model, dropout=dropout))
      # Add ReZero parameter (α) initialized to zero for attention
      self.attn_alphas.append(nn.Parameter(torch.zeros(1)))

    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(n_freq, n_freq)
    self.norm_layer = nn.LayerNorm(n_freq)

  def forward(self, x):
    B, T, F = x.shape
    x = x.reshape(B * T, self.n_group, self.d_model)
    # Replace get_device() with x.device to ensure same device usage
    pe = positional_encoding(batch_size=x.shape[0], n_time=x.shape[1], n_feature=x.shape[2]).to(x.device)
    x = x + pe * self.pr  # use out-of-place addition instead of x += pe * self.pr

    # Make sure x requires gradients
    if not x.requires_grad:
        x.requires_grad_()

    for i, (attn, ff) in enumerate(zip(self.attn_layer, self.ff_layer)):
      residual = x

      # IMPORTANT: Also ensure residual has requires_grad=True since it's passed to checkpoint
      if not residual.requires_grad:
          residual.requires_grad_()

      # Modified layer_fn to use ReZero
      def layer_fn(x, residual, alpha):
        out, _ = attn(x, x, x, need_weights=False)
        # Apply ReZero: residual + α * attention_output
        out = residual + alpha * out
        # Pass to feed-forward (which has its own ReZero)
        return ff(out)

      # Pass the alpha parameter to the checkpoint function
      x = checkpoint.checkpoint(layer_fn, x, residual, self.attn_alphas[i])  # All inputs now have requires_grad=True

    y = x.reshape(B, T, self.n_freq)
    y = self.dropout(y)
    y = self.fc(y)
    y = self.norm_layer(y)

    return y

class EncoderT(nn.Module):
  def __init__(self, n_freq, n_head=8, n_layer=5, dropout=0.2, pr=0.02):
    super().__init__()
    self.n_freq = n_freq
    self.n_layer = n_layer
    self.pr = pr

    self.attn_layer = nn.ModuleList()
    self.ff_layer = nn.ModuleList()
    # Add ReZero parameters for each attention layer
    self.attn_alphas = nn.ParameterList()

    for _ in range(n_layer):
      self.attn_layer.append(nn.MultiheadAttention(n_freq, n_head, batch_first=True))
      # FeedForward uses n_freq as n_feature here
      self.ff_layer.append(FeedForward(n_feature=n_freq, dropout=dropout))
      # Add ReZero parameter (α) initialized to zero for attention
      self.attn_alphas.append(nn.Parameter(torch.zeros(1)))

    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(n_freq, n_freq)
    self.norm_layer = nn.LayerNorm(n_freq)

  def forward(self, x):
    B, T, F = x.shape
    # Replace get_device() with x.device to ensure same device usage
    x = x + positional_encoding(B, T, F).to(x.device) * self.pr  # out-of-place addition

    for i, (attn, ff) in enumerate(zip(self.attn_layer, self.ff_layer)):
      residual = x
      # Apply attention
      attn_out, _ = attn(x, x, x, need_weights=False)
      # Apply ReZero: residual + α * attention_output
      x = residual + self.attn_alphas[i] * attn_out
      # Pass to feed-forward (which has its own ReZero)
      x = ff(x)

    x = self.dropout(x)
    x = self.fc(x)
    x = self.norm_layer(x)

    return x

class Decoder(nn.Module):
  def __init__(self, d_model=512, n_head=8, n_layer=5, dropout=0.5, r1=1.0, r2=1.0, wr=1.0, pr=0.01):
    super().__init__()
    self.r1 = r1
    self.r2 = r2
    self.wr = wr
    self.n_layer = n_layer
    self.pr = pr

    self.attn_layer1 = nn.ModuleList()
    self.attn_layer2 = nn.ModuleList()
    self.ff_layer = nn.ModuleList()
    # Add ReZero parameters for each attention layer
    self.attn1_alphas = nn.ParameterList()
    self.attn2_alphas = nn.ParameterList()

    for _ in range(n_layer):
      self.attn_layer1.append(nn.MultiheadAttention(d_model, n_head, batch_first=True))
      self.attn_layer2.append(nn.MultiheadAttention(d_model, n_head, batch_first=True))
      # FeedForward uses d_model as n_feature here
      self.ff_layer.append(FeedForward(n_feature=d_model, dropout=dropout))
      # Add ReZero parameters (α) initialized to zero for both attention layers
      self.attn1_alphas.append(nn.Parameter(torch.zeros(1)))
      self.attn2_alphas.append(nn.Parameter(torch.zeros(1)))

    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(d_model, d_model)
    self.norm_layer = nn.LayerNorm(d_model)

  def forward(self, x1, x2, weight=None):
    y = x1 * self.r1 + x2 * self.r2
    if weight is not None:
      # Fix: Properly reshape weight for broadcasting if dimensions don't match
      if weight.dim() < y.dim():
        # Adjust weight tensor shape to match y's dimensions for proper broadcasting
        # This handles the case when weight is [B, T] and needs to be [B, T, d_model]
        for _ in range(y.dim() - weight.dim()):
          weight = weight.unsqueeze(-1)
        # If weight doesn't have the same last dimension as y, expand it
        if weight.shape[-1] == 1 and y.shape[-1] > 1:
          weight = weight.expand(-1, -1, y.shape[-1])
      y = y + weight * self.wr

    # Replace get_device() with y.device to ensure same device usage
    y = y + positional_encoding(y.shape[0], y.shape[1], y.shape[2]).to(y.device) * self.pr  # out-of-place

    for i in range(self.n_layer):
      # Self-attention with ReZero
      residual = y
      attn1_out, _ = self.attn_layer1[i](y, y, y, need_weights=False)
      attn1_out = self.dropout(attn1_out)
      # Apply ReZero: residual + α * attention_output
      y = residual + self.attn1_alphas[i] * attn1_out
      y = self.norm_layer(y)

      # Cross-attention with ReZero
      residual = y
      attn2_out, _ = self.attn_layer2[i](y, x2, x2, need_weights=False)
      attn2_out = self.dropout(attn2_out)
      # Apply ReZero: residual + α * attention_output
      y = residual + self.attn2_alphas[i] * attn2_out
      y = self.norm_layer(y)

      # Feed-forward (which has its own ReZero)
      y = self.ff_layer[i](y)

    output = self.dropout(y)
    output = self.fc(output)

    return output, y

class BaseTransformer(nn.Module):
    def __init__(self, n_channel=2, n_freq=2048, n_group=16,
                 f_layer=2, f_head=8, f_dropout=0.2, f_pr=0.01,
                 t_layer=2, t_head=4, t_dropout=0.2, t_pr=0.02,
                 d_layer=2, d_head=4, d_dropout=0.5, d_pr=0.02,
                 r1=1.0, r2=1.0, wr=0.2):
        super().__init__()
        self.n_channel = n_channel
        self.encoder_f = nn.ModuleList()
        self.encoder_t = nn.ModuleList()
        for _ in range(n_channel):
            # EncoderF uses FeedForward with n_feature = n_freq // n_group
            self.encoder_f.append(EncoderF(n_freq=n_freq, n_group=n_group, n_head=f_head,
                                            n_layer=f_layer, dropout=f_dropout, pr=f_pr))
            # EncoderT uses FeedForward with n_feature = n_freq
            self.encoder_t.append(EncoderT(n_freq=n_freq, n_head=t_head, n_layer=t_layer,
                                            dropout=t_dropout, pr=t_pr))
        # Decoder uses FeedForward with n_feature = d_model = n_freq
        self.decoder = Decoder(d_model=n_freq, n_head=d_head, n_layer=d_layer,
                               dropout=d_dropout, r1=r1, r2=r2, wr=wr, pr=d_pr)

    def forward(self, x, weight=None):
        # Ensure input is [B, n_channel, T, F]
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, T, F]
            if self.n_channel > 1:
                # Duplicate the single channel to match expected n_channel
                x = x.expand(-1, self.n_channel, -1, -1)  # [B, n_channel, T, F]

        # Handle case where we have fewer channels than expected
        actual_channels = x.shape[1]
        if actual_channels < self.n_channel:
            # Warning: input has fewer channels than the model expects
            # Use only the available channels
            ff, tf = [], []
            for i in range(actual_channels):
                x1 = self.encoder_f[i](x[:, i, :, :])
                x2 = self.encoder_t[i](x[:, i, :, :])
                ff.append(x1)
                tf.append(x2)

            # For missing channels, use zeros
            for i in range(actual_channels, self.n_channel):
                # Use the first channel shape as reference
                ff.append(torch.zeros_like(ff[0]))
                tf.append(torch.zeros_like(tf[0]))
        else:
            # Normal case - we have enough channels
            ff, tf = [], []
            for i in range(self.n_channel):
                x1 = self.encoder_f[i](x[:, i, :, :])
                x2 = self.encoder_t[i](x[:, i, :, :])
                ff.append(x1)
                tf.append(x2)

        y1 = torch.sum(torch.stack(ff, dim=0), dim=0)
        y2 = torch.sum(torch.stack(tf, dim=0), dim=0)
        y, w = self.decoder(y1, y2, weight)
        return y, w

if __name__ == '__main__':
  # Test initialization using BaseTransformer instead of Transformer
  model = BaseTransformer(n_freq=2048)
  x = torch.randn(1, 2, 1024, 2048)
  y, logits = model(x)
  print(y.shape, logits.shape)
