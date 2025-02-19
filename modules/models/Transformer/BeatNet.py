import torch
import torch.nn as nn
from modules.models.Transformer.BaseTransformer import BaseTransformer

class BeatNet(nn.Module):
  def __init__(self, 
               source=3,
               n_class=3,
               weights=(0.4, 0.3, 0.3),
               n_freq=2048,
               n_group=32,
               f_layer=2,
               f_head=4,
               t_layer=2,
               t_head=4,
               d_layer=2, 
               d_head=4,
               dropout=0.2,
               *args, **kwargs):
    super().__init__()
    self.weights = weights
    self.transfomer_layer = nn.ModuleList()
    for _ in range(source):
      self.transformer_layer.append(BaseTransformer(n_freq=n_freq, n_group=n_group,
                                                     f_layer=f_layer, f_head=f_head, f_dropout=dropout,
                                                     t_layer=t_layer, t_head=t_head, t_dropout=dropout,
                                                     d_layer=d_layer, d_head=d_head, d_dropout=dropout))
      self.dropout = nn.Dropout(dropout)
      self.fc = nn.Linear(n_freq, n_class)

      self.reset_parameters(0.05)
    
    def reset_parameters(self, stdv):
      self.fc.bias.data.fill_(-torch.log(torch.tensor(1 / stdv - 1)))

    def forward(self, input):
      y, logits = [], []
      for i, layer in enumerate(self.transformer_layer):
        x = input[:, i, :, :]
        x, _f = layer(x)
        w = self.weights[i]

        y.append(x * w)
        logits.append(_f * w)
      
      y = torch.sum(torch.stack(y, dim=0), dim=0)
      logits = torch.sum(torch.stack(logits, dim=0), dim=0)

      y = self.dropout(y)
      beats = self.fc(y)
      beats = torch.argmax(beats, dim=-1)
      return beats, logits
  
if __name__ == '__main__':
  # testing
  model = BeatNet()
  print(model)

  x = torch.randn(2, 3, 2048, 128)
  y, weights = model(x)
  print(y.shape, weights.shape)
