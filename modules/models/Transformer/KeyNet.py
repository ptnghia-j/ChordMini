import torch
import torch.nn as nn
from modules.models.Transformer import BaseTransformer


class KeyNet(nn.Module):
    def __init__(self,
                 n_freq=2048,
                 n_classes=24,  # typically 12 major + 12 minor keys
                 n_group=32,
                 f_layer=5,
                 f_head=8,
                 t_layer=5,
                 t_head=8,
                 d_layer=5,
                 d_head=8,
                 dropout=0.2,
                 *args, **kwargs):
        super().__init__()
        self.transformer = BaseTransformer(n_freq=n_freq, n_group=n_group,
                                       f_layer=f_layer, f_head=f_head, f_dropout=dropout,
                                       t_layer=t_layer, t_head=t_head, t_dropout=dropout,
                                       d_layer=d_layer, d_head=d_head, d_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_classes)

    def forward(self, x, weight=None):
        # x shape: [B, channels, time, n_freq]
        o, logits = self.transformer(x, weight)
        o = self.dropout(o)
        key_logits = self.fc(o)  
        # For local key detection, predict key class per time step
        keys_pred = torch.argmax(key_logits, dim=-1)
        return keys_pred, key_logits


if __name__ == '__main__':
    # Test initialization
    model = KeyNet()
    print(model)

    # Simulate input tensor with shape [batch, channels, time, n_freq]
    # For instance, 2 channels, time=128 segments, n_freq=2048
    x = torch.randn(2, 2, 128, 2048)
    keys, logits = model(x)
    print('Predicted keys shape:', keys.shape)
    print('Logits shape:', logits.shape)

