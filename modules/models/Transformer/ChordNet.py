import torch
import torch.nn as nn
from modules.models.Transformer.BaseTransformer import BaseTransformer

class ChordNet(nn.Module):
    def __init__(self, n_freq=2048, n_classes=122, n_group=32,
                 f_layer=5, f_head=8,
                 t_layer=5, t_head=8,
                 d_layer=5, d_head=8,
                 dropout=0.2, ignore_index=None, *args, **kwargs):
        super().__init__()
        self.transformer = BaseTransformer(n_freq=n_freq, n_group=n_group,
                                           f_layer=f_layer, f_head=f_head, f_dropout=dropout,
                                           t_layer=t_layer, t_head=t_head, t_dropout=dropout,
                                           d_layer=d_layer, d_head=d_head, d_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_classes)
        self.ignore_index = ignore_index

    def forward(self, x, weight=None):
        o, logits = self.transformer(x, weight)
        o = self.dropout(o)
        logits = self.fc(o)
        
        # Apply a penalty to the ignore_index if specified
        if self.ignore_index is not None and self.ignore_index < logits.shape[-1]:
            penalty_mask = torch.zeros_like(logits)
            penalty_mask[..., self.ignore_index] = -10.0  # Large negative bias
            logits = logits + penalty_mask
            
        return logits, o

    def predict(self, x, weight=None):
        logits, _ = self.forward(x, weight)
        if logits.ndim == 3:
            logits = logits.mean(dim=1)

        return torch.argmax(logits, dim=-1)

if __name__ == '__main__':
    model = ChordNet()
    print(model)
    x = torch.randn(2, 2, 2048, 128)
    y, weights = model(x)
    print(y.shape, weights.shape)
    y_pred = model.predict(x)
    print(y_pred.shape)