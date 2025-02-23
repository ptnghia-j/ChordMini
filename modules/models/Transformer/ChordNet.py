import torch
import torch.nn as nn
from modules.models.Transformer.BaseTransformer import BaseTransformer

class ChordNet(nn.Module):
    def __init__(self, n_freq=2048, n_classes=122, n_group=32,
                 f_layer=5, f_head=8,
                 t_layer=5, t_head=8,
                 d_layer=5, d_head=8,
                 dropout=0.2, *args, **kwargs):
        super().__init__()
        self.transformer = BaseTransformer(n_freq=n_freq, n_group=n_group,
                                           f_layer=f_layer, f_head=f_head, f_dropout=dropout,
                                           t_layer=t_layer, t_head=t_head, t_dropout=dropout,
                                           d_layer=d_layer, d_head=d_head, d_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_classes)

    def forward(self, x, weight=None):
        o, logits = self.transformer(x, weight)
        o = self.dropout(o)
        logits = self.fc(o)
        return logits, o

    def predict(self, x, weight=None):
        logits, _ = self.forward(x, weight)
        return torch.argmax(logits, dim=-1)

if __name__ == '__main__':
    model = ChordNet()
    print(model)
    x = torch.randn(2, 2, 2048, 128)
    y, weights = model(x)
    print(y.shape, weights.shape)
    y_pred = model.predict(x)
    print(y_pred.shape)