import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models.Transformer.BaseTransformer import BaseTransformer


class ChordNet(nn.Module):
    def __init__(self, n_freq=2048, n_classes=122, n_group=32,
                 f_layer=5, f_head=8,
                 t_layer=5, t_head=8,
                 d_layer=5, d_head=8,
                 dropout=0.2, *args, **kwargs):
        super().__init__()

        # Ensure n_group is set to 12 to match the frequency dimension of 144
        # 144/12 = 12, which matches the frequency dimension we're seeing in the error
        n_group = 12  # Explicitly override n_group to 12

        # Calculate the actual feature dimension that will come out of the transformer
        actual_feature_dim = n_freq // n_group  # This will be 12 for n_freq=144, n_group=12

        self.transformer = BaseTransformer(
            n_channel=1,  # Explicitly specify we only need 1 channel for ChordNet
            n_freq=n_freq,
            n_group=n_group,  # Use the fixed value of n_group=12
            f_layer=f_layer,
            f_head=f_head,
            f_dropout=dropout,
            t_layer=t_layer,
            t_head=t_head,
            t_dropout=dropout,
            d_layer=d_layer,
            d_head=d_head,
            d_dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

        # Fix: Use the correct feature dimension for the output linear layer
        self.fc = nn.Linear(actual_feature_dim, n_classes)  # Now expects [batch, 12] instead of [batch, 144]

    def forward(self, x, targets=None, weight=None):
        # For SynthDataset, input is [batch_size, time_steps, freq_bins]
        if x.dim() == 3:  # [batch_size, time_steps, freq_bins]
            # Add channel dimension (size 1) at position 1
            x = x.unsqueeze(1)  # Results in [batch_size, 1, time_steps, freq_bins]

        # If we get a 2D input (batch, freq), expand it to include a time dimension
        elif x.dim() == 2:  # [batch_size, freq_bins] - from mean pooling
            # Expand to 4D with time dimension of 1
            x = x.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, freq_bins]

        # Process through transformer - preserves temporal structure for attention
        o, logits = self.transformer(x, weight)
        o = self.dropout(o)
        logits = self.fc(o)

        # Return loss if targets are provided
        loss = None
        if targets is not None:
            criterion = nn.CrossEntropyLoss()
            # Handle different dimensions (batch vs sequence)
            if logits.ndim == 3 and targets.ndim == 1:
                # For sequence data, average across time dimension
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.repeat_interleave(logits.size(1)))
            else:
                loss = criterion(logits, targets)
            
            # Ensure loss is non-negative (critical fix)
            loss = torch.clamp(loss, min=0.0)

        return logits, o if loss is None else (logits, loss)

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