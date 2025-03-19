import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models.Transformer.BaseTransformer import BaseTransformer


class ChordNet(nn.Module):
    def __init__(self, n_freq=2048, n_classes=122, n_group=32,
                 f_layer=5, f_head=8,
                 t_layer=5, t_head=8,
                 d_layer=5, d_head=8,
                 dropout=0.2, ignore_index=None, *args, **kwargs):
        super().__init__()
        
        # CQT vs STFT handling
        # For CQT input (typically n_freq=144), we want n_group=12 to get actual_feature_dim=12
        # For STFT input (typically n_freq=2048), the provided n_group would be used
        is_cqt = n_freq <= 256  # Simple heuristic to detect CQT input
        if is_cqt:
            # Note: forcing n_group=12 when using CQT leads to an actual feature dimension of 144/12 = 12.
            # For better performance you might reconsider this setting.
            n_group = 12
            print(f"Detected CQT input (n_freq={n_freq}), setting n_group={n_group}")
        else:
            print(f"Using standard STFT configuration with n_freq={n_freq}, n_group={n_group}")

        # Calculate the actual feature dimension that will come out of the transformer
        actual_feature_dim = n_freq // n_group
        print(f"Actual feature dimension: {actual_feature_dim}")
        
        # Ensure n_freq is divisible by n_group
        assert n_freq % n_group == 0, f"n_freq ({n_freq}) must be divisible by n_group ({n_group})"

        self.transformer = BaseTransformer(
            n_channel=1,  # Explicitly specify we only need 1 channel for ChordNet
            n_freq=n_freq,
            n_group=n_group,
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
        
        # Use the correct feature dimension for the output linear layer
        self.fc = nn.Linear(actual_feature_dim, n_classes)
        self.ignore_index = ignore_index

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
            criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            # Handle different dimensions (batch vs sequence)
            if logits.ndim == 3 and targets.ndim == 1:
                # For sequence data, average across time dimension
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.repeat_interleave(logits.size(1)))
            else:
                loss = criterion(logits, targets)
            
            # Ensure loss is non-negative (critical fix)
            loss = torch.clamp(loss, min=0.0)

        return logits, o if loss is None else (logits, loss)

    def predict(self, x, weight=None, per_frame=False):
        """
        Make chord predictions from input spectrograms.
        
        Args:
            x: Input tensor
            weight: Optional weight parameter
            per_frame: If True, return predictions per frame (time step)
                      If False (default), average over time dimension
        
        Returns:
            Tensor of predictions with shape [batch] or [batch, time] depending on per_frame
        """
        logits, _ = self.forward(x, weight)
        
        # Return per-frame predictions if requested
        if per_frame and logits.ndim == 3:
            return torch.argmax(logits, dim=2)  # [batch, time]
        
        # Otherwise use the standard approach with averaging
        if logits.ndim == 3:
            logits = logits.mean(dim=1)  # Average over time dimension
        
        return torch.argmax(logits, dim=-1)  # [batch]


if __name__ == '__main__':
    model = ChordNet()
    print(model)
    x = torch.randn(2, 2, 2048, 128)
    y, weights = model(x)
    print(y.shape, weights.shape)
    y_pred = model.predict(x)
    print(y_pred.shape)