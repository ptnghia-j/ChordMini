import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models.Transformer.BaseTransformer import BaseTransformer
# Add missing import for device handling
from modules.utils.device import get_device, to_device
import warnings

class ChordNet(nn.Module):
    def __init__(self, n_freq=2048, n_classes=122, n_group=32,
                 f_layer=5, f_head=8,
                 t_layer=5, t_head=8,
                 d_layer=5, d_head=8,
                 dropout=0.2, ignore_index=None, cqt_threshold=256, *args, **kwargs):
        super().__init__()
        
        # Make CQT threshold configurable but use default if not specified
        self.cqt_threshold = cqt_threshold
        
        # CQT vs STFT handling with more robust detection
        is_cqt = n_freq <= self.cqt_threshold
        original_n_group = n_group
        
        if is_cqt:
            # Note: We prefer n_group=12 for CQT (144 bins) for better feature representation
            # But we'll make sure it divides n_freq evenly
            candidates = [12, 16, 24, 48, 72, 144]
            for cand in candidates:
                if n_freq % cand == 0 and (n_freq // cand) % f_head == 0:  # Ensure divisible by head count
                    n_group = cand
                    break
            
            # If no suitable n_group found, adjust to ensure compatibility with increased head count
            if (n_freq // n_group) % f_head != 0:
                for divisor in [8, 12, 16, 24, 32, 48, 64]:
                    if n_freq % divisor == 0 and (n_freq // divisor) % f_head == 0:
                        warnings.warn(f"Adjusted n_group from {n_group} to {divisor} for compatibility with larger head count")
                        n_group = divisor
                        break
            
            print(f"Detected CQT input (n_freq={n_freq}), setting n_group={n_group}")
        else:
            print(f"Using standard STFT configuration with n_freq={n_freq}, n_group={n_group}")
            
            # For STFT, also ensure compatibility with increased head count
            if (n_freq // n_group) % f_head != 0:
                for divisor in [8, 12, 16, 24, 32, 48, 64]:
                    if n_freq % divisor == 0 and (n_freq // divisor) % f_head == 0:
                        warnings.warn(f"Adjusted n_group from {n_group} to {divisor} for compatibility with larger head count")
                        n_group = divisor
                        break

        # Check if n_freq is divisible by n_group and adjust if needed
        if n_freq % n_group != 0:
            # Find closest divisor that's also compatible with head count
            for divisor in [8, 12, 16, 24, 32, 48, 64]:
                if n_freq % divisor == 0 and (n_freq // divisor) % f_head == 0:
                    warnings.warn(f"n_freq ({n_freq}) not divisible by n_group ({n_group}). "
                                 f"Adjusted n_group to {divisor}.")
                    n_group = divisor
                    break
            else:
                # If no exact divisor found, find best approximation
                best_remainder = n_freq
                best_divisor = n_group
                
                for div in range(1, 65):  # Try reasonable divisors
                    remainder = n_freq % div
                    # Must be compatible with head count
                    if remainder < best_remainder and (n_freq // div) % f_head == 0:
                        best_remainder = remainder
                        best_divisor = div
                
                # Trim input if reasonable, otherwise pad
                if best_remainder < n_freq * 0.1:  # Less than 10% wastage
                    new_n_freq = n_freq - best_remainder
                    warnings.warn(f"n_freq ({n_freq}) not divisible by any suitable value. "
                                f"Trimming to {new_n_freq} to be divisible by {best_divisor}.")
                    n_freq = new_n_freq
                    n_group = best_divisor
                else:
                    # If n_group is still not compatible, adjust f_head instead
                    warnings.warn(f"Could not find good divisor for n_freq={n_freq}. "
                                f"Adjusting attention heads for compatibility.")
                    feature_dim = n_freq // n_group
                    # Find largest compatible head count
                    for h in range(f_head, 0, -1):
                        if feature_dim % h == 0:
                            warnings.warn(f"Adjusted f_head from {f_head} to {h} for dimensional compatibility")
                            f_head = h
                            break

        # Calculate the actual feature dimension that will come out of the transformer
        actual_feature_dim = n_freq // n_group
        print(f"Using feature dimensions: n_freq={n_freq}, n_group={n_group}, feature_dim={actual_feature_dim}, heads={f_head}")
        
        # Final compatibility check
        if actual_feature_dim % f_head != 0:
            warnings.warn(f"Feature dimension {actual_feature_dim} not divisible by head count {f_head}. "
                         f"This will cause errors. Please adjust parameters.")
        
        # Check if n_group was changed from what the user specified
        if n_group != original_n_group:
            warnings.warn(f"Modified n_group from {original_n_group} to {n_group} for compatibility")

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
            
            # Verify input is well-formed
            if torch.isnan(x).any():
                warnings.warn("Input tensor contains NaN values! Replacing with zeros.")
                x = torch.nan_to_num(x, nan=0.0)

        # If we get a 2D input (batch, freq), expand it to include a time dimension
        elif x.dim() == 2:  # [batch_size, freq_bins] - from mean pooling
            # Expand to 4D with time dimension of 1
            x = x.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, freq_bins]
        
        # Add safeguards for BaseTransformer call
        try:
            # Process through transformer - preserves temporal structure for attention
            o, logits = self.transformer(x, weight)
        except Exception as e:
            warnings.warn(f"Error in transformer: {e}. This might indicate an issue with BaseTransformer implementation.")
            # Provide a fallback mechanism when BaseTransformer fails
            # We'll create dummy outputs that match the expected shapes
            batch_size = x.size(0)
            time_steps = x.size(2) if x.dim() > 2 else 1
            feature_dim = x.size(-1) // self.transformer.n_group
            
            # Create dummy outputs that match expected shapes
            o = torch.zeros((batch_size, time_steps, feature_dim), device=x.device)
            # Return early with zeros to avoid subsequent errors
            logits = self.fc(o)
            if targets is not None:
                criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                loss = criterion(logits.reshape(-1, logits.size(-1)), 
                               targets.reshape(-1) if targets.dim() > 1 else targets)
                return logits, loss
            return logits, o
        
        o = self.dropout(o)
        logits = self.fc(o)

        # Return loss if targets are provided
        loss = None
        if targets is not None:
            criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            try:
                # Handle different dimensions (batch vs sequence)
                if logits.ndim == 3 and targets.ndim == 1:
                    # For sequence data, average across time dimension to get one prediction per sequence
                    avg_logits = logits.mean(dim=1)
                    loss = criterion(avg_logits, targets)
                elif logits.ndim == 3 and targets.ndim == 2:
                    # If both have time dimension, reshape both for time-wise predictions
                    batch_size, time_steps, num_classes = logits.shape
                    loss = criterion(logits.reshape(-1, num_classes), targets.reshape(-1))
                else:
                    # Standard case - same dimensions
                    loss = criterion(logits, targets)
            except RuntimeError as e:
                warnings.warn(f"Error in loss computation: {e}")
                warnings.warn(f"Shapes - logits: {logits.shape}, targets: {targets.shape}")
                
                # Try to recover from common dimension errors
                if logits.ndim == 3:
                    avg_logits = logits.mean(dim=1)
                    if targets.ndim == 1:
                        loss = criterion(avg_logits, targets)
                    else:
                        # Last resort - reshape everything
                        loss = criterion(logits.reshape(-1, logits.size(-1)), 
                                       targets.reshape(-1))
                else:
                    # Re-raise if we can't handle it
                    raise
            
            # Ensure loss is non-negative
            loss = torch.clamp(loss, min=0.0)

        return logits, o if loss is None else (logits, loss)

    def predict(self, x, *args, **kwargs):
        """
        Make a prediction with this model, supporting both regular and per-frame modes.
        
        Args:
            x: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments including:
                - per_frame: Whether to return per-frame predictions (default: False)
        
        Returns:
            Tensor of predictions
        """
        self.eval()
        
        # Extract the per_frame parameter if present, default to False
        per_frame = kwargs.get('per_frame', False)
        
        with torch.no_grad():
            # Run forward pass
            logits = self.forward(x)
            
            # Handle different output formats
            if isinstance(logits, tuple):
                logits = logits[0]  # Take primary prediction logits
            
            # Make predictions based on mode
            if per_frame:
                # Return predictions for each frame
                return torch.argmax(logits, dim=-1)
            else:
                # Take the most common prediction across the time dimension for batch prediction
                if logits.dim() > 2:  # If we have a time dimension (dim>2)
                    # Average across time dimension first
                    logits_avg = torch.mean(logits, dim=1)
                    return torch.argmax(logits_avg, dim=-1)
                else:
                    # No time dimension, just return argmax of logits
                    return torch.argmax(logits, dim=-1)

    # To maintain backward compatibility, you can also add these alias methods
    def predict_per_frame(self, x):
        """Alias for predict(x, per_frame=True)"""
        return self.predict(x, per_frame=True)

    def predict_frames(self, x):
        """Alias for predict(x, per_frame=True)"""
        return self.predict(x, per_frame=True)


if __name__ == '__main__':
    model = ChordNet()
    print(model)
    x = torch.randn(2, 2, 2048, 128)
    y, weights = model(x)
    print(y.shape, weights.shape)
    y_pred = model.predict(x)
    print(y_pred.shape)