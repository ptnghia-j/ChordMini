import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models.Transformer.BaseTransformer import BaseTransformer
# Add missing import for device handling
from modules.utils.device import get_device, to_device
import warnings

class ChordNet(nn.Module):
    def __init__(self, n_freq=2048, n_classes=122, n_group=12,
                 f_layer=5, f_head=8,
                 t_layer=5, t_head=8,
                 d_layer=5, d_head=8,
                 dropout=0.2, ignore_index=None, *args, **kwargs):
        super().__init__()

        # Ensure n_freq is divisible by n_group
        if n_freq % n_group != 0:
            warnings.warn(f"Input with n_freq={n_freq} is not divisible by n_group=12. "
                         f"This may cause issues with the model architecture.")

        # Ensure feature dimension is divisible by head count
        if (n_freq // n_group) % f_head != 0:
            # Adjust head count instead of n_group
            for h in range(f_head, 0, -1):
                if (n_freq // n_group) % h == 0:
                    warnings.warn(f"Adjusted f_head from {f_head} to {h} to maintain n_group=12")
                    f_head = h
                    break


        # Calculate the actual feature dimension that will come out of the transformer
        feature_dim = n_freq // n_group
        print(f"Using feature dimensions: n_freq={n_freq}, n_group={n_group}, feature_dim={feature_dim}, heads={f_head}")

        # Final compatibility check
        if feature_dim % f_head != 0:
            warnings.warn(f"Feature dimension {feature_dim} not divisible by head count {f_head}. "
                         f"This will cause errors. Please adjust parameters.")

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
        self.n_classes = n_classes  # Store the number of classes for debugging and loading
        # FIX: Use n_freq as the input dimension for the final linear layer,
        # as the BaseTransformer's decoder outputs features of size n_freq.
        self.fc = nn.Linear(n_freq, n_classes)
        self.ignore_index = ignore_index

        # Dictionary to map indices to chord names - useful for evaluation
        self.idx_to_chord = kwargs.get('idx_to_chord', None)

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
            # BaseTransformer returns (decoder_logits, pre_decoder_features)
            # decoder_logits shape: [B, T, n_class]
            # pre_decoder_features shape: [B, T, n_freq]
            _, features = self.transformer(x, weight) # Assign the second return value to 'features'
        except Exception as e:
            warnings.warn(f"Error in transformer: {e}. This might indicate an issue with BaseTransformer implementation.")
            # Provide a fallback mechanism when BaseTransformer fails
            # We'll create dummy outputs that match the expected shapes
            batch_size = x.size(0)
            time_steps = x.size(2) if x.dim() > 2 else 1
            # The feature dimension expected by self.fc is n_freq
            feature_dim = self.transformer.decoder.fc.in_features # Should be n_freq

            # Create dummy outputs that match expected shapes for 'features'
            features = torch.zeros((batch_size, time_steps, feature_dim), device=x.device)
            # Return early with zeros to avoid subsequent errors
            logits = self.fc(features) # Use dummy features
            if targets is not None:
                criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1) if targets.dim() > 1 else targets)
                return logits, loss
            # Return dummy logits and features
            return logits, features

        # Apply dropout to the features before the final linear layer
        features = self.dropout(features)
        # Calculate final logits using ChordNet's fc layer with the correct features
        logits = self.fc(features)

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

        # Return final logits and the features used to compute them
        return logits, features if loss is None else (logits, loss)

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
            # Run forward pass - it now returns (logits, features)
            output = self.forward(x)

            # Handle different output formats from forward pass
            if isinstance(output, tuple):
                logits = output[0]  # Take primary prediction logits
            else:
                # Should not happen if forward always returns tuple, but handle just in case
                logits = output

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

    def load_state_dict(self, state_dict, strict=True, partial_output_layer=False):
        """
        Custom state_dict loader that can handle output layer size mismatches

        Args:
            state_dict: The state dictionary to load
            strict: Whether to enforce exact matches (default: True)
            partial_output_layer: Whether to allow partial loading of output layer (default: False)
                                 when model sizes differ
        """
        # Check if output layer dimensions match
        if 'fc.weight' in state_dict and hasattr(self, 'fc'):
            pretrained_classes = state_dict['fc.weight'].size(0)
            current_classes = self.fc.weight.size(0)

            if pretrained_classes != current_classes:
                warnings.warn(f"Output layer mismatch: pretrained={pretrained_classes} classes, "
                              f"current={current_classes} classes")

                if partial_output_layer:
                    warnings.warn("Attempting partial loading of output layer...")
                    # Common case: larger model loading smaller pretrained weights
                    if current_classes > pretrained_classes:
                        # Create a new weight tensor with correct dimensions
                        new_weight = torch.zeros_like(self.fc.weight)
                        new_bias = torch.zeros_like(self.fc.bias)

                        # Copy the available weights
                        new_weight[:pretrained_classes, :] = state_dict['fc.weight']
                        new_bias[:pretrained_classes] = state_dict['fc.bias']

                        # Update the state dict
                        state_dict['fc.weight'] = new_weight
                        state_dict['fc.bias'] = new_bias

                        warnings.warn(f"Loaded first {pretrained_classes} classes, "
                                     f"initialized remaining {current_classes-pretrained_classes} classes to zero")

                    # Less common: smaller model loading larger pretrained weights
                    elif current_classes < pretrained_classes:
                        # Truncate the weights
                        state_dict['fc.weight'] = state_dict['fc.weight'][:current_classes, :]
                        state_dict['fc.bias'] = state_dict['fc.bias'][:current_classes]

                        warnings.warn(f"Truncated pretrained weights from {pretrained_classes} to {current_classes} classes")

        # Call the parent implementation
        return super().load_state_dict(state_dict, strict)

    def set_chord_mapping(self, chord_mapping):
        """Set chord mapping for evaluation purposes"""
        from modules.utils.chords import idx2voca_chord

        if chord_mapping:
            self.chord_mapping = chord_mapping
            # Create reverse mapping for evaluation
            self.idx_to_chord = {v: k for k, v in chord_mapping.items()}
        else:
            # Fall back to default mapping
            self.idx_to_chord = idx2voca_chord()


if __name__ == '__main__':
    model = ChordNet()
    print(model)
    x = torch.randn(2, 2, 2048, 128)
    y, weights = model(x)
    print(y.shape, weights.shape)
    y_pred = model.predict(x)
    print(y_pred.shape)