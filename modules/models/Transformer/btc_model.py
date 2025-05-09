from modules.models.Transformer.transformer_modules import *
from modules.models.Transformer.transformer_modules import _gen_timing_signal, _gen_bias_mask
from modules.utils.logger import warning # Added import for warning

use_cuda = torch.cuda.is_available()

class self_attention_block(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, attention_map=False):
        super(self_attention_block, self).__init__()

        self.attention_map = attention_map
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,hidden_size, num_heads, bias_mask, attention_dropout, attention_map)
        self.positionwise_convolution = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        if self.attention_map is True:
            y, weights = self.multi_head_attention(x_norm, x_norm, x_norm)
        else:
            y = self.multi_head_attention(x_norm, x_norm, x_norm)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_convolution(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        if self.attention_map is True:
            return y, weights
        return y

class bi_directional_self_attention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):

        super(bi_directional_self_attention, self).__init__()

        self.weights_list = list()

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.attn_block = self_attention_block(*params)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.backward_attn_block = self_attention_block(*params)

        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        x, list = inputs

        # Forward Self-attention Block
        encoder_outputs, weights = self.attn_block(x)
        # Backward Self-attention Block
        reverse_outputs, reverse_weights = self.backward_attn_block(x)
        # Concatenation and Fully-connected Layer
        outputs = torch.cat((encoder_outputs, reverse_outputs), dim=2)
        y = self.linear(outputs)

        # Attention weights for Visualization
        self.weights_list = list
        self.weights_list.append(weights)
        self.weights_list.append(reverse_weights)
        return y, self.weights_list

class bi_directional_self_attention_layers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0):
        super(bi_directional_self_attention_layers, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        # Store max_length for internal use if needed, e.g., for bias mask generation
        self.max_length = max_length
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  max_length, # Pass max_length for bias mask generation
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[bi_directional_self_attention(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        # Input 'x' is now expected to have x.shape[1] == self.max_length (timing_signal.shape[1])
        # due to segmentation/padding in BTC_model.forward
        if x.shape[1] != self.timing_signal.shape[1]:
            # This case should ideally not be hit if BTC_model.forward prepares input correctly.
            # However, as a safeguard or for standalone use of this layer:
            # warning(
            #     f"bi_directional_self_attention_layers received input of length {x.shape[1]}, "
            #     f"but expected {self.timing_signal.shape[1]}. This might lead to errors or unexpected behavior."
            # )
            # Pad or truncate if necessary, though BTC_model should handle this.
            # For now, we assume BTC_model has prepared 'x' to be self.max_length.
            # If not, the original error would occur here or in self_attn_layers.
            # To be robust, let's ensure the timing signal slice matches x's current length,
            # but this relies on BTC_model sending fixed-size chunks.
            x = x + self.timing_signal[:, :x.shape[1], :].type_as(x)
        else:
            # Standard case: x.shape[1] == self.max_length
            x = x + self.timing_signal.type_as(x)


        # A Stack of Bi-directional Self-attention Layers
        y, weights_list = self.self_attn_layers((x, []))

        # Layer Normalization
        y = self.layer_norm(y)
        return y, weights_list

class BTC_model(nn.Module):
    def __init__(self, config):
        super(BTC_model, self).__init__()

        # Use get with default values for safer config access
        self.timestep = config.get('seq_len', 108) # This is the max_length for components
        # probs_out is removed as SoftmaxOutputLayer always returns logits
        # self.probs_out = config.get('probs_out', False)

        params = (config.get('feature_size', 144),
                  config.get('hidden_size', 128), # Default from btc_config
                  config.get('num_layers', 8),    # Default from btc_config
                  config.get('num_heads', 4),     # Default from btc_config
                  config.get('total_key_depth', 128), # Default from btc_config
                  config.get('total_value_depth', 128), # Default from btc_config
                  config.get('filter_size', 128), # Default from btc_config
                  self.timestep, # Use seq_len
                  config.get('input_dropout', 0.2), # Default from btc_config
                  config.get('layer_dropout', 0.2), # Default from btc_config
                  config.get('attention_dropout', 0.2), # Default from btc_config
                  config.get('relu_dropout', 0.2)) # Default from btc_config

        self.self_attn_layers = bi_directional_self_attention_layers(*params)
        # Pass probs_out=True is no longer needed for SoftmaxOutputLayer
        self.output_layer = SoftmaxOutputLayer(hidden_size=config.get('hidden_size', 128),
                                               output_size=config.get('num_chords', 170))
                                               # Removed probs_out=True

    def forward(self, x):
        # Removed labels argument
        # labels = labels.view(-1, self.timestep) # No longer needed here

        # Output of Bi-directional Self-attention Layers
        # Ensure input has 3 dimensions [batch, time, features]
        if x.dim() == 4 and x.size(1) == 1: # Handle potential [batch, 1, time, features]
             x = x.squeeze(1)
        elif x.dim() != 3:
             # Attempt to reshape common incorrect inputs
             if x.dim() == 2: # Maybe [batch*time, features]? Need batch size info.
                 # Cannot reliably reshape without batch size and time steps.
                 # Assuming a default time step if possible, or raise error.
                 # This part might need adjustment based on how data is fed.
                 # For now, let's assume x is already [batch, time, features]
                 pass # Or raise ValueError(f"Expected 3D input [batch, time, features], got {x.shape}")
             elif x.dim() == 4: # Maybe [batch, time, features, 1]?
                 if x.size(3) == 1:
                     x = x.squeeze(3)
                 # else: raise ValueError(f"Expected 3D input [batch, time, features], got {x.shape}")

        max_chunk_len = self.timestep # Max sequence length for self_attn_layers
        original_input_len = x.shape[1]
        processed_outputs = []

        if original_input_len == 0:
            # Handle empty sequence input gracefully
            # Output layer expects [batch, time, hidden_size]
            # Assuming hidden_size can be inferred from output_layer or a default
            hidden_size = self.output_layer.fc.in_features # Infer from output layer
            # Or use: hidden_size = self.self_attn_layers.embedding_proj.out_features

            # Create zero logits of shape [batch, 0, num_chords]
            # num_chords can be inferred from self.output_layer.output_size
            num_chords = self.output_layer.output_size
            return torch.zeros(x.shape[0], 0, num_chords, device=x.device, dtype=x.dtype)


        # Segment input if it's longer than max_chunk_len, or pad if shorter/equal.
        # All chunks passed to self.self_attn_layers will have length max_chunk_len.
        
        current_pos = 0
        while current_pos < original_input_len:
            chunk = x[:, current_pos : current_pos + max_chunk_len, :]
            current_chunk_len = chunk.shape[1]

            if current_chunk_len < max_chunk_len:
                padding_size = max_chunk_len - current_chunk_len
                padding = torch.zeros(chunk.shape[0], padding_size, chunk.shape[2], 
                                      device=x.device, dtype=x.dtype)
                chunk = torch.cat((chunk, padding), dim=1)
            
            # Now chunk.shape[1] is guaranteed to be max_chunk_len
            attn_output_chunk, _ = self.self_attn_layers(chunk) # weights_list is ignored
            processed_outputs.append(attn_output_chunk)
            current_pos += max_chunk_len
            
        if not processed_outputs:
             # This should not happen if original_input_len > 0
             # But as a safeguard, if somehow no chunks were processed (e.g. original_input_len was 0 and not caught)
             # Create a zero tensor of appropriate shape to avoid crashing output_layer
            hidden_size = self.output_layer.fc.in_features
            self_attn_output = torch.zeros(x.shape[0], original_input_len, hidden_size, device=x.device, dtype=x.dtype)
        else:
            # Concatenate all processed chunks
            self_attn_output_possibly_padded = torch.cat(processed_outputs, dim=1)
    
            # Truncate the concatenated output back to the original_input_len
            self_attn_output = self_attn_output_possibly_padded[:, :original_input_len, :]


        # Pass the output through the final layer to get logits
        logits = self.output_layer(self_attn_output)

        # Return only logits. Loss calculation is handled by the Trainer.
        # Weights list can be returned for analysis if needed, but Trainer might not use it.
        # For compatibility with StudentTrainer, just return logits.
        # If attention weights are needed later, modify Trainer or return a tuple/dict.
        return logits

    def predict(self, x):
        """Predict method for compatibility with evaluation functions.

        Args:
            x: Input tensor of shape [batch, time, features] or [batch, 1, time, features]

        Returns:
            Predicted class indices of shape [batch, time]
        """
        with torch.no_grad():
            # Get logits from forward pass
            logits = self(x)

            # For BTC model, output is [batch, time, classes]
            # We need to get the predicted class indices
            if logits.dim() == 3:
                predictions = logits.argmax(dim=2)  # Get predicted class indices
            else:
                predictions = logits.argmax(dim=1)  # Handle case where time dimension is collapsed

        return predictions

        # Removed prediction, loss calculation, and second prediction return values
        # prediction,second = self.output_layer(self_attn_output)
        # prediction = prediction.view(-1)
        # second = second.view(-1)
        # loss = self.output_layer.loss(self_attn_output, labels)
        # return prediction, loss, weights_list, second

if __name__ == "__main__":
    # Example usage demonstrating the new forward pass
    # Load config from the yaml file for consistency
    from modules.utils.hparams import HParams
    try:
        # Assuming the script is run from the project root
        config = HParams.load("./config/btc_config.yaml")
        print("Loaded config from ./config/btc_config.yaml")
    except FileNotFoundError:
        # Fallback if run from within the modules directory
        config = HParams.load("../../config/btc_config.yaml")
        print("Loaded config from ../../config/btc_config.yaml")

    device = torch.device("cuda" if use_cuda else "cpu")

    # Use config values
    batch_size = config.experiment.get('batch_size', 2) # Example batch size
    timestep = config.model.get('seq_len', 108)
    feature_size = config.model.get('feature_size', 144)
    num_chords = config.model.get('num_chords', 170)

    features = torch.randn(batch_size, timestep, feature_size, requires_grad=True).to(device)
    # labels are no longer passed to the model's forward method
    # chords = torch.randint(num_chords,(batch_size*timestep,)).to(device)

    # Pass the loaded config object (HParams object) directly
    model = BTC_model(config=config.model).to(device) # Pass the model sub-config

    # Call the forward method with only features
    logits = model(features)

    # Logits shape should be [batch_size, timestep, num_chords]
    print(f"Output logits shape: {logits.shape}") # Expected: [batch_size, 108, 170]

    # Example external loss calculation
    dummy_labels = torch.randint(num_chords, (batch_size, timestep)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Flatten logits and labels for CrossEntropyLoss
    loss = loss_fn(logits.reshape(-1, num_chords), dummy_labels.reshape(-1))
    print(f"Example external loss: {loss.item()}")


