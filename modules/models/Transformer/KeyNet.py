import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.models.Transformer.BaseTransformer import BaseTransformer

class KeyNet(nn.Module):
    def __init__(self, n_freq=2048, n_classes=24, n_group=32,
                 f_layer=5, f_head=8,
                 t_layer=5, t_head=8,
                 d_layer=5, d_head=8,
                 dropout=0.2, ignore_index=None, *args, **kwargs):
        super().__init__()
        # Shared encoder
        self.encoder = BaseTransformer(n_freq=n_freq, n_group=n_group,
                                       f_layer=f_layer, f_head=f_head, f_dropout=dropout,
                                       t_layer=t_layer, t_head=t_head, t_dropout=dropout,
                                       d_layer=d_layer, d_head=d_head, d_dropout=dropout)
        # Local head: predicts key per time step
        self.local_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_freq, n_classes)
        )
        # Global head: uses a learnable query to attend over encoder outputs
        self.attn_query = nn.Parameter(torch.randn(n_freq))
        self.global_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_freq, n_classes)
        )
        self.ignore_index = ignore_index

    def forward(self, x, mode="local", weight=None):
        """
        Args:
            x: input tensor of shape [B, channels, T, F]. If fewer dimensions, unsqueeze.
            mode: either "local" to obtain per-segment key predictions or "global" to combine segments.
            weight: optional extra weight input to the encoder.
        """
        # Ensure input shape: [B, channels, T, F]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        # Shared encoding: Out shape [B, T, n_freq]
        encoder_out, _ = self.encoder(x, weight)  # Use the shared transformer encoder
        
        if mode == "local":
            # Local prediction: apply the local head on each time step
            local_logits = self.local_head(encoder_out)  # [B, T, n_classes]
            # For inference, one may take the prediction from the last time step:
            return local_logits  # caller can choose dim to pool if needed
        elif mode == "global":
            # Global prediction: compute attention weight for each time step using a learned query
            # encoder_out: [B, T, n_freq]
            # Compute similarity between each time step and query: [B, T]
            scores = torch.matmul(encoder_out, self.attn_query)  # [B, T]
            attn_weights = F.softmax(scores, dim=1)  # [B, T]
            # Weighted sum of local features: [B, n_freq]
            global_feat = torch.sum(encoder_out * attn_weights.unsqueeze(-1), dim=1)
            global_logits = self.global_classifier(global_feat)  # [B, n_classes]
            return global_logits
        else:
            # If "both", return both local and global predictions
            local_logits = self.local_head(encoder_out)
            scores = torch.matmul(encoder_out, self.attn_query)
            attn_weights = F.softmax(scores, dim=1)
            global_feat = torch.sum(encoder_out * attn_weights.unsqueeze(-1), dim=1)
            global_logits = self.global_classifier(global_feat)
            return local_logits, global_logits

    def predict(self, x, mode="local", weight=None):
        with torch.no_grad():
            if mode == "local":
                logits = self.forward(x, mode="local", weight=weight)
                # For local prediction, take prediction for the last segment
                # Alternatively, one can use some pooling over time.
                pred = torch.argmax(logits[:, -1, :], dim=-1)
            elif mode == "global":
                logits = self.forward(x, mode="global", weight=weight)
                pred = torch.argmax(logits, dim=-1)
            else:
                # If both mode, return both predictions
                local_logits, global_logits = self.forward(x, mode="both", weight=weight)
                pred = (torch.argmax(local_logits[:, -1, :], dim=-1), torch.argmax(global_logits, dim=-1))
            return pred

if __name__ == '__main__':
    model = KeyNet()
    print(model)
    # Simulate input: batch=2, channels=2, T=128 segments, n_freq=2048
    x = torch.randn(2, 2, 128, 2048)
    # Local predictions (per-segment)
    local_logits = model(x, mode="local")
    print("Local logits shape:", local_logits.shape)
    # Global prediction: weighted combination of segments
    global_logits = model(x, mode="global")
    print("Global logits shape:", global_logits.shape)
    # Predict
    local_pred = model.predict(x, mode="local")
    global_pred = model.predict(x, mode="global")
    print("Local prediction shape:", local_pred.shape)
    print("Global prediction shape:", global_pred.shape)

