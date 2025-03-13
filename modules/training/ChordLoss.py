import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.utils.chord_metrics import chord_similarity

def build_chord_similarity_matrix(idx_to_chord):
    """
    Build a chord similarity matrix where M[i][j] represents
    the similarity between chord i and chord j.
    
    Args:
        idx_to_chord: Dictionary mapping indices to chord symbols
        
    Returns:
        A tensor of shape (num_classes, num_classes) with similarity values
    """
    num_classes = len(idx_to_chord)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    # Fill the similarity matrix
    for i in range(num_classes):
        for j in range(num_classes):
            chord_i = idx_to_chord.get(i, "Unknown")
            chord_j = idx_to_chord.get(j, "Unknown")
            matrix[i][j] = chord_similarity(chord_i, chord_j)
    
    return torch.tensor(matrix)

class ChordAwareLoss(nn.Module):
    """
    A loss function that considers harmonic relationships between chords.
    Errors between similar chords are penalized less than errors between
    harmonically distant chords.
    """
    def __init__(self, idx_to_chord, ignore_index=-100, class_weights=None, device=None):
        super(ChordAwareLoss, self).__init__()
        self.ignore_index = ignore_index
        self.device = device
        
        # Build similarity matrix
        self.sim_matrix = build_chord_similarity_matrix(idx_to_chord)
        if device:
            self.sim_matrix = self.sim_matrix.to(device)
        
        # Convert similarity (higher=more similar) to distance (lower=more similar)
        self.dist_matrix = 1.0 - self.sim_matrix
        
        # Apply class weights if provided
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, device=device)
        else:
            self.class_weights = None
            
    def forward(self, logits, targets):
        """
        Calculate chord-aware loss.
        
        Args:
            logits: Model predictions of shape (batch_size, num_classes)
            targets: Target labels of shape (batch_size,)
            
        Returns:
            A scalar loss value
        """
        batch_size, num_classes = logits.shape
        
        # Convert targets to one-hot
        target_oh = F.one_hot(targets, num_classes=num_classes).float()
        
        # For ignored indices, set the one-hot vector to all zeros
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            target_oh = target_oh * mask
        
        # Apply softmax to get probability distribution
        probs = F.softmax(logits, dim=1)
        
        # Calculate weighted cross-entropy
        # For each example, calculate distance-weighted error
        # Instead of just penalizing the wrong class, penalize based on distance
        dist_weighted_error = torch.matmul(probs, self.dist_matrix)
        weighted_loss = torch.sum(target_oh * dist_weighted_error, dim=1)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            class_weights_per_sample = self.class_weights[targets]
            if self.ignore_index >= 0:
                class_weights_per_sample = torch.where(
                    targets == self.ignore_index,
                    torch.zeros_like(class_weights_per_sample),
                    class_weights_per_sample
                )
            weighted_loss = weighted_loss * class_weights_per_sample
            
        # Return mean loss over batch
        weighted_loss = weighted_loss.mean()
        
        # Ensure loss is non-negative
        weighted_loss = torch.clamp(weighted_loss, min=0.0)
        return weighted_loss

class SoftLabelLoss(nn.Module):
    """
    Alternative approach using soft labels instead of hard one-hot encoding.
    Each chord is represented by a probability distribution over related chords.
    """
    def __init__(self, idx_to_chord, ignore_index=-100, smoothing_factor=0.2, device=None):
        super(SoftLabelLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smoothing_factor = smoothing_factor
        self.device = device
        
        # Build similarity matrix
        self.sim_matrix = build_chord_similarity_matrix(idx_to_chord)
        if device:
            self.sim_matrix = self.sim_matrix.to(device)
            
    def generate_soft_labels(self, targets, num_classes):
        """
        Generate soft labels based on chord similarity.
        """
        batch_size = targets.shape[0]
        soft_labels = torch.zeros(batch_size, num_classes, device=self.device)
        
        for i in range(batch_size):
            target_idx = targets[i].item()
            if target_idx == self.ignore_index:
                continue
                
            # Get similarities for this chord
            similarities = self.sim_matrix[target_idx, :].clone()
            
            # Normalize similarities to create a probability distribution
            similarities = similarities / similarities.sum()
            
            # Apply smoothing: blend between one-hot and similarity distribution
            one_hot = torch.zeros(num_classes, device=self.device)
            one_hot[target_idx] = 1.0
            
            soft_labels[i] = (1 - self.smoothing_factor) * one_hot + self.smoothing_factor * similarities
            
        return soft_labels
        
    def forward(self, logits, targets):
        # Generate soft labels from targets using your similarity matrix
        soft_labels = self.generate_soft_labels(targets, logits.size(-1))
        log_probs = F.log_softmax(logits, dim=1)
        # Compute KL divergence loss between predicted log-probs and soft label distribution
        loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        # Clamp loss to ensure it stays non-negative
        loss = torch.clamp(loss, min=0.0)
        return loss