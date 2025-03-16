import torch
import torch.nn as nn
import torch.nn.functional as F

class HMM(nn.Module):
    """
    Implements Hidden Markov Model that can be trained via
    backpropagation instead of Baum-Welch.
    """
    def __init__(self, num_states, num_emissions):
        super(HMM, self).__init__()
        
        # Initial state probabilities (in log space)
        self.start_probs = nn.Parameter(torch.randn(num_states))
        
        # Transition probabilities (in log space): from_state -> to_state
        self.transitions = nn.Parameter(torch.Tensor(num_states, num_states))
        
        # Emission probabilities (in log space): state -> emission
        self.emissions = nn.Parameter(torch.Tensor(num_states, num_emissions))
        
        # Initialize parameters
        nn.init.xavier_normal_(self.transitions)
        nn.init.xavier_normal_(self.emissions)
    
    def forward(self, observations):
        """
        Run Viterbi algorithm to find most likely state sequence
        
        Parameters:
            observations: [batch_size, seq_length] - indices of observations
        Returns:
            Most likely state sequence [batch_size, seq_length]
        """
        return self._viterbi(observations)
    
    def loss(self, observations, states):
        """
        Compute negative log likelihood of sequences
        
        Parameters:
            observations: [batch_size, seq_length] - indices of observations
            states: [batch_size, seq_length] - indices of true states
        Returns:
            Negative log likelihood
        """
        log_likelihood = self._forward_algorithm(observations)
        return -log_likelihood.mean()
    
    def _forward_algorithm(self, observations):
        """
        Compute log probability of observation sequences
        
        Parameters:
            observations: [batch_size, seq_length] - indices of observations
        Returns:
            Log probabilities [batch_size]
        """
        batch_size, seq_length = observations.shape
        num_states = self.transitions.size(0)
        
        # Get emission probabilities for observations
        # First convert observations to one-hot to use as indices
        obs_one_hot = F.one_hot(observations, num_classes=self.emissions.size(1)).float()
        # Gather emission probabilities for each observation
        emit_probs = torch.bmm(
            obs_one_hot.view(-1, 1, self.emissions.size(1)),
            self.emissions.transpose(0, 1).expand(
                batch_size * seq_length, -1, -1)
        ).view(batch_size, seq_length, num_states)
        
        # Initialize forward variables with start probabilities + first emission
        alphas = self.start_probs.unsqueeze(0) + emit_probs[:, 0]  # [batch_size, num_states]
        
        # Iterate through the sequence
        for t in range(1, seq_length):
            # Previous step's alphas expanded for matrix op
            prev_alphas = alphas.unsqueeze(2)  # [batch_size, num_states, 1]
            
            # Add transition probabilities
            trans_probs = self.transitions.unsqueeze(0)  # [1, num_states, num_states]
            
            # Combine previous alphas with transitions
            next_alphas = prev_alphas + trans_probs  # [batch_size, num_states, num_states]
            
            # Sum over previous states (log-space -> use logsumexp)
            alphas = self._log_sum_exp(next_alphas, dim=1)  # [batch_size, num_states]
            
            # Add emission probabilities
            alphas = alphas + emit_probs[:, t]  # [batch_size, num_states]
        
        # Final step - sum over all final states
        return self._log_sum_exp(alphas, dim=1)  # [batch_size]
    
    def _viterbi(self, observations):
        """
        Find most likely state sequence using Viterbi algorithm
        
        Parameters:
            observations: [batch_size, seq_length] - indices of observations
        Returns:
            Most likely state sequence [batch_size, seq_length]
        """
        batch_size, seq_length = observations.shape
        num_states = self.transitions.size(0)
        
        # Get emission probabilities for observations
        obs_one_hot = F.one_hot(observations, num_classes=self.emissions.size(1)).float()
        emit_probs = torch.bmm(
            obs_one_hot.view(-1, 1, self.emissions.size(1)),
            self.emissions.transpose(0, 1).expand(
                batch_size * seq_length, -1, -1)
        ).view(batch_size, seq_length, num_states)
        
        # Initialize viterbi and backpointer arrays
        v = self.start_probs.unsqueeze(0) + emit_probs[:, 0]  # [batch_size, num_states]
        backpointers = []
        
        # Iterate through the sequence
        for t in range(1, seq_length):
            # Add transition probabilities to previous scores
            scores = v.unsqueeze(2) + self.transitions.unsqueeze(0)  # [batch_size, num_states, num_states]
            
            # Get max score and corresponding index
            v, idx = scores.max(1)  # [batch_size, num_states], [batch_size, num_states]
            backpointers.append(idx)
            
            # Add emission score for current observation
            v = v + emit_probs[:, t]  # [batch_size, num_states]
        
        # Get final best path and score
        best_last_state = v.argmax(1, keepdim=True)  # [batch_size, 1]
        
        # Follow the backpointers to get the best path
        best_path = [best_last_state]
        for bp in reversed(backpointers):
            best_last_state = bp.gather(1, best_last_state)
            best_path.append(best_last_state)
            
        # Reverse and concatenate
        best_path.reverse()
        return torch.cat(best_path, 1)
    
    def _log_sum_exp(self, tensor, dim):
        """Stable log-sum-exp implementation"""
        max_val, _ = tensor.max(dim, keepdim=True)
        return max_val.squeeze(dim) + (tensor - max_val).exp().sum(dim).log()