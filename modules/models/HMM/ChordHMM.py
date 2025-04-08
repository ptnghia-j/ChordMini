import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from modules.utils import logger
# Import the trainer from its new location
from modules.training.ChordHMMTrainer import ChordHMMTrainer
# Import visualization functions from their new location
from modules.utils.visualize import visualize_transitions

class ChordHMM(nn.Module):
    """
    Chord-specific Hidden Markov Model that:
    1. Uses pretrained chord model's emission probabilities
    2. Trains transition probabilities between chords
    3. Uses dataset chord distribution as initial probabilities
    """
    def __init__(self, pretrained_model, num_states, init_distribution=None, transition_stats=None, 
                 device='cpu', temperature=0.5, self_transition_bias=0.2, emission_weight=0.8,
                 transition_boost=0.3, min_segment_penalty=0.1):
        super(ChordHMM, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.num_states = num_states  # Number of chords in vocabulary
        self.device = device
        self.transition_stats = transition_stats  # Add transition statistics
        
        # Store hyperparameters for transition control
        self.temperature = temperature  # Lower temperature = sharper distribution
        self.self_transition_bias = self_transition_bias  # Reduced from 0.3 to 0.2 for even fewer self-transitions
        self.emission_weight = emission_weight  # Increased from 0.7 to 0.8 to trust emissions more
        self.transition_boost = transition_boost  # New: explicitly encourage transitions
        self.min_segment_penalty = min_segment_penalty  # New: penalize very long segments
        
        # Dynamic inference parameters (can be modified at inference time)
        self.smoothing_level = 1.0  # 1.0 = normal, <1 = fewer transitions, >1 = more transitions
        self.max_segment_length = 15  # Max segment length in seconds (at feature_rate)
        self.segment_confidence_threshold = 0.7  # Min confidence to maintain a segment
        
        # Freeze the pretrained model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
            
        # Initial state probabilities based on dataset distribution (in log space)
        if init_distribution is not None:
            # Ensure the distribution is valid
            init_distribution = np.asarray(init_distribution)
            init_distribution = np.clip(init_distribution, 1e-5, 1.0)  # Clip to avoid zeros
            init_distribution = init_distribution / init_distribution.sum()  # Normalize
            
            # Convert to log probabilities with proper scaling to avoid numerical issues
            log_init = torch.log(torch.tensor(init_distribution, dtype=torch.float32))
            self.start_probs = nn.Parameter(log_init)
        else:
            # Initialize with uniform distribution
            uniform = torch.ones(num_states, dtype=torch.float32) / num_states
            self.start_probs = nn.Parameter(torch.log(uniform))
        
        # Transition probabilities (in log space): from_chord -> to_chord
        # Use a separate matrix to ensure proper normalization
        self.raw_transitions = nn.Parameter(torch.zeros(num_states, num_states))
        
        # Initialize transition matrix with musical prior knowledge and/or dataset statistics
        self._initialize_transitions()
        
        # Create a dictionary to store chord quality relationships
        self.quality_relationships = self._build_quality_relationships()
        
        # Runtime book-keeping
        self.feature_rate = 22050 / 2048  # Default feature frame rate (frames/sec) based on config hop_length
    
    def _build_quality_relationships(self):
        """Build a dictionary mapping chord qualities to related qualities"""
        # Define groups of related chord qualities
        quality_groups = [
            # Major-type chords
            ['maj', 'maj7', '7', 'sus4', 'sus2', '6', 'maj9', 'add9'],
            # Minor-type chords
            ['min', 'min7', 'min6', 'min9', 'minmaj7'],
            # Diminished-type chords
            ['dim', 'dim7', 'hdim7'],
            # Augmented-type chords
            ['aug', 'aug7'],
        ]
        
        # Build relationship dictionary
        relationships = {}
        for group in quality_groups:
            for quality in group:
                relationships[quality] = [q for q in group if q != quality]
        
        return relationships
    
    def _initialize_transitions(self):
        """Initialize transition matrix with musical priors based on music theory"""
        with torch.no_grad():
            # Get chord-to-index mapping
            idx_to_chord = self.pretrained_model.idx_to_chord if hasattr(self.pretrained_model, 'idx_to_chord') else {}
            chord_to_idx = {chord: idx for idx, chord in idx_to_chord.items()}
            
            # Initialize with small random values
            self.raw_transitions.data = 0.01 * torch.randn_like(self.raw_transitions)
            
            # Apply anti-self-transition bias (boosting transitions between different chords)
            all_chord_transitions = torch.ones_like(self.raw_transitions) - torch.eye(self.num_states, device=self.raw_transitions.device)
            self.raw_transitions.data += self.transition_boost * all_chord_transitions
            
            # Then apply a reduced self-transition bias (now smaller to reduce over-smoothing)
            diagonal = torch.eye(self.num_states)
            self.raw_transitions.data += self.self_transition_bias * diagonal
            
            # Define roots for all 12 keys/pitches
            roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Parse chord symbols to get root and quality information
            chord_info = {}
            chord_qualities = set()  # Track all qualities used in the dataset
            
            for idx, chord_symbol in idx_to_chord.items():
                if chord_symbol == 'N':  # No chord
                    continue
                    
                # Split into root and quality
                parts = chord_symbol.split(':')
                root = parts[0]
                quality = parts[1] if len(parts) > 1 else 'maj'  # Default to major if not specified
                
                chord_info[idx] = {
                    'root': root,
                    'quality': quality
                }
                chord_qualities.add(quality)
            
            # Create dictionaries for indexing chords by root and quality
            root_to_chord_indices = {}
            quality_to_chord_indices = {}
            
            for idx, info in chord_info.items():
                root = info['root']
                quality = info['quality']
                
                # Index by root
                if root not in root_to_chord_indices:
                    root_to_chord_indices[root] = {}
                if quality not in root_to_chord_indices[root]:
                    root_to_chord_indices[root][quality] = []
                root_to_chord_indices[root][quality].append(idx)
                
                # Index by quality
                if quality not in quality_to_chord_indices:
                    quality_to_chord_indices[quality] = []
                quality_to_chord_indices[quality].append(idx)
            
            # Apply smaller bias for transitions between same-root but different qualities
            # Enhanced for specific quality transitions
            for root, qualities in root_to_chord_indices.items():
                # Create pairs of all chord qualities for this root
                quality_pairs = []
                for q1 in qualities.keys():
                    for q2 in qualities.keys():
                        if q1 != q2:
                            quality_pairs.append((q1, q2))
                
                # Apply biases based on quality relationships
                for from_quality, to_quality in quality_pairs:
                    bias = 0.1  # Base bias
                    
                    # Specific quality transitions that are common
                    quality_transitions = {
                        # Common transitions between qualities
                        ('maj', 'sus4'): 0.4,
                        ('sus4', 'maj'): 0.4,
                        ('min', 'min7'): 0.4,
                        ('min7', 'min'): 0.3,
                        ('maj', '7'): 0.3,
                        ('7', 'maj'): 0.3,
                        ('min', 'sus4'): 0.2,
                        ('sus4', 'min'): 0.2,
                        ('7', 'sus4'): 0.3,
                        ('sus4', '7'): 0.3,
                    }
                    
                    # Use specific bias if defined, otherwise use base bias
                    if (from_quality, to_quality) in quality_transitions:
                        bias = quality_transitions[(from_quality, to_quality)]
                    
                    # Apply the bias to all chords with this root and quality pair
                    from_indices = root_to_chord_indices[root][from_quality]
                    to_indices = root_to_chord_indices[root][to_quality]
                    
                    for i in from_indices:
                        for j in to_indices:
                            self.raw_transitions.data[i, j] += bias
            
            # Define common chord progressions using Roman numeral notation
            # Format: (from_degree, from_quality, to_degree, to_quality, weight)
            major_progressions = [
                # Major key progressions
                (1, 'maj', 4, 'maj', 0.3),    # I -> IV
                (1, 'maj', 5, 'maj', 0.4),    # I -> V
                (1, 'maj', 6, 'min', 0.2),    # I -> vi
                (2, 'min', 5, 'maj', 0.4),    # ii -> V
                (4, 'maj', 1, 'maj', 0.3),    # IV -> I (plagal cadence)
                (4, 'maj', 5, 'maj', 0.3),    # IV -> V
                (5, 'maj', 1, 'maj', 0.5),    # V -> I (perfect cadence)
                (5, 'maj', 6, 'min', 0.2),    # V -> vi (deceptive cadence)
                (6, 'min', 2, 'min', 0.2),    # vi -> ii
                (6, 'min', 4, 'maj', 0.3),    # vi -> IV
                
                # Common progressions with 7th chords
                (2, 'min7', 5, '7', 0.4),    # ii7 -> V7 
                (5, '7', 1, 'maj', 0.5),     # V7 -> I
                (1, 'maj7', 6, 'min7', 0.2), # Imaj7 -> vi7
                (6, 'min7', 2, 'min7', 0.2), # vi7 -> ii7
                (5, 'sus4', 5, '7', 0.4),    # Vsus4 -> V7
                (5, 'sus4', 1, 'maj', 0.3),  # Vsus4 -> I
                
                # Additional sus4 progressions
                (1, 'sus4', 1, 'maj', 0.5),  # Isus4 -> I
                (4, 'sus4', 4, 'maj', 0.5),  # IVsus4 -> IV
            ]
            
            minor_progressions = [
                # Natural minor key progressions
                (1, 'min', 4, 'min', 0.3),    # i -> iv
                (1, 'min', 5, 'min', 0.2),    # i -> v
                (1, 'min', 5, 'maj', 0.3),    # i -> V (harmonic minor)
                (1, 'min', 7, 'maj', 0.2),    # i -> VII
                (3, 'maj', 6, 'maj', 0.2),    # III -> VI
                (4, 'min', 1, 'min', 0.3),    # iv -> i
                (4, 'min', 5, 'min', 0.2),    # iv -> v
                (4, 'min', 5, 'maj', 0.3),    # iv -> V (harmonic minor)
                (5, 'min', 1, 'min', 0.3),    # v -> i
                (5, 'maj', 1, 'min', 0.4),    # V -> i (harmonic minor)
                (6, 'maj', 4, 'min', 0.2),    # VI -> iv
                (6, 'maj', 2, 'dim', 0.1),    # VI -> ii°
                (7, 'maj', 3, 'maj', 0.2),    # VII -> III
                
                # Common progressions with 7th chords in minor
                (1, 'min7', 4, 'min7', 0.3),   # i7 -> iv7
                (5, '7', 1, 'min', 0.4),       # V7 -> i
                (5, '7', 1, 'min7', 0.4),      # V7 -> i7
                (4, 'min7', 5, '7', 0.3),      # iv7 -> V7
            ]
            
            # Major scale degrees to semitones
            major_scale = [0, 2, 4, 5, 7, 9, 11]
            
            # Apply progressions across all keys
            for key_idx, key_root in enumerate(roots):
                
                # Process major key progressions
                for from_degree, from_quality, to_degree, to_quality, weight in major_progressions:
                    # Calculate absolute root note indices
                    from_root_idx = (key_idx + major_scale[(from_degree - 1) % 7]) % 12
                    to_root_idx = (key_idx + major_scale[(to_degree - 1) % 7]) % 12
                    
                    # Get corresponding root notes
                    from_root = roots[from_root_idx]
                    to_root = roots[to_root_idx]
                    
                    # Check if we have chords with these roots and qualities
                    if (from_root in root_to_chord_indices and 
                        from_quality in root_to_chord_indices[from_root] and
                        to_root in root_to_chord_indices and
                        to_quality in root_to_chord_indices[to_root]):
                        
                        # Get all chord indices with these roots and qualities
                        from_indices = root_to_chord_indices[from_root][from_quality]
                        to_indices = root_to_chord_indices[to_root][to_quality]
                        
                        # Apply progression weight to all transitions between these chords
                        for i in from_indices:
                            for j in to_indices:
                                self.raw_transitions.data[i, j] += weight
                
                # Process minor key progressions
                for from_degree, from_quality, to_degree, to_quality, weight in minor_progressions:
                    # Minor scale degrees to semitones (using natural minor)
                    minor_scale = [0, 2, 3, 5, 7, 8, 10]
                    
                    # Calculate absolute root note indices
                    from_root_idx = (key_idx + minor_scale[(from_degree - 1) % 7]) % 12
                    to_root_idx = (key_idx + minor_scale[(to_degree - 1) % 7]) % 12
                    
                    # Get corresponding root notes
                    from_root = roots[from_root_idx]
                    to_root = roots[to_root_idx]
                    
                    # Check if we have chords with these roots and qualities
                    if (from_root in root_to_chord_indices and 
                        from_quality in root_to_chord_indices[from_root] and
                        to_root in root_to_chord_indices and
                        to_quality in root_to_chord_indices[to_root]):
                        
                        # Get all chord indices with these roots and qualities
                        from_indices = root_to_chord_indices[from_root][from_quality]
                        to_indices = root_to_chord_indices[to_root][to_quality]
                        
                        # Apply progression weight to all transitions between these chords
                        for i in from_indices:
                            for j in to_indices:
                                self.raw_transitions.data[i, j] += weight
            
            # Add additional common chord patterns
            common_patterns = [
                # Common jazz turnarounds and substitutions
                ('2-5-1', [(2, 'min7'), (5, '7'), (1, 'maj7')], 0.4),
                ('backdoor', [(4, 'min7'), (7, '7'), (1, 'maj7')], 0.3),
                ('1-6-2-5', [(1, 'maj7'), (6, 'min7'), (2, 'min7'), (5, '7')], 0.25),
                
                # Common pop patterns
                ('pop1', [(1, 'maj'), (5, 'maj'), (6, 'min'), (4, 'maj')], 0.3),
                ('pop2', [(1, 'maj'), (6, 'min'), (4, 'maj'), (5, 'maj')], 0.3),
                
                # Add specific pattern for sus4 -> major resolution
                ('sus4_resolution', [(5, 'sus4'), (5, 'maj'), (1, 'maj')], 0.4),
            ]
            
            # Quality mapping for various chord types
            quality_map = {
                'maj7': 'maj', 'maj': 'maj', 'maj9': 'maj',
                'min7': 'min', 'min': 'min', 'min9': 'min',
                'dom7': '7', '7': '7', '9': '7', '13': '7',
                'dim7': 'dim', 'dim': 'dim', 'hdim7': 'dim',
                'aug': 'aug', 'aug7': 'aug',
                'sus4': 'sus4', 'sus2': 'sus4'
            }
            
            # Process these patterns in all keys
            for pattern_name, pattern_chords, weight in common_patterns:
                for key_idx, key_root in enumerate(roots):
                    # Process pattern through each key
                    for i in range(len(pattern_chords) - 1):
                        from_degree, from_quality_base = pattern_chords[i]
                        to_degree, to_quality_base = pattern_chords[i + 1]
                        
                        # Map qualities to our recognized qualities
                        from_quality = quality_map.get(from_quality_base, from_quality_base)
                        to_quality = quality_map.get(to_quality_base, to_quality_base)
                        
                        # Calculate absolute root note indices (using major scale as reference)
                        from_root_idx = (key_idx + major_scale[(from_degree - 1) % 7]) % 12
                        to_root_idx = (key_idx + major_scale[(to_degree - 1) % 7]) % 12
                        
                        # Get corresponding root notes
                        from_root = roots[from_root_idx]
                        to_root = roots[to_root_idx]
                        
                        # Apply transition weight if chords exist
                        if (from_root in root_to_chord_indices and 
                            from_quality in root_to_chord_indices[from_root] and
                            to_root in root_to_chord_indices and
                            to_quality in root_to_chord_indices[to_root]):
                            
                            from_indices = root_to_chord_indices[from_root][from_quality]
                            to_indices = root_to_chord_indices[to_root][to_quality]
                            
                            for i in from_indices:
                                for j in to_indices:
                                    self.raw_transitions.data[i, j] += weight
            
            # Enhance quality-specific transitions
            quality_transitions = {
                # Enhanced for sus4 chords which often serve as passing chords
                ('sus4', 'maj'): 0.5,
                ('sus4', '7'): 0.4,
                ('7', 'maj'): 0.4,
                ('min', 'min7'): 0.4,
                ('min7', 'min'): 0.3,
                ('min7', '7'): 0.3,
                ('7', 'min7'): 0.2,
                ('maj', 'sus4'): 0.3,
                ('maj', 'min'): 0.2,
                ('min', 'maj'): 0.2,
            }
            
            # Apply quality-specific transitions globally (across all roots)
            for (from_qual, to_qual), bias in quality_transitions.items():
                if from_qual in quality_to_chord_indices and to_qual in quality_to_chord_indices:
                    for from_idx in quality_to_chord_indices[from_qual]:
                        for to_idx in quality_to_chord_indices[to_qual]:
                            if chord_info[from_idx]['root'] == chord_info[to_idx]['root']:
                                # Higher bias for same-root quality transitions
                                self.raw_transitions.data[from_idx, to_idx] += bias * 1.5
                            else:
                                # Lower bias for different-root quality transitions
                                self.raw_transitions.data[from_idx, to_idx] += bias * 0.5
            
            # Add even stronger handling for N (no chord) transitions to reduce its prevalence
            if 'N' in chord_to_idx:
                n_idx = chord_to_idx['N']
                
                # N can transition to any chord with moderate probability
                for idx in range(self.num_states):
                    if idx != n_idx:
                        # Get chord info if available
                        if idx in chord_info:
                            # Favor transitions to major/minor chords from N
                            quality = chord_info[idx]['quality']
                            if quality in ['maj', 'min']:
                                self.raw_transitions.data[n_idx, idx] += 0.2  # Higher for major/minor (increased from 0.15)
                            else:
                                self.raw_transitions.data[n_idx, idx] += 0.1  # Lower for others (increased from 0.08)
                        else:
                            # Default if no chord info
                            self.raw_transitions.data[n_idx, idx] += 0.1
                
                # Any chord can transition to N with low probability (kept very low)
                for idx in range(self.num_states):
                    if idx != n_idx:
                        self.raw_transitions.data[idx, n_idx] += 0.02  # Reduced from 0.05
                        
                # N can also self-transition but with lower probability
                self.raw_transitions.data[n_idx, n_idx] += 0.1  # Reduced from 0.15
    
    def _normalize_transitions(self):
        """
        Convert raw transition weights to valid log probabilities.
        Applies temperature and dynamic smoothing level.
        """
        # Apply dynamic smoothing level - lower value = smoother transitions
        effective_temperature = self.temperature / self.smoothing_level
        
        # Use temperature for sharper/smoother distributions
        scaled_transitions = self.raw_transitions / effective_temperature
        normalized = F.softmax(scaled_transitions, dim=1)
        normalized = torch.clamp(normalized, min=1e-10)
        
        return torch.log(normalized)
    
    @property
    def transitions(self):
        """
        Property to return normalized transition probabilities.
        This ensures we always work with proper transition distributions.
        """
        return self._normalize_transitions()
    
    def forward(self, features, feature_rate=None):
        """
        Run Viterbi algorithm to find most likely chord sequence
        
        Parameters:
            features: [batch_size, seq_length, feature_dim] - audio features
            feature_rate: Optional, frames per second for time-based penalties
            
        Returns:
            Most likely chord sequence [batch_size, seq_length]
        """
        # Update feature rate if provided (for time-based penalties)
        if feature_rate is not None:
            self.feature_rate = feature_rate
        
        # Get emission probabilities from pretrained model
        emission_probs = self._get_emission_probs(features)
        
        # Run Viterbi algorithm
        predictions = self._viterbi(emission_probs)
        
        # Post-process to split overly long segments if needed
        if self.max_segment_length > 0:
            predictions = self._split_long_segments(predictions, emission_probs)
            
        return predictions
    
    def _get_emission_probs(self, features):
        """
        Get emission probabilities from pretrained model
        
        Parameters:
            features: [batch_size, seq_length, feature_dim] - audio features
        Returns:
            Log emission probabilities [batch_size, seq_length, num_states]
        """
        self.pretrained_model.eval()
        with torch.no_grad():
            # Get batch size and sequence length
            if features.dim() == 2:  # [seq_len, features]
                features = features.unsqueeze(0)  # Add batch dimension
            
            batch_size, seq_len, _ = features.shape
            
            # Process through pretrained model to get chord probabilities
            # Handle different sequence lengths to avoid memory issues
            all_logits = []
            max_batch_size = 32  # Process in smaller batches
            
            for i in range(0, batch_size, max_batch_size):
                batch_features = features[i:i+max_batch_size]
                batch_logits = self.pretrained_model(batch_features)
                
                if isinstance(batch_logits, tuple):
                    batch_logits = batch_logits[0]
                    
                all_logits.append(batch_logits)
            
            logits = torch.cat(all_logits, dim=0)
                
            # Convert logits to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            return log_probs
    
    def loss(self, features, chord_labels):
        """
        Compute negative log likelihood of chord sequences
        
        Parameters:
            features: [batch_size, seq_length, feature_dim] - audio features
            chord_labels: [batch_size, seq_length] - indices of true chords
        Returns:
            Negative log likelihood
        """
        # Get emission probabilities from pretrained model
        emission_probs = self._get_emission_probs(features)
        
        # Get normalized transitions
        trans_probs = self.transitions
        
        # Compute log likelihood using forward algorithm
        batch_size, seq_length, num_states = emission_probs.shape
        
        # Initialize forward variables with start probabilities + first emission
        alphas = self.start_probs.unsqueeze(0) + emission_probs[:, 0]  # [batch_size, num_states]
        
        # Iterate through the sequence
        for t in range(1, seq_length):
            prev_alphas = alphas.unsqueeze(2)  # [batch_size, num_states, 1]
            
            # Add transition probabilities
            next_alphas = prev_alphas + trans_probs.unsqueeze(0)  # [batch_size, num_states, num_states]
            
            # Sum over previous states (log-space -> logsumexp)
            alphas = torch.logsumexp(next_alphas, dim=1)  # [batch_size, num_states]
            
            # Add emission probabilities
            alphas = alphas + emission_probs[:, t]  # [batch_size, num_states]
        
        # Final step - sum over all final states
        log_likelihood = torch.logsumexp(alphas, dim=1)  # [batch_size]
        
        # Use milder L2 regularization to allow more expressive transitions
        l2_reg = 0.001 * torch.norm(self.raw_transitions)**2  # Reduced from 0.01
        
        # Use stronger entropy regularization to encourage more peaked distributions
        transition_probs = torch.exp(self.transitions)
        entropy_reg = -0.05 * torch.mean(torch.sum(transition_probs * torch.log(transition_probs + 1e-10), dim=1))  # Reduced from 0.1
        
        # Negative log likelihood plus regularization terms
        return -log_likelihood.mean() + l2_reg - entropy_reg
    
    def _viterbi(self, emission_probs):
        """
        Find most likely chord sequence using Viterbi algorithm
        Enhanced with emission confidence weighting and segment length penalties
        
        Parameters:
            emission_probs: [batch_size, seq_length, num_states] - log emission probabilities
        Returns:
            Most likely chord sequence [batch_size, seq_length]
        """
        batch_size, seq_length, num_states = emission_probs.shape
        
        # Get normalized transition probabilities
        trans_probs = self.transitions
        
        # Calculate emission confidence for weighting
        emission_confidence = torch.exp(emission_probs)
        max_confidence, _ = torch.max(emission_confidence, dim=2, keepdim=True)
        
        # Weight balance between emissions and transitions based on confidence
        emission_weight = self.emission_weight * max_confidence + (1 - self.emission_weight)
        
        # Initialize viterbi variables and backpointers
        v = self.start_probs.unsqueeze(0) + emission_probs[:, 0]  # [batch_size, num_states]
        backpointers = torch.zeros(batch_size, seq_length, num_states, dtype=torch.long, device=self.device)
        
        # Track the current state and duration for segment length penalties
        current_state = torch.zeros(batch_size, num_states, dtype=torch.long, device=self.device)
        state_duration = torch.zeros(batch_size, num_states, dtype=torch.long, device=self.device)
        
        # Iterate through the sequence
        for t in range(1, seq_length):
            # Calculate potential scores for all transitions
            transition_scores = v.unsqueeze(2) + trans_probs.unsqueeze(0)  # [batch_size, from_state, to_state]
            
            # Apply segment length penalties to discourage overly long segments
            # This creates a gradual penalty that increases with segment duration
            if self.min_segment_penalty > 0:
                # Calculate how many seconds each state has been active
                seconds_active = state_duration / max(1.0, self.feature_rate)
                
                # Create penalty that increases with duration (sigmoid-like curve)
                # This gives increasing incentive to change states as duration increases
                duration_factor = torch.clamp(seconds_active / 5.0, 0, 2.0)  # Normalized by 5 seconds
                penalty = self.min_segment_penalty * duration_factor
                
                # Apply penalty to self-transitions (diagonal elements)
                for i in range(num_states):
                    # Only penalize continuing in the same state (self-transitions)
                    transition_scores[:, i, i] -= penalty[:, i].unsqueeze(0)
            
            # Find best previous state for each current state
            max_scores, indices = torch.max(transition_scores, dim=1)  # [batch_size, num_states]
            
            # Update backpointers
            backpointers[:, t] = indices
            
            # Apply emission confidence weighting
            # Higher confidence = rely more on emission, less on transition
            weighted_emissions = emission_probs[:, t] * emission_weight[:, t]
            weighted_transitions = max_scores * (1.0 - emission_weight[:, t])
            
            # Update viterbi scores with weighted sum
            v = weighted_transitions + weighted_emissions
            
            # Update state duration tracking (for next iteration)
            # For each possible current state, check if it's continuing or new
            new_state_duration = torch.zeros_like(state_duration)
            for i in range(num_states):
                # For each potential current state, get the previous state that would lead to it
                prev_states = indices[:, i]
                
                # For each sample in batch, check if same state continues or new one starts
                for b in range(batch_size):
                    prev_state = prev_states[b].item()
                    if prev_state == i:  # Same state continues
                        new_state_duration[b, i] = state_duration[b, i] + 1
                    else:  # New state starts
                        new_state_duration[b, i] = 1
            
            state_duration = new_state_duration
        
        # Final step - get best final state
        best_path_scores, best_last_states = torch.max(v, dim=1)  # [batch_size]
        
        # Backtrack to get best path
        path = torch.zeros(batch_size, seq_length, dtype=torch.long, device=self.device)
        path[:, -1] = best_last_states
        
        # Follow backpointers to find best path
        for t in range(seq_length - 2, -1, -1):
            path[:, t] = backpointers[torch.arange(batch_size), t + 1, path[:, t + 1]]
            
        return path
    
    def _split_long_segments(self, predictions, emission_probs):
        """
        Post-process predictions to split overly long segments based on emission probabilities
        
        Parameters:
            predictions: [batch_size, seq_length] - predicted chord indices
            emission_probs: [batch_size, seq_length, num_states] - log emission probabilities
            
        Returns:
            Refined predictions with long segments potentially split
        """
        batch_size, seq_length = predictions.shape
        refined_predictions = predictions.clone()
        
        # Get emission probabilities in normal space
        emission_probs_exp = torch.exp(emission_probs)
        
        # Process each sequence in the batch
        for b in range(batch_size):
            # Find segment boundaries
            segments = []
            current_chord = predictions[b, 0].item()
            segment_start = 0
            
            for t in range(1, seq_length):
                if predictions[b, t].item() != current_chord:
                    # End of segment
                    segments.append((segment_start, t-1, current_chord))
                    segment_start = t
                    current_chord = predictions[b, t].item()
            
            # Add the last segment
            segments.append((segment_start, seq_length-1, current_chord))
            
            # Check each segment for potential splitting
            for start, end, chord_idx in segments:
                segment_length = end - start + 1
                segment_duration = segment_length / max(1.0, self.feature_rate)
                
                # Only process segments longer than the threshold
                if segment_duration > self.max_segment_length:
                    # Calculate confidence of the current chord across the segment
                    segment_confidence = emission_probs_exp[b, start:end+1, chord_idx]
                    
                    # Find potential splitting points where confidence drops below threshold
                    low_confidence_points = []
                    
                    # Skip the very beginning of the segment (first 5%)
                    min_idx = start + max(1, int(segment_length * 0.05))
                    # Skip the very end of the segment (last 5%)
                    max_idx = end - max(1, int(segment_length * 0.05))
                    
                    for t in range(min_idx, max_idx + 1):
                        # Look for points where confidence drops below threshold
                        if segment_confidence[t - start] < self.segment_confidence_threshold:
                            # Get the highest confidence chord at this point
                            _, best_chord = torch.max(emission_probs_exp[b, t], dim=0)
                            
                            # Only consider splitting if the highest confidence chord is different
                            if best_chord.item() != chord_idx:
                                low_confidence_points.append((t, best_chord.item()))
                    
                    # Group adjacent splitting points to avoid over-fragmentation
                    if low_confidence_points:
                        split_regions = []
                        region_start = low_confidence_points[0][0]
                        region_chord = low_confidence_points[0][1]
                        
                        for i in range(1, len(low_confidence_points)):
                            t, chord = low_confidence_points[i]
                            
                            # If this point is adjacent to previous and has same chord
                            if t == low_confidence_points[i-1][0] + 1 and chord == region_chord:
                                # Continue the region
                                continue
                            else:
                                # End the current region and start a new one
                                region_end = low_confidence_points[i-1][0]
                                
                                # Only keep regions with minimum length (at least 0.5 seconds)
                                min_frames = max(2, int(0.5 * self.feature_rate))
                                if region_end - region_start + 1 >= min_frames:
                                    split_regions.append((region_start, region_end, region_chord))
                                
                                region_start = t
                                region_chord = chord
                        
                        # Don't forget the last region
                        region_end = low_confidence_points[-1][0]
                        if region_end - region_start + 1 >= min_frames:
                            split_regions.append((region_start, region_end, region_chord))
                        
                        # Apply the splits
                        for split_start, split_end, split_chord in split_regions:
                            refined_predictions[b, split_start:split_end+1] = split_chord
        
        return refined_predictions
    
    def decode(self, features, feature_rate=None, smoothing_level=None, max_segment_length=None):
        """
        Decode feature sequence to chord sequence using Viterbi algorithm.
        
        Args:
            features: Feature tensor of shape [seq_len, num_features] or [batch_size, seq_len, num_features]
            feature_rate: Optional, frames per second for time-based penalties
            smoothing_level: Optional, override smoothing level (1.0 = normal, <1 = fewer transitions, >1 = more)
            max_segment_length: Optional, maximum segment length in seconds
            
        Returns:
            Most likely chord sequence
        """
        # Handle both single sequence and batch input cases
        is_single_sequence = features.dim() == 2
        
        if is_single_sequence:
            features = features.unsqueeze(0)  # Add batch dimension
        
        # Set parameters for this decoding
        if feature_rate is not None:
            self.feature_rate = feature_rate
        
        # Temporarily override smoothing level if provided
        original_smoothing_level = self.smoothing_level
        if smoothing_level is not None:
            self.smoothing_level = smoothing_level
            
        # Temporarily override max segment length if provided
        original_max_segment_length = self.max_segment_length
        if max_segment_length is not None:
            self.max_segment_length = max_segment_length
            
        try:
            # Get chord predictions using forward method
            predictions = self.forward(features, feature_rate)
            
            # Remove batch dimension for single sequence input
            if is_single_sequence:
                predictions = predictions[0]
                
            return predictions
            
        finally:
            # Restore original parameters
            self.smoothing_level = original_smoothing_level
            self.max_segment_length = original_max_segment_length
    
    def set_temperature(self, temperature):
        """
        Set the temperature for transition probabilities.
        Lower temperature = sharper distributions = less smoothing
        
        Args:
            temperature: Temperature value (default is 0.5)
        """
        self.temperature = temperature
        logger.info(f"Set HMM transition temperature to {temperature}")
        
    def set_emission_weight(self, weight):
        """
        Set the weight for emission probabilities.
        Higher weight = trust emissions more than transitions.
        
        Args:
            weight: Value between 0 and 1 (default is 0.8)
        """
        self.emission_weight = max(0.0, min(1.0, weight))
        logger.info(f"Set emission weight to {self.emission_weight}")
    
    def set_smoothing_level(self, level):
        """
        Set dynamic smoothing level.
        Higher = more transitions, lower = fewer transitions
        
        Args:
            level: Value > 0 (1.0 is normal, <1 is smoother, >1 is less smooth)
        """
        self.smoothing_level = max(0.1, level)
        logger.info(f"Set smoothing level to {self.smoothing_level}")
    
    def set_max_segment_length(self, length):
        """
        Set maximum segment length in seconds.
        Longer segments will be checked for potential splitting.
        Set to 0 to disable splitting.
        
        Args:
            length: Maximum segment length in seconds
        """
        self.max_segment_length = max(0, length)
        logger.info(f"Set maximum segment length to {self.max_segment_length} seconds")
    
    def set_segment_confidence_threshold(self, threshold):
        """
        Set confidence threshold for segment splitting.
        Lower values = more aggressive splitting of long segments.
        
        Args:
            threshold: Value between 0 and 1 (default is 0.7)
        """
        self.segment_confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Set segment confidence threshold to {self.segment_confidence_threshold}")
        
    def get_chord_quality(self, chord_symbol):
        """Extract the quality from a chord symbol"""
        if chord_symbol == 'N':
            return None
            
        parts = chord_symbol.split(':')
        if len(parts) == 1:
            return 'maj'  # Default to major when no quality specified
        return parts[1]
        
    def get_chord_root(self, chord_symbol):
        """Extract the root from a chord symbol"""
        if chord_symbol == 'N':
            return None
            
        parts = chord_symbol.split(':')
        return parts[0]
