import numpy as np

def parse_chord(chord_str):
    """Parse chord string to extract root and chord quality."""
    # Handle "N" (no chord)
    if chord_str == "N":
        return None, None
    
    # Basic parsing - assumes format like "C", "Cm", "G7", "Bb_maj7", etc.
    if "_" in chord_str:
        parts = chord_str.split("_")
        root = parts[0]
        quality = "_".join(parts[1:])
    else:
        # For simple formats like "C", "Cm", "G7"
        root = chord_str[0]
        if len(chord_str) > 1 and chord_str[1] in ["#", "b"]:
            root += chord_str[1]
            quality = chord_str[2:] if len(chord_str) > 2 else "maj"
        else:
            quality = chord_str[1:] if len(chord_str) > 1 else "maj"
            
    return root, quality

def calculate_root_similarity(root1, root2):
    """Calculate similarity between chord roots based on circle of fifths."""
    if root1 is None or root2 is None:
        return 0.0
    
    # Simplified circle of fifths distances
    notes = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
    # Handle equivalents (e.g., F# = Gb)
    equivalents = {
        "Gb": "F#", "Db": "C#", "Ab": "G#", "Eb": "D#", "Bb": "A#"
    }
    
    # Normalize roots
    root1 = equivalents.get(root1, root1)
    root2 = equivalents.get(root2, root2)
    
    if root1 not in notes or root2 not in notes:
        return 0.5  # Default if roots not in our circle
    
    # Calculate distance in circle of fifths
    idx1, idx2 = notes.index(root1), notes.index(root2)
    distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
    
    # Convert to similarity (0-1)
    return 1.0 - (distance / 6.0)

def calculate_quality_similarity(quality1, quality2):
    """Calculate similarity between chord qualities."""
    if quality1 is None or quality2 is None:
        return 0.0
    
    # Define chord quality groups
    major_like = ["maj", "maj7", "maj9", "6", "69", "maj13"]
    minor_like = ["m", "min", "m7", "min7", "m9", "min9", "m11", "m13"]
    dominant_like = ["7", "9", "13", "7b9", "7#9", "7b5", "7#5", "7b13"]
    diminished_like = ["dim", "dim7", "m7b5", "°", "ø"]
    augmented_like = ["aug", "+", "+7"]
    
    # Find which groups the qualities belong to
    groups1, groups2 = [], []
    for group_name, group in [
        ("major", major_like), ("minor", minor_like), 
        ("dominant", dominant_like), ("diminished", diminished_like),
        ("augmented", augmented_like)
    ]:
        if quality1 in group:
            groups1.append(group_name)
        if quality2 in group:
            groups2.append(group_name)
    
    # Exact match
    if quality1 == quality2:
        return 1.0
    
    # Same chord family
    if any(g in groups1 and g in groups2 for g in ["major", "minor", "dominant", "diminished", "augmented"]):
        return 0.8
    
    # Related families (e.g., major and dominant)
    related_pairs = [
        ("major", "dominant"), ("minor", "diminished")
    ]
    if any((g1, g2) in related_pairs or (g2, g1) in related_pairs 
           for g1 in groups1 for g2 in groups2):
        return 0.5
    
    # Different families
    return 0.2

def chord_similarity(chord1, chord2):
    """Calculate similarity between two chord symbols."""
    # Exact match
    if chord1 == chord2:
        return 1.0
    
    # No chord case
    if chord1 == "N" or chord2 == "N":
        return 0.0
    
    # Parse chords
    root1, quality1 = parse_chord(chord1)
    root2, quality2 = parse_chord(chord2)
    
    # Calculate similarities
    root_sim = calculate_root_similarity(root1, root2)
    quality_sim = calculate_quality_similarity(quality1, quality2)
    
    # Weight root more heavily than quality (60%-40%)
    return 0.6 * root_sim + 0.4 * quality_sim

def weighted_chord_symbol_recall(y_true, y_pred, idx_to_chord):
    """
    Calculate Weighted Chord Symbol Recall (WCSR)
    
    Args:
        y_true: Array of true chord indices
        y_pred: Array of predicted chord indices
        idx_to_chord: Dictionary mapping indices to chord symbols
        
    Returns:
        WCSR score (0-1)
    """
    if len(y_true) == 0:
        return 0.0
    
    total_score = 0.0
    for i in range(len(y_true)):
        true_idx = y_true[i]
        pred_idx = y_pred[i]
        
        true_chord = idx_to_chord.get(true_idx, "Unknown")
        pred_chord = idx_to_chord.get(pred_idx, "Unknown")
        
        sim_score = chord_similarity(true_chord, pred_chord)
        total_score += sim_score
    
    return total_score / len(y_true)