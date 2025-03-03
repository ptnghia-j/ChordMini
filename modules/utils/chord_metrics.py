import numpy as np

def parse_chord(chord_str):
    """Parse chord string to extract root and chord quality."""
    # Handle "N" (no chord)
    if (chord_str == "N"):
        return (None, None)
    # Handle different notation formats
    if ("_" in chord_str):
        # Handle underscore notation (e.g., 'c_maj_maj7', 'c_min_min7')
        parts = chord_str.split("_")
        root = parts[0]
        quality = "_".join(parts[1:])
    else:
        # Extract root note (special case for sharps)
        if ((len(chord_str) > 1) and (chord_str[1] == "#")):
            root = chord_str[:2]
            quality_start = 2
        else:
            root = chord_str[0]
            quality_start = 1
        # Extract quality
        quality = (chord_str[quality_start:] if (len(chord_str) > quality_start) else "maj")
        # Convert empty quality to "maj" (e.g., "c" -> "c", "maj")
        if (not quality):
            quality = "maj"
    # Normalize quality
    quality_mapping = {'': 'maj', 'm': 'min', 'dim': 'dim', 'aug': 'aug', 'maj7': 'maj7', 'm7': 'min7', '7': '7', 'dim7': 'dim7'}
    quality = quality_mapping.get(quality, quality)
    return (root, quality)
def calculate_root_similarity(root1, root2):
    """Calculate similarity between chord roots based on circle of fifths."""
    if ((root1 is None) or (root2 is None)):
        return 0.0
    # Comprehensive circle of fifths with enharmonic equivalents
    notes_circle = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    # Map for enharmonic equivalents (case-insensitive)
    equivalents = {'Gb': 'F#', 'Db': 'C#', 'Ab': 'G#', 'Eb': 'D#', 'Bb': 'A#', 'gb': 'f#', 'db': 'c#', 'ab': 'g#', 'eb': 'd#', 'bb': 'a#'}
    # Normalize to uppercase for comparison
    root1 = root1.upper()
    root2 = root2.upper()
    # Map enharmonic equivalents
    root1 = equivalents.get(root1, root1)
    root2 = equivalents.get(root2, root2)
    if ((root1 not in notes_circle) or (root2 not in notes_circle)):
        return 0.5
    # Calculate distance in circle of fifths
    (idx1, idx2) = (notes_circle.index(root1), notes_circle.index(root2))
    distance = min(abs((idx1 - idx2)), (12 - abs((idx1 - idx2))))
    # Convert to similarity (0-1)
    return (1.0 - (distance / 6.0))
def calculate_quality_similarity(quality1, quality2):
    """Calculate similarity between chord qualities."""
    if ((quality1 is None) or (quality2 is None)):
        return 0.0
    # Comprehensive chord quality groups that match the dataset
    major_like = ['maj', 'maj7', 'maj_maj7', '6', '6/9', 'add9', 'maj9', 'maj13']
    minor_like = ['min', 'm', 'm6', 'm7', 'min7', 'min_min7', 'm9', 'm11', 'm13']
    dominant_like = ['7', '9', '13', '7b9', '7#9', '7b5', '7#5', 'maj_min7']
    diminished_like = ['dim', 'dim7', 'dim_dim7', 'dim_min7', 'm7b5', '°', 'ø']
    augmented_like = ['aug', '_aug']
    # Map each quality to its group(s)
    def get_groups(quality):
        groups = []
        if ((quality in major_like) or quality.endswith("maj7")):
            groups.append("major")
        if (((quality in minor_like) or quality.startswith("min_")) or quality.startswith("m")):
            groups.append("minor")
        if (((quality in dominant_like) or quality.endswith("7")) and (not quality.endswith("maj7")) and (not quality.endswith("min7")) and (not quality.endswith("dim7"))):
            groups.append("dominant")
        if ((quality in diminished_like) or ("dim" in quality)):
            groups.append("diminished")
        if ((quality in augmented_like) or ("aug" in quality)):
            groups.append("augmented")
        return groups
    # Get groups for both qualities
    groups1 = get_groups(quality1)
    groups2 = get_groups(quality2)
    # Exact match
    if (quality1 == quality2):
        return 1.0
    # Same chord family
    if any(((g in groups1) and (g in groups2)) for g in ['major', 'minor', 'dominant', 'diminished', 'augmented']):
        return 0.8
    # Related families
    related_pairs = [('major', 'dominant'), ('minor', 'diminished')]
    if any((((g1, g2) in related_pairs) or ((g2, g1) in related_pairs)) for g1 in groups1 for g2 in groups2):
        return 0.5
    # Different families
    return 0.2
def chord_similarity(chord1, chord2):
    """Calculate similarity between two chord symbols."""
    # Exact match
    if (chord1 == chord2):
        return 1.0
    # No chord case
    if ((chord1 == "N") or (chord2 == "N")):
        return 0.0
    # Parse chords
    (root1, quality1) = parse_chord(chord1)
    (root2, quality2) = parse_chord(chord2)
    # Calculate similarities
    root_sim = calculate_root_similarity(root1, root2)
    quality_sim = calculate_quality_similarity(quality1, quality2)
    # Weight root 60% and quality 40%
    return ((0.6 * root_sim) + (0.4 * quality_sim))
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
    if (len(y_true) == 0):
        return 0.0
    total_score = 0.0
    for i in range(len(y_true)):
        true_idx = y_true[i]
        pred_idx = y_pred[i]
        true_chord = idx_to_chord.get(true_idx, "Unknown")
        pred_chord = idx_to_chord.get(pred_idx, "Unknown")
        sim_score = chord_similarity(true_chord, pred_chord)
        total_score += sim_score
    return (total_score / len(y_true))