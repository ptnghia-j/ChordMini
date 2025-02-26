import re
from modules.utils.inversion_mapping import inversion_mapping  # NEW: use external inversion map

def normalize_chord(chord: str) -> str:
    """
    Normalize a chord label to its canonical non-inversion representation.
    Uses inversion_mapping to translate inverted chord labels.
    """
    chord_clean = chord.strip().strip('"').lower()
    if chord_clean in ["nan", "n"]:
        return "N"
    
    if '/' in chord_clean:
        # Use the imported inversion_mapping; if not found, use the base chord.
        return inversion_mapping.get(chord_clean, chord_clean.split('/')[0])
    
    if chord_clean.endswith("_maj"):
        return chord_clean[:-4]
    if chord_clean.endswith("_min"):
        return chord_clean[:-4] + "m"
    
    return chord_clean
