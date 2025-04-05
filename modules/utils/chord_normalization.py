import re

inversion_mapping = {
    "a/c#": "a",
    "a/e": "a",
    "a/g": "a7",
    "a7/c#": "a7",
    "ab/bb": "ab",
    "ab/c": "ab",
    "ab/eb": "ab",
    "abdim7/gb": "abdim7",
    "b/a": "b7",
    "b/d#": "b7",
    "b/f#": "b7",
    "b7/d#": "b7",
    "bb/ab": "bb",
    "bb/c": "bb",
    "bb/d": "bb",
    "bb/f": "bb",
    "bb7/d": "bb7",
    "bbdim7/ab": "bbdim7",
    "c/e": "cmaj7",
    "c/g": "c",
    "c/b": "cmaj7",
    "c/g/b": "cmaj7",
    "d/f#": "d",
    "d/a": "d",
    "d7/f#": "d7",
    "d7/a": "d7",
    "dm/f": "dm",
    "e/g#": "e",
    "e/b": "e",
    "e7/g#": "e7",
    "e7/b": "e7",
    "em/g": "em",
    "em/b": "em",
    "f/a": "f",
    "f/c": "f",
    "f7/a": "f7",
    "f7/c": "f7",
    "g/b": "g",
    "g/d": "g",
    "g7/b": "g7",
    "g7/d": "g7",
    "g/f": "g6",
    "bm/d": "bm",
    "bm/f#": "bm",
    "c#/e#": "c#7",
    "c#/g#": "c#",
    "c#7/e#": "c#7",
    "c#/dim7/b": "c#dim7",
    "cdim7/bb": "cdim7",
    "ddim7/c": "ddim7",
    "adim7/g": "adim7",
    "bdim7/a": "bdim7"
}

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
