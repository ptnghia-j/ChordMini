# encoding: utf-8
"""
This module contains chord evaluation functionality and utilities.
"""

import numpy as np
import pandas as pd
import mir_eval
import logging
import re
# Add defaultdict for grouping
from collections import defaultdict

logger = logging.getLogger(__name__)

# Module-level constants
PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_L = [0, 1, 1, 0, 1, 1, 1]
_CHROMA_ID = (np.arange(len(_L) * 2) + 1) + np.array(_L + _L).cumsum() - 1

# Define maps once
ENHARMONIC_MAP = {
    'Bb': 'A#', 'A#': 'Bb',
    'Db': 'C#', 'C#': 'Db',
    'Eb': 'D#', 'D#': 'Eb',
    'Gb': 'F#', 'F#': 'Gb',
    'Ab': 'G#', 'G#': 'Ab',
    # Add common enharmonic spellings for parsing
    'B#': 'C', 'Cb': 'B',
    'E#': 'F', 'Fb': 'E',
}
# Map for normalization (prefer common flats, otherwise use sharps/naturals)
# Maps from the default PITCH_CLASS representation to the preferred spelling
PREFERRED_SPELLING_MAP = {
    'A#': 'Bb', 'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab',
}


QUALITY_NORM_MAP = {
    # Basic Qualities & Common Alternatives
    '': 'maj',          # Empty string means major chord
    ':': 'maj',         # Handle colon-only as major
    'M': 'maj',         # Alternative notation for major
    'm': 'min',         # Alternative notation for minor
    '-': 'min',         # Common shorthand
    'mi': 'min',
    'major': 'maj',     # Full word notation
    'minor': 'min',     # Full word notation
    'ma': 'maj',
    'dom': '7',         # Dominant
    'ø': 'hdim7',       # Half-diminished symbol
    'o': 'dim',         # Diminished symbol
    '°': 'dim',         # Diminished symbol
    '+': 'aug',         # Augmented symbol
    'aug': 'aug',       # Ensure aug maps to aug
    'dim': 'dim',       # Ensure dim maps to dim
    'sus': 'sus4',      # Default sus to sus4

    # Sevenths & Common Alternatives
    'maj7': 'maj7',
    'M7': 'maj7',
    '^': 'maj7',        # Common shorthand (e.g., C^7)
    '^7': 'maj7',
    'major-seventh': 'maj7',
    'min7': 'min7',
    'm7': 'min7',
    '-7': 'min7',
    'minor-seventh': 'min7',
    '7': '7',
    'dom7': '7',
    'dominant-seventh': '7',
    'alt': '7',         # Altered dominant
    'dim7': 'dim7',
    '°7': 'dim7',
    'o7': 'dim7',
    'diminished-seventh': 'dim7',
    'hdim7': 'hdim7',
    'm7b5': 'hdim7',
    'half-diminished-seventh': 'hdim7',
    'minmaj7': 'minmaj7',
    'mmaj7': 'minmaj7',
    'min-maj7': 'minmaj7',
    'aug7': 'aug7',
    '7+5': 'aug7',
    '7#5': 'aug7',

    # Sixths
    '6': 'maj6',        # Ambiguous '6' usually means maj6
    'maj6': 'maj6',
    'M6': 'maj6',
    'major-sixth': 'maj6',
    'min6': 'min6',
    'm6': 'min6',
    '-6': 'min6',
    'minor-sixth': 'min6',

    # Suspended & Power Chords
    'sus2': 'sus2',
    'suspended-second': 'sus2',
    'sus4': 'sus4',
    'suspended-fourth': 'sus4',
    '5': '5',           # Keep power chord distinct
    'power': '5',

    # Extended Chords (Normalize to base 7th/6th/triad)
    '9': '7',
    '11': '7',
    '13': '7',
    'maj9': 'maj7',
    'maj11': 'maj7',
    'maj13': 'maj7',
    'min9': 'min7',
    'min11': 'min7',
    'min13': 'min7',
    '69': 'maj6',       # Simplify 6/9 chords to maj6
    'maj6(9)': 'maj6',
    'min69': 'min6',    # Add min6/9

    # Altered Chords (Normalize to base quality)
    # Parentheses content is removed by regex first, but handle common non-parentheses alterations
    '7#9': '7',
    '7b9': '7',
    '7b5': '7',         # Often implies hdim7 if root is minor, but simplify to 7 for dom root
    'maj7#5': 'maj7',   # Or augmaj7? Simplify to maj7
    'maj7+5': 'maj7',
    'maj7b5': 'maj7',
    'maj7-5': 'maj7',
    'min7+5': 'min7',   # Uncommon
    'min7b5': 'hdim7',  # This is standard hdim7
    'min7-5': 'hdim7',
    'aug9': 'aug7',     # Simplify augmented extensions
    'maj7#11': 'maj7',  # Lydian sound, simplify
    '7#11': '7',

    # MIREX/JAA notations with parentheses (Parentheses usually removed by regex, but add explicit maps for robustness)
    'maj(9)': 'maj7',
    'maj(11)': 'maj7',
    'maj(13)': 'maj7',
    'min(9)': 'min7',
    'min(11)': 'min7',
    'min(13)': 'min7',
    'min(7)': 'min7',
    'maj(7)': 'maj7',
    'min(6)': 'min6',
    'maj(6)': 'maj6',
    'maj(2)': 'maj',    # Add9 simplified
    'min(2)': 'min',    # Add9 simplified
    'maj(4)': 'maj',    # Add11 simplified
    'min(4)': 'min',    # Add11 simplified
    'sus(2)': 'sus2',
    'sus(4)': 'sus4',
    '7(b9)': '7',
    '7(#9)': '7',
    '7(11)': '7',
    '7(13)': '7',
    '7(b5)': '7',
    '7(#5)': 'aug7',
    '7(4)': '7',        # Treat 7sus4 as 7 for simplicity here
    'maj7(b5)': 'maj7',
    'maj7(#5)': 'maj7',
    'maj7(9)': 'maj7',
    'maj7(11)': 'maj7',
    'maj7(#11)': 'maj7',
    'maj7(b9)': 'maj7',
    'min7(b5)': 'hdim7',
    'min7(9)': 'min7',
    'min7(11)': 'min7',
    'min7(13)': 'min7',
    'min7(4)': 'min7',  # Treat min7sus4 as min7
    'maj9(*7)': 'maj7',
    'sus4(9)': 'sus4',
    'sus4(b7)': 'sus4',
    'sus4(2)': 'sus4',
    'sus2(b7)': 'sus2',
    'sus2(4)': 'sus4', # Treat sus2(4) as sus4
    'aug(9)': 'aug',
    'aug(11)': 'aug',
    'aug(9,11)': 'aug',

    # MIREX/* alterations (Parentheses usually removed by regex, but add explicit maps)
    'maj(*1)': 'maj',
    'maj(*3)': 'maj',
    'maj(*5)': 'maj',
    'min(*b3)': 'min',
    'min(*5)': 'min',
    'min(*b3,*5)': 'min',
    'maj7(*5)': 'maj7',
    'maj7(*b5)': 'maj7',
    'min7(*5)': 'min7',
    'min7(*b3)': 'min7',
    'min7(*1,*5)': 'min7',
    'min7(*5,b6)': 'min7',
    'min7(2,*b3,4)': 'min7',
    '7(*3)': '7',
    '7(*5)': '7',
    '9(*3)': '7',
    '9(*3,11)': '7',
    '7(*5,13)': '7',
    'dim(*b3)': 'dim',

    # Interval lists (Parentheses removed by regex, but handle common resulting strings)
    '(1)': 'maj',       # Root only -> Major
    '(1,5)': '5',       # Power Chord
    '(1,3,5)': 'maj',
    '(1,b3,5)': 'min',
    '(1,b3,b5)': 'dim',
    '(1,3,#5)': 'aug',
    '(1,2,5)': 'sus2',
    '(1,4,5)': 'sus4',
    '(1,b7)': '7',      # Root + 7th -> Dominant 7
    '(1,4)': 'sus4',    # Root + 4th -> Sus4
    '(1,b3)': 'min',    # Root + min 3rd -> Minor
    '(b3,5)': 'min',    # Implied root minor triad
    '(1,4,b7)': 'sus4', # Dominant sus
    '(1,b3,4)': 'min',  # Complex minor variant
    '(1,4,b5)': 'dim',  # Complex diminished variant
    '(1,2,5,b6)': 'min6',# Complex min6 variant
    '(1,2,4)': 'sus2',  # Complex sus variant
    '(1,b3,b5,6)': 'min6', # Explicit intervals for min6addb5? Treat as min6.
    '(6)': 'maj6',      # Assume major for single interval 6
    '(7)': 'maj7',      # Assume major for single interval 7

    # Ensure target qualities map to themselves
    **{q: q for q in ['maj', 'min', 'dim', 'aug', 'maj7', 'min7', '7', 'dim7', 'hdim7',
                      'minmaj7', 'maj6', 'min6', 'sus2', 'sus4', '5', 'aug7',
                      # Add extended qualities that INTERVAL_SETS_TO_QUALITY might produce
                      'maj9', 'min9', '9', 'maj11', 'min11', '11', 'maj13', 'min13', '13']}
}


# Simplification map for reducing vocabulary further if needed (used in simplify_chord)
QUALITY_SIMPLIFY_MAP = {
    'maj9': 'maj7', 'maj11': 'maj7', 'maj13': 'maj7',
    '9': '7', '11': '7', '13': '7',
    'min9': 'min7', 'min11': 'min7', 'min13': 'min7',
    'minmaj7': 'min7', # Simplify minmaj7 to min7
    'aug7': '7', # Simplify aug7 to 7
    'maj6': 'maj', # Simplify maj6 to maj
    'min6': 'min', # Simplify min6 to min
    'sus2': 'sus4', # Group sus chords
    'hdim7': 'dim7', # Group diminished chords
    'dim': 'dim7',
    'aug': '7', # Group augmented with dominant
    '5': 'maj', # Power chord as major
    # Keep core qualities: maj, min, 7, maj7, min7, dim7, sus4
}

# Define standard quality categories for reporting
QUALITY_CATEGORIES = {
    "maj": "Major", "": "Major", "M": "Major",
    "min": "Minor", "m": "Minor", "-": "Minor", "mi": "Minor",
    "7": "Dom7", "dom7": "Dom7", "dominant": "Dom7", "alt": "Dom7",
    "maj7": "Maj7", "M7": "Maj7", "major7": "Maj7", "^": "Maj7", "^7": "Maj7",
    "min7": "Min7", "m7": "Min7", "minor7": "Min7", "-7": "Min7",
    "dim": "Dim", "°": "Dim", "o": "Dim", "diminished": "Dim",
    "dim7": "Dim7", "°7": "Dim7", "o7": "Dim7", "diminished7": "Dim7",
    "hdim7": "Half-Dim", "m7b5": "Half-Dim", "ø": "Half-Dim", "half-diminished": "Half-Dim",
    "aug": "Aug", "+": "Aug", "augmented": "Aug",
    "sus2": "Sus", "sus4": "Sus", "sus": "Sus", "suspended": "Sus",
    "min6": "Min6", "m6": "Min6", "-6": "Min6",
    "maj6": "Maj6", "6": "Maj6", "M6": "Maj6",
    "minmaj7": "Min-Maj7", "mmaj7": "Min-Maj7", "min-maj7": "Min-Maj7",
    "N": "No Chord", "X": "No Chord", # Map X to No Chord for reporting consistency
    "5": "Other", # Power chords
    "1": "Other", # Root only
    "aug7": "Other",
    # Default
    "Other": "Other"
}


# Helper function for modifying pitch based on accidentals (standalone)
def _modify_pitch_standalone(base_pitch, modifier_str):
    """
    Modifies a base pitch integer by a series of accidentals.
    Args:
        base_pitch (int): The base pitch class (0-11).
        modifier_str (str): String of modifiers (e.g., "b", "##", "bb").
    Returns:
        int: The modified pitch class (0-11).
    """
    pitch = base_pitch
    for mod_char in modifier_str:
        if mod_char == 'b':
            pitch -= 1
        elif mod_char == '#':
            pitch += 1
    return pitch % 12

# Helper function to parse a single interval string to an integer (0-11)
def _parse_single_interval_to_int(interval_str):
    """
    Parses a single interval string (e.g., "b3", "5", "#5", "11") to a pitch class integer.
    Assumes root is 0.
    Args:
        interval_str (str): The interval string.
    Returns:
        int or None: The pitch class integer (0-11) or None if parsing fails.
    """
    interval_str = interval_str.strip()
    if not interval_str:
        return None

    # Handle special cases like 'bb7' first if necessary, or rely on _CHROMA_ID and _modify_pitch_standalone
    # For 'bb7': base is 7th (_CHROMA_ID[6]), modifier is 'bb'
    
    digit_part_str = ""
    modifier_str = ""
    
    for i, char_in_interval in enumerate(interval_str):
        if char_in_interval.isdigit():
            modifier_str = interval_str[:i]
            digit_part_str = interval_str[i:]
            break
    else: # No digit found, maybe just modifier (e.g. 'b') or invalid
        if not interval_str.isalpha(): # e.g. just '#' or 'b'
             logger.debug(f"Invalid interval string (no digit): {interval_str}")
             return None # Or handle as alteration of root? For now, invalid.
        # If it's all alpha, it might be a quality string itself, not an interval
        return None


    try:
        digit_part = int(digit_part_str)
        if not (1 <= digit_part <= 14): # _CHROMA_ID has 14 elements
            logger.debug(f"Interval digit out of range (1-14): {digit_part} in '{interval_str}'")
            return None
        
        # _CHROMA_ID gives major/perfect interval in semitones from C=0
        # For interval '1', index is 0. For '14', index is 13.
        base_interval_pitch = _CHROMA_ID[digit_part - 1]
        
        # Apply modifiers and take modulo 12 for pitch class
        return _modify_pitch_standalone(base_interval_pitch, modifier_str)
    except (ValueError, IndexError) as e:
        logger.debug(f"Error parsing interval string '{interval_str}': {e}")
        return None

# Map of interval sets (frozenset of pitch classes relative to root 0) to quality strings
INTERVAL_SETS_TO_QUALITY = {
    # Triads
    frozenset({0, 4, 7}): "maj",
    frozenset({0, 3, 7}): "min",
    frozenset({0, 3, 6}): "dim",
    frozenset({0, 4, 8}): "aug",
    # Sevenths
    frozenset({0, 4, 7, 11}): "maj7",
    frozenset({0, 3, 7, 10}): "min7",
    frozenset({0, 4, 7, 10}): "7",
    frozenset({0, 3, 6, 9}): "dim7",  # bb7 is 9
    frozenset({0, 3, 6, 10}): "hdim7", # m7b5
    # Suspended
    frozenset({0, 5, 7}): "sus4", # 1, 4, 5
    frozenset({0, 2, 7}): "sus2", # 1, 2, 5
    # Power Chord
    frozenset({0, 7}): "5",
    # Extended Chords (examples)
    # Ninths (Major 9th is interval class 2)
    frozenset({0, 4, 7, 11, 2}): "maj9", # 1, 3, 5, 7, 9
    frozenset({0, 3, 7, 10, 2}): "min9", # 1, b3, 5, b7, 9
    frozenset({0, 4, 7, 10, 2}): "9",    # 1, 3, 5, b7, 9
    # Elevenths (Perfect 11th is interval class 5)
    frozenset({0, 4, 7, 11, 2, 5}): "maj11", # 1, 3, 5, 7, 9, 11
    frozenset({0, 3, 7, 10, 2, 5}): "min11", # 1, b3, 5, b7, 9, 11
    frozenset({0, 4, 7, 10, 2, 5}): "11",    # 1, 3, 5, b7, 9, 11
    # For the specific example F:(b3,5,b7,11) -> intervals {b3,5,b7,11} -> pcs {0,3,7,10,5}
    frozenset({0, 3, 7, 10, 5}): "min11", # Root, b3, 5, b7, 11(pc 5)
    frozenset({0, 4, 7, 10, 5}): "11",    # Root, 3, 5, b7, 11(pc 5) - dominant 11th
    frozenset({0, 4, 7, 11, 5}): "maj11", # Root, 3, 5, maj7, 11(pc 5) - major 11th

    # More specific extended chords if needed, ensure they map to something in QUALITY_NORM_MAP
    # Example: (b3,5,b7,9,11) -> {0,3,7,10,5,9} -> "min11" (already covered by QUALITY_NORM_MAP)
}

# Helper function to parse an interval list string (e.g., "(b3,5,b7)") to a quality string
def _parse_intervals_to_quality(intervals_str):
    """
    Parses an interval list string (e.g., "(b3,5,b7,11)") into a set of pitch classes
    and attempts to map it to a chord quality string using INTERVAL_SETS_TO_QUALITY.
    Args:
        intervals_str (str): The interval list string, e.g., "(b3,5,b7,11)".
    Returns:
        str or None: The determined quality string (e.g., "min11", "maj7") or None if no match.
    """
    if not (intervals_str.startswith('(') and intervals_str.endswith(')')):
        return None

    content = intervals_str[1:-1].strip()
    if not content: # Empty parentheses like "()"
        return None # Or map to "maj"? For now, None.

    interval_parts = [part.strip() for part in content.split(',')]
    
    pitch_classes = {0} # Always include the root (0)
    valid_intervals_found = False
    for part in interval_parts:
        if not part: continue # Skip empty parts from multiple commas e.g. (b3,,5)
        pc = _parse_single_interval_to_int(part)
        if pc is not None:
            pitch_classes.add(pc)
            valid_intervals_found = True
        else:
            # If any part fails to parse as an interval, the whole string is suspect
            logger.debug(f"Failed to parse interval part '{part}' in list '{intervals_str}'")
            return None 
    
    if not valid_intervals_found and len(pitch_classes) == 1: # Only root, no other valid intervals
        return None

    # Try to match the full set of pitch classes
    # For more robustness, could try matching subsets if full set fails,
    # e.g., check for 7th, then triad if 7th fails.
    # For now, direct match on the generated frozenset.
    current_fset = frozenset(pitch_classes)
    
    # Iterate through INTERVAL_SETS_TO_QUALITY by decreasing size of interval set for best match
    # This helps match e.g. min11 before min7 if both are subsets.
    # However, a direct lookup is simpler if the map is comprehensive for expected inputs.
    # Let's try direct lookup first. If it becomes an issue, can implement subset matching.
    
    # A more robust matching: iterate defined qualities from most complex to simplest
    # This is complex. Let's rely on a good INTERVAL_SETS_TO_QUALITY map.
    # If the exact set is in the map, use it.
    if current_fset in INTERVAL_SETS_TO_QUALITY:
        return INTERVAL_SETS_TO_QUALITY[current_fset]

    # Fallback: if no exact match, try to find the largest subset that matches.
    # This is to handle cases like (1,b3,5,b7,9,11,13) -> min13, but if only min7 is defined, it should match min7.
    # For simplicity, we'll require an exact match for now. The map should be reasonably complete for common interval lists.
    # If more advanced matching is needed, this part can be expanded.
    # Example: (b3,5,b7,11,13) -> {0,3,7,10,5,9} -> "min13"
    # If "min13" is not in map, but "min11" ({0,3,7,10,5}) is, should it match "min11"?
    # This requires careful definition of "best match".
    # For now, if no exact match, return None.
    
    logger.debug(f"Interval set {current_fset} from '{intervals_str}' not found in INTERVAL_SETS_TO_QUALITY.")
    return None


def _parse_root(root_str):
    """Internal helper to parse root string to integer, handling enharmonics."""
    if not root_str or not root_str[0].isalpha():
        raise ValueError(f"Invalid root note: {root_str}")

    # Handle explicit enharmonic notations first (e.g., B#, Cb) using full map
    if root_str in ENHARMONIC_MAP:
        # Use the mapped value (e.g., B# -> C) then parse 'C'
        root_str = ENHARMONIC_MAP[root_str]

    # Standard parsing
    base_char = root_str[0].upper()
    modifier = root_str[1:]
    if base_char < 'A' or base_char > 'G':
         raise ValueError(f"Invalid root note base: {base_char}")

    # Calculate base pitch relative to C=0
    # C=0, D=2, E=4, F=5, G=7, A=9, B=11
    base_offset = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    base_pitch = base_offset[base_char]

    # Apply modifiers (#, b)
    return _modify_pitch_standalone(base_pitch, modifier) # Use standalone and it already does % 12


def _parse_chord_string(chord_label):
    """
    Internal helper to parse a chord label into root, quality, and bass.
    Applies cleaning and normalization.

    Args:
        chord_label (str): Raw chord label.

    Returns:
        tuple: (root_str, quality_str, bass_str) or ("N", None, None) / ("X", None, None).
               Returns standardized strings (e.g., root normalized, quality normalized).
               Returns None for components if parsing fails and results in "X".
    """
    if not chord_label or chord_label.strip().upper() in ["N", "NC"]:
        return "N", None, None
    if chord_label.strip().upper() == "X":
        return "X", None, None

    # Basic cleaning
    chord_label = chord_label.strip()

    # --- Fix common specific issues first ---
    # These often break general parsing rules
    specific_fixes = {
        'Emin/4': 'E:min/4', 'A7/3': 'A:7/3', 'Bb7/3': 'Bb:7/3', 'Bb7/5': 'Bb:7/5',
        # Add more specific known issues if needed
    }
    if chord_label in specific_fixes:
        chord_label = specific_fixes[chord_label]

    root_str, quality_str, bass_str = None, None, None
    chord_part = chord_label # Part representing the chord itself (root+quality)

    # 1. Split bass note first
    if '/' in chord_label:
        parts = chord_label.split('/', 1)
        if len(parts) == 2 and parts[1]: # Ensure bass part is not empty
            chord_part = parts[0].strip()
            bass_str = parts[1].strip()
            # Try parsing bass as pitch class, otherwise keep as string (scale degree)
            try:
                bass_int = _parse_root(bass_str)
                # Normalize bass pitch class spelling like the root
                default_bass_name = PITCH_CLASS[bass_int]
                bass_str = PREFERRED_SPELLING_MAP.get(default_bass_name, default_bass_name)
            except ValueError:
                # Keep bass_str as is (e.g., '3', 'b7', '#4', '9')
                # Validate if it looks like a scale degree
                if not re.match(r'^[#b]?\d+$', bass_str):
                     logger.debug(f"Invalid bass note string '{bass_str}' derived from '{chord_label}'. Ignoring bass.")
                     bass_str = None # Treat as root position if bass is invalid string
        else: # Malformed bass? Ignore it. Treat as root position.
            chord_part = chord_label.replace('/', '').strip()
            bass_str = None # Explicitly set bass to None

    # Handle case where chord_part might be empty after splitting bass
    if not chord_part:
         logger.debug(f"Empty chord part after splitting bass from '{chord_label}'")
         return "X", None, None

    # 2. Split root and quality
    if ':' in chord_part:
        parts = chord_part.split(':', 1)
        if len(parts) == 2:
            root_str = parts[0].strip()
            quality_str = parts[1].strip()
            if not root_str: # Handle labels like ":min"
                 logger.debug(f"Missing root in colon notation: '{chord_label}'")
                 return "X", None, None
        else: # Malformed colon? Treat as root only.
            root_str = chord_part.replace(':', '').strip()
            quality_str = '' # Assume major if only root is given
    else:
        # No colon, need to find split point using regex
        # Match root note (A-G, optional #/b) at the beginning
        match = re.match(r'^([A-G][#b]?)', chord_part)
        if match:
            root_str = match.group(1)
            quality_str = chord_part[len(root_str):].strip()
            # If quality_str is empty here, it means only root was provided (e.g., "C", "Bb")
        else:
            # Cannot identify a valid root note at the start
            logger.debug(f"Could not parse root from: {chord_part} in label {chord_label}")
            return "X", None, None # Treat as unknown if root is unparseable

    # 3. Normalize Root
    try:
        root_int = _parse_root(root_str) # Handles B#, Cb etc via _parse_root's use of ENHARMONIC_MAP
        # Get the default pitch class name (preferring sharps for C#, D#, F#, G#, A#)
        default_root_name = PITCH_CLASS[root_int]
        # Use preferred spelling if available (e.g., A# -> Bb), otherwise use default
        root_str = PREFERRED_SPELLING_MAP.get(default_root_name, default_root_name)

    except ValueError as e:
        logger.debug(f"Error parsing root '{root_str}' from label '{chord_label}': {e}")
        return "X", None, None # Treat as unknown

    # 4. Normalize Quality
    # Remove parentheses content for normalization (e.g., maj(9) -> maj)
    # This simplifies handling of complex MIREX notations like * and alterations within ()
    original_quality_str = quality_str # Keep original for logging if needed

    # Check if quality_str is an explicit interval list like (b3,5,b7)
    # It must start with '(', end with ')', and contain at least one comma if multiple intervals.
    # Single interval like (b3) won't have a comma but should still be parsed if we want.
    # For now, align with prompt: "contains commas like (b3,5,b7,11)"
    # A more general check for "is it purely an interval list?" could be:
    # quality_str.startswith('(') and quality_str.endswith(')') and not any(c.isalpha() for c in quality_str[1:-1] if c not in 'b#(),')
    
    is_interval_list_notation = False
    if quality_str.startswith('(') and quality_str.endswith(')'):
        # Check if content looks like intervals rather than text e.g. (maj)
        # A simple check: contains digits or 'b'/'#' and not too many letters.
        # Or, more strictly, require a comma for multi-interval lists as per prompt.
        # If it's like "(b3,5)", it should be parsed.
        # If it's like "(maj)", it should not.
        # If it's like "(9)", it could be an interval.
        # Let's use the comma criteria from the prompt for now for multi-item lists.
        # For single items like "(b3)", _parse_intervals_to_quality handles it if no comma.
        # The key is to distinguish "(b3,5)" from "(add9)".
        
        # If it's purely parenthesized, try parsing as intervals.
        # The _parse_intervals_to_quality function will return None if it's not a valid list.
        parsed_quality_from_intervals = _parse_intervals_to_quality(quality_str)
        if parsed_quality_from_intervals:
            quality_str = parsed_quality_from_intervals
            is_interval_list_notation = True # Mark that we used this path
            logger.debug(f"Parsed interval list '{original_quality_str}' to quality '{quality_str}' for label '{chord_label}'")
        # else: it was parenthesized but not a valid interval list (e.g. "(maj)", "(nonsense)")
        # or parsing failed, so it will fall through to re.sub.

    if not is_interval_list_notation:
        # If not parsed as an interval list, apply standard regex stripping for parentheses
        # This handles cases like "maj(9)" -> "maj", or if "(b3,5)" failed interval parsing.
        quality_str = re.sub(r'\(.*\)', '', quality_str).strip()


    # Apply standard quality mappings
    normalized_quality = QUALITY_NORM_MAP.get(quality_str, quality_str)

    # Handle empty quality -> 'maj' AFTER normalization lookup
    # This covers cases like "C", "C/5", "C:", "C:M" -> 'maj'
    if not quality_str or normalized_quality == '':
        normalized_quality = 'maj'

    # If after all this, the quality is still not in our known list, map to 'X' or keep?
    # Let's map unknown qualities to 'maj' for now, assuming it's a typo or simple chord.
    # A stricter approach would map to 'X'.
    if normalized_quality not in QUALITY_NORM_MAP.values():
         logger.debug(f"Unknown quality '{original_quality_str}' (parsed as '{quality_str}', normalized to '{normalized_quality}') in label '{chord_label}'. Mapping to 'maj'.")
         normalized_quality = 'maj' # Default unknown qualities to major

    return root_str, normalized_quality, bass_str


def get_chord_quality(chord_label):
    """
    Extract the quality from a chord label and map it to a standard category.
    (Static/Standalone version)

    Args:
        chord_label (str): Chord label like "C:maj", "G:7", etc.

    Returns:
        str: Standardized chord quality category (e.g., "Major", "Minor", "Dom7").
    """
    root, quality, bass = _parse_chord_string(chord_label)

    if root == "N" or root == "X":
        return "No Chord"
    if quality is None:
        # This case should ideally not happen if _parse_chord_string returns 'maj' for unknowns
        logger.warning(f"Parsed quality is None for label '{chord_label}'. Returning 'Other'.")
        return "Other"

    # Map the parsed quality string to a reporting category
    return QUALITY_CATEGORIES.get(quality, "Other")


class Chords:
    def __init__(self):
        # Shorthands remain useful for interval calculations if needed
        self._shorthands = {
            'maj': self.interval_list('(1,3,5)'),
            'min': self.interval_list('(1,b3,5)'),
            'dim': self.interval_list('(1,b3,b5)'),
            'aug': self.interval_list('(1,3,#5)'),
            'maj7': self.interval_list('(1,3,5,7)'),
            'min7': self.interval_list('(1,b3,5,b7)'),
            '7': self.interval_list('(1,3,5,b7)'),
            # '6': self.interval_list('(1,6)'), # Ambiguous, use maj6
            '5': self.interval_list('(1,5)'),
            # '4': self.interval_list('(1,4)'), # Ambiguous
            '1': self.interval_list('(1)'),
            'dim7': self.interval_list('(1,b3,b5,bb7)'),
            'hdim7': self.interval_list('(1,b3,b5,b7)'),
            'minmaj7': self.interval_list('(1,b3,5,7)'),
            'maj6': self.interval_list('(1,3,5,6)'),
            'min6': self.interval_list('(1,b3,5,6)'),
            # '9': self.interval_list('(1,3,5,b7,9)'), # Covered by '7' in normalization
            # 'maj9': self.interval_list('(1,3,5,7,9)'), # Covered by 'maj7'
            # 'min9': self.interval_list('(1,b3,5,b7,9)'), # Covered by 'min7'
            'sus2': self.interval_list('(1,2,5)'),
            'sus4': self.interval_list('(1,4,5)'),
            'aug7': self.interval_list('(1,3,#5,b7)'), # Added for completeness if needed
        }
        # Initialize chord mapping dictionary
        self.chord_mapping = {}

    # --- Interval/Pitch calculation methods (Keep if reduce_to_triads is used) ---
    def _modify(self, base_pitch, modifier):
        """Internal helper - kept for interval calculations."""
        return _modify_pitch_standalone(base_pitch, modifier)

    def interval(self, interval_str):
        """Kept for interval calculations."""
        interval_str = interval_str.strip()
        if not interval_str: return 0 # Handle empty string case
        
        # Use the standalone parser for consistency if desired, or keep this specialized one.
        # For now, keeping this as it's part of the class's original structure for interval math.
        # The new _parse_single_interval_to_int is for quality string parsing.
        # This method returns the raw semitone count, not % 12.
        
        # Handle special cases like 'bb7'
        if interval_str == 'bb7':
            # _CHROMA_ID[6] is 7th (11 semitones). 'bb' means -2. So 9.
            return self._modify(_CHROMA_ID[7 - 1], 'bb') # Original logic was modulo 12, but interval() seems to return raw sometimes.
                                                        # Let's ensure it returns raw semitones for _shorthands.
                                                        # _shorthands expect pitch class bitmasks, so %12 is needed eventually.
                                                        # The Chords.interval_list applies %12 via self.interval() then uses it as index.
                                                        # Let's make Chords.interval return pitch class (0-11) for consistency.
                                                        # This means _shorthands might need adjustment if they relied on raw semitones > 11.
                                                        # However, _shorthands build a 12-element array, so %12 is implicit.

        for i, c in enumerate(interval_str):
            if c.isdigit():
                try:
                    digit_part = int(interval_str[i:])
                    if digit_part == 0: return 0 # Interval '0' is unison
                    # Use module-level constant _CHROMA_ID
                    base_interval_pitch = _CHROMA_ID[digit_part - 1]
                    modifier = interval_str[:i]
                    return self._modify(base_interval_pitch, modifier) % 12
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid interval string: {interval_str}")
        # If no digit found, it might be just modifiers (e.g., 'b'), treat as error or default
        raise ValueError(f"Invalid interval string (no digit): {interval_str}")


    def interval_list(self, intervals_str, given_pitch_classes=None):
        """Kept for interval calculations."""
        if given_pitch_classes is None:
            given_pitch_classes = np.zeros(12, dtype=np.int32)
        # Handle edge case of empty interval string "()"
        if intervals_str == '()':
            return given_pitch_classes
        try:
            for int_def in intervals_str[1:-1].split(','):
                int_def = int_def.strip()
                if not int_def: continue # Skip empty parts
                if int_def[0] == '*':
                    # Remove the interval
                    interval_to_remove = self.interval(int_def[1:])
                    given_pitch_classes[interval_to_remove] = 0
                else:
                    # Add the interval
                    interval_to_add = self.interval(int_def) # This now returns 0-11
                    given_pitch_classes[interval_to_add] = 1
            return given_pitch_classes
        except Exception as e:
            logger.error(f"Error parsing interval list '{intervals_str}': {e}")
            # Return the array as is or raise error? Returning as is might hide issues.
            raise ValueError(f"Failed to parse interval list: {intervals_str}") from e


    def chord_intervals(self, quality_str):
        """Kept for interval calculations."""
        # Use normalized quality string for lookup/parsing
        norm_quality = QUALITY_NORM_MAP.get(quality_str, quality_str)
        if norm_quality == 'maj': norm_quality = 'maj' # Ensure empty maps to maj shorthand
        # Ensure norm_quality itself is a key in _shorthands or parsable as interval list
        # This part might need review if norm_quality can be complex after new parsing.
        # For now, assume it's a basic quality string or direct interval list.

        list_idx = norm_quality.find('(') # Check normalized quality for interval lists
        if list_idx == -1:
            # Direct shorthand lookup
            if norm_quality in self._shorthands:
                return self._shorthands[norm_quality].copy()
            else:
                # Attempt to parse as interval list directly if format matches, e.g., "(1,3,5)"
                # This path is less likely now with extensive QUALITY_NORM_MAP
                if norm_quality.startswith('(') and norm_quality.endswith(')'):
                    try:
                        return self.interval_list(norm_quality)
                    except ValueError:
                        raise ValueError(f"Unknown quality shorthand and invalid interval list: {norm_quality} (from original {quality_str})")
                else:
                     raise ValueError(f"Unknown quality shorthand: {norm_quality} (from original {quality_str})")
        else:
            # Combined shorthand and interval list (less likely needed now)
            base_quality = norm_quality[:list_idx]
            interval_part = norm_quality[list_idx:]
            if base_quality:
                if base_quality in self._shorthands:
                    ivs = self._shorthands[base_quality].copy()
                else:
                    # Try normalizing the base part again?
                    norm_base_quality = QUALITY_NORM_MAP.get(base_quality, base_quality)
                    if norm_base_quality in self._shorthands:
                         ivs = self._shorthands[norm_base_quality].copy()
                    else:
                         raise ValueError(f"Unknown base quality shorthand: {base_quality} in {norm_quality}")
            else:
                ivs = np.zeros(12, dtype=np.int32)

            return self.interval_list(interval_part, ivs)

    # --- Core Public Methods ---

    def label_error_modify(self, label):
        """
        Correct common errors and inconsistencies in chord label format.
        Uses the centralized parser and reconstructs the label in mir_eval format.

        Parameters:
            label (str): Chord label to correct.

        Returns:
            str: Corrected and normalized chord label (e.g., "C", "C:min", "Bb:7/3").
        """
        root, quality, bass = _parse_chord_string(label)

        if root == "N" or root == "X":
            return root
        if root is None: # Parsing failed completely
             logger.warning(f"Complete parsing failure for label '{label}'. Returning 'X'.")
             return "X" # Return unknown

        # Reconstruct the normalized label using mir_eval standard
        # Omit ':maj' for major chords.
        quality_str = quality if quality and quality != 'maj' else None

        if quality_str:
            corrected_label = f"{root}:{quality_str}"
        else:
            corrected_label = root # Just root for major

        if bass:
            # Append bass note (already normalized pitch class or original scale degree)
            corrected_label += f"/{bass}"

        return corrected_label

    def normalize_chord(self, chord_name):
        """
        Normalize a chord name to a canonical form (Root[:Quality][/Bass]).
        Uses the centralized parser. Quality is omitted for major chords.

        Args:
            chord_name (str): Input chord name.

        Returns:
            tuple: (normalized_chord_str, root_str, quality_str, bass_str)
                   Returns ("N", "N", None, None) or ("X", "X", None, None) for N/X.
                   Returns ("X", None, None, None) if parsing fails.
        """
        root, quality, bass = _parse_chord_string(chord_name)

        if root == "N" or root == "X":
            return root, root, None, None
        if root is None:
            logger.warning(f"Complete parsing failure for label '{chord_name}' in normalize_chord. Returning 'X'.")
            return "X", None, None, None # Parsing failed

        # Reconstruct the normalized label string (mir_eval standard)
        quality_str_for_label = quality if quality and quality != 'maj' else None

        if quality_str_for_label:
            normalized_label = f"{root}:{quality_str_for_label}"
        else:
            normalized_label = root # Just root for major

        if bass:
            normalized_label += f"/{bass}" # Append the parsed bass string

        return normalized_label, root, quality, bass


    def simplify_chord(self, chord_name):
        """
        Simplify a chord to its basic root position form using QUALITY_SIMPLIFY_MAP.
        Uses the centralized parser first.

        Args:
            chord_name (str): The chord name to simplify.

        Returns:
            str: Simplified chord name (Root:Quality or Root for major).
                 Returns "N" or "X" for no chord/unknown.
        """
        root, quality, bass = _parse_chord_string(chord_name)

        if root == "N" or root == "X":
            return root
        if root is None:
             logger.warning(f"Complete parsing failure for label '{chord_name}' in simplify_chord. Returning 'X'.")
             return "X" # Parsing failed

        # Apply simplification map to the *normalized* quality
        simplified_quality = QUALITY_SIMPLIFY_MAP.get(quality, quality)

        # Further simplify based on target categories if needed (e.g., map all dim to dim7)
        # This step depends on the desired level of simplification
        simplified_quality = QUALITY_SIMPLIFY_MAP.get(simplified_quality, simplified_quality) # Apply again if needed

        # Reconstruct simplified chord name (root position, mir_eval standard)
        if simplified_quality == 'maj':
            return root # Just root for major
        else:
            # Ensure the simplified quality is valid before formatting
            if simplified_quality in QUALITY_CATEGORIES: # Check against known categories/qualities
                 return f"{root}:{simplified_quality}"
            else:
                 logger.warning(f"Unknown simplified quality '{simplified_quality}' for chord '{chord_name}'. Defaulting to root '{root}'.")
                 return root # Fallback to just root if simplified quality is weird


    def set_chord_mapping(self, chord_mapping):
        """
        Set the chord mapping to be used for chord index lookup.
        Also initializes extended mappings for variants.

        Args:
            chord_mapping (dict): Dictionary mapping chord names to indices
        """
        if not chord_mapping or not isinstance(chord_mapping, dict):
            logger.warning("Invalid or empty chord mapping provided. Chord lookup will fail.")
            self.chord_mapping = {}
            self.base_mapping = {}
            return

        # Store a copy of the original mapping
        self.base_mapping = chord_mapping.copy()
        logger.info(f"Base chord mapping set with {len(self.base_mapping)} entries")

        # Initialize extended mappings (inversions, variants, etc.)
        self.initialize_chord_mapping()

    def initialize_chord_mapping(self):
        """
        Initialize the full chord mapping including normalized, simplified,
        enharmonic, and root-position variants based on self.base_mapping.
        """
        if not hasattr(self, 'base_mapping') or not self.base_mapping:
            logger.error("Cannot initialize chord mapping: No base mapping available.")
            self.chord_mapping = {}
            return

        # Start with a fresh copy of the base mapping
        self.chord_mapping = self.base_mapping.copy()
        extended_mapping = {}
        logger.info(f"Initializing full chord mapping from {len(self.chord_mapping)} base entries.")

        processed_indices = set() # Keep track of indices already processed

        for base_chord, base_idx in self.base_mapping.items():
            # If we've already generated variants for this index, skip
            if base_idx in processed_indices:
                continue

            variants = set()
            # Add the original base chord
            variants.add(base_chord)

            try:
                # Get normalized components (handles N/X correctly)
                norm_label, norm_root, norm_quality, norm_bass = self.normalize_chord(base_chord)

                if norm_root == "N" or norm_root == "X":
                    variants.add(norm_root) # Add N or X directly
                elif norm_root and norm_quality: # Ensure parsing was successful
                    # Add normalized label (e.g., C:min/G, Bb:7/D, A)
                    variants.add(norm_label)

                    # Add normalized root position (e.g., C:min, Bb:7, A)
                    norm_root_pos_qual_str = norm_quality if norm_quality != 'maj' else None
                    if norm_root_pos_qual_str:
                        norm_root_pos = f"{norm_root}:{norm_root_pos_qual_str}"
                    else:
                        norm_root_pos = norm_root
                    variants.add(norm_root_pos)

                    # Get simplified root position (e.g., C:min -> C:min, Bb:7 -> Bb:7, A -> A)
                    # Simplify the *normalized root position* chord
                    simplified_chord = self.simplify_chord(norm_root_pos)
                    variants.add(simplified_chord)

                    # --- Generate enharmonic variants ---
                    # Only generate if the root has an enharmonic equivalent
                    if norm_root in ENHARMONIC_MAP:
                        enh_root = ENHARMONIC_MAP[norm_root]

                        # Enharmonic of normalized root position
                        if norm_root_pos_qual_str:
                            enh_norm_root_pos = f"{enh_root}:{norm_root_pos_qual_str}"
                        else:
                            enh_norm_root_pos = enh_root
                        variants.add(enh_norm_root_pos)

                        # Enharmonic of simplified root position
                        # Re-parse the simplified chord to get its root and quality
                        simp_root_enh, simp_qual_enh, _ = _parse_chord_string(simplified_chord)
                        # Check if the simplified root is the same as the normalized root
                        # (it might change if simplification maps aug->7 etc.)
                        if simp_root_enh == norm_root:
                             simp_qual_enh_str = simp_qual_enh if simp_qual_enh and simp_qual_enh != 'maj' else None
                             if simp_qual_enh_str:
                                 enh_simplified = f"{enh_root}:{simp_qual_enh_str}"
                             else:
                                 enh_simplified = enh_root
                             variants.add(enh_simplified)
                        # else: Simplified root differs, don't create enharmonic based on original root


            except Exception as e:
                logger.warning(f"Error generating variants for base chord '{base_chord}' (idx {base_idx}): {e}", exc_info=True)

            # Add all generated variants to the extended mapping, pointing to the base index
            for variant in variants:
                # Only add if not already present with a *different* index
                if variant and variant not in self.chord_mapping:
                    extended_mapping[variant] = base_idx
                # Optional: Check for conflicts where a variant maps to multiple base indices
                # elif variant and self.chord_mapping[variant] != base_idx:
                #     logger.warning(f"Mapping conflict: Variant '{variant}' from base '{base_chord}' (idx {base_idx}) already maps to index {self.chord_mapping[variant]}. Keeping existing mapping.")


            processed_indices.add(base_idx) # Mark this index as done

        # Update the main mapping
        original_size = len(self.chord_mapping)
        # Prioritize existing base_mapping entries over newly generated extended_mapping entries in case of conflict
        final_mapping = extended_mapping.copy()
        final_mapping.update(self.chord_mapping) # Overwrite extended with base if keys overlap
        self.chord_mapping = final_mapping

        added_count = len(self.chord_mapping) - len(self.base_mapping) # Count added vs original base
        logger.info(f"Enhanced chord mapping with {added_count} variants (total: {len(self.chord_mapping)})")

        # --- BEGIN ADDED LOGGING ---
        # Group the final mapping by index to show all variants mapping to the same index
        index_to_variants = defaultdict(set)
        for chord_str, index in self.chord_mapping.items():
            index_to_variants[index].add(chord_str)

        logger.debug("--- Final Chord Mapping (Grouped by Index) ---")
        # Sort by index for clarity
        for index in sorted(index_to_variants.keys()):
            variants_list = sorted(list(index_to_variants[index]))
            # Log only if there are multiple variants or if it's N/X for clarity
            if len(variants_list) > 1 or index >= (len(self.base_mapping) - 2): # Assuming N/X are last indices
                 logger.debug(f"Index {index}: {variants_list}")
            # Optionally log single-entry mappings too if needed for full picture
            # else:
            #     logger.debug(f"Index {index}: {variants_list}")
        logger.debug("--- End Final Chord Mapping ---")
        # --- END ADDED LOGGING ---

        # Log a few examples if needed
        # if added_count > 0:
        #     sample_added = {k: v for k, v in extended_mapping.items() if k not in self.base_mapping}
        #     logger.debug(f"Sample added mappings: {dict(list(sample_added.items())[:5])}")

    def get_chord_idx(self, chord_name, use_large_voca=True):
        """
        Get chord index using the pre-initialized comprehensive mapping.
        Normalizes the input chord name before lookup.
        Falls back to root position lookup if inversion is not found.

        Args:
            chord_name (str): Chord name to look up.
            use_large_voca (bool): Determines N/X index values.

        Returns:
            int: Chord index.
        """
        # Define N/X indices based on vocabulary size
        n_chord_idx = 169 if use_large_voca else 24
        x_chord_idx = 168 if use_large_voca else 25

        # 1. Handle N/X directly (using raw input)
        if not chord_name or chord_name.strip().upper() in ["N", "NC"]:
            return n_chord_idx
        if chord_name.strip().upper() == "X":
            return x_chord_idx

        # 2. Perform normalization using label_error_modify (which uses _parse_chord_string)
        # This produces the canonical mir_eval format string (e.g., "Bb:7/3", "C", "A:min")
        lookup_key = self.label_error_modify(chord_name)

        # Handle case where normalization itself failed and returned 'X'
        if lookup_key == "X":
            return x_chord_idx
        if lookup_key == "N": # Should not happen if input wasn't N, but handle defensively
            return n_chord_idx

        # 3. Direct lookup in the comprehensive mapping
        if hasattr(self, 'chord_mapping') and lookup_key in self.chord_mapping:
            return self.chord_mapping[lookup_key]
        else:
            # 4. Fallback: If direct lookup failed and it's an inversion, try root position
            root_pos_key = lookup_key
            if '/' in lookup_key:
                root_pos_key = lookup_key.split('/')[0]
                if root_pos_key != lookup_key and root_pos_key in self.chord_mapping:
                    # logger.debug(f"Chord '{chord_name}' (normalized to '{lookup_key}') not found, but root position '{root_pos_key}' found.")
                    return self.chord_mapping[root_pos_key]

            # 5. Fallback: Try looking up the simplified version of the root position key
            # Use root_pos_key here to simplify the root position chord
            simplified_key = self.simplify_chord(root_pos_key)
            # Check if simplification actually changed the key and if the simplified key exists
            if simplified_key != root_pos_key and simplified_key in self.chord_mapping:
                 # logger.debug(f"Chord '{chord_name}' (normalized root pos '{root_pos_key}') not found, but simplified '{simplified_key}' found.")
                 return self.chord_mapping[simplified_key]
            # Also check if the original simplified key (if different) exists, just in case
            elif simplified_key != lookup_key and simplified_key in self.chord_mapping:
                 # logger.debug(f"Chord '{chord_name}' (normalized '{lookup_key}') not found, but simplified '{simplified_key}' found.")
                 return self.chord_mapping[simplified_key]
            else:
                 # If still not found, return unknown index
                 # logger.debug(f"Chord '{chord_name}' (normalized to '{lookup_key}', root pos '{root_pos_key}', simplified to '{simplified_key}') not found in mapping. Returning unknown index {x_chord_idx}.")
                 return x_chord_idx

    # --- Kept for compatibility if reduce_to_triads is needed ---
    def reduce_to_triads(self, chords_struct_array, keep_bass=False):
        """
        Reduce chords (structured array) to triads.

        Parameters:
            chords_struct_array : numpy structured array (CHORD_DTYPE)
                Chords to be reduced. Requires fields 'root', 'bass', 'intervals'.
            keep_bass : bool
                Indicates whether to keep the bass note or set it to 0 (root).

        Returns:
            reduced_chords : numpy structured array
                Chords reduced to triads.
        """
        # This method requires input structured array generation, which is outside the scope
        # of the current refactoring focused on string parsing.
        # Keeping the logic but adding input validation.
        expected_fields = ('root', 'bass', 'intervals')
        if not isinstance(chords_struct_array, np.ndarray) or \
           not all(field in chords_struct_array.dtype.names for field in expected_fields):
             logger.error(f"reduce_to_triads requires input as numpy structured array with fields {expected_fields}.")
             raise ValueError("Invalid input format for reduce_to_triads")


        intervals = chords_struct_array['intervals']
        # Ensure intervals has 12 columns
        if intervals.shape[1] != 12:
             raise ValueError(f"Input 'intervals' field must have 12 columns, found {intervals.shape[1]}")

        unison = intervals[:, 0].astype(bool)
        maj_sec = intervals[:, 2].astype(bool)
        min_third = intervals[:, 3].astype(bool)
        maj_third = intervals[:, 4].astype(bool)
        perf_fourth = intervals[:, 5].astype(bool)
        dim_fifth = intervals[:, 6].astype(bool)
        perf_fifth = intervals[:, 7].astype(bool)
        aug_fifth = intervals[:, 8].astype(bool)
        # Check for NO_CHORD based on root = -1 (assuming this convention)
        no_chord = (chords_struct_array['root'] == -1)

        reduced_chords = chords_struct_array.copy()
        ivs = reduced_chords['intervals']

        # Reset non 'no_chord' intervals to root only initially
        # Ensure interval_list returns a compatible shape
        root_only_ivs = self.interval_list('(1)').reshape(1, -1)
        ivs[~no_chord] = np.tile(root_only_ivs, (np.sum(~no_chord), 1))


        # Apply reduction rules based on intervals present
        # Power chord
        is_power = unison & perf_fifth & ~min_third & ~maj_third & ~no_chord
        ivs[is_power] = np.tile(self._shorthands['5'].reshape(1, -1), (np.sum(is_power), 1))

        # Sus chords (handle carefully to avoid overwriting triads)
        # Ensure sus2 doesn't have 3rd/4th, sus4 doesn't have 2nd/3rd
        is_sus2 = unison & maj_sec & perf_fifth & ~min_third & ~maj_third & ~perf_fourth & ~no_chord
        is_sus4 = unison & perf_fourth & perf_fifth & ~maj_sec & ~min_third & ~maj_third & ~no_chord
        ivs[is_sus2] = np.tile(self._shorthands['sus2'].reshape(1, -1), (np.sum(is_sus2), 1))
        ivs[is_sus4] = np.tile(self._shorthands['sus4'].reshape(1, -1), (np.sum(is_sus4), 1))

        # Triads (apply after sus chords)
        # Basic major/minor (assuming perfect fifth if not dim/aug)
        is_min_triad = unison & min_third & perf_fifth & ~maj_third & ~no_chord
        is_maj_triad = unison & maj_third & perf_fifth & ~min_third & ~no_chord
        ivs[is_min_triad] = np.tile(self._shorthands['min'].reshape(1, -1), (np.sum(is_min_triad), 1))
        ivs[is_maj_triad] = np.tile(self._shorthands['maj'].reshape(1, -1), (np.sum(is_maj_triad), 1))

        # Diminished/Augmented (based on fifth)
        is_dim = unison & min_third & dim_fifth & ~perf_fifth & ~maj_third & ~no_chord
        is_aug = unison & maj_third & aug_fifth & ~perf_fifth & ~min_third & ~no_chord
        ivs[is_dim] = np.tile(self._shorthands['dim'].reshape(1, -1), (np.sum(is_dim), 1))
        ivs[is_aug] = np.tile(self._shorthands['aug'].reshape(1, -1), (np.sum(is_aug), 1))

        # Handle less common triads (e.g., maj-dim, min-aug) if needed, currently overwritten by dim/aug logic
        # is_min_aug = unison & min_third & aug_fifth & ~perf_fifth & ~maj_third & ~no_chord
        # is_maj_dim = unison & maj_third & dim_fifth & ~perf_fifth & ~min_third & ~no_chord
        # ivs[is_min_aug] = self.interval_list('(1,b3,#5)')
        # ivs[is_maj_dim] = self.interval_list('(1,3,b5)')


        # Handle bass note
        if not keep_bass:
            reduced_chords['bass'][~no_chord] = reduced_chords['root'][~no_chord] # Set bass to root for non 'no_chord'
        else:
            # Ensure bass note is valid for the reduced chord (relative to root=0)
            for i in range(len(reduced_chords)):
                if not no_chord[i]:
                    root_note = reduced_chords['root'][i]
                    bass_note = reduced_chords['bass'][i]
                    # Calculate bass interval relative to root
                    bass_interval = (bass_note - root_note) % 12
                    # If bass interval is not in the reduced intervals, set bass to root
                    if bass_interval < 0 or bass_interval >= 12 or ivs[i, bass_interval] == 0:
                        reduced_chords['bass'][i] = root_note

        # Ensure bass is -1 for no chords
        reduced_chords['bass'][no_chord] = -1
        # Update is_major flag based on reduced intervals (if field exists)
        if 'is_major' in reduced_chords.dtype.names:
             maj_pattern = self._shorthands['maj']
             reduced_chords['is_major'] = np.array([np.array_equal(row, maj_pattern) for row in ivs])


        return reduced_chords

    # --- Removed Legacy / Unused Methods ---
    # def pitch(self, pitch_str): ... # Removed, use _parse_root internally
    # def load_chords(self, filename): ... # Removed, had bug, unused
    # def convert_to_id(self, root, is_major): ... # Removed, legacy
    # def convert_to_id_voca(self, root, quality): ... # Removed, legacy, use get_chord_idx
    # def assign_chord_id(self, entry): ... # Removed, legacy
    # def get_converted_chord(self, filename): ... # Removed, legacy
    # def get_converted_chord_voca(self, filename): ... # Removed, legacy, buggy


# --- Standalone Function ---
def idx2voca_chord():
    """
    Create a mapping from chord index (0-169) to chord label for the large vocabulary.
    Uses standard mir_eval-compatible notations (e.g., C, G:min, D:7).
    Index 168 = X (Unknown), Index 169 = N (No Chord).
    """
    mapping = {}
    # Quality names in a standard order matching the typical 168-class setup
    # C:maj=0*14+1=1, C:min=0*14+0=0? No, usually Maj=0, Min=1? Let's verify common practice.
    # mir_eval uses alphabetical: aug, dim, hdim7, maj, maj6, maj7, min, min6, min7, minmaj7, sus2, sus4, 7, dim7?
    # Let's stick to the order implied by the original code if possible, assuming it matches model output.
    # Original order seems to be: min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4
    quality_list = [
        'min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7',
        'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4'
    ]
    num_qualities = len(quality_list) # Should be 14

    if num_qualities != 14:
        logger.warning(f"Expected 14 qualities for idx2voca_chord, found {num_qualities}. Mapping might be incorrect.")

    # Create the standard mapping
    for root_idx in range(12):
        # Use preferred spelling for root note
        default_root_name = PITCH_CLASS[root_idx]
        root_note = PREFERRED_SPELLING_MAP.get(default_root_name, default_root_name)

        for quality_idx in range(num_qualities):
            chord_idx = root_idx * num_qualities + quality_idx
            quality = quality_list[quality_idx]

            # Construct label: Root:Quality (omit :maj for major triads)
            if quality == 'maj':
                label = root_note
            else:
                label = f"{root_note}:{quality}"
            mapping[chord_idx] = label

    # Add special chords
    mapping[168] = "X"  # Unknown chord
    mapping[169] = "N"  # No chord

    # Verify expected size
    expected_size = 12 * num_qualities + 2
    if len(mapping) != expected_size:
         logger.warning(f"Generated idx2voca_chord mapping has {len(mapping)} entries, expected {expected_size}.")
         # Log missing/extra indices if needed for debugging
         # expected_indices = set(range(expected_size))
         # actual_indices = set(mapping.keys())
         # logger.debug(f"Missing indices: {expected_indices - actual_indices}")
         # logger.debug(f"Extra indices: {actual_indices - expected_indices}")


    return mapping
