# encoding: utf-8
"""
This module contains chord evaluation functionality and utilities.
"""

import numpy as np
import pandas as pd
import mir_eval
import logging
import re

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
# Reverse map for normalization (prefer sharps generally, but keep common flats)
ENHARMONIC_NORM_MAP = {
    'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#',
    'B#': 'C', 'Cb': 'B', 'E#': 'F', 'Fb': 'E',
}

QUALITY_NORM_MAP = {
    '': 'maj',          # Empty string means major chord
    ':': 'maj',         # Handle colon-only as major (or potentially error)
    'M': 'maj',         # Alternative notation for major
    'm': 'min',         # Alternative notation for minor
    'major': 'maj',     # Full word notation
    'minor': 'min',     # Full word notation
    'dom': '7',         # Dominant
    'ø': 'hdim7',       # Half-diminished symbol
    'o': 'dim',         # Diminished symbol
    '°': 'dim',         # Diminished symbol
    '+': 'aug',         # Augmented symbol
    'major-seventh': 'maj7',
    'minor-seventh': 'min7',
    'dominant-seventh': '7',
    'diminished-seventh': 'dim7',
    'augmented-seventh': 'aug7', # Often simplified
    'half-diminished-seventh': 'hdim7',
    'major-sixth': 'maj6',
    'minor-sixth': 'min6',
    'suspended-second': 'sus2',
    'suspended-fourth': 'sus4',
    'power': '5',
    # Simplify extended/altered chords to base quality for normalization
    'maj7+5': 'maj7',
    'maj7-5': 'maj7',
    'min7+5': 'min7',
    'min7-5': 'hdim7', # More specific than just min7
    '7+5': 'aug7', # Or just 7 depending on rules
    '7-5': '7',
    '7b9': '7',
    '7#9': '7',
    '7(b9)': '7',       # Added
    '7(#9)': '7',       # Added
    '7(13)': '7',       # Added
    '7(*5,13)': '7',    # Added
    'maj9': 'maj7',
    'maj11': 'maj7',
    'maj13': 'maj7',
    '9': '7',
    '11': '7',
    '13': '7',
    '9(*3)': '7',       # Added
    '9(*3,11)': '7',    # Added
    'min9': 'min7',
    'min11': 'min7',
    'min13': 'min7',
    'minmaj7': 'minmaj7', # Keep this distinct
    '69': 'maj6', # Simplify 6/9 chords
    'maj6(9)': 'maj6',  # Added
    'maj(9)': 'maj7',
    'maj(11)': 'maj7',  # Added
    'maj(#11)': 'maj7', # Added
    'maj(#4)': 'maj7',  # Added (Lydian)
    'maj(b9)': 'maj7',  # Added
    'maj(2)': 'maj',    # Added (Add9 simplified)
    'maj(4)': 'maj',    # Added (Add11 simplified)
    'maj(*1)': 'maj',   # Added (Treat as major)
    'maj(*3)': 'maj',   # Added (Treat as major)
    'maj(*5)': 'maj',   # Keep existing
    'maj7(*5)': 'maj7', # Added
    'maj7(*b5)': 'maj7',# Added
    'min(9)': 'min7',   # Added
    'min(2)': 'min',    # Added (Add9 simplified)
    'min(4)': 'min',    # Added (Add11 simplified)
    'min(*3)': 'min',   # Added (Treat as minor)
    'min(*5)': 'min',   # Added (Treat as minor)
    'min(*b3)': 'min',  # Keep existing
    'min(*b3,*5)': 'min',# Keep existing
    'min7(2,*b3,4)': 'min7', # Added (Complex altered)
    'min7(4)': 'min7',  # Added
    'min7(*5)': 'min7', # Added
    'min7(*5,b6)': 'min7', # Added (Complex altered)
    'min7(*b3)': 'min7',# Added (Treat as min7)
    'sus4(9)': 'sus4',
    'sus4(2)': 'sus4',  # Added
    'sus4(b7)': 'sus4', # Added
    'sus2(b7)': 'sus2',
    'aug(9,11)': 'aug', # Added
    '(1)': 'maj',       # Added (Root only)
    '(1,5)': 'maj',     # Added (Power chord)
    '(1,b3)': 'min',    # Added (Root + min 3rd)
    '(b3,5)': 'min',    # Added (Implied root minor triad)
    '(1,b7)': '7',      # Added (Root + dom 7th)
    '(1,4)': 'sus4',    # Added (Implied root sus4)
    '(1,4,b7)': 'sus4', # Added (Dominant sus)
    '(1,b3,4)': 'min',  # Added (Complex minor variant)
    '(1,4,b5)': 'dim',  # Added (Complex diminished variant)
    '(1,2,5,b6)': 'min6',# Added (Complex minor 6 variant)
    '(1,2,4)': 'sus2',  # Added (Complex sus variant)
    'maj9(*7)': 'maj7', # Added
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
    "min": "Minor", "m": "Minor",
    "7": "Dom7", "dom7": "Dom7", "dominant": "Dom7",
    "maj7": "Maj7", "M7": "Maj7", "major7": "Maj7",
    "min7": "Min7", "m7": "Min7", "minor7": "Min7",
    "dim": "Dim", "°": "Dim", "o": "Dim", "diminished": "Dim",
    "dim7": "Dim7", "°7": "Dim7", "o7": "Dim7", "diminished7": "Dim7",
    "hdim7": "Half-Dim", "m7b5": "Half-Dim", "ø": "Half-Dim", "half-diminished": "Half-Dim",
    "aug": "Aug", "+": "Aug", "augmented": "Aug",
    "sus2": "Sus", "sus4": "Sus", "sus": "Sus", "suspended": "Sus",
    "min6": "Min6", "m6": "Min6",
    "maj6": "Maj6", "6": "Maj6",
    "minmaj7": "Min-Maj7", "mmaj7": "Min-Maj7", "min-maj7": "Min-Maj7",
    "N": "No Chord", "X": "No Chord", # Map X to No Chord for reporting consistency
    "5": "Other", # Power chords
    "1": "Other", # Root only
    "aug7": "Other",
    # Default
    "Other": "Other"
}


def _modify_pitch(base_pitch, modifier):
    """Internal helper to modify a pitch class integer by modifiers."""
    for m in modifier:
        if m == 'b':
            base_pitch -= 1
        elif m == '#':
            base_pitch += 1
        else:
            # Ignore unexpected characters in modifiers
            pass
    return base_pitch

def _parse_root(root_str):
    """Internal helper to parse root string to integer, handling enharmonics."""
    if not root_str or not root_str[0].isalpha():
        raise ValueError(f"Invalid root note: {root_str}")

    # Handle explicit enharmonic notations first
    if root_str in ENHARMONIC_MAP:
        # Use the mapped value (e.g., B# -> C) then parse 'C'
        root_str = ENHARMONIC_MAP[root_str]

    # Standard parsing
    base_char = root_str[0].upper()
    modifier = root_str[1:]
    if base_char < 'A' or base_char > 'G':
         raise ValueError(f"Invalid root note base: {base_char}")

    base_pitch = _CHROMA_ID[(ord(base_char) - ord('C')) % 7]
    return _modify_pitch(base_pitch, modifier) % 12

def _parse_chord_string(chord_label):
    """
    Internal helper to parse a chord label into root, quality, and bass.
    Applies basic cleaning and normalization.

    Args:
        chord_label (str): Raw chord label.

    Returns:
        tuple: (root_str, quality_str, bass_str) or (None, None, None) for N/X.
               Returns standardized strings (e.g., root normalized, quality normalized).
               Returns None for components if parsing fails.
    """
    if not chord_label or chord_label in ["N", "NC"]:
        return "N", None, None
    if chord_label == "X":
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
        if len(parts) == 2:
            chord_part = parts[0]
            bass_str = parts[1]
            # Normalize bass note representation if it's a pitch class
            try:
                bass_int = _parse_root(bass_str)
                bass_str = PITCH_CLASS[bass_int] # Standardize bass notation
            except ValueError:
                # Bass might be a scale degree (e.g., /3), keep as is for now
                pass
        else: # Malformed bass? Ignore it.
            chord_part = chord_label.replace('/', '')

    # 2. Split root and quality
    if ':' in chord_part:
        parts = chord_part.split(':', 1)
        if len(parts) == 2:
            root_str = parts[0]
            quality_str = parts[1]
        else: # Malformed colon? Treat as root only.
            root_str = chord_part.replace(':', '')
            quality_str = ''
    else:
        # No colon, need to find split point
        # Use regex to find the root note (including accidentals)
        match = re.match(r'^([A-G][#b]?)', chord_part)
        if match:
            root_str = match.group(1)
            quality_str = chord_part[len(root_str):]
        else:
            # Cannot identify a valid root note
            logger.debug(f"Could not parse root from: {chord_part}")
            return "X", None, None # Treat as unknown if root is unparseable

    # 3. Normalize Root
    try:
        root_int = _parse_root(root_str)
        # Prefer sharps for normalization generally, but keep common flats like Bb, Eb
        norm_root_str = PITCH_CLASS[root_int]
        if norm_root_str in ENHARMONIC_NORM_MAP:
             # If the normalized sharp version has a common flat equivalent, use the flat
             if ENHARMONIC_NORM_MAP[norm_root_str] == root_str:
                 root_str = norm_root_str # Keep original if it matches preferred flat
             else:
                 # Otherwise, use the standard sharp/natural name
                 root_str = norm_root_str
        else:
            root_str = norm_root_str

    except ValueError as e:
        logger.debug(f"Error parsing root '{root_str}' from label '{chord_label}': {e}")
        return "X", None, None # Treat as unknown

    # 4. Normalize Quality
    # Remove parentheses content for normalization (e.g., maj(9) -> maj)
    quality_str = re.sub(r'\(.*\)', '', quality_str).strip()
    # Apply standard quality mappings
    quality_str = QUALITY_NORM_MAP.get(quality_str, quality_str)

    # Handle empty quality -> 'maj'
    if not quality_str:
        quality_str = 'maj'

    return root_str, quality_str, bass_str


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
        return "Other" # Parsing failed

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
            '6': self.interval_list('(1,6)'), # Ambiguous, assume maj6
            '5': self.interval_list('(1,5)'),
            '4': self.interval_list('(1,4)'), # Ambiguous, assume sus4? No, just interval.
            '1': self.interval_list('(1)'),
            'dim7': self.interval_list('(1,b3,b5,bb7)'),
            'hdim7': self.interval_list('(1,b3,b5,b7)'),
            'minmaj7': self.interval_list('(1,b3,5,7)'),
            'maj6': self.interval_list('(1,3,5,6)'),
            'min6': self.interval_list('(1,b3,5,6)'),
            '9': self.interval_list('(1,3,5,b7,9)'),
            'maj9': self.interval_list('(1,3,5,7,9)'),
            'min9': self.interval_list('(1,b3,5,b7,9)'),
            'sus2': self.interval_list('(1,2,5)'),
            'sus4': self.interval_list('(1,4,5)'),
            # Add more if needed by interval logic, but parsing handles string qualities
        }
        # Initialize chord mapping dictionary
        self.chord_mapping = {}

    # --- Interval/Pitch calculation methods (Keep if reduce_to_triads is used) ---
    def _modify(self, base_pitch, modifier):
        """Internal helper - kept for interval calculations."""
        return _modify_pitch(base_pitch, modifier)

    def interval(self, interval_str):
        """Kept for interval calculations."""
        interval_str = interval_str.strip()
        if not interval_str: return 0 # Handle empty string case
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
                    interval_to_add = self.interval(int_def)
                    given_pitch_classes[interval_to_add] = 1
            return given_pitch_classes
        except Exception as e:
            logger.error(f"Error parsing interval list '{intervals_str}': {e}")
            # Return the array as is or raise error? Returning as is might hide issues.
            raise ValueError(f"Failed to parse interval list: {intervals_str}") from e


    def chord_intervals(self, quality_str):
        """Kept for interval calculations."""
        list_idx = quality_str.find('(')
        if list_idx == -1:
            # Direct shorthand lookup
            if quality_str in self._shorthands:
                return self._shorthands[quality_str].copy()
            else:
                # Attempt to parse as interval list directly if format matches, e.g., "(1,3,5)"
                if quality_str.startswith('(') and quality_str.endswith(')'):
                    try:
                        return self.interval_list(quality_str)
                    except ValueError:
                        raise ValueError(f"Unknown quality shorthand and invalid interval list: {quality_str}")
                else:
                     raise ValueError(f"Unknown quality shorthand: {quality_str}")
        else:
            # Combined shorthand and interval list
            base_quality = quality_str[:list_idx]
            interval_part = quality_str[list_idx:]
            if base_quality:
                if base_quality in self._shorthands:
                    ivs = self._shorthands[base_quality].copy()
                else:
                    raise ValueError(f"Unknown base quality shorthand: {base_quality}")
            else:
                ivs = np.zeros(12, dtype=np.int32)

            return self.interval_list(interval_part, ivs)

    # --- Core Public Methods ---

    def label_error_modify(self, label):
        """
        Correct common errors and inconsistencies in chord label format.
        Uses the centralized parser and reconstructs the label.

        Parameters:
            label (str): Chord label to correct.

        Returns:
            str: Corrected and normalized chord label.
        """
        root, quality, bass = _parse_chord_string(label)

        if root == "N" or root == "X":
            return root
        if root is None: # Parsing failed completely
             return "X" # Return unknown

        # Reconstruct the normalized label
        # Use standard mir_eval format: Root:Quality/Bass
        # Handle major quality explicitly as ':maj'
        quality_str = quality if quality and quality != 'maj' else 'maj'
        corrected_label = f"{root}:{quality_str}"

        if bass:
            corrected_label += f"/{bass}"

        return corrected_label

    def normalize_chord(self, chord_name):
        """
        Normalize a chord name to a canonical form (Root:Quality/Bass).
        Uses the centralized parser.

        Args:
            chord_name (str): Input chord name.

        Returns:
            tuple: (normalized_chord_str, root_str, quality_str, bass_str)
        """
        root, quality, bass = _parse_chord_string(chord_name)

        if root == "N" or root == "X":
            return root, root, None, None
        if root is None:
            return "X", None, None, None # Parsing failed

        # Reconstruct the normalized label string
        # Ensure quality is 'maj' if it was originally empty or normalized to it
        quality_str_for_label = quality if quality and quality != 'maj' else 'maj'
        normalized_label = f"{root}:{quality_str_for_label}"
        if bass:
            normalized_label += f"/{bass}"

        return normalized_label, root, quality, bass


    def simplify_chord(self, chord_name):
        """
        Simplify a chord to its basic root position form using QUALITY_SIMPLIFY_MAP.

        Args:
            chord_name (str): The chord name to simplify.

        Returns:
            str: Simplified chord name (Root:Quality or Root for major).
        """
        root, quality, bass = _parse_chord_string(chord_name)

        if root == "N" or root == "X":
            return root
        if root is None:
            return "X" # Parsing failed

        # Apply simplification map to the quality
        simplified_quality = QUALITY_SIMPLIFY_MAP.get(quality, quality)

        # Reconstruct simplified chord name (root position)
        if simplified_quality == 'maj':
            return root # Just root for major
        else:
            return f"{root}:{simplified_quality}"

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
                # Get normalized components
                norm_label, norm_root, norm_quality, norm_bass = self.normalize_chord(base_chord)
                if norm_root == "N" or norm_root == "X":
                    variants.add(norm_root) # Add N or X directly
                elif norm_root:
                    # Add normalized label
                    variants.add(norm_label)
                    # Add normalized root position
                    norm_root_pos_qual = norm_quality if norm_quality and norm_quality != 'maj' else 'maj'
                    norm_root_pos = f"{norm_root}:{norm_root_pos_qual}"
                    variants.add(norm_root_pos)

                    # Get simplified root position
                    simplified_chord = self.simplify_chord(norm_label) # Simplify the normalized version
                    variants.add(simplified_chord)

                    # Generate enharmonic variants for normalized root position and simplified
                    # Normalized Root Position Enharmonic
                    if norm_root in ENHARMONIC_MAP:
                        enh_root = ENHARMONIC_MAP[norm_root]
                        enh_norm_root_pos = f"{enh_root}:{norm_root_pos_qual}"
                        variants.add(enh_norm_root_pos)

                    # Simplified Enharmonic
                    simp_root, simp_qual, _ = _parse_chord_string(simplified_chord)
                    if simp_root and simp_root in ENHARMONIC_MAP:
                         enh_simp_root = ENHARMONIC_MAP[simp_root]
                         enh_simp_qual_str = simp_qual if simp_qual and simp_qual != 'maj' else 'maj'
                         enh_simplified = f"{enh_simp_root}:{enh_simp_qual_str}"
                         variants.add(enh_simplified)

            except Exception as e:
                logger.warning(f"Error generating variants for base chord '{base_chord}' (idx {base_idx}): {e}")

            # Add all generated variants to the extended mapping, pointing to the base index
            for variant in variants:
                if variant and variant not in self.chord_mapping:
                    extended_mapping[variant] = base_idx

            processed_indices.add(base_idx) # Mark this index as done

        # Update the main mapping
        original_size = len(self.chord_mapping)
        self.chord_mapping.update(extended_mapping)
        added_count = len(self.chord_mapping) - original_size
        logger.info(f"Enhanced chord mapping with {added_count} variants (total: {len(self.chord_mapping)})")

        # Log a few examples if needed
        # if added_count > 0:
        #     logger.debug(f"Sample added mappings: {dict(list(extended_mapping.items())[:5])}")


    def get_chord_idx(self, chord_name, use_large_voca=True):
        """
        Get chord index using the pre-initialized comprehensive mapping.

        Args:
            chord_name (str): Chord name to look up.
            use_large_voca (bool): Determines N/X index values.

        Returns:
            int: Chord index.
        """
        # Define N/X indices based on vocabulary size
        n_chord_idx = 169 if use_large_voca else 24
        x_chord_idx = 168 if use_large_voca else 25

        # 1. Handle N/X directly
        if not chord_name or chord_name in ["N", "NC"]:
            return n_chord_idx
        if chord_name == "X":
            return x_chord_idx

        # 2. Perform basic cleaning/normalization using the centralized parser approach
        # We use label_error_modify as it calls _parse_chord_string and reconstructs
        lookup_key = self.label_error_modify(chord_name)

        # 3. Direct lookup in the comprehensive mapping
        if hasattr(self, 'chord_mapping') and lookup_key in self.chord_mapping:
            return self.chord_mapping[lookup_key]
        else:
            # If not found after normalization, return unknown index
            # logger.debug(f"Chord '{chord_name}' (normalized to '{lookup_key}') not found in mapping. Returning unknown index {x_chord_idx}.")
            return x_chord_idx

    # --- Kept for compatibility if reduce_to_triads is needed ---
    def reduce_to_triads(self, chords_struct_array, keep_bass=False):
        """
        Reduce chords (structured array) to triads.

        Parameters:
            chords_struct_array : numpy structured array (CHORD_DTYPE)
                Chords to be reduced.
            keep_bass : bool
                Indicates whether to keep the bass note or set it to 0.

        Returns:
            reduced_chords : numpy structured array
                Chords reduced to triads.
        """
        # Ensure input is a structured array with the expected dtype
        # This method assumes the input `chords` is already parsed into the CHORD_DTYPE structure
        # which is not directly done by the refactored parsing methods.
        # If this method is needed, the input generation needs review.
        # For now, keeping the logic assuming correct input structure.
        if not isinstance(chords_struct_array, np.ndarray) or chords_struct_array.dtype.names != ('root', 'bass', 'intervals', 'is_major'):
             logger.error("reduce_to_triads requires input as numpy structured array with CHORD_DTYPE.")
             # Return empty array or raise error? Returning copy for now.
             return chords_struct_array.copy()


        intervals = chords_struct_array['intervals']
        unison = intervals[:, 0].astype(bool)
        maj_sec = intervals[:, 2].astype(bool)
        min_third = intervals[:, 3].astype(bool)
        maj_third = intervals[:, 4].astype(bool)
        perf_fourth = intervals[:, 5].astype(bool)
        dim_fifth = intervals[:, 6].astype(bool)
        perf_fifth = intervals[:, 7].astype(bool)
        aug_fifth = intervals[:, 8].astype(bool)
        # Check for NO_CHORD based on root = -1
        no_chord = (chords_struct_array['root'] == -1)

        reduced_chords = chords_struct_array.copy()
        ivs = reduced_chords['intervals']

        # Reset non 'no_chord' intervals to root only initially
        ivs[~no_chord] = self.interval_list('(1)')

        # Apply reduction rules based on intervals present
        # Power chord
        ivs[unison & perf_fifth & ~no_chord] = self.interval_list('(1,5)')
        # Sus chords (handle carefully to avoid overwriting triads)
        is_sus2 = ~perf_fourth & maj_sec & ~min_third & ~maj_third & ~no_chord
        is_sus4 = perf_fourth & ~maj_sec & ~min_third & ~maj_third & ~no_chord
        ivs[is_sus2] = self._shorthands['sus2']
        ivs[is_sus4] = self._shorthands['sus4']

        # Triads (apply after sus chords)
        is_min = min_third & ~maj_third & ~no_chord
        is_maj = maj_third & ~min_third & ~no_chord

        ivs[is_min] = self._shorthands['min']
        ivs[is_min & aug_fifth & ~perf_fifth] = self.interval_list('(1,b3,#5)') # min-aug (uncommon)
        ivs[is_min & dim_fifth & ~perf_fifth] = self._shorthands['dim']

        ivs[is_maj] = self._shorthands['maj']
        ivs[is_maj & dim_fifth & ~perf_fifth] = self.interval_list('(1,3,b5)') # maj-dim (uncommon)
        ivs[is_maj & aug_fifth & ~perf_fifth] = self._shorthands['aug']

        # Handle bass note
        if not keep_bass:
            reduced_chords['bass'][~no_chord] = 0 # Set bass to root for non 'no_chord'
        else:
            # Ensure bass note is valid for the reduced chord
            for i in range(len(reduced_chords)):
                if not no_chord[i]:
                    bass_note = reduced_chords['bass'][i]
                    # If bass note is not in the reduced intervals, set to root
                    if bass_note < 0 or bass_note >= 12 or ivs[i, bass_note] == 0:
                        reduced_chords['bass'][i] = 0

        # Ensure bass is -1 for no chords
        reduced_chords['bass'][no_chord] = -1
        # Update is_major flag based on reduced intervals
        reduced_chords['is_major'] = np.array_equal(ivs, self._shorthands['maj'])

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
    Uses standard mir_eval-compatible notations (e.g., C:maj, G:min, D:7).
    Index 168 = X (Unknown), Index 169 = N (No Chord).
    """
    mapping = {}
    # Quality names in a standard order
    quality_list = [
        'min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7',
        'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4'
    ]
    num_qualities = len(quality_list) # Should be 14

    if num_qualities != 14:
        logger.warning(f"Expected 14 qualities for idx2voca_chord, found {num_qualities}. Mapping might be incorrect.")

    # Create the standard mapping
    for root_idx in range(12):
        root_note = PITCH_CLASS[root_idx]
        for quality_idx in range(num_qualities):
            chord_idx = root_idx * num_qualities + quality_idx
            quality = quality_list[quality_idx]

            # Construct label: Root:Quality (use ':maj' for major triads)
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
