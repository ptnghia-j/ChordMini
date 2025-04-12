# encoding: utf-8
"""
This module contains chord evaluation functionality.

It provides the evaluation measures used for the MIREX ACE task, and
tries to follow [1]_ and [2]_ as closely as possible.
"""

import numpy as np
import pandas as pd
import mir_eval
import logging

logger = logging.getLogger(__name__)

CHORD_DTYPE = [('root', int),
               ('bass', int),
               ('intervals', int, (12,)),
               ('is_major', bool)]

CHORD_ANN_DTYPE = [('start', float),
                   ('end', float),
                   ('chord', CHORD_DTYPE)]

NO_CHORD = (-1, -1, np.zeros(12, dtype=int), False)
UNKNOWN_CHORD = (-1, -1, np.ones(12, dtype=int) * -1, False)

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def idx_to_chord(idx):
    if idx == 24:
        return "-"
    elif idx == 25:
        return u"\u03B5"

    minmaj = idx % 2
    root = idx // 2

    return PITCH_CLASS[root] + ("M" if minmaj == 0 else "m")


def get_chord_quality(chord_label):
    """
    Extract the quality from a chord label and map it to a standard category.
    
    Args:
        chord_label (str): Chord label like "C:maj", "G:7", etc.
        
    Returns:
        str: Standardized chord quality category
    """
    # Handle special non-chord labels
    if chord_label in ["N", "X"]:
        return "No Chord"
        
    # Use chord parsing utilities from this module
    try:
        # Handle any error correction and standardization
        chord_label = Chords().label_error_modify(chord_label)
        
        # Extract chord parts (internally handles format variations)
        if ':' in chord_label:
            root, quality = chord_label.split(':', 1)
            if '/' in quality:
                quality = quality.split('/', 1)[0]  # Remove bass note
        else:
            # Handle no-separator format like "Cmaj7" or "Dm"
            for root_note in PITCH_CLASS:
                if chord_label.startswith(root_note):
                    if len(chord_label) == len(root_note):
                        # Just a root note like "C" - implied major
                        return "Major"
                    quality = chord_label[len(root_note):]
                    break
            else:
                return "Other"  # Couldn't identify root note
                
        # Map quality to standard categories
        quality_map = {
            # Major family
            "maj": "Major", "": "Major", "M": "Major", 
            
            # Minor family
            "min": "Minor", "m": "Minor",
            
            # Dominant seventh
            "7": "Dom7",
            
            # Major seventh
            "maj7": "Maj7", "M7": "Maj7",
            
            # Minor seventh
            "min7": "Min7", "m7": "Min7",
            
            # Diminished
            "dim": "Dim", "°": "Dim", "o": "Dim",
            
            # Diminished seventh
            "dim7": "Dim7", "°7": "Dim7", "o7": "Dim7",
            
            # Half-diminished seventh
            "hdim7": "Half-Dim", "m7b5": "Half-Dim", "ø": "Half-Dim",
            
            # Augmented
            "aug": "Aug", "+": "Aug",
            
            # Suspended
            "sus2": "Sus", "sus4": "Sus", "sus": "Sus",
            
            # Extended chords
            "9": "Extended", "11": "Extended", "13": "Extended",
            "maj9": "Extended", "min9": "Extended",
            
            # Minor/Major sixth
            "min6": "Min6", "m6": "Min6",
            "maj6": "Maj6", "6": "Maj6",
            
            # Minor/Major seventh
            "minmaj7": "Min-Maj7", "mmaj7": "Min-Maj7"
        }
        
        return quality_map.get(quality, "Other")
        
    except Exception as e:
        logger.warning(f"Error determining chord quality for '{chord_label}': {e}")
        return "Other"


class Chords:
    def __init__(self):
        self._shorthands = {
            'maj': self.interval_list('(1,3,5)'),
            'min': self.interval_list('(1,b3,5)'),
            'dim': self.interval_list('(1,b3,b5)'),
            'aug': self.interval_list('(1,3,#5)'),
            'maj7': self.interval_list('(1,3,5,7)'),
            'min7': self.interval_list('(1,b3,5,b7)'),
            '7': self.interval_list('(1,3,5,b7)'),
            '6': self.interval_list('(1,6)'),
            '5': self.interval_list('(1,5)'),
            '4': self.interval_list('(1,4)'),
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
            '11': self.interval_list('(1,3,5,b7,9,11)'),
            'min11': self.interval_list('(1,b3,5,b7,9,11)'),
            '13': self.interval_list('(1,3,5,b7,13)'),
            'maj13': self.interval_list('(1,3,5,7,13)'),
            'min13': self.interval_list('(1,b3,5,b7,13)')
        }

        # Common mapping tables - consolidated from multiple places
        self.enharmonic_map = {
            'Bb': 'A#', 'A#': 'Bb',
            'Db': 'C#', 'C#': 'Db',
            'Eb': 'D#', 'D#': 'Eb',
            'Gb': 'F#', 'F#': 'Gb',
            'Ab': 'G#', 'G#': 'Ab',
        }

        self.quality_map = {
            '': 'maj',          # Empty string means major chord
            'M': 'maj',         # Alternative notation for major
            'm': 'min',         # Alternative notation for minor
            'major': 'maj',     # Full word notation
            'minor': 'min',     # Full word notation
            'maj7+5': 'maj7',   # Simplify extended chords
            'maj7-5': 'maj7',
            'min7+5': 'min7',
            'min7-5': 'min7',
        }
        
        # Simplification rules for extended qualities and special notations
        self.simplification_map = {
            # Extensions to base qualities
            'maj9': 'maj7',      # Cmaj9 -> Cmaj7
            'maj11': 'maj7',     # Cmaj11 -> Cmaj7
            'maj13': 'maj7',     # Cmaj13 -> Cmaj7
            '9': '7',            # C9 -> C7
            '11': '7',           # C11 -> C7
            '13': '7',           # C13 -> C7
            'min9': 'min7',      # Cmin9 -> Cmin7
            'min11': 'min7',     # Cmin11 -> Cmin7
            'min13': 'min7',     # Cmin13 -> Cmin7
            
            # Suspended and special chords
            'sus4(2)': 'sus4',   # Csus4(2) -> Csus4
            'maj(9)': 'maj7',    # Cmaj(9) -> Cmaj7
            'sus4(9)': 'sus4',   # Csus4(9) -> Csus4
            'sus2(b7)': 'sus2',  # Csus2(b7) -> Csus2
            '7(#9)': '7',        # C7(#9) -> C7
            'maj(b9)': 'maj7',   # Cmaj(b9) -> Cmaj7
            
            # Special notation patterns
            '(1)': 'maj',        # C(1) -> C
            '(1,5)': 'maj',      # C(1,5) -> C
            '(1,b7)': '7',       # C(1,b7) -> C7
            '(1,b3)': 'min',     # C(1,b3) -> Cmin
            'maj(*5)': 'maj',    # Cmaj(*5) -> C
            'min(*b3)': 'min',   # Cmin(*b3) -> Cmin
            'min(*b3,*5)': 'min' # Cmin(*b3,*5) -> Cmin
        }
        
        # Initialize chord mapping dictionary
        self.chord_mapping = {}

    _l = [0, 1, 1, 0, 1, 1, 1]
    _chroma_id = (np.arange(len(_l) * 2) + 1) + np.array(_l + _l).cumsum() - 1

    def modify(self, base_pitch, modifier):
        """
        Modify a pitch class in integer representation by a given modifier string.

        A modifier string can be any sequence of 'b' (one semitone down)
        and '#' (one semitone up).

        Parameters
        ----------
        base_pitch : int
            Pitch class as integer.
        modifier : str
            String of modifiers ('b' or '#').

        Returns
        -------
        modified_pitch : int
            Modified root note.

        """
        for m in modifier:
            if m == 'b':
                base_pitch -= 1
            elif m == '#':
                base_pitch += 1
            else:
                raise ValueError('Unknown modifier: {}'.format(m))
        return base_pitch

    def pitch(self, pitch_str):
        """
        Convert a string representation of a pitch class (consisting of root
        note and modifiers) to an integer representation.

        Parameters
        ----------
        pitch_str : str
            String representation of a pitch class.

        Returns
        -------
        pitch : int
            Integer representation of a pitch class.

        """
        return self.modify(self._chroma_id[(ord(pitch_str[0]) - ord('C')) % 7],
                      pitch_str[1:]) % 12

    def interval(self, interval_str):
        """
        Convert a string representation of a musical interval into a pitch class
        (e.g. a minor seventh 'b7' into 10, because it is 10 semitones above its
        base note).

        Parameters
        ----------
        interval_str : str
            Musical interval.

        Returns
        -------
        pitch_class : int
            Number of semitones to base note of interval.

        """
        for i, c in enumerate(interval_str):
            if c.isdigit():
                return self.modify(self._chroma_id[int(interval_str[i:]) - 1],
                              interval_str[:i]) % 12

    def interval_list(self, intervals_str, given_pitch_classes=None):
        """
        Convert a list of intervals given as string to a binary pitch class
        representation. For example, 'b3, 5' would become
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0].

        Parameters
        ----------
        intervals_str : str
            List of intervals as comma-separated string (e.g. 'b3, 5').
        given_pitch_classes : None or numpy array
            If None, start with empty pitch class array, if numpy array of length
            12, this array will be modified.

        Returns
        -------
        pitch_classes : numpy array
            Binary pitch class representation of intervals.

        """
        if given_pitch_classes is None:
            given_pitch_classes = np.zeros(12, dtype=np.int32)
        for int_def in intervals_str[1:-1].split(','):
            int_def = int_def.strip()
            if int_def[0] == '*':
                given_pitch_classes[self.interval(int_def[1:])] = 0
            else:
                given_pitch_classes[self.interval(int_def)] = 1
        return given_pitch_classes

    def chord_intervals(self, quality_str):
        """
        Convert a chord quality string to a pitch class representation. For
        example, 'maj' becomes [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0].

        Parameters
        ----------
        quality_str : str
            String defining the chord quality.

        Returns
        -------
        pitch_classes : numpy array
            Binary pitch class representation of chord quality.

        """
        list_idx = quality_str.find('(')
        if list_idx == -1:
            return self._shorthands[quality_str].copy()
        if list_idx != 0:
            ivs = self._shorthands[quality_str[:list_idx]].copy()
        else:
            ivs = np.zeros(12, dtype=np.int32)

        return self.interval_list(quality_str[list_idx:], ivs)

    def load_chords(self, filename):
        """
        Load chords from a text file.

        The chord must follow the syntax defined in [1]_.

        Parameters
        ----------
        filename : str
            File containing chord segments.

        Returns
        -------
        crds : numpy structured array
            Structured array with columns "start", "end", and "chord",
            containing the beginning, end, and chord definition of chord
            segments.

        References
        ----------
        .. [1] Christopher Harte, "Towards Automatic Extraction of Harmony
               Information from Music Signals." Dissertation,
               Department for Electronic Engineering, Queen Mary University of
               London, 2010.

        """
        start, end, chord_labels = [], [], []
        with open(filename, 'r') as f:
            for line in f:
                if line:

                    splits = line.split()
                    if len(splits) == 3:

                        s = splits[0]
                        e = splits[1]
                        l = splits[2]

                        start.append(float(s))
                        end.append(float(e))
                        chord_labels.append(l)

        crds = np.zeros(len(start), dtype=CHORD_ANN_DTYPE)
        crds['start'] = start
        crds['end'] = end
        crds['chord'] = self.chords(chord_labels)

        return crds

    def reduce_to_triads(self, chords, keep_bass=False):
        """
        Reduce chords to triads.

        The function follows the reduction rules implemented in [1]_. If a chord
        chord does not contain a third, major second or fourth, it is reduced to
        a power chord. If it does not contain neither a third nor a fifth, it is
        reduced to a single note "chord".

        Parameters
        ----------
        chords : numpy structured array
            Chords to be reduced.
        keep_bass : bool
            Indicates whether to keep the bass note or set it to 0.

        Returns
        -------
        reduced_chords : numpy structured array
            Chords reduced to triads.

        References
        ----------
        .. [1] Johan Pauwels and Geoffroy Peeters.
               "Evaluating Automatically Estimated Chord Sequences."
               In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

        """
        unison = chords['intervals'][:, 0].astype(bool)
        maj_sec = chords['intervals'][:, 2].astype(bool)
        min_third = chords['intervals'][:, 3].astype(bool)
        maj_third = chords['intervals'][:, 4].astype(bool)
        perf_fourth = chords['intervals'][:, 5].astype(bool)
        dim_fifth = chords['intervals'][:, 6].astype(bool)
        perf_fifth = chords['intervals'][:, 7].astype(bool)
        aug_fifth = chords['intervals'][:, 8].astype(bool)
        no_chord = (chords['intervals'] == NO_CHORD[-1]).all(axis=1)

        reduced_chords = chords.copy()
        ivs = reduced_chords['intervals']

        ivs[~no_chord] = self.interval_list('(1)')
        ivs[unison & perf_fifth] = self.interval_list('(1,5)')
        ivs[~perf_fourth & maj_sec] = self._shorthands['sus2']
        ivs[perf_fourth & ~maj_sec] = self._shorthands['sus4']

        ivs[min_third] = self._shorthands['min']
        ivs[min_third & aug_fifth & ~perf_fifth] = self.interval_list('(1,b3,#5)')
        ivs[min_third & dim_fifth & ~perf_fifth] = self._shorthands['dim']

        ivs[maj_third] = self._shorthands['maj']
        ivs[maj_third & dim_fifth & ~perf_fifth] = self.interval_list('(1,3,b5)')
        ivs[maj_third & aug_fifth & ~perf_fifth] = self._shorthands['aug']

        if not keep_bass:
            reduced_chords['bass'] = 0
        else:
            # remove bass notes if they are not part of the intervals anymore
            reduced_chords['bass'] *= ivs[range(len(reduced_chords)),
                                          reduced_chords['bass']]
        # keep -1 in bass for no chords
        reduced_chords['bass'][no_chord] = -1

        return reduced_chords

    def normalize_chord(self, chord_name):
        """
        Normalize a chord name to a canonical form for consistent lookup.
        Handles various notations, enharmonic equivalents, and format issues.
        
        Args:
            chord_name (str): Input chord name in any supported format
            
        Returns:
            tuple: (normalized_chord, root, quality, bass)
        """
        # Handle special chords
        if chord_name in ['N', 'NC', 'X']:
            return chord_name, None, None, None
        
        # Apply basic error correction
        chord_name = self.label_error_modify(chord_name)
        
        # Split into components
        root, quality, bass = None, None, None
        
        # Handle format with separator (:)
        if ':' in chord_name:
            parts = chord_name.split(':')
            root = parts[0]
            quality_with_bass = parts[1]
            
            # Handle bass note
            if '/' in quality_with_bass:
                quality, bass = quality_with_bass.split('/')
            else:
                quality = quality_with_bass
        else:
            # Handle format without separator
            # First check for bass note
            if '/' in chord_name:
                main_chord, bass = chord_name.split('/')
            else:
                main_chord = chord_name
                bass = None
                
            # Then extract root and quality
            # Find where the root note ends and quality starts
            for i, c in enumerate(main_chord):
                if i > 0 and c.islower():  # Start of a quality like 'min', 'maj', etc.
                    root = main_chord[:i]
                    quality = main_chord[i:]
                    break
                elif i > 0 and c.isdigit():  # Start of a quality like '7', '9', etc.
                    root = main_chord[:i]
                    quality = main_chord[i:]
                    break
            else:
                # If no quality found, it's just a root (major chord)
                root = main_chord
                quality = ''
        
        # Normalize root (prefer sharps)
        if root in self.enharmonic_map:
            root = self.enharmonic_map.get(root, root)
        
        # Normalize quality
        if quality in self.quality_map:
            quality = self.quality_map[quality]
        
        # Construct normalized chord name
        if not quality or quality == 'maj':
            normalized = root  # Just root for major
        else:
            normalized = f"{root}:{quality}"
        
        # Add bass if present
        if bass:
            normalized += f"/{bass}"
            
        return normalized, root, quality, bass

    def simplify_chord(self, chord_name):
        """
        Simplify a chord to its basic form by reducing extensions and removing inversions.
        This is helpful for mapping complex chords to a smaller vocabulary.
        
        Args:
            chord_name (str): The chord name to simplify
            
        Returns:
            str: Simplified chord name
        """
        # Special cases
        if chord_name in ['N', 'NC', 'X']:
            return chord_name
            
        # First normalize the chord
        normalized, root, quality, bass = self.normalize_chord(chord_name)
        if root is None:  # Special chord
            return normalized
            
        # Remove inversion (bass note)
        simplified = f"{root}:{quality}" if quality else root
        
        # Apply simplification for extended qualities
        if quality in self.simplification_map:
            simple_quality = self.simplification_map[quality]
            simplified = f"{root}:{simple_quality}" if simple_quality != 'maj' else root
        
        # Handle special cases with parentheses
        elif '(' in quality and quality not in self._shorthands:
            # Extract base quality if present
            base_quality = quality.split('(')[0]
            if base_quality in self._shorthands:
                simplified = f"{root}:{base_quality}" if base_quality != 'maj' else root
            elif any(pattern in quality for pattern in ['(1)', '(*']):
                # Patterns like (1) or (*3) indicate root position major or altered chords
                simplified = root
        
        return simplified

    def label_error_modify(self, label):
        """
        Correct common errors and inconsistencies in chord label format
        """
        if not label or label in ['N', 'X', 'NC']:
            return label
            
        # Common format fixes
        if label == 'Emin/4': 
            return 'E:min/4'
        elif label == 'A7/3': 
            return 'A:7/3'
        elif label == 'Bb7/3': 
            return 'Bb:7/3'
        elif label == 'Bb7/5': 
            return 'Bb:7/5'
            
        # Handle format without colon separator
        if ':' not in label:
            # Check for common quality strings and insert separator
            for quality in ['min', 'maj', 'dim', 'aug', 'sus']:
                if quality in label:
                    idx = label.find(quality)
                    return label[:idx] + ':' + label[idx:]
            
            # Check for numeric qualities like 7, 9, etc.
            for i, c in enumerate(label):
                if i > 0 and c.isdigit():
                    return label[:i] + ':' + label[i:]
                    
        return label

    def set_chord_mapping(self, chord_mapping):
        """
        Set the chord mapping to be used for chord index lookup.
        Also initializes extended mappings for variants.
        
        Args:
            chord_mapping (dict): Dictionary mapping chord names to indices
        """
        if not chord_mapping:
            logger.warning("Empty chord mapping provided. Using formula-based approach.")
            self.chord_mapping = {}
            return
            
        # Store a copy of the original mapping
        self.chord_mapping = chord_mapping.copy()
        logger.info(f"Chord mapping set with {len(chord_mapping)} entries")
        
        # Initialize extended mappings (inversions, variants, etc.)
        self.initialize_chord_mapping()

    def initialize_chord_mapping(self, chord_mapping=None):
        """
        Ensure chord_mapping is initialized, with extensive handling of variants
        
        Args:
            chord_mapping: Chord mapping to use, if specified
        """
        if chord_mapping:
            self.chord_mapping = chord_mapping.copy()  # Create a copy to avoid modifying the original
            logger.info(f"Chord mapping set with {len(chord_mapping)} entries")
        elif not hasattr(self, 'chord_mapping') or not self.chord_mapping:
            # Create empty mapping if none exists
            self.chord_mapping = {}
            logger.warning("Using empty chord mapping - chord lookups will use formula-based approach")
        
        # Make a copy of the original mapping before we start adding to it
        original_mapping = dict(self.chord_mapping)
        
        # ---- PART 1: ADD BARE ROOTS FOR ENHARMONIC EQUIVALENTS ----
        
        # Define the flat and sharp equivalents including more variants
        all_enharmonics = [
            ('Bb', 'A#'), ('Db', 'C#'), ('Eb', 'D#'), ('Gb', 'F#'), ('Ab', 'G#')
        ]
        
        # First make sure ALL bare roots are correctly mapped (both ways)
        for flat, sharp in all_enharmonics:
            # If EITHER form exists, ensure BOTH forms are mapped
            if flat in self.chord_mapping:
                flat_idx = self.chord_mapping[flat]
                self.chord_mapping[sharp] = flat_idx
                logger.debug(f"Added bare root enharmonic: {sharp} -> {flat_idx}")
            elif sharp in self.chord_mapping:
                sharp_idx = self.chord_mapping[sharp]
                self.chord_mapping[flat] = sharp_idx
                logger.debug(f"Added bare root enharmonic: {flat} -> {sharp_idx}")
            
            # Also ensure the :maj forms exist
            flat_maj = f"{flat}:maj"
            sharp_maj = f"{sharp}:maj"
            
            if flat in self.chord_mapping and flat_maj not in self.chord_mapping:
                self.chord_mapping[flat_maj] = self.chord_mapping[flat]
                logger.debug(f"Added maj form: {flat_maj} -> {self.chord_mapping[flat]}")
            
            if sharp in self.chord_mapping and sharp_maj not in self.chord_mapping:
                self.chord_mapping[sharp_maj] = self.chord_mapping[sharp]
                logger.debug(f"Added maj form: {sharp_maj} -> {self.chord_mapping[sharp]}")
            
            # Ensure bare roots are mapped to :maj forms
            if flat_maj in self.chord_mapping and flat not in self.chord_mapping:
                self.chord_mapping[flat] = self.chord_mapping[flat_maj]
                logger.debug(f"Added bare root from maj: {flat} -> {self.chord_mapping[flat_maj]}")
            
            if sharp_maj in self.chord_mapping and sharp not in self.chord_mapping:
                self.chord_mapping[sharp] = self.chord_mapping[sharp_maj]
                logger.debug(f"Added bare root from maj: {sharp} -> {self.chord_mapping[sharp_maj]}")
                
        # ---- PART 2: GENERATE ALL QUALITY VARIANTS FOR ENHARMONICS ----
        
        # Generate all quality variants for enharmonic equivalents
        qualities = ['', ':maj', ':min', ':7', ':maj7', ':min7', ':dim', ':aug', ':sus2', ':sus4', 
                    ':6', ':9', ':maj9', ':min9', ':13', ':maj13', ':min13', ':dim7', ':hdim7', 
                    ':minmaj7', ':sus', ':11', ':maj11', ':min11']
        
        # For each enharmonic pair, make sure all qualities are mapped
        for flat, sharp in all_enharmonics:
            # For each quality, map between flat and sharp
            for q in qualities:
                flat_chord = flat + q
                sharp_chord = sharp + q
                
                # Map from flat to sharp
                if flat_chord in self.chord_mapping and sharp_chord not in self.chord_mapping:
                    self.chord_mapping[sharp_chord] = self.chord_mapping[flat_chord]
                    logger.debug(f"Added enharmonic quality: {sharp_chord} -> {self.chord_mapping[flat_chord]}")
                # Map from sharp to flat
                elif sharp_chord in self.chord_mapping and flat_chord not in self.chord_mapping:
                    self.chord_mapping[flat_chord] = self.chord_mapping[sharp_chord]
                    logger.debug(f"Added enharmonic quality: {flat_chord} -> {self.chord_mapping[sharp_chord]}")
                
                # Also handle all inversions for each quality
                for inversion in ['2', '3', '4', '5', '6', '7', '9', 'b2', 'b3', 'b5', 'b6', 'b7', 'b9', '#5', '#9']:
                    flat_inv = f"{flat_chord}/{inversion}"
                    sharp_inv = f"{sharp_chord}/{inversion}"
                    
                    # Map flat inversions to their sharp equivalents
                    if flat_chord in self.chord_mapping:
                        self.chord_mapping[flat_inv] = self.chord_mapping[flat_chord]
                        self.chord_mapping[sharp_inv] = self.chord_mapping[flat_chord]
                    elif sharp_chord in self.chord_mapping:
                        self.chord_mapping[sharp_inv] = self.chord_mapping[sharp_chord]
                        self.chord_mapping[flat_inv] = self.chord_mapping[sharp_chord]
        
        # ---- PART 3: HANDLE MAJOR/MINOR BASE FORM VARIATIONS ----
        
        # Handle implicit and explicit major chord forms
        for root in ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']:
            # Handle both root forms (G and G:maj)
            root_form = root
            maj_form = f"{root}:maj"
            
            # Map between the two forms
            if root_form in self.chord_mapping and maj_form not in self.chord_mapping:
                self.chord_mapping[maj_form] = self.chord_mapping[root_form]
                logger.debug(f"Added major mapping: {maj_form} -> {self.chord_mapping[root_form]}")
            elif maj_form in self.chord_mapping and root_form not in self.chord_mapping:
                self.chord_mapping[root_form] = self.chord_mapping[maj_form]
                logger.debug(f"Added major mapping: {root_form} -> {self.chord_mapping[maj_form]}")
        
        # ---- PART 4: HANDLE INVERSIONS ----
        
        # Create a new mapping of chords to process inversions and other variants
        extended_mapping = dict(self.chord_mapping)
        
        # Process all chords in our mapping to add their inversions
        for chord, idx in extended_mapping.items():
            # Skip special chords like N or X
            if chord in ['N', 'X']:
                continue
                
            # Extract the root and quality
            if ':' in chord:
                root, quality = chord.split(':', 1)
            else:
                root = chord
                quality = 'maj'  # Implicit major
            
            # Handle basic inversions: map G/3, G/5, G/7, etc. to G
            # Include ALL common inversion numbers and bass notes
            for inversion in ['2', '3', '4', '5', '6', '7', '9', 'b2', 'b3', 'b5', 'b6', 'b7', 'b9', '#5', '#9']:
                # For quality-specified chords (e.g., "E:min/5")
                inv_chord = f"{root}:{quality}/{inversion}"
                if inv_chord not in self.chord_mapping:
                    self.chord_mapping[inv_chord] = idx
                    logger.debug(f"Added inversion mapping: {inv_chord} -> {idx}")
                
                # For implicit major chords (e.g., "A/3")
                if quality == 'maj':
                    implicit_inv = f"{root}/{inversion}"
                    if implicit_inv not in self.chord_mapping:
                        self.chord_mapping[implicit_inv] = idx
                        logger.debug(f"Added implicit major inversion: {implicit_inv} -> {idx}")
        
        # ---- PART 5: HANDLE EXTENSIONS AND ALTERATIONS ----
        
        # Map extended chords (9, 11, 13) to their base 7th chord forms
        extension_mappings = {
            'maj9': 'maj7',      # Cmaj9 -> Cmaj7
            'maj11': 'maj7',     # Cmaj11 -> Cmaj7
            'maj13': 'maj7',     # Cmaj13 -> Cmaj7
            'maj9(*7)': 'maj7',  # Cmaj9(*7) -> Cmaj7
            '9': '7',            # C9 -> C7
            '11': '7',           # C11 -> C7
            '13': '7',           # C13 -> C7
            '9(11)': '7',        # C9(11) -> C7
            'min9': 'min7',      # Cmin9 -> Cmin7
            'min11': 'min7',     # Cmin11 -> Cmin7
            'min13': 'min7',     # Cmin13 -> Cmin7
            '7(#9)': '7',        # C7(#9) -> C7
            '7(b9)': '7',        # C7(b9) -> C7
            '7(#5)': '7',        # C7(#5) -> C7
            'maj(9)': 'maj7',    # Cmaj(9) -> Cmaj7
            'maj(b9)': 'maj7',   # Cmaj(b9) -> Cmaj7
            'maj(4)': 'maj',     # Cmaj(4) -> Cmaj
            'dim/b3': 'dim',     # Cdim/b3 -> Cdim
            'sus4(2)': 'sus4',   # Csus4(2) -> Csus4
            'sus4(9)': 'sus4',   # Csus4(9) -> Csus4
            'sus4(b7)': 'sus4',  # Csus4(b7) -> Csus4
            'sus2(b7)': 'sus2',  # Csus2(b7) -> Csus2
            '(1)': 'maj',        # C(1) -> C
            '(1,5)': 'maj',      # C(1,5) -> C
            '(1,b7)': '7',       # C(1,b7) -> C7
            '(1,b3)': 'min',     # C(1,b3) -> Cmin
            '(1,4,b7)': '7',     # C(1,4,b7) -> C7
            'maj(*5)': 'maj',    # Cmaj(*5) -> C
            'min(*b3)': 'min',   # Cmin(*b3) -> Cmin
            'min(*b3,*5)': 'min' # Cmin(*b3,*5) -> Cmin
        }
        
        # Adding extended mappings for each root note
        all_roots = []
        # Include all possible root notes, both sharps and flats
        for root in ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']:
            if root not in all_roots:
                all_roots.append(root)
                
        for root in all_roots:
            for ext, base in extension_mappings.items():
                ext_chord = f"{root}:{ext}"
                base_chord = f"{root}:{base}"
                
                # Try both with and without colon for the base chord (for major quality)
                if base_chord in self.chord_mapping and ext_chord not in self.chord_mapping:
                    self.chord_mapping[ext_chord] = self.chord_mapping[base_chord]
                    logger.debug(f"Added extension mapping: {ext_chord} -> {self.chord_mapping[base_chord]}")
                
                # Special case for 'maj' quality which might be implicit
                if base == 'maj' and root in self.chord_mapping and ext_chord not in self.chord_mapping:
                    self.chord_mapping[ext_chord] = self.chord_mapping[root]
                    logger.debug(f"Added extension to implicit major: {ext_chord} -> {self.chord_mapping[root]}")
                
                # Also handle inversions of extended chords - CRITICAL for patterns like "E:9/5"
                for inversion in ['2', '3', '4', '5', '6', '7', '9', 'b2', 'b3', 'b5', 'b6', 'b7', 'b9', '#5', '#9']:
                    ext_inv_chord = f"{root}:{ext}/{inversion}"
                    if base_chord in self.chord_mapping and ext_inv_chord not in self.chord_mapping:
                        self.chord_mapping[ext_inv_chord] = self.chord_mapping[base_chord]
                        logger.debug(f"Added extension inversion: {ext_inv_chord} -> {self.chord_mapping[base_chord]}")
                    
                    # Handle special case for maj quality
                    if base == 'maj' and root in self.chord_mapping and ext_inv_chord not in self.chord_mapping:
                        self.chord_mapping[ext_inv_chord] = self.chord_mapping[root]
                        logger.debug(f"Added extension inversion to implicit major: {ext_inv_chord} -> {self.chord_mapping[root]}")
        
        # ---- PART 6: VERIFY PROBLEMATIC CHORDS ----
        
        # Verify common problematic chords seen in logs
        problem_chords = [
            # Bare enharmonics
            'Bb', 'Eb', 'Db', 'Gb', 'Ab',
            # Common inversions
            'G/3', 'A/3', 'E:min/5', 'A/5', 'E:min/b7', 'D/5', 'C#:min/5',
            # Altered and extended chords
            'E:9', 'D:9', 'F:9', 'C:9', 'D:min9', 'F:maj9(*7)', 'C:maj(4)', 'E:(1)',
            'A:min/b3', 'E:7/3', 'A:7/b7', 'A:7/5', 'G:sus4(b7)', 'F#:(1,4,b7)'
        ]
        
        # Verify the mappings and log results
        logger.info("\n==== Verifying Critical Chord Mappings ====")
        mapped_count = 0
        for chord in problem_chords:
            if chord in self.chord_mapping:
                mapped_count += 1
                logger.info(f"  ✓ {chord} -> {self.chord_mapping[chord]}")
            else:
                logger.warning(f"  ✗ {chord} -> Not mapped")
                
                # Try to find the root/quality to see where mapping is failing
                if ':' in chord:
                    root, quality = chord.split(':', 1)
                    if '/' in quality:
                        quality, _ = quality.split('/', 1)
                    
                    # Check if we have the base quality
                    base_chord = f"{root}:{quality}"
                    if base_chord in self.chord_mapping:
                        logger.warning(f"    - Base chord {base_chord} IS mapped -> {self.chord_mapping[base_chord]}")
                    
                    # If enharmonic, check alternative form
                    if root in self.enharmonic_map:
                        alt_root = self.enharmonic_map[root]
                        alt_chord = f"{alt_root}:{quality}"
                        if alt_chord in self.chord_mapping:
                            logger.warning(f"    - Enharmonic {alt_chord} IS mapped -> {self.chord_mapping[alt_chord]}")
                
        # Log success rate
        logger.info(f"Successfully mapped {mapped_count}/{len(problem_chords)} critical chords")
        logger.info(f"Chord mapping expanded from {len(original_mapping)} to {len(self.chord_mapping)} entries")
    
    def get_chord_idx(self, chord_name, use_large_voca=False):
        """
        Get chord index with enhanced fallback and simplification logic
        
        Args:
            chord_name (str): Chord name to look up
            use_large_voca (bool): Whether to use large vocabulary
            
        Returns:
            int: Chord index
        """
        # Handle special cases
        if chord_name == "N" or chord_name == "NC":
            return 169 if use_large_voca else 24
        if chord_name == "X":
            return 168 if use_large_voca else 25
        
        # Direct lookup first (most efficient)
        if chord_name in self.chord_mapping:
            return self.chord_mapping[chord_name]
        
        # FALLBACK 1: Try both implicit and explicit forms for major chords
        if ':' not in chord_name and '/' not in chord_name:
            # This is a bare root, try with explicit major
            maj_form = f"{chord_name}:maj"
            if maj_form in self.chord_mapping:
                logger.debug(f"Using explicit major: {chord_name} -> {maj_form}")
                return self.chord_mapping[maj_form]
        elif ':maj' in chord_name:
            # This is an explicit major, try implicit form
            root = chord_name.split(':')[0]
            if root in self.chord_mapping:
                logger.debug(f"Using implicit major: {chord_name} -> {root}")
                return self.chord_mapping[root]
                
        # FALLBACK 2: Try enharmonic equivalents
        enharmonic = self._get_enharmonic(chord_name)
        if enharmonic and enharmonic in self.chord_mapping:
            logger.debug(f"Using enharmonic: {chord_name} -> {enharmonic}")
            return self.chord_mapping[enharmonic]
        
        # FALLBACK 3: Try removing inversions
        if '/' in chord_name:
            base_chord = chord_name.split('/')[0]
            if base_chord in self.chord_mapping:
                logger.debug(f"Removed inversion: {chord_name} -> {base_chord}")
                return self.chord_mapping[base_chord]
            
            # Try enharmonic of base chord
            base_enharmonic = self._get_enharmonic(base_chord)
            if base_enharmonic and base_enharmonic in self.chord_mapping:
                logger.debug(f"Using enharmonic base: {chord_name} -> {base_enharmonic}")
                return self.chord_mapping[base_enharmonic]
        
        # FALLBACK 4: Try simplification (more complex/slower)
        simplified_chord = self._simplify_chord(chord_name)
        if simplified_chord in self.chord_mapping:
            logger.debug(f"Using simplified: {chord_name} -> {simplified_chord}")
            return self.chord_mapping[simplified_chord]
            
        # FALLBACK 5: Try simplified enharmonic
        simplified_enharmonic = self._get_enharmonic(simplified_chord)
        if simplified_enharmonic and simplified_enharmonic in self.chord_mapping:
            logger.debug(f"Using simplified enharmonic: {chord_name} -> {simplified_enharmonic}")
            return self.chord_mapping[simplified_enharmonic]
        
        # FALLBACK 6: Try to extract root and directly map to major/minor
        if ':' in chord_name:
            root, quality = chord_name.split(':', 1)
            if '/' in quality:  # Remove inversion if present
                quality = quality.split('/')[0]
                
            # For minor-like chords
            if 'min' in quality or 'm' == quality or 'dim' in quality:
                minor_form = f"{root}:min"
                if minor_form in self.chord_mapping:
                    logger.debug(f"Mapped to minor: {chord_name} -> {minor_form}")
                    return self.chord_mapping[minor_form]
                    
            # For dominant-like chords
            if '7' in quality or '9' in quality or '11' in quality or '13' in quality:
                dominant_form = f"{root}:7"
                if dominant_form in self.chord_mapping:
                    logger.debug(f"Mapped to dominant: {chord_name} -> {dominant_form}")
                    return self.chord_mapping[dominant_form]
                    
            # For major-like chords
            if quality.startswith('maj') or quality in ['', '6', 'sus2', 'sus4']:
                if root in self.chord_mapping:
                    logger.debug(f"Mapped to major: {chord_name} -> {root}")
                    return self.chord_mapping[root]
        
        # If everything fails, map to unknown chord
        logger.warning(f"Could not map chord: {chord_name}, using unknown chord index")
        return 169 if use_large_voca else 24
    
    def _get_enharmonic(self, chord_name):
        """
        Get enharmonic equivalent of a chord name if possible
        
        Args:
            chord_name (str): Chord name to find enharmonic for
            
        Returns:
            str: Enharmonic equivalent or None
        """
        if not chord_name or chord_name in ["N", "X", "NC"]:
            return None
            
        # Define flat-sharp conversion map
        enharmonic_map = {
            'Bb': 'A#', 'A#': 'Bb',
            'Db': 'C#', 'C#': 'Db',
            'Eb': 'D#', 'D#': 'Eb',
            'Gb': 'F#', 'F#': 'Gb',
            'Ab': 'G#', 'G#': 'Ab',
        }
        
        # Handle chord without quality (e.g., "Bb")
        if ':' not in chord_name and '/' not in chord_name:
            return enharmonic_map.get(chord_name)
            
        # Handle chord with quality and/or bass (e.g., "Bb:maj7/5")
        parts = chord_name.split(':')
        root = parts[0]
        
        if root in enharmonic_map:
            # Replace root with enharmonic equivalent
            enharmonic_root = enharmonic_map[root]
            if len(parts) == 1:
                return enharmonic_root
            else:
                # Keep the quality and bass parts intact
                return f"{enharmonic_root}:{parts[1]}"
                
        return None

    def get_chord_idx(self, chord_name, use_large_voca=False):
        """
        Get the index for a chord name, with fallbacks for unknown chords.
        
        Args:
            chord_name (str): Chord name to look up
            use_large_voca (bool): Whether to use large vocabulary mode
            
        Returns:
            int: Chord index
        """
        # Special cases
        if chord_name == "N" or chord_name == "NC":
            return 169 if use_large_voca else 24
        if chord_name == "X":
            return 168 if use_large_voca else 25
            
        # Direct lookup (most efficient)
        if chord_name in self.chord_mapping:
            return self.chord_mapping[chord_name]
            
        # Try normalized form
        normalized, root, quality, _ = self.normalize_chord(chord_name)
        if normalized in self.chord_mapping:
            return self.chord_mapping[normalized]
            
        # Try simplified form
        simplified = self.simplify_chord(chord_name)
        if simplified in self.chord_mapping:
            return self.chord_mapping[simplified]
            
        # If root exists, try basic forms
        if root:
            # Try major form of root (both implicit and explicit)
            if root in self.chord_mapping:
                return self.chord_mapping[root]
            elif f"{root}:maj" in self.chord_mapping:
                return self.chord_mapping[f"{root}:maj"]
                
            # If it's a major quality variant, try using just the root
            if quality and (quality.startswith('maj') or quality in ['6', '9', '13']):
                if root in self.chord_mapping:
                    return self.chord_mapping[root]
                    
            # If it's a dominant quality, try using root:7
            if quality and ('7' in quality or '9' in quality or '11' in quality or '13' in quality):
                if f"{root}:7" in self.chord_mapping:
                    return self.chord_mapping[f"{root}:7"]
                    
            # If it's a minor quality variant, try using root:min
            if quality and ('min' in quality or 'm' == quality):
                if f"{root}:min" in self.chord_mapping:
                    return self.chord_mapping[f"{root}:min"]
        
        # Log warning and return unknown chord as last resort
        logger.warning(f"Could not map chord: {chord_name}")
        return 169 if use_large_voca else 24  # Default to no-chord

    def convert_to_id(self, root, is_major):
        """Legacy method for small vocabulary (25 chords)"""
        if root == -1:
            return 24  # No chord
        else:
            if is_major:
                return root * 2  # Even indices for major
            else:
                return root * 2 + 1  # Odd indices for minor

    def convert_to_id_voca(self, root, quality):
        """Legacy method for large vocabulary (170 chords)"""
        if root == -1:
            return 169  # No chord
            
        # First normalize the quality
        quality = self.normalize_quality(quality) if hasattr(self, 'normalize_quality') else quality
        
        # Try mapping directly if available
        chord_key = f"{PITCH_CLASS[root]}:{quality}" if quality and quality != 'maj' else PITCH_CLASS[root]
        if hasattr(self, 'chord_mapping') and chord_key in self.chord_mapping:
            return self.chord_mapping[chord_key]
            
        # Fall back to formula-based mapping
        if quality == 'min':
            return root * 14
        elif quality == 'maj' or not quality:
            return root * 14 + 1
        elif quality == 'dim':
            return root * 14 + 2
        elif quality == 'aug':
            return root * 14 + 3
        elif quality == 'min6':
            return root * 14 + 4
        elif quality == 'maj6':
            return root * 14 + 5
        elif quality == 'min7':
            return root * 14 + 6
        elif quality == 'minmaj7':
            return root * 14 + 7
        elif quality == 'maj7':
            return root * 14 + 8
        elif quality == '7':
            return root * 14 + 9
        elif quality == 'dim7':
            return root * 14 + 10
        elif quality == 'hdim7':
            return root * 14 + 11
        elif quality == 'sus2':
            return root * 14 + 12
        elif quality == 'sus4':
            return root * 14 + 13
        else:
            return 168  # Unknown chord

    def assign_chord_id(self, entry):
        """Legacy method for compatibility"""
        df = pd.DataFrame(data=entry[['root', 'is_major']])
        df['chord_id'] = df.apply(lambda row: self.convert_to_id(row['root'], row['is_major']), axis=1)
        return df

    def get_converted_chord(self, filename):
        """Legacy method for compatibility"""
        loaded_chord = self.load_chords(filename)
        triads = self.reduce_to_triads(loaded_chord['chord'])
        df = self.assign_chord_id(triads)
        df['start'] = loaded_chord['start']
        df['end'] = loaded_chord['end']
        return df

    def get_converted_chord_voca(self, filename):
        """Legacy method for compatibility"""
        # ...existing code...


def idx2voca_chord():
    """
    Create a mapping from chord index to chord label.
    Adjust the mapping as needed to match your evaluation conventions.
    """
    # For example, assuming indices 0..167 map to chords via convert_to_id_voca,
    # and 168, 169 map to unknown and no-chord respectively.
    mapping = {}
    for i in range(168):
        # This assumes a predefined order; customize as needed.
        root = i // 14
        quality_idx = i % 14
        quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']
        # Use chords from PITCH_CLASS defined in this module.
        label = PITCH_CLASS[root] + (":" + quality_list[quality_idx] if quality_idx != 1 else "")
        mapping[i] = label
    mapping[168] = "X"
    mapping[169] = "N"
    return mapping