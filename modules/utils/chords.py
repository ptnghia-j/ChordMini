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
        Initialize chord mapping including enharmonic equivalents, simplified forms,
        and inversions mapping to root position.

        Args:
            chord_mapping (dict, optional): Base chord mapping from string representations to indices.
                                           If None, uses the existing self.chord_mapping.
        """
        # Use provided mapping or existing mapping
        base_mapping = chord_mapping if chord_mapping else self.chord_mapping
        if not base_mapping:
            logger.warning("Cannot initialize chord mapping: No base mapping provided or set.")
            self.chord_mapping = {} # Ensure it's at least an empty dict
            return

        # Start with a fresh copy of the base mapping
        self.chord_mapping = base_mapping.copy()
        logger.info(f"Initializing chord mapping with {len(self.chord_mapping)} base entries.")

        # Define enharmonic equivalents (both directions)
        enharmonic_map = {
            'Bb': 'A#', 'Eb': 'D#', 'Ab': 'G#', 'Db': 'C#', 'Gb': 'F#',
            'A#': 'Bb', 'D#': 'Eb', 'G#': 'Ab', 'C#': 'Db', 'F#': 'Gb'
        }

        # Store additions temporarily to avoid modifying during iteration
        extended_mapping = {}

        for chord, idx in base_mapping.items():
            # Skip special chords N, X
            if chord in ['N', 'X']:
                continue

            try:
                # Normalize and simplify the original chord from the mapping
                # This helps establish the canonical form for this index
                normalized, root, quality, bass = self.normalize_chord(chord)
                simplified_chord = self.simplify_chord(chord) # Gets root position, simplified quality

                # 1. Add mapping for the explicitly simplified form
                if simplified_chord not in self.chord_mapping:
                    extended_mapping[simplified_chord] = idx

                # 2. Add mapping for the normalized form (if different and not present)
                if normalized != chord and normalized not in self.chord_mapping:
                     extended_mapping[normalized] = idx

                # 3. Add mappings for enharmonic roots (for original, normalized, and simplified)
                if root and root in enharmonic_map:
                    enharmonic_root = enharmonic_map[root]

                    # Enharmonic of original chord (if it has root, quality, bass structure)
                    if quality is not None or bass is not None:
                        enharmonic_original = f"{enharmonic_root}"
                        if quality: enharmonic_original += f":{quality}"
                        if bass: enharmonic_original += f"/{bass}"
                        if enharmonic_original not in self.chord_mapping:
                             extended_mapping[enharmonic_original] = idx

                    # Enharmonic of normalized chord
                    if normalized:
                        norm_root, norm_qual, norm_bass = self.normalize_chord(normalized)[1:]
                        if norm_root and norm_root in enharmonic_map: # Check again after normalization
                            enh_norm_root = enharmonic_map[norm_root]
                            enharmonic_normalized = f"{enh_norm_root}"
                            if norm_qual: enharmonic_normalized += f":{norm_qual}"
                            # Bass is removed by simplify_chord, don't add it here for normalized simplified
                            # if norm_bass: enharmonic_normalized += f"/{norm_bass}" # Bass already removed by simplify
                            if enharmonic_normalized not in self.chord_mapping:
                                 extended_mapping[enharmonic_normalized] = idx

                    # Enharmonic of simplified chord
                    if simplified_chord:
                        simp_root, simp_qual, _ = self.normalize_chord(simplified_chord)[1:] # Bass is already removed
                        if simp_root and simp_root in enharmonic_map:
                            enh_simp_root = enharmonic_map[simp_root]
                            enharmonic_simplified = f"{enh_simp_root}"
                            if simp_qual: enharmonic_simplified += f":{simp_qual}"
                            if enharmonic_simplified not in self.chord_mapping:
                                 extended_mapping[enharmonic_simplified] = idx

                # 4. Ensure root position maps correctly even if only inverted forms are in base map
                if bass:
                    root_pos_chord = chord.split('/')[0]
                    if root_pos_chord not in self.chord_mapping:
                         extended_mapping[root_pos_chord] = idx
                    # Also add simplified root position
                    if simplified_chord != root_pos_chord and simplified_chord not in self.chord_mapping:
                         extended_mapping[simplified_chord] = idx


            except Exception as e:
                logger.warning(f"Skipping initialization for base chord '{chord}' due to error: {e}")

        # Update the main mapping with all additions
        original_size = len(self.chord_mapping)
        self.chord_mapping.update(extended_mapping)
        added_count = len(self.chord_mapping) - original_size
        logger.info(f"Enhanced chord mapping with {added_count} additional variants (total: {len(self.chord_mapping)})")
        # Log a few examples of added mappings if debugging needed
        # if added_count > 0 and logger.isEnabledFor(logging.DEBUG):
        #     logger.debug(f"Sample added mappings: {dict(list(extended_mapping.items())[:5])}")


    def get_chord_idx(self, chord_name, use_large_voca=False):
        """
        Get chord index using comprehensive mapping strategy.
        Handles enharmonic equivalents, simplification of extensions, and inversions.

        Args:
            chord_name (str): Chord name to look up
            use_large_voca (bool): Whether to use large vocabulary (170 chords)

        Returns:
            int: Chord index
        """
        # Define N/X indices based on vocabulary size
        n_chord_idx = 169 if use_large_voca else 24
        x_chord_idx = 168 if use_large_voca else 25

        # Handle empty or explicit N/X chords first
        if not chord_name or chord_name in ["N", "NC"]:
            return n_chord_idx
        if chord_name == "X":
            return x_chord_idx

        # Step 0: Basic error correction
        chord_name = self.label_error_modify(chord_name)

        # Step 1: Direct lookup in the potentially extended mapping
        if chord_name in self.chord_mapping:
            return self.chord_mapping[chord_name]

        # Step 2: Try normalized form
        try:
            normalized_chord, root, quality, bass = self.normalize_chord(chord_name)
            if normalized_chord != chord_name and normalized_chord in self.chord_mapping:
                return self.chord_mapping[normalized_chord]

            # Step 3: Try simplified form (removes inversion, reduces extensions)
            simplified_chord = self.simplify_chord(chord_name) # Use the potentially modified chord_name
            if simplified_chord != chord_name and simplified_chord != normalized_chord and simplified_chord in self.chord_mapping:
                return self.chord_mapping[simplified_chord]

            # Step 4: Try enharmonic versions of simplified and normalized forms
            # (initialize_chord_mapping should ideally cover this, but as a fallback)
            enharmonic_map = {
                'Bb': 'A#', 'Eb': 'D#', 'Ab': 'G#', 'Db': 'C#', 'Gb': 'F#',
                'A#': 'Bb', 'D#': 'Eb', 'G#': 'Ab', 'C#': 'Db', 'F#': 'Gb'
            }

            # Try enharmonic of simplified
            simp_root, simp_qual, _ = self.normalize_chord(simplified_chord)[1:]
            if simp_root and simp_root in enharmonic_map:
                enh_simp_root = enharmonic_map[simp_root]
                enh_simplified = f"{enh_simp_root}" + (f":{simp_qual}" if simp_qual else "")
                if enh_simplified in self.chord_mapping:
                    return self.chord_mapping[enh_simplified]

            # Try enharmonic of normalized (root position)
            norm_root, norm_qual, _ = self.normalize_chord(normalized.split('/')[0])[1:] # Use root position
            if norm_root and norm_root in enharmonic_map:
                enh_norm_root = enharmonic_map[norm_root]
                enh_normalized_root_pos = f"{enh_norm_root}" + (f":{norm_qual}" if norm_qual else "")
                if enh_normalized_root_pos in self.chord_mapping:
                    return self.chord_mapping[enh_normalized_root_pos]

        except Exception as e:
             logger.debug(f"Error during intermediate lookup for '{chord_name}': {e}") # Debug level

        # Step 5: If still not found, log warning and return unknown index
        # Avoid overly verbose logging for common misses during training
        # logger.warning(f"Could not map chord: '{chord_name}'. Returning unknown index.")
        return x_chord_idx # Use unknown chord index

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