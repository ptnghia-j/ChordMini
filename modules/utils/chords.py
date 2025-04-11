# encoding: utf-8
"""
This module contains chord evaluation functionality.

It provides the evaluation measures used for the MIREX ACE task, and
tries to follow [1]_ and [2]_ as closely as possible.

Notes
-----
This implementation tries to follow the references and their implementation
(e.g., https://github.com/jpauwels/MusOOEvaluator for [2]_). However, there
are some known (and possibly some unknown) differences. If you find one not
listed in the following, please file an issue:

 - Detected chord segments are adjusted to fit the length of the annotations.
   In particular, this means that, if necessary, filler segments of 'no chord'
   are added at beginnings and ends. This can result in different segmentation
   scores compared to the original implementation.

References
----------
.. [1] Christopher Harte, "Towards Automatic Extraction of Harmony Information
       from Music Signals." Dissertation,
       Department for Electronic Engineering, Queen Mary University of London,
       2010.
.. [2] Johan Pauwels and Geoffroy Peeters.
       "Evaluating Automatically Estimated Chord Sequences."
       In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

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
            '6': self.interval_list('(1,6)'),  # custom
            '5': self.interval_list('(1,5)'),
            '4': self.interval_list('(1,4)'),  # custom
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

        # Add enharmonic mappings for flat to sharp conversions
        self.enharmonic_map = {
            'Bb': 'A#',
            'Db': 'C#',
            'Eb': 'D#',
            'Gb': 'F#',
            'Ab': 'G#',
        }

        # Add quality name normalization mappings to match teacher model's vocabulary
        self.quality_map = {
            '': 'maj',   # Empty string means major chord
            'M': 'maj',  # Alternative notation for major
            'm': 'min',  # Alternative notation for minor
            'major': 'maj',
            'minor': 'min',
            'maj7+5': 'maj7',  # Simplify some extended chords
            'maj7-5': 'maj7',
            'min7+5': 'min7',
            'min7-5': 'min7',
        }

    def chords(self, labels):

        """
        Transform a list of chord labels into an array of internal numeric
        representations.

        Parameters
        ----------
        labels : list
            List of chord labels (str).

        Returns
        -------
        chords : numpy.array
            Structured array with columns 'root', 'bass', and 'intervals',
            containing a numeric representation of chords.

        """
        crds = np.zeros(len(labels), dtype=CHORD_DTYPE)
        cache = {}
        for i, lbl in enumerate(labels):
            cv = cache.get(lbl, None)
            if cv is None:
                cv = self.chord(lbl)
                cache[lbl] = cv
            crds[i] = cv

        return crds

    def label_error_modify(self, label):
        if label == 'Emin/4': label = 'E:min/4'
        elif label == 'A7/3': label = 'A:7/3'
        elif label == 'Bb7/3': label = 'Bb:7/3'
        elif label == 'Bb7/5': label = 'Bb:7/5'
        elif label.find(':') == -1:
            if label.find('min') != -1:
                label = label[:label.find('min')] + ':' + label[label.find('min'):]
        return label

    def chord(self, label):
        """
        Transform a chord label into the internal numeric represenation of
        (root, bass, intervals array).

        Parameters
        ----------
        label : str
            Chord label.

        Returns
        -------
        chord : tuple
            Numeric representation of the chord: (root, bass, intervals array).

        """

        try:
            is_major = False

            if label == 'N':
                return NO_CHORD
            if label == 'X':
                return UNKNOWN_CHORD

            label = self.label_error_modify(label)

            c_idx = label.find(':')
            s_idx = label.find('/')

            if c_idx == -1:
                quality_str = 'maj'  # Default to major if no quality specified
                if s_idx == -1:
                    root_str = label
                    bass_str = ''
                else:
                    root_str = label[:s_idx]
                    bass_str = label[s_idx + 1:]
            else:
                root_str = label[:c_idx]
                if s_idx == -1:
                    quality_str = label[c_idx + 1:]
                    bass_str = ''
                else:
                    quality_str = label[c_idx + 1:s_idx]
                    bass_str = label[s_idx + 1:]

            # Normalize root string for enharmonic equivalents
            root_str = self.normalize_root(root_str)
            
            # Normalize quality string
            quality_str = self.normalize_quality(quality_str)
            
            root = self.pitch(root_str)
            bass = self.interval(bass_str) if bass_str else 0
            ivs = self.chord_intervals(quality_str)
            ivs[bass] = 1

            if 'min' in quality_str:
                is_major = False
            else:
                is_major = True

        except Exception as e:
            print(e, label)

        return root, bass, ivs, is_major

    def normalize_root(self, root_str):
        """
        Normalize a root note by converting flat notes to their enharmonic sharp equivalents.
        This helps with consistent chord vocabulary lookups.
        
        Parameters
        ----------
        root_str : str
            Root note string (e.g. 'Bb', 'Ab', 'F#')
            
        Returns
        -------
        normalized_root : str
            Normalized root note, preferring sharps over flats
        """
        return self.enharmonic_map.get(root_str, root_str)
        
    def normalize_quality(self, quality_str):
        """
        Normalize quality string to match vocabulary standards.
        
        Parameters
        ----------
        quality_str : str
            Quality string (e.g., '', 'm', 'maj7')
            
        Returns
        -------
        normalized_quality : str
            Normalized quality string matching vocabulary
        """
        # Handle parentheses and additional indicators
        # Strip away scale degree indicators like (*3,*5)
        base_quality = quality_str.split('(')[0].strip()
        
        # Apply quality mapping
        return self.quality_map.get(base_quality, base_quality)

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

    # mapping of shorthand interval notations to the actual interval representation

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

    def convert_to_id(self, root, is_major):
        if root == -1:
            return 24
        else:
            if is_major:
                return root * 2
            else:
                return root * 2 + 1

    def get_converted_chord(self, filename):
        loaded_chord = self.load_chords(filename)
        triads = self.reduce_to_triads(loaded_chord['chord'])

        df = self.assign_chord_id(triads)
        df['start'] = loaded_chord['start']
        df['end'] = loaded_chord['end']

        return df

    def assign_chord_id(self, entry):
        # maj, min chord only
        # if you want to add other chord, change this part and get_converted_chord(reduce_to_triads)
        df = pd.DataFrame(data=entry[['root', 'is_major']])
        df['chord_id'] = df.apply(lambda row: self.convert_to_id(row['root'], row['is_major']), axis=1)
        return df

    def convert_to_id_voca(self, root, quality):
        if root == -1:
            return 169  # "N" (no chord) is mapped to 169
        else:
            # First normalize the quality for better matching
            quality = self.normalize_quality(quality)
            
            # Handle extended chord qualities by reducing them to their base quality
            base_quality = self._reduce_extended_quality(quality)
            
            # Special case for major chord: teacher model expects empty string
            if base_quality == 'maj':
                chord_key = PITCH_CLASS[root]
                if chord_key in self.chord_mapping:
                    return self.chord_mapping[chord_key]
                    
            # Try to construct the canonical chord name using the pattern from teacher model
            chord_key = f"{PITCH_CLASS[root]}:{base_quality}"
            if chord_key in self.chord_mapping:
                return self.chord_mapping[chord_key]
                
            # If still not found, use the original algorithm
            if base_quality == 'min':
                return root * 14
            elif base_quality == 'maj':
                return root * 14 + 1
            elif base_quality == 'dim':
                return root * 14 + 2
            elif base_quality == 'aug':
                return root * 14 + 3
            elif base_quality == 'min6':
                return root * 14 + 4
            elif base_quality == 'maj6':
                return root * 14 + 5
            elif base_quality == 'min7':
                return root * 14 + 6
            elif base_quality == 'minmaj7':
                return root * 14 + 7
            elif base_quality == 'maj7':
                return root * 14 + 8
            elif base_quality == '7':
                return root * 14 + 9
            elif base_quality == 'dim7':
                return root * 14 + 10
            elif base_quality == 'hdim7':
                return root * 14 + 11
            elif base_quality == 'sus2':
                return root * 14 + 12
            elif base_quality == 'sus4':
                return root * 14 + 13
            else:
                return 168  # Unknown chord mapped to 168

    def _reduce_extended_quality(self, quality):
        """
        Reduce extended chord qualities to their base quality for better mapping.
        For example, min9 -> min7, maj13 -> maj7, etc.
        """
        # Map extended chords to their base qualities
        if quality in ['min9', 'min11', 'min13']:
            return 'min7'  # Reduce to min7
        elif quality in ['maj9', 'maj11', 'maj13']:
            return 'maj7'  # Reduce to maj7
        elif quality in ['9', '11', '13']:
            return '7'     # Reduce to dominant 7
        elif quality in ['add9', 'add2']:
            return 'maj'   # Reduce to major
        elif quality in ['minAdd9', 'min(add9)']:
            return 'min'   # Reduce to minor
        # Handle special cases: (*3,*5) notation for omitted notes
        elif '(*' in quality:
            # Extract base quality before the parenthesis
            base = quality.split('(')[0]
            return base if base else 'maj'
        
        return quality     # Return unchanged if not an extended chord

    def get_converted_chord_voca(self, filename):
        loaded_chord = self.load_chords(filename)
        triads = self.reduce_to_triads(loaded_chord['chord'])
        df = pd.DataFrame(data=triads[['root', 'is_major']])

        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(filename)
        ref_labels = self.lab_file_error_modify(ref_labels)
        idxs = list()
        
        # Add debugging for chord mapping
        if not hasattr(self, 'chord_mapping'):
            logger.info("Warning: chord_mapping attribute not set. Using formula-based mapping.")
            self.chord_mapping = {}
            
        # DEBUG: Add logging to show what's in the chord mapping
        logger.debug(f"Chord mapping contains {len(self.chord_mapping)} entries")
        
        for i in ref_labels:
            try:
                # This split returns the root and quality separately
                chord_root, quality, scale_degrees, bass = mir_eval.chord.split(i, reduce_extended_chords=True)
                
                # Apply enharmonic mapping for flat notes
                orig_root = chord_root  # Save original root for debugging
                if chord_root in self.enharmonic_map:
                    # Convert flats to sharps
                    sharp_root = self.enharmonic_map[chord_root]
                    chord_root = sharp_root
                    
                    # Reconstruct i with the sharp notation for use in chord function
                    if quality:
                        i_normalized = f"{sharp_root}:{quality}"
                        if bass:
                            i_normalized += f"/{bass}"
                    else:
                        i_normalized = sharp_root
                        if bass:
                            i_normalized += f"/{bass}"
                    
                    logger.debug(f"Converted {i} to {i_normalized}")
                else:
                    i_normalized = i
                
                # Get the root index and other data from the transformed chord
                root, bass, ivs, is_major = self.chord(i_normalized)
                
                # CRITICAL FIX: Major chords in the teacher model are represented without :maj suffix
                # Try lookup in the following order:
                
                # 1. For major chords, first try without quality (e.g., "C" instead of "C:maj")
                if quality == 'maj' or not quality:
                    lookup_key = PITCH_CLASS[root]
                    if lookup_key in self.chord_mapping:
                        logger.debug(f"Found major chord {i} as {lookup_key} -> {self.chord_mapping[lookup_key]}")
                        idxs.append(self.chord_mapping[lookup_key])
                        continue
                
                # 2. Try with explicit quality
                lookup_key = f"{PITCH_CLASS[root]}:{quality}" if quality else PITCH_CLASS[root]
                if lookup_key in self.chord_mapping:
                    logger.debug(f"Found chord {i} as {lookup_key} -> {self.chord_mapping[lookup_key]}")
                    idxs.append(self.chord_mapping[lookup_key])
                    continue
                    
                # 3. For extended chords, try the reduced base quality
                base_quality = self._reduce_extended_quality(quality)
                base_lookup_key = f"{PITCH_CLASS[root]}:{base_quality}" if base_quality and base_quality != 'maj' else PITCH_CLASS[root]
                if base_lookup_key in self.chord_mapping:
                    logger.debug(f"Found extended chord {i} reduced to {base_lookup_key} -> {self.chord_mapping[base_lookup_key]}")
                    idxs.append(self.chord_mapping[base_lookup_key])
                    continue
                
                # 4. If all else fails, use the numeric formula
                formula_result = self.convert_to_id_voca(root=root, quality=quality)
                logger.debug(f"Using formula for chord {i}: root={root}, quality={quality} -> {formula_result}")
                idxs.append(formula_result)
                
            except Exception as e:
                logger.warning(f"Error processing chord {i}: {e}")
                idxs.append(168)  # Unknown chord
        
        df['chord_id'] = idxs
        df['start'] = loaded_chord['start']
        df['end'] = loaded_chord['end']
        
        return df

    def lab_file_error_modify(self, ref_labels):
        for i in range(len(ref_labels)):
            # Handle various notation formats
            if ref_labels[i][-2:] == ':4':
                ref_labels[i] = ref_labels[i].replace(':4', ':sus4')
            elif ref_labels[i][-2:] == ':6':
                ref_labels[i] = ref_labels[i].replace(':6', ':maj6')
            elif ref_labels[i][-4:] == ':6/2':
                ref_labels[i] = ref_labels[i].replace(':6/2', ':maj6/2')
            # Keep extended chords as-is but remove extensions for lookup
            elif ref_labels[i][-2:] == ':9':
                # Keep the label but will be processed in get_converted_chord_voca
                pass
            elif ref_labels[i][-4:] == ':11':
                # Keep the label but will be processed in get_converted_chord_voca
                pass
            elif ref_labels[i][-4:] == ':13':
                # Keep the label but will be processed in get_converted_chord_voca
                pass
            # Handle specific error cases
            elif ref_labels[i] == 'Emin/4':
                ref_labels[i] = 'E:min/4'
            elif ref_labels[i] == 'A7/3':
                ref_labels[i] = 'A:7/3'
            elif ref_labels[i] == 'Bb7/3':
                ref_labels[i] = 'A#:7/3'  # Convert directly to sharp notation
            elif ref_labels[i] == 'Bb7/5':
                ref_labels[i] = 'A#:7/5'  # Convert directly to sharp notation
            # Handle format without colon separator
            elif ref_labels[i].find(':') == -1:
                if ref_labels[i].find('min') != -1:
                    ref_labels[i] = ref_labels[i][:ref_labels[i].find('min')] + ':' + ref_labels[i][ref_labels[i].find('min'):]
                elif ref_labels[i].find('maj') != -1:
                    ref_labels[i] = ref_labels[i][:ref_labels[i].find('maj')] + ':' + ref_labels[i][ref_labels[i].find('maj'):]
                elif ref_labels[i].find('dim') != -1:
                    ref_labels[i] = ref_labels[i][:ref_labels[i].find('dim')] + ':' + ref_labels[i][ref_labels[i].find('dim'):]
                elif ref_labels[i].find('aug') != -1:
                    ref_labels[i] = ref_labels[i][:ref_labels[i].find('aug')] + ':' + ref_labels[i][ref_labels[i].find('aug'):]
                elif ref_labels[i].find('sus') != -1:
                    ref_labels[i] = ref_labels[i][:ref_labels[i].find('sus')] + ':' + ref_labels[i][ref_labels[i].find('sus'):]
                elif ref_labels[i].find('7') != -1:
                    ref_labels[i] = ref_labels[i][:ref_labels[i].find('7')] + ':' + ref_labels[i][ref_labels[i].find('7'):]
            
            # Handle enharmonic equivalents at the label level
            parts = ref_labels[i].split(':')
            if len(parts) > 1 and parts[0] in self.enharmonic_map:
                ref_labels[i] = self.enharmonic_map[parts[0]] + ':' + parts[1]

        return ref_labels

    def set_chord_mapping(self, chord_mapping):
        """
        Set the chord mapping to be used when looking up chord indices.
        
        Parameters
        ----------
        chord_mapping : dict
            Dictionary mapping chord names to indices
        """
        if not chord_mapping:
            logger.warning("Empty chord mapping provided. This may cause issues with chord recognition.")
            self.chord_mapping = {}
            return
            
        self.chord_mapping = chord_mapping
        logger.info(f"Chord mapping set with {len(chord_mapping)} entries")
        
        # Log some example mappings for debugging purposes
        debug_sample = {
            'Major chords': {k: chord_mapping.get(k) for k in ['C', 'F', 'G'] if k in chord_mapping},
            'Minor chords': {k: chord_mapping.get(k) for k in ['C:min', 'A:min', 'D:min'] if k in chord_mapping},
            'Special chords': {k: chord_mapping.get(k) for k in ['N', 'X'] if k in chord_mapping}
        }
        logger.debug(f"Chord mapping examples: {debug_sample}")
        
        # Check for expected chord mappings
        if 'N' not in chord_mapping:
            logger.warning("No mapping for 'N' (no chord) found. Will default to 24 or 169 depending on vocabulary size.")
            
        # Check if we're using large vocabulary (170 chords) by examining the mapping
        max_index = max(chord_mapping.values()) if chord_mapping else 0
        if max_index > 25:
            logger.info(f"Large vocabulary detected (max index: {max_index})")
        else:
            logger.info(f"Standard vocabulary detected (max index: {max_index})")
        
        # Initialize the enharmonic mapping for consistent chord lookup
        self._initialize_enharmonic_mapping()

    def _initialize_enharmonic_mapping(self):
        """Initialize mappings between flat and sharp notes for consistent chord lookup"""
        # Create bidirectional mapping for enharmonic equivalents
        self.enharmonic_map = {
            'Bb': 'A#', 'A#': 'Bb',
            'Db': 'C#', 'C#': 'Db',
            'Eb': 'D#', 'D#': 'Eb',
            'Gb': 'F#', 'F#': 'Gb',
            'Ab': 'G#', 'G#': 'Ab',
        }
        
        # Add chord mappings for enharmonic equivalents if one exists but the other doesn't
        if hasattr(self, 'chord_mapping') and self.chord_mapping:
            # Check each root note with both flat and sharp variants
            for flat, sharp in [('Bb', 'A#'), ('Db', 'C#'), ('Eb', 'D#'), ('Gb', 'F#'), ('Ab', 'G#')]:
                # For major chords (root only)
                if flat in self.chord_mapping and sharp not in self.chord_mapping:
                    self.chord_mapping[sharp] = self.chord_mapping[flat]
                elif sharp in self.chord_mapping and flat not in self.chord_mapping:
                    self.chord_mapping[flat] = self.chord_mapping[sharp]
                
                # For common qualities
                for quality in ['min', 'maj', 'dim', 'aug', '7', 'maj7', 'min7']:
                    flat_chord = f"{flat}:{quality}"
                    sharp_chord = f"{sharp}:{quality}"
                    
                    if flat_chord in self.chord_mapping and sharp_chord not in self.chord_mapping:
                        self.chord_mapping[sharp_chord] = self.chord_mapping[flat_chord]
                    elif sharp_chord in self.chord_mapping and flat_chord not in self.chord_mapping:
                        self.chord_mapping[flat_chord] = self.chord_mapping[sharp_chord]

    def get_chord_idx(self, chord_name, use_large_voca=False):
        """
        Get chord index for a given chord name, with fallbacks for error cases.
        
        Parameters
        ----------
        chord_name : str
            The chord name to look up
        use_large_voca : bool
            Whether to use the large vocabulary (170 chords)
            
        Returns
        -------
        int
            The chord index
        """
        # Direct lookup first (most efficient)
        if hasattr(self, 'chord_mapping') and self.chord_mapping and chord_name in self.chord_mapping:
            return self.chord_mapping[chord_name]
            
        # Handle special cases
        if chord_name == "N":
            return 169 if use_large_voca else 24
        if chord_name == "X":
            return 168 if use_large_voca else 25
            
        # Try parsing the chord
        try:
            # Process the chord label with error correction
            modified_chord = self.label_error_modify(chord_name)
            
            # Parse the chord
            root, bass, intervals, is_major = self.chord(modified_chord)
            
            # For large vocabulary, extract quality
            if use_large_voca:
                quality = None
                if ':' in modified_chord:
                    _, quality = modified_chord.split(':', 1)
                    if '/' in quality:
                        quality = quality.split('/', 1)[0]
                else:
                    quality = 'maj'  # Default
                
                return self.convert_to_id_voca(root=root, quality=quality)
            else:
                # For standard vocabulary
                return self.convert_to_id(root=root, is_major=is_major)
                
        except Exception as e:
            logger.warning(f"Error parsing chord '{chord_name}': {e}")
            return 169 if use_large_voca else 24  # Default to N

    def initialize_chord_mapping(self, chord_mapping=None):
        """
        Ensure chord_mapping is initialized, logging availability for debugging
        
        Args:
            chord_mapping: Chord mapping to use, if specified
        """
        if chord_mapping:
            self.chord_mapping = chord_mapping
            logger.info(f"Chord mapping set with {len(chord_mapping)} entries")
        elif not hasattr(self, 'chord_mapping') or not self.chord_mapping:
            # Create empty mapping if none exists
            self.chord_mapping = {}
            logger.warning("Using empty chord mapping - chord lookups will use formula-based approach")
            
        # Log the mapping of standard notes to verify correctness
        debug_mappings = {
            'C': self.chord_mapping.get('C', None),
            'C#': self.chord_mapping.get('C#', None),
            'D': self.chord_mapping.get('D', None),
            'D#': self.chord_mapping.get('D#', None),
            'C:min': self.chord_mapping.get('C:min', None),
            'C:maj': self.chord_mapping.get('C:maj', None),
            'C:min7': self.chord_mapping.get('C:min7', None)
        }
        logger.debug(f"Sample mappings: {debug_mappings}")

    def check_chord_mapping_completeness(self):
        """
        Check that our chord mapping contains all expected chord types
        and log any missing entries.
        """
        # Only run check if chord_mapping is set
        if not hasattr(self, 'chord_mapping') or not self.chord_mapping:
            logger.warning("Cannot check chord mapping completeness - mapping not set")
            return False
            
        # Check all key chord types
        missing_chords = []
        
        # Check for major chords (just the root note, no quality suffix)
        for root in PITCH_CLASS:
            if root not in self.chord_mapping:
                missing_chords.append(root)
                
        # Check for other common qualities
        qualities = ['min', 'dim', 'aug', 'sus4', '7', 'maj7', 'min7']
        for root in PITCH_CLASS:
            for quality in qualities:
                chord_key = f"{root}:{quality}"
                if chord_key not in self.chord_mapping:
                    missing_chords.append(chord_key)
        
        if missing_chords:
            logger.warning(f"Missing {len(missing_chords)} expected chords in mapping")
            if len(missing_chords) < 20:
                logger.warning(f"Missing chords: {missing_chords}")
            return False
        else:
            logger.info("Chord mapping is complete with all expected chord types")
            return True

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