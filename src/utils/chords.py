"""
Standalone chord vocabulary and parsing utilities for the ChordMini pipeline.

Provides:
- idx2voca_chord(): maps chord index (0-169) to chord label strings
- Chords class: parsing, normalization, chord-to-index lookup
- PITCH_CLASS, PREFERRED_SPELLING_MAP constants
"""
import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_L = [0, 1, 1, 0, 1, 1, 1]
_CHROMA_ID = (np.arange(len(_L) * 2) + 1) + np.array(_L + _L).cumsum() - 1

ENHARMONIC_MAP = {
    'Bb': 'A#', 'A#': 'Bb', 'Db': 'C#', 'C#': 'Db',
    'Eb': 'D#', 'D#': 'Eb', 'Gb': 'F#', 'F#': 'Gb',
    'Ab': 'G#', 'G#': 'Ab', 'B#': 'C', 'Cb': 'B',
    'E#': 'F', 'Fb': 'E',
}

PREFERRED_SPELLING_MAP = {
    'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#',
}

ENHARMONIC_NORMALIZE_MAP = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'B#': 'C', 'Cb': 'B', 'E#': 'F', 'Fb': 'E',
}

QUALITY_NORM_MAP = {
    '': 'maj', ':': 'maj', 'M': 'maj', 'm': 'min', '-': 'min', 'mi': 'min',
    'major': 'maj', 'minor': 'min', 'ma': 'maj', 'dom': '7',
    'ø': 'hdim7', 'o': 'dim', '°': 'dim', '+': 'aug',
    'aug': 'aug', 'dim': 'dim', 'sus': 'sus4',
    'maj7': 'maj7', 'M7': 'maj7', '^': 'maj7', '^7': 'maj7',
    'min7': 'min7', 'm7': 'min7', '-7': 'min7',
    '7': '7', 'dom7': '7', 'dim7': 'dim7', '°7': 'dim7', 'o7': 'dim7',
    'hdim7': 'hdim7', 'm7b5': 'hdim7', 'min7b5': 'hdim7', 'min7-5': 'hdim7',
    'minmaj7': 'minmaj7', 'mmaj7': 'minmaj7',
    '6': 'maj6', 'maj6': 'maj6', 'M6': 'maj6',
    'min6': 'min6', 'm6': 'min6', '-6': 'min6',
    'sus2': 'sus2', 'sus4': 'sus4', '5': '5',
    '9': '7', '11': '7', '13': '7',
    'maj9': 'maj7', 'maj11': 'maj7', 'maj13': 'maj7',
    'min9': 'min7', 'min11': 'min7', 'min13': 'min7',
    'aug7': 'aug7', '7#5': 'aug7', '7+5': 'aug7',
    '69': 'maj6', '6/9': 'maj6',
    '7#9': '7', '7b9': '7', '7b5': '7',
    'alt': '7',
    # Parenthesized MIREX variants
    'maj(9)': 'maj7', 'min(9)': 'min7', 'min(7)': 'min7', 'maj(7)': 'maj7',
    'min(6)': 'min6', 'maj(6)': 'maj6', '7(b9)': '7', '7(#9)': '7',
    '7(b5)': '7', '7(#5)': 'aug7', 'min7(b5)': 'hdim7',
    'sus4(9)': 'sus4', 'sus4(b7)': 'sus4',
    # Self-maps for normalized values
    **{q: q for q in ['maj', 'min', 'dim', 'aug', 'maj7', 'min7', '7', 'dim7',
                      'hdim7', 'minmaj7', 'maj6', 'min6', 'sus2', 'sus4', '5', 'aug7']},
}


def _modify_pitch(base_pitch, modifier_str):
    pitch = base_pitch
    for c in modifier_str:
        if c == 'b':
            pitch -= 1
        elif c == '#':
            pitch += 1
    return pitch % 12


def _parse_single_interval(interval_str):
    interval_str = interval_str.strip()
    if not interval_str:
        return None
    digit_part_str, modifier_str = "", ""
    for i, c in enumerate(interval_str):
        if c.isdigit():
            modifier_str = interval_str[:i]
            digit_part_str = interval_str[i:]
            break
    else:
        return None
    try:
        digit = int(digit_part_str)
        if not (1 <= digit <= 14):
            return None
        return _modify_pitch(_CHROMA_ID[digit - 1], modifier_str)
    except (ValueError, IndexError):
        return None


INTERVAL_SETS_TO_QUALITY = {
    frozenset({0, 4, 7}): "maj", frozenset({0, 4}): "maj", frozenset({0}): "maj",
    frozenset({0, 3, 7}): "min", frozenset({0, 3}): "min",
    frozenset({0, 4, 8}): "aug", frozenset({0, 3, 6}): "dim",
    frozenset({0, 4, 7, 11}): "maj7", frozenset({0, 4, 11}): "maj7",
    frozenset({0, 4, 7, 10}): "7", frozenset({0, 4, 10}): "7",
    frozenset({0, 3, 7, 10}): "min7", frozenset({0, 3, 10}): "min7",
    frozenset({0, 3, 7, 11}): "minmaj7",
    frozenset({0, 4, 7, 9}): "maj6", frozenset({0, 3, 7, 9}): "min6",
    frozenset({0, 5, 7}): "sus4", frozenset({0, 5}): "sus4",
    frozenset({0, 2, 7}): "sus2", frozenset({0, 2}): "sus2",
    frozenset({0, 3, 6, 9}): "dim7", frozenset({0, 3, 6, 10}): "hdim7",
    frozenset({0, 7}): "maj", frozenset({0, 5, 7, 10}): "sus4",
    frozenset({0, 4, 8, 10}): "aug",
    frozenset({0, 2, 4, 7, 10}): "7", frozenset({0, 2, 3, 7, 10}): "min7",
    frozenset({0, 2, 4, 7, 11}): "maj7",
}


def _parse_intervals_to_quality(intervals_str):
    if not (intervals_str.startswith('(') and intervals_str.endswith(')')):
        return None
    content = intervals_str[1:-1].strip()
    if not content:
        return None
    pitch_classes = {0}
    for part in content.split(','):
        part = part.strip()
        if not part:
            continue
        pc = _parse_single_interval(part)
        if pc is not None:
            pitch_classes.add(pc)
        else:
            return None
    fset = frozenset(pitch_classes)
    return INTERVAL_SETS_TO_QUALITY.get(fset)


def _parse_root(root_str):
    if not root_str or not root_str[0].isalpha():
        raise ValueError(f"Invalid root note: {root_str}")
    if root_str in ENHARMONIC_MAP:
        root_str = ENHARMONIC_MAP[root_str]
    base_char = root_str[0].upper()
    modifier = root_str[1:]
    base_offset = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    if base_char not in base_offset:
        raise ValueError(f"Invalid root note base: {base_char}")
    return _modify_pitch(base_offset[base_char], modifier)


def _parse_chord_string(chord_label):
    if not chord_label or chord_label.strip().upper() in ("N", "NC"):
        return "N", None, None
    if chord_label.strip().upper() == "X":
        return "X", None, None

    cleaned = chord_label.strip()

    # Separate bass note
    chord_part_str = cleaned
    raw_bass = None
    if '/' in cleaned:
        pot_chord, pot_bass = cleaned.rsplit('/', 1)
        pot_bass = pot_bass.strip()
        bass_valid = False
        if pot_bass:
            try:
                _parse_root(pot_bass)
                bass_valid = True
            except ValueError:
                if _parse_single_interval(pot_bass) is not None:
                    bass_valid = True
        if bass_valid:
            chord_part_str = pot_chord.strip()
            raw_bass = pot_bass
            if not chord_part_str:
                return "X", None, None

    # Parse root and quality
    parsed_root = parsed_quality = None
    if ':' in chord_part_str:
        parts = chord_part_str.split(':', 1)
        parsed_root = parts[0].strip()
        parsed_quality = parts[1].strip() if len(parts) > 1 else ''
        if not parsed_root:
            return "X", None, None
    else:
        m = re.match(r'^([a-gA-G][#b]?)', chord_part_str)
        if m:
            parsed_root = m.group(1)
            parsed_quality = chord_part_str[len(parsed_root):].strip()
        else:
            return "X", None, None

    # Normalize root
    try:
        root_int = _parse_root(parsed_root)
        default_name = PITCH_CLASS[root_int]
        final_root = PREFERRED_SPELLING_MAP.get(default_name, default_name)
    except ValueError:
        return "X", None, None

    # Normalize bass
    final_bass = None
    if raw_bass:
        is_interval = re.match(r'^[#b]?\d+$', raw_bass)
        parsed_as_interval = False
        if is_interval:
            ipc = _parse_single_interval(raw_bass)
            if ipc is not None:
                abs_pc = (root_int + ipc) % 12
                bn = PITCH_CLASS[abs_pc]
                final_bass = PREFERRED_SPELLING_MAP.get(bn, bn)
                parsed_as_interval = True
        if not parsed_as_interval:
            try:
                bpc = _parse_root(raw_bass)
                bn = PITCH_CLASS[bpc]
                final_bass = PREFERRED_SPELLING_MAP.get(bn, bn)
            except ValueError:
                final_bass = raw_bass if is_interval else None

    # Normalize quality
    q = parsed_quality
    final_q = QUALITY_NORM_MAP.get(q)

    if final_q is None and q.startswith('(') and q.endswith(')'):
        piq = _parse_intervals_to_quality(q)
        if piq:
            final_q = QUALITY_NORM_MAP.get(piq, piq)

    if final_q is None:
        stripped = re.sub(r'\(.*\)', '', q).strip()
        if stripped != q and q:
            final_q = QUALITY_NORM_MAP.get(stripped, stripped)

    if final_q is None:
        candidate = re.sub(r'\(.*\)', '', q).strip()
        final_q = QUALITY_NORM_MAP.get(candidate, candidate)

    if not final_q:
        final_q = 'maj'

    valid_qualities = set(QUALITY_NORM_MAP.values())
    if final_q not in valid_qualities:
        final_q = 'maj'

    return final_root, final_q, final_bass


def normalize_enharmonic_label(label):
    if not label or label in ('N', 'X'):
        return label
    if '/' in label:
        base, bass = label.rsplit('/', 1)
        for flat, sharp in ENHARMONIC_NORMALIZE_MAP.items():
            if bass.startswith(flat):
                bass = sharp + bass[len(flat):]
                break
        base = normalize_enharmonic_label(base)
        return f"{base}/{bass}"
    for flat, sharp in ENHARMONIC_NORMALIZE_MAP.items():
        if label.startswith(flat):
            return sharp + label[len(flat):]
    return label


def transpose_chord_label(chord_label, semitones):
    if chord_label in ('N', 'X', ''):
        return chord_label
    try:
        if '/' in chord_label:
            chord_part, bass_part = chord_label.rsplit('/', 1)
        else:
            chord_part, bass_part = chord_label, None

        if ':' in chord_part:
            root, quality = chord_part.split(':', 1)
        else:
            match = re.match(r'^([A-Ga-g][#b]?)', chord_part)
            if match:
                root = match.group(1)
                quality = chord_part[len(root):] or 'maj'
            else:
                return chord_label

        base_notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        root_upper = root[0].upper() + root[1:]
        if root_upper[0] not in base_notes:
            return chord_label
        pitch_class = base_notes[root_upper[0]]
        if len(root_upper) > 1:
            pitch_class += 1 if root_upper[1] == '#' else (-1 if root_upper[1] == 'b' else 0)
        new_pitch_class = (pitch_class + semitones) % 12
        new_root = PREFERRED_SPELLING_MAP.get(PITCH_CLASS[new_pitch_class], PITCH_CLASS[new_pitch_class])

        new_bass = None
        if bass_part:
            if bass_part[0].isdigit() or (bass_part[0] in '#b' and len(bass_part) > 1 and bass_part[1].isdigit()):
                new_bass = bass_part
            else:
                bass_upper = bass_part[0].upper() + bass_part[1:]
                if bass_upper[0] in base_notes:
                    bass_pitch_class = base_notes[bass_upper[0]]
                    if len(bass_upper) > 1:
                        bass_pitch_class += 1 if bass_upper[1] == '#' else (-1 if bass_upper[1] == 'b' else 0)
                    new_bass_pc = (bass_pitch_class + semitones) % 12
                    new_bass = PREFERRED_SPELLING_MAP.get(PITCH_CLASS[new_bass_pc], PITCH_CLASS[new_bass_pc])

        result = f"{new_root}:{quality}" if quality and quality != 'maj' else new_root
        if new_bass:
            result += f"/{new_bass}"
        return result
    except Exception:
        return chord_label


class Chords:
    def __init__(self):
        self._shorthands = {
            'maj': self._ivl('(1,3,5)'), 'min': self._ivl('(1,b3,5)'),
            'dim': self._ivl('(1,b3,b5)'), 'aug': self._ivl('(1,3,#5)'),
            'maj7': self._ivl('(1,3,5,7)'), 'min7': self._ivl('(1,b3,5,b7)'),
            '7': self._ivl('(1,3,5,b7)'), '5': self._ivl('(1,5)'),
            'dim7': self._ivl('(1,b3,b5,bb7)'), 'hdim7': self._ivl('(1,b3,b5,b7)'),
            'minmaj7': self._ivl('(1,b3,5,7)'), 'maj6': self._ivl('(1,3,5,6)'),
            'min6': self._ivl('(1,b3,5,6)'), 'sus2': self._ivl('(1,2,5)'),
            'sus4': self._ivl('(1,4,5)'), 'aug7': self._ivl('(1,3,#5,b7)'),
        }
        self.chord_mapping = {}

    @staticmethod
    def _interval_pc(interval_str):
        interval_str = interval_str.strip()
        for i, c in enumerate(interval_str):
            if c.isdigit():
                digit = int(interval_str[i:])
                if digit == 0:
                    return 0
                return _modify_pitch(_CHROMA_ID[digit - 1], interval_str[:i])
        raise ValueError(f"Invalid interval: {interval_str}")

    def _ivl(self, intervals_str):
        arr = np.zeros(12, dtype=np.int32)
        for part in intervals_str[1:-1].split(','):
            part = part.strip()
            if not part:
                continue
            if part[0] == '*':
                arr[self._interval_pc(part[1:]) % 12] = 0
            else:
                arr[self._interval_pc(part) % 12] = 1
        return arr

    def label_error_modify(self, label):
        root, quality, bass = _parse_chord_string(label)
        if root in ("N", "X"):
            return root
        if root is None:
            return "X"
        q_str = quality if quality and quality != 'maj' else None
        result = f"{root}:{q_str}" if q_str else root
        if bass:
            result += f"/{bass}"
        return result

    def set_chord_mapping(self, chord_mapping):
        if not chord_mapping or not isinstance(chord_mapping, dict):
            self.chord_mapping = {}
            return
        self.base_mapping = chord_mapping.copy()
        self._build_extended_mapping()

    def _build_extended_mapping(self):
        self.chord_mapping = self.base_mapping.copy()
        extended = {}
        for base_chord, base_idx in self.base_mapping.items():
            variants = {base_chord}
            try:
                root, quality, bass = _parse_chord_string(base_chord)
                if root in ("N", "X"):
                    variants.add(root)
                elif root and quality:
                    q_str = quality if quality != 'maj' else None
                    norm_label = f"{root}:{q_str}" if q_str else root
                    variants.add(norm_label)
                    if root in ENHARMONIC_MAP:
                        enh = ENHARMONIC_MAP[root]
                        enh_label = f"{enh}:{q_str}" if q_str else enh
                        variants.add(enh_label)
            except Exception:
                pass
            for v in variants:
                if v and v not in self.chord_mapping:
                    extended[v] = base_idx
        final = extended.copy()
        final.update(self.chord_mapping)
        self.chord_mapping = final

    def get_chord_idx(self, chord_name, use_large_voca=True):
        n_idx = 169 if use_large_voca else 24
        x_idx = 168 if use_large_voca else 25
        if not chord_name or chord_name.strip().upper() in ("N", "NC"):
            return n_idx
        if chord_name.strip().upper() == "X":
            return x_idx
        lookup = self.label_error_modify(chord_name)
        if lookup == "X":
            return x_idx
        if lookup == "N":
            return n_idx
        if lookup in self.chord_mapping:
            return self.chord_mapping[lookup]
        if '/' in lookup:
            root_pos = lookup.split('/')[0]
            if root_pos in self.chord_mapping:
                return self.chord_mapping[root_pos]
        return x_idx


def idx2voca_chord():
    """Create mapping from chord index (0-169) to chord label for 170-class vocabulary."""
    mapping = {}
    quality_list = [
        'min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7',
        'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4'
    ]
    for root_idx in range(12):
        default_name = PITCH_CLASS[root_idx]
        root = PREFERRED_SPELLING_MAP.get(default_name, default_name)
        for qi, quality in enumerate(quality_list):
            idx = root_idx * 14 + qi
            mapping[idx] = root if quality == 'maj' else f"{root}:{quality}"
    mapping[168] = "X"
    mapping[169] = "N"
    return mapping
