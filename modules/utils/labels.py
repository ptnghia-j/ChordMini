# Chord labels and mappings

# Define the 12 roots
MAJOR_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Define chord suffix types. We choose 10 types so that total chords per key become 10.
# The chosen types: (empty string for basic major triad), 'maj7', 'min', 'min7', '7', 'dim', 'aug', 'sus4', 'sus2', '6'
CHORD_SUFFIXES = ["", "maj7", "min", "min7", "7", "dim", "aug", "sus4", "sus2", "6"]

# Construct the chord labels list:
# First label is 'NC' (no chord), then for each key and each suffix, then an extra label 'Other' to reach 122 classes.
CHORD_LABELS = ["NC"] + [ key + suffix for key in MAJOR_KEYS for suffix in CHORD_SUFFIXES ] + ["Other"]

# Validate total number of classes: should be 1 + (12*10) + 1 = 122
assert len(CHORD_LABELS) == 122, f"Expected 122 chord labels, got {len(CHORD_LABELS)}"

# Define mapping from major keys to their relative minor chords.
MAJOR_TO_RELATIVE_MINOR = {}
for i, key in enumerate(MAJOR_KEYS):
    relative_index = (i - 3) % len(MAJOR_KEYS)
    relative_minor = MAJOR_KEYS[relative_index] + "min"
    MAJOR_TO_RELATIVE_MINOR[key] = relative_minor

# Enhance: define enharmonic labels for all 12 roots. For each note, +1 is the next semitone, -1 is the previous semitone (wrap around).
ENHARMONIC_LABELS = {}
num_keys = len(MAJOR_KEYS)
for i, key in enumerate(MAJOR_KEYS):
    plus = MAJOR_KEYS[(i+1) % num_keys]
    minus = MAJOR_KEYS[(i-1) % num_keys]
    ENHARMONIC_LABELS[key] = {"+1": plus, "-1": minus}

import unittest

class TestLabels(unittest.TestCase):
    def test_chord_labels_length(self):
        # CHORD_LABELS should have exactly 122 chords.
        self.assertEqual(len(CHORD_LABELS), 122)

    def test_major_to_relative_minor_mapping(self):
        # Validate few known mappings.
        self.assertEqual(MAJOR_TO_RELATIVE_MINOR["C"], "Amin")
        self.assertEqual(MAJOR_TO_RELATIVE_MINOR["D"], "Bmin")
        self.assertEqual(MAJOR_TO_RELATIVE_MINOR["E"], "C#min")

    def test_enharmonic_labels(self):
        # Example: for "C", +1 should be "C#" and -1 should be "B"
        self.assertEqual(ENHARMONIC_LABELS["C"]["+1"], "C#")
        self.assertEqual(ENHARMONIC_LABELS["C"]["-1"], "B")

if __name__ == "__main__":
    unittest.main()
