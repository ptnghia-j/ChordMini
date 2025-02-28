import sys
import os
import unittest
import numpy as np

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.utils.chord_metrics import (
    parse_chord,
    calculate_root_similarity,
    calculate_quality_similarity,
    chord_similarity,
    weighted_chord_symbol_recall
)

class TestChordMetrics(unittest.TestCase):
    def test_parse_chord(self):
        """Test chord parsing function"""
        # Test basic chords
        self.assertEqual(parse_chord("C"), ("C", "maj"))
        self.assertEqual(parse_chord("Cm"), ("C", "m"))
        self.assertEqual(parse_chord("G7"), ("G", "7"))
        
        # Test complex chords
        self.assertEqual(parse_chord("Bb_maj7"), ("Bb", "maj7"))
        self.assertEqual(parse_chord("F#_m7"), ("F#", "m7"))
        
        # Test no chord
        self.assertEqual(parse_chord("N"), (None, None))
        
        # Test chords with accidentals
        self.assertEqual(parse_chord("C#m"), ("C#", "m"))
        self.assertEqual(parse_chord("Bbdim"), ("Bb", "dim"))

    def test_root_similarity(self):
        """Test root similarity calculation"""
        # Same roots should have maximum similarity
        self.assertEqual(calculate_root_similarity("C", "C"), 1.0)
        
        # Test circle of fifths relationships
        self.assertAlmostEqual(calculate_root_similarity("C", "G"), 1.0 - 1/6)
        self.assertAlmostEqual(calculate_root_similarity("C", "F"), 1.0 - 1/6)
        
        # Test opposite keys (furthest in circle of fifths)
        self.assertAlmostEqual(calculate_root_similarity("C", "F#"), 0.0)
        
        # Test normalization of enharmonic equivalents
        self.assertEqual(calculate_root_similarity("F#", "Gb"), 1.0)
        
        # Test null roots
        self.assertEqual(calculate_root_similarity(None, "C"), 0.0)
        self.assertEqual(calculate_root_similarity("C", None), 0.0)

    def test_quality_similarity(self):
        """Test chord quality similarity calculation"""
        # Exact match should be 1.0
        self.assertEqual(calculate_quality_similarity("maj", "maj"), 1.0)
        self.assertEqual(calculate_quality_similarity("m", "m"), 1.0)
        
        # Same family should be 0.8
        self.assertEqual(calculate_quality_similarity("maj", "maj7"), 0.8)
        self.assertEqual(calculate_quality_similarity("m", "m7"), 0.8)
        
        # Related families should be 0.5
        self.assertEqual(calculate_quality_similarity("maj", "7"), 0.5)
        
        # Different families should be 0.2
        self.assertEqual(calculate_quality_similarity("maj", "dim"), 0.2)
        
        # Null qualities
        self.assertEqual(calculate_quality_similarity(None, "maj"), 0.0)

    def test_chord_similarity(self):
        """Test overall chord similarity calculation"""
        # Exact match
        self.assertEqual(chord_similarity("C", "C"), 1.0)
        self.assertEqual(chord_similarity("Cm", "Cm"), 1.0)
        
        # No chord
        self.assertEqual(chord_similarity("N", "C"), 0.0)
        self.assertEqual(chord_similarity("C", "N"), 0.0)
        
        # Related chords
        # For C and G: 
        # - Root similarity: 1.0 - 1/6 = 0.833
        # - Quality similarity: 1.0
        # - Final: 0.6 * 0.833 + 0.4 * 1.0 = 0.9
        self.assertAlmostEqual(chord_similarity("C", "G"), 0.6 * (1.0 - 1/6) + 0.4 * 1.0)
        
        # Different root and quality
        # For C and Am:
        # - Root similarity: 1.0 - 3/6 = 0.5
        # - Quality similarity: 0.2
        # - Final: 0.6 * 0.5 + 0.4 * 0.2 = 0.38
        self.assertAlmostEqual(chord_similarity("C", "Am"), 0.6 * (1.0 - 3/6) + 0.4 * 0.2)

    def test_weighted_chord_symbol_recall(self):
        """Test WCSR calculation"""
        # Create a test mapping of indices to chord names
        idx_to_chord = {
            0: "C",
            1: "G",
            2: "Am",
            3: "F",
            4: "N"
        }
        
        # Test exact match
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        self.assertEqual(weighted_chord_symbol_recall(y_true, y_pred, idx_to_chord), 1.0)
        
        # Test no match
        y_true = [0, 1, 2, 3]
        y_pred = [4, 4, 4, 4]  # All predictions are "N"
        self.assertEqual(weighted_chord_symbol_recall(y_true, y_pred, idx_to_chord), 0.0)
        
        # Test partial match (computed manually based on chord_similarity values)
        y_true = [0, 1]  # C, G
        y_pred = [1, 0]  # G, C
        expected = (chord_similarity("C", "G") + chord_similarity("G", "C")) / 2
        self.assertAlmostEqual(weighted_chord_symbol_recall(y_true, y_pred, idx_to_chord), expected)
        
        # Test empty array
        self.assertEqual(weighted_chord_symbol_recall([], [], idx_to_chord), 0.0)

if __name__ == '__main__':
    unittest.main()