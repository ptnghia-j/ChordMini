def fix_chord_labels(chord_mapping):
    """
    Add missing chord label mappings for common shorthand notations.
    This helps avoid "Unknown chord label" warnings in CrossDataset.
    
    Args:
        chord_mapping (dict): Original chord mapping dict
        
    Returns:
        dict: Enhanced chord mapping with shorthand notation mappings added
    """
    new_mapping = chord_mapping.copy()
    
    # Add common shorthand notations if their expanded forms exist
    shorthand_fixes = {
        # Major chords
        'c': 'c:maj', 'd': 'd:maj', 'e': 'e:maj', 
        'f': 'f:maj', 'g': 'g:maj', 'a': 'a:maj', 'b': 'b:maj',
        'c#': 'c#:maj', 'd#': 'd#:maj', 'f#': 'f#:maj', 
        'g#': 'g#:maj', 'a#': 'a#:maj',
        # Alternative formats
        'f#': 'f#maj', 'f#maj': 'f#:maj',
        'c#': 'c#maj', 'c#maj': 'c#:maj',
        'g#': 'g#maj', 'g#maj': 'g#:maj',
        'd#': 'd#maj', 'd#maj': 'd#:maj',
        'a#': 'a#maj', 'a#maj': 'a#:maj',
    }
    
    # Try multiple alternatives for each shorthand
    for short, expanded in shorthand_fixes.items():
        if short not in new_mapping:
            # Try first expanded form
            if expanded in new_mapping:
                new_mapping[short] = new_mapping[expanded]
            # Try alternative expanded forms
            elif expanded + "or" in new_mapping:
                new_mapping[short] = new_mapping[expanded + "or"]
            # Try other common variations
            elif short + ":major" in new_mapping:
                new_mapping[short] = new_mapping[short + ":major"]
    
    return new_mapping

if __name__ == "__main__":
    # Example usage
    sample_mapping = {'c:maj': 0, 'c:min': 1, 'f#:maj': 2}
    enhanced = fix_chord_labels(sample_mapping)
    print(f"Original: {sample_mapping}")
    print(f"Enhanced: {enhanced}")
