import os  # Add this import at the top level
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from scipy.spatial.distance import pdist, squareform

# Import chord metrics for similarity calculations
from modules.utils.chord_metrics import chord_similarity, parse_chord as metrics_parse_chord

# Root notes in the circle of fifths order
CIRCLE_OF_FIFTHS = ['c', 'g', 'd', 'a', 'e', 'b', 'f#', 'c#', 'ab', 'eb', 'bb', 'f']
# Alternative spellings for enharmonic equivalents
ENHARMONIC = {
    'gb': 'f#', 'db': 'c#', 'ab': 'g#', 'eb': 'eb', 'bb': 'bb',
    'g#': 'ab', 'd#': 'eb', 'a#': 'bb', 'cb': 'b', 'fb': 'e'
}
# Chord types organized by complexity/similarity
CHORD_TYPES = ['', 'm', '7', 'm7', 'maj7', 'dim', 'aug', 
               'dim7', 'm7b5', '6', 'm6', '9', 'maj9', 'add9',
               'sus2', 'sus4', '7sus4', '11', '13']

def parse_chord(chord):
    """Parse a chord string into root and type components."""
    if chord == "N":
        return "N", ""  # No-chord case
    
    # Basic regex to identify root and type
    match = re.match(r'([a-g][b#]?)(.*)$', chord.lower())
    if match:
        root, chord_type = match.groups()
        # Normalize root using enharmonic equivalents
        root = ENHARMONIC.get(root, root)
        return root, chord_type
    return chord.lower(), ""  # Default fallback

def generate_chord_coordinates(chords):
    """Generate 2D coordinates for chords based on music theory."""
    coordinates = {}
    
    for chord in chords:
        if chord == "N":
            # Place "N" (no chord) at origin
            coordinates[chord] = (0, 0)
            continue
            
        root, chord_type = parse_chord(chord)
        
        # X-coordinate: based on root position in circle of fifths
        if root in CIRCLE_OF_FIFTHS:
            x = CIRCLE_OF_FIFTHS.index(root)
        else:
            # Handle any roots not found in the circle
            x = len(CIRCLE_OF_FIFTHS) // 2  # Place in middle as fallback
        
        # Y-coordinate: based on chord type/complexity
        if chord_type in CHORD_TYPES:
            y = CHORD_TYPES.index(chord_type)
        else:
            # For complex/unusual chord types, place them higher
            y = len(CHORD_TYPES)
            
        # Adjust x to make the circle more spread out
        x = np.cos(x * 2 * np.pi / len(CIRCLE_OF_FIFTHS)) * 10
        y = np.sin(y * 2 * np.pi / len(CHORD_TYPES)) * 10 + y * 0.5
        
        coordinates[chord] = (x, y)
    
    return coordinates

def generate_advanced_coordinates(chords, chord_vectors=None):
    """Generate 2D coordinates using dimensionality reduction if needed."""
    if len(chords) <= 3:  # Not enough data for dimensionality reduction
        return generate_chord_coordinates(chords)
    
    if chord_vectors is None:
        # Create simple feature vectors for chords
        chord_vectors = {}
        for chord in chords:
            root, chord_type = parse_chord(chord)
            
            # One-hot encoding for root (12 dimensions)
            root_vec = np.zeros(12)
            if root in CIRCLE_OF_FIFTHS:
                root_vec[CIRCLE_OF_FIFTHS.index(root)] = 1
                
            # One-hot encoding for chord type (using position in CHORD_TYPES)
            type_vec = np.zeros(len(CHORD_TYPES))
            if chord_type in CHORD_TYPES:
                type_vec[CHORD_TYPES.index(chord_type)] = 1
                
            # Combine the features
            chord_vectors[chord] = np.concatenate([root_vec, type_vec])
    
    # Convert to array for dimensionality reduction
    chord_names = list(chords)
    feature_matrix = np.array([chord_vectors[chord] for chord in chord_names])
    
    # Use t-SNE for dimensionality reduction
    if len(chord_names) > 5:  # t-SNE works better with more data points
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(chord_names)//2))
        embedding = tsne.fit_transform(feature_matrix)
    else:
        # Fallback to PCA for very small datasets
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(feature_matrix)
    
    # Create the coordinates dictionary
    coordinates = {chord: (embedding[i, 0], embedding[i, 1]) 
                  for i, chord in enumerate(chord_names)}
    
    # Special handling for "N" if present
    if "N" in chords:
        coordinates["N"] = (0, 0)  # Place at origin
        
    return coordinates

def generate_similarity_based_coordinates(chords, max_chords=300):
    """Generate 2D coordinates for chords based on chord similarity metrics with sampling."""
    if len(chords) <= 1:
        return generate_chord_coordinates(chords)

    # Sample chords if there are too many
    original_chords = list(chords)
    if len(original_chords) > max_chords:
        print(f"Sampling {max_chords} chords from {len(original_chords)} for similarity calculation...")
        # Ensure N is included in sampled chords if present
        if "N" in original_chords:
            sampled_chords = ["N"]
            original_chords.remove("N")
            # Sample from remaining chords
            import random
            sampled_chords.extend(random.sample(original_chords, min(max_chords - 1, len(original_chords))))
        else:
            import random
            sampled_chords = random.sample(original_chords, min(max_chords, len(original_chords)))
        
        # Calculate coordinates for sample
        sample_coordinates = _compute_similarity_coordinates(sampled_chords)
        
        # Use nearest neighbor interpolation for other chords
        coordinates = {}
        for chord in original_chords:
            if chord in sample_coordinates:
                coordinates[chord] = sample_coordinates[chord]
            else:
                # Find most similar chord in sample
                most_similar = max(sampled_chords, key=lambda c: chord_similarity(chord, c))
                coordinates[chord] = sample_coordinates[most_similar]
                
        return coordinates
    else:
        print(f"Generating similarity-based coordinates for {len(chords)} chords...")
        return _compute_similarity_coordinates(chords)

def _compute_similarity_coordinates(chords):
    """Helper function to compute similarity coordinates for a manageable number of chords."""
    # Create similarity matrix
    n_chords = len(chords)
    similarity_matrix = np.zeros((n_chords, n_chords))
    
    # Use parallelization for larger sets
    if n_chords > 50:
        from concurrent.futures import ThreadPoolExecutor
        from itertools import product
        
        def compute_similarity(i, j):
            if i == j:
                return 1.0
            return chord_similarity(chords[i], chords[j])
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i, j in product(range(n_chords), range(n_chords)):
                if i <= j:  # Only compute upper triangle
                    futures.append(executor.submit(compute_similarity, i, j))
            
            # Get results
            results = [f.result() for f in futures]
            
            # Fill in the matrix (symmetric)
            idx = 0
            for i in range(n_chords):
                for j in range(i, n_chords):
                    similarity_matrix[i, j] = results[idx]
                    similarity_matrix[j, i] = results[idx]  # Mirror
                    idx += 1
    else:
        # Standard computation for smaller matrices
        for i, chord1 in enumerate(chords):
            for j, chord2 in enumerate(chords):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = chord_similarity(chord1, chord2)
    
    # Convert similarity matrix to distance matrix
    distance_matrix = 1.0 - similarity_matrix
    
    # Special handling for "N" (no chord)
    if "N" in chords:
        n_index = chords.index("N")
        distance_matrix[n_index, :] = 1.0
        distance_matrix[:, n_index] = 1.0
        distance_matrix[n_index, n_index] = 0.0
    
    # Apply MDS with optimized parameters
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
              normalized_stress='auto', n_init=1 if n_chords > 100 else 5,
              max_iter=200 if n_chords > 100 else 300)
    embedding = mds.fit_transform(distance_matrix)
    
    # Create the coordinates dictionary
    coordinates = {chord: (embedding[i, 0], embedding[i, 1]) 
                  for i, chord in enumerate(chords)}
    
    return coordinates

def generate_enhanced_coordinates(chords, min_chord_count=5):
    """
    Generate coordinates using a hybrid approach: 
    - For small datasets: Use music theory based positioning
    - For moderate datasets: Use similarity-based MDS
    - For large datasets: Use t-SNE with similarity as input
    """
    if len(chords) <= 3:
        return generate_chord_coordinates(chords)
    elif len(chords) <= min_chord_count:
        return generate_advanced_coordinates(chords)
    else:
        return generate_similarity_based_coordinates(chords)

def create_chord_scatter_plot(chord_counts, title, filename, min_freq_pct=None, figsize=(12, 10)):
    """
    Create a scatter plot of chord distribution.
    
    Args:
        chord_counts (Counter): Counter of chord frequencies
        title (str): Title for the plot
        filename (str): Filename to save the plot
        min_freq_pct (float, optional): Minimum frequency percentage to include
        figsize (tuple): Figure size
    """
    # Filter chords if min_freq_pct is provided
    total = sum(chord_counts.values())
    if min_freq_pct is not None:
        filtered_chords = {chord: count for chord, count in chord_counts.items() 
                          if (count / total) * 100 >= min_freq_pct}
    else:
        filtered_chords = chord_counts
    
    chords = list(filtered_chords.keys())
    frequencies = list(filtered_chords.values())
    
    # Get 2D coordinates for the chords - use the enhanced version
    coordinates = generate_enhanced_coordinates(chords)
    
    # Extract x and y coordinates
    x_coords = [coordinates[chord][0] for chord in chords]
    y_coords = [coordinates[chord][1] for chord in chords]
    
    # Calculate marker sizes based on frequency (using square root to moderate the size difference)
    sizes = [50 + 2000 * (freq / max(frequencies))**0.5 for freq in frequencies]
    
    # Create color groups based on chord types
    color_groups = {}
    for chord in chords:
        root, chord_type = parse_chord(chord)
        
        if chord == "N":
            color_groups[chord] = "No Chord"
        elif chord_type == "":
            color_groups[chord] = "Major"
        elif chord_type == "m":
            color_groups[chord] = "Minor"
        elif "7" in chord_type:
            color_groups[chord] = "Seventh"
        elif "dim" in chord_type:
            color_groups[chord] = "Diminished"
        elif "aug" in chord_type:
            color_groups[chord] = "Augmented"
        elif "sus" in chord_type:
            color_groups[chord] = "Suspended"
        else:
            color_groups[chord] = "Other"
    
    # Define colors for each group
    group_colors = {
        "Major": "#1f77b4",       # Blue
        "Minor": "#ff7f0e",       # Orange
        "Seventh": "#2ca02c",     # Green
        "Diminished": "#d62728",  # Red
        "Augmented": "#9467bd",   # Purple
        "Suspended": "#8c564b",   # Brown
        "No Chord": "#e377c2",    # Pink
        "Other": "#7f7f7f"        # Gray
    }
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Track which groups we've seen for the legend
    seen_groups = set()
    
    # Plot each chord
    for i, chord in enumerate(chords):
        group = color_groups[chord]
        color = group_colors[group]
        seen_groups.add(group)
        
        # Calculate relative frequency percentage
        freq_pct = (filtered_chords[chord] / total) * 100
        
        # Plot with custom hover annotation
        plt.scatter(x_coords[i], y_coords[i], s=sizes[i], color=color, alpha=0.6, edgecolors='black')
        
        # Use different fontsize based on frequency
        if freq_pct > 5:  # Very frequent chords
            fontsize = 12
        elif freq_pct > 1:  # Moderately frequent
            fontsize = 10
        else:  # Less frequent
            fontsize = 8
            
        plt.annotate(chord, (x_coords[i], y_coords[i]), 
                    fontsize=fontsize, ha='center', va='center')
    
    # Create custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=group_colors[group], 
                             markersize=10, label=group)
                      for group in sorted(seen_groups)]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel("← Root Note Position →")
    plt.ylabel("← Chord Complexity →")
    
    # Remove axis ticks for cleaner look
    plt.xticks([])
    plt.yticks([])
    
    # Add count information
    plt.figtext(0.5, 0.01, f"Total unique chords: {len(filtered_chords)}, Total instances: {sum(filtered_chords.values())}", 
               ha="center", fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_root_grouped_visualization(chord_counts, title, filename, figsize=(15, 12)):
    """Create a visualization where chords are grouped by their root notes."""
    total = sum(chord_counts.values())
    chords = list(chord_counts.keys())
    frequencies = list(chord_counts.values())
    
    # Extract root notes and types from all chords
    chord_data = {}
    roots = set()
    types = set()
    
    for chord in chords:
        if chord == "N":
            chord_data[chord] = {"root": "N", "type": "", "freq": chord_counts[chord]}
            continue
        
        # Use the more comprehensive parser from chord_metrics
        root, quality = metrics_parse_chord(chord)
        chord_data[chord] = {
            "root": root or "unknown",
            "type": quality or "",
            "freq": chord_counts[chord]
        }
        if root:
            roots.add(root)
        if quality:
            types.add(quality)
    
    # Sort roots by circle of fifths
    circle_of_fifths = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
    roots_lower = {r.lower() for r in roots if r}
    
    # Create plot with grouped layout
    plt.figure(figsize=figsize)
    
    # Define color map for chord types
    type_colors = {
        "maj": "#1f77b4",    # Blue
        "min": "#ff7f0e",    # Orange
        "7": "#2ca02c",      # Green
        "maj7": "#9467bd",   # Purple
        "min7": "#d62728",   # Red
        "dim": "#8c564b",    # Brown
        "aug": "#e377c2",    # Pink
        "sus": "#7f7f7f",    # Gray
        "": "#17becf",       # Cyan
        "N": "#bcbd22"       # Yellow
    }
    
    # Create a mapping for chord types not in our predefined list
    for chord_type in types:
        if chord_type not in type_colors:
            # Assign a color based on some simple rules
            if "7" in chord_type:
                type_colors[chord_type] = "#2ca02c"  # Green for seventh chords
            elif "min" in chord_type or "m" == chord_type:
                type_colors[chord_type] = "#ff7f0e"  # Orange for minor-related chords
            elif "maj" in chord_type:
                type_colors[chord_type] = "#1f77b4"  # Blue for major-related chords
            elif "dim" in chord_type:
                type_colors[chord_type] = "#8c564b"  # Brown for diminished chords
            elif "aug" in chord_type:
                type_colors[chord_type] = "#e377c2"  # Pink for augmented chords
            elif "sus" in chord_type:
                type_colors[chord_type] = "#7f7f7f"  # Gray for suspended chords
            else:
                type_colors[chord_type] = "#17becf"  # Cyan for others
    
    # Get chord positions using similarity metrics
    coordinates = generate_similarity_based_coordinates(chords)
    
    # Extract coordinates
    x_coords = [coordinates[chord][0] for chord in chords]
    y_coords = [coordinates[chord][1] for chord in chords]
    
    # Calculate marker sizes
    max_freq = max(frequencies)
    sizes = [100 + 3000 * (freq / max_freq)**0.5 for freq in frequencies]
    
    # Create the scatter plot
    for i, chord in enumerate(chords):
        data = chord_data[chord]
        chord_type = data["type"]
        color = type_colors.get(chord_type, "#17becf")  # Cyan for unknown types
        
        # Calculate relative frequency percentage
        freq_pct = (chord_counts[chord] / total) * 100
        
        # Plot point
        plt.scatter(x_coords[i], y_coords[i], s=sizes[i], color=color, 
                   alpha=0.7, edgecolors='black')
        
        # Add label with font size based on frequency
        if freq_pct > 5:
            fontsize = 12
        elif freq_pct > 1:
            fontsize = 10
        else:
            fontsize = 8
        
        plt.annotate(chord, (x_coords[i], y_coords[i]), 
                    fontsize=fontsize, ha='center', va='center')
    
    # Add legend for chord types
    legend_elements = []
    seen_types = set()
    
    # Group similar chord types
    chord_type_groups = {
        "Major": ["maj", ""],
        "Minor": ["min", "m"],
        "Dominant 7th": ["7"],
        "Major 7th": ["maj7"],
        "Minor 7th": ["min7", "m7"],
        "Diminished": ["dim", "dim7", "°"],
        "Augmented": ["aug"],
        "Suspended": ["sus", "sus2", "sus4", "7sus4"]
    }
    
    group_colors = {}
    for group, types_list in chord_type_groups.items():
        # Use the color of the first type in the list
        for t in types_list:
            if t in type_colors:
                group_colors[group] = type_colors[t]
                break
        else:
            # If none found, use default
            group_colors[group] = "#17becf"
    
    # Create legend entries
    for group, color in group_colors.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                  markersize=10, label=group)
        )
    
    # Add "Other" category if needed
    other_types = [t for t in types if not any(t in group_types for group_types in chord_type_groups.values())]
    if other_types:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#17becf", 
                  markersize=10, label="Other")
        )
    
    # Add "No Chord" if present
    if "N" in chords:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=type_colors["N"], 
                  markersize=10, label="No Chord")
        )
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel("Musical Similarity Dimension 1")
    plt.ylabel("Musical Similarity Dimension 2")
    
    # Remove axis ticks for cleaner look
    plt.xticks([])
    plt.yticks([])
    
    # Add count information
    plt.figtext(0.5, 0.01, f"Total unique chords: {len(chord_counts)}, Total instances: {sum(chord_counts.values())}", 
               ha="center", fontsize=12)
    
    # Add explanatory note
    plt.figtext(0.5, 0.03, "Positioning based on musical similarity metrics. Larger circles = higher frequency.", 
               ha="center", fontsize=10, style='italic')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_chord_distributions(dataset_list, output_dir=None, skip_full_viz=True):
    """
    Visualize chord distributions with performance optimizations.
    
    Args:
        dataset_list (list): List of CrossDataset objects
        output_dir (str): Directory to save plots
        skip_full_viz (bool): Skip the full visualization to save time
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "visualizations")
        os.makedirs(output_dir, exist_ok=True)
    
    # Gather chord counts from all datasets
    combined_counter = Counter()
    for ds in dataset_list:
        chord_counts = Counter([s['chord_label'] for s in ds.samples])
        combined_counter.update(chord_counts)
    
    # Calculate total for percentage
    total_samples = sum(combined_counter.values())
    
    # Print statistics
    print(f"Chord Distribution Statistics:")
    print(f"Total unique chords: {len(combined_counter)}")
    print(f"Total chord instances: {total_samples}")
    
    # Create visualizations with optimizations
    
    # Only generate the filtered visualization by default (skip full distribution to save time)
    if not skip_full_viz:
        viz_path1 = os.path.join(output_dir, "chord_distribution_full.png")
        create_chord_scatter_plot(
            combined_counter,
            f"Chord Distribution (All {len(combined_counter)} chords, {total_samples} instances)",
            filename=viz_path1
        )
    
    # Always generate the filtered visualization (most useful)
    filtered_counter = {chord: count for chord, count in combined_counter.items() 
                       if (count / total_samples) * 100 >= 0.1}
    
    viz_path2 = os.path.join(output_dir, "chord_distribution_filtered.png")
    create_chord_scatter_plot(
        filtered_counter,
        "Chord Distribution (Filtered, frequency ≥ 0.1%)",
        filename=viz_path2,
        min_freq_pct=0.1
    )
    
    # Generate similarity-based visualization for filtered data only
    viz_path4 = os.path.join(output_dir, "chord_distribution_similarity_filtered.png")
    create_root_grouped_visualization(
        filtered_counter,
        "Musical Similarity-Based Chord Distribution (Filtered, frequency ≥ 0.1%)",
        filename=viz_path4
    )
    
    # Print paths to the generated files
    print(f"Visualization files saved to visualization directory ({output_dir}):")
    print(f"1. Standard visualization (full): {viz_path1}")
    print(f"2. Standard visualization (filtered): {viz_path2}")
    print(f"3. Similarity-based visualization (full): {viz_path3}")
    print(f"4. Similarity-based visualization (filtered): {viz_path4}")
