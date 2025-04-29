import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
import os
import re
import traceback
# Fix import path to use correct module location
from modules.utils.logger import info, warning, error, debug, logging_verbosity, is_debug

# Define chord quality mappings - will be used if not using chords.py functions
DEFAULT_CHORD_QUALITIES = [
    "Major", "Minor", "Dom7", "Maj7", "Min7", "Dim",
    "Dim7", "Half-Dim", "Aug", "Sus", "No Chord", "Other"
]

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, title=None,
                          figsize=(12, 10), cmap='Blues', max_classes=12, text_size=8):
    """
    Plot a confusion matrix with improved visualization for many classes.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Dictionary mapping class indices to names, or list of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        figsize: Figure size (width, height) in inches
        cmap: Color map for heatmap
        max_classes: Maximum number of classes to show (None for all)
        text_size: Font size for text annotations

    Returns:
        matplotlib figure
    """
    # Create mapping from class indices to names if provided
    if class_names is None:
        # Default to index strings if no mapping provided
        unique_labels = sorted(set(np.concatenate([y_true, y_pred])))
        class_names = {i: str(i) for i in unique_labels}

    # Handle both dict and list formats for class_names
    if isinstance(class_names, dict):
        # Convert dict to list for confusion matrix labels
        unique_labels = sorted(set(np.concatenate([y_true, y_pred])))
        class_list = [class_names.get(i, str(i)) for i in unique_labels]
        indices = unique_labels
    else:
        # Assume class_names is already a list
        class_list = class_names
        indices = list(range(len(class_names)))

    # If we have too many classes, select only the most common ones
    if max_classes is not None and len(indices) > max_classes:
        # Count occurrences of each class in true labels
        class_counts = {}
        for i, idx in enumerate(indices):
            count = np.sum(y_true == idx)
            class_counts[idx] = count

        # Select top max_classes by count
        top_indices = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)[:max_classes]

        # Filter labels and class list to only include top classes
        mask_true = np.isin(y_true, top_indices)
        mask_pred = np.isin(y_pred, top_indices)

        # Keep only samples where both true and pred are in top classes
        mask = mask_true & mask_pred
        filtered_y_true = y_true[mask]
        filtered_y_pred = y_pred[mask]

        # Remap indices to be consecutive
        index_map = {idx: i for i, idx in enumerate(top_indices)}
        remapped_y_true = np.array([index_map[idx] for idx in filtered_y_true])
        remapped_y_pred = np.array([index_map[idx] for idx in filtered_y_pred])

        # Filter class list
        filtered_class_list = [class_list[indices.index(idx)] for idx in top_indices]

        # Use filtered data
        cm_indices = top_indices
        cm_classes = filtered_class_list
        y_true_cm = remapped_y_true
        y_pred_cm = remapped_y_pred
    else:
        # Use all classes
        cm_indices = indices
        cm_classes = class_list
        y_true_cm = y_true
        y_pred_cm = y_pred

    # Create the confusion matrix
    cm = confusion_matrix(y_true_cm, y_pred_cm)

    # Normalize if requested
    if normalize:
        # Handle division by zero by adding a small epsilon to avoid warnings
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        # Add a small epsilon to avoid division by zero
        row_sums = np.where(row_sums == 0, 1e-10, row_sums)
        cm = cm.astype('float') / row_sums
        cm = np.nan_to_num(cm)  # Replace any remaining NaN with zero

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate suitable font size for axis ticks based on number of classes
    n_classes = len(cm_classes)
    font_size = max(5, min(10, 20 - 0.1 * n_classes))

    # Plot the confusion matrix with seaborn for better coloring
    sns.heatmap(cm, annot=n_classes <= 20, fmt='.2f' if normalize else 'd',
                cmap=cmap, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=cm_classes, yticklabels=cm_classes, annot_kws={"size": text_size})

    # Improve the layout
    plt.tight_layout()
    ax.set_xlabel('Predicted', fontsize=font_size + 2)
    ax.set_ylabel('True', fontsize=font_size + 2)

    if title:
        ax.set_title(title, fontsize=font_size + 4)

    # Set tick font sizes
    plt.xticks(rotation=45, ha='right', fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)

    return fig

def plot_class_distribution(y_true, class_names=None, figsize=(12, 8), title='Class Distribution',
                            max_classes=20, save_path=None, dpi=300, vertical=True):
    """
    Plot the distribution of classes in a dataset.

    Args:
        y_true: Array of true class indices or labels
        class_names: Dictionary mapping class indices to names, or list of class names
        figsize: Figure size (width, height) in inches
        title: Plot title
        max_classes: Maximum number of classes to show (None for all)
        save_path: Path to save the figure (if None, just returns the figure)
        dpi: DPI for saved figure
        vertical: If True, use vertical bars; otherwise, horizontal

    Returns:
        matplotlib figure
    """
    # Count frequency of each class
    class_counts = Counter(y_true)

    # Create mapping from class indices to names if provided
    if class_names is None:
        # Default to index strings if no mapping provided
        class_mapping = {idx: str(idx) for idx in class_counts.keys()}
    elif isinstance(class_names, dict):
        # Use provided dictionary
        class_mapping = class_names
    else:
        # Assume class_names is a list and map indices to it
        class_mapping = {i: name for i, name in enumerate(class_names) if i in class_counts}

    # Sort by frequency (most common first) and limit to max_classes
    sorted_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)
    if max_classes is not None and len(sorted_classes) > max_classes:
        sorted_classes = sorted_classes[:max_classes]

    # Get class names and counts for plotting
    class_labels = [class_mapping.get(cls, str(cls)) for cls in sorted_classes]
    class_values = [class_counts[cls] for cls in sorted_classes]

    # Calculate percentages
    total_samples = sum(class_counts.values())
    percentages = [100.0 * count / total_samples for count in class_values]

    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot horizontal or vertical bars
    if vertical:
        bars = ax.bar(class_labels, class_values, color='steelblue')
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}',
                    ha='center', va='bottom', rotation=0)
        # Set labels
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
    else:
        # Horizontal bars with class names on y-axis
        bars = ax.barh(class_labels, class_values, color='steelblue')
        # Add count and percentage labels inside bars
        for i, (bar, percentage) in enumerate(zip(bars, percentages)):
            width = bar.get_width()
            if width > 0:
                ax.text(width * 0.5, bar.get_y() + bar.get_height()/2,
                        f'{int(width)} ({percentage:.1f}%)',
                        ha='center', va='center', color='white')
        # Set labels
        ax.set_ylabel('Class')
        ax.set_xlabel('Count')

    # Set title and adjust layout
    ax.set_title(title)
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # Close the figure to free memory
        plt.close(fig)

    return fig

def map_chord_to_quality(chord_name):
    """
    Map a chord name to its quality group.

    Args:
        chord_name (str): The chord name (e.g., "C:maj", "A:min", "G:7", "N")

    Returns:
        str: The chord quality group name
    """
    # Handle non-string input
    if not isinstance(chord_name, str):
        return "Other"

    # Try to import chord functions from chords.py
    try:
        from modules.utils.chords import get_chord_quality
        return get_chord_quality(chord_name)
    except (ImportError, AttributeError):
        # Fallback implementation if get_chord_quality isn't available

        # Handle special cases
        if chord_name in ["N", "X", "None", "Unknown", "NC"]:
            return "No Chord"

        # Extract quality using extract_chord_quality function
        quality = extract_chord_quality(chord_name)

        # Map extracted quality to broader categories
        quality_groups = {
            # Major family
            "maj": "Major", "": "Major", "M": "Major", "major": "Major",

            # Minor family
            "min": "Minor", "m": "Minor", "minor": "Minor",

            # Dominant seventh family
            "7": "Dom7", "dom7": "Dom7", "dominant": "Dom7",

            # Major seventh family
            "maj7": "Maj7", "M7": "Maj7", "major7": "Maj7",

            # Minor seventh family
            "min7": "Min7", "m7": "Min7", "minor7": "Min7",

            # Diminished family
            "dim": "Dim", "°": "Dim", "o": "Dim", "diminished": "Dim",

            # Diminished seventh family
            "dim7": "Dim7", "°7": "Dim7", "o7": "Dim7", "diminished7": "Dim7",

            # Half-diminished family
            "hdim7": "Half-Dim", "m7b5": "Half-Dim", "ø": "Half-Dim", "half-diminished": "Half-Dim",

            # Augmented family
            "aug": "Aug", "+": "Aug", "augmented": "Aug",

            # Suspended family
            "sus2": "Sus", "sus4": "Sus", "sus": "Sus", "suspended": "Sus",

            # Extended chords
            "9": "Extended", "11": "Extended", "13": "Extended",
            "maj9": "Extended", "maj11": "Extended", "maj13": "Extended",
            "min9": "Extended", "min11": "Extended", "min13": "Extended",

            # Additional common chord qualities
            "min6": "Min6", "m6": "Min6",
            "maj6": "Maj6", "6": "Maj6",
            "minmaj7": "Min-Maj7", "mmaj7": "Min-Maj7", "min-maj7": "Min-Maj7",

            # Special cases
            "N": "No Chord",
            "X": "No Chord",

            # Add additional chord qualities that appear in your dataset
            "min(9)": "Minor", "min/5": "Minor",  # Minor with extensions or inversions
            "maj/3": "Major", "maj/5": "Major",   # Major with inversions
            "7/3": "Dom7",                        # Dominant seventh with inversion
        }

        # Return mapped quality or "Other" if not found
        return quality_groups.get(quality, "Other")

def extract_chord_quality(chord):
    """
    Extract chord quality from a chord label, handling different formats.
    Supports both colon format (C:maj) and direct format (Cmaj).

    Args:
        chord: A chord label string

    Returns:
        The chord quality as a string
    """
    # Handle None or empty strings
    if not chord:
        return "N"  # Default to "N" for empty chords

    # Handle special cases
    if chord in ["N", "None", "NC"]:
        return "N"  # No chord
    if chord in ["X", "Unknown"]:
        return "X"  # Unknown chord

    # Handle common case where chord might be a numeric index instead of a string
    if not isinstance(chord, str):
        return str(chord)

    # Handle colon format (e.g., "C:min")
    if ':' in chord:
        parts = chord.split(':')
        if len(parts) > 1:
            # Handle bass notes (e.g., "C:min/G")
            quality = parts[1].split('/')[0] if '/' in parts[1] else parts[1]
            return quality

    # Handle direct format without colon (e.g., "Cmin")
    import re
    root_pattern = r'^[A-G][#b]?'
    match = re.match(root_pattern, chord)
    if match:
        quality = chord[match.end():]
        if quality:
            # Handle bass notes (e.g., "Cmin/G")
            return quality.split('/')[0] if '/' in quality else quality

    # Default to major if we couldn't extract a quality
    return "maj"

def get_chord_quality_groups():
    """
    Get the list of chord quality groups from chords.py if available,
    otherwise return a default list.
    """
    try:
        from modules.utils.chords import CHORD_QUALITIES
        return CHORD_QUALITIES
    except (ImportError, AttributeError):
        return DEFAULT_CHORD_QUALITIES

def group_chords_by_focus_qualities(predictions, targets, idx_to_chord, focus_qualities=None):
    """
    Group chords by specific focus qualities.

    Args:
        predictions: List of predicted chord indices
        targets: List of target chord indices
        idx_to_chord: Dictionary mapping indices to chord names
        focus_qualities: List of qualities to focus on

    Returns:
        tuple: (pred_qualities_idx, target_qualities_idx, qualities_list, quality_to_idx, idx_to_quality)
    """
    if focus_qualities is None:
        focus_qualities = ["maj", "min", "dim", "aug", "min6", "maj6", "min7",
                          "min-maj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]

    # Add "N" and "other" to cover all cases
    if "N" not in focus_qualities:
        focus_qualities.append("N")
    if "other" not in focus_qualities:
        focus_qualities.append("other")

    # Create mapping dictionaries
    quality_to_idx = {q: i for i, q in enumerate(focus_qualities)}
    idx_to_quality = {}

    # Add logging to see what's in idx_to_chord for debugging
    info(f"First 5 idx_to_chord mappings for debugging:")
    for idx, chord_name in list(idx_to_chord.items())[:5]:
        info(f"  {idx}: {chord_name}")

    # Map chord indices to qualities
    for idx, chord in idx_to_chord.items():
        try:
            # Get the raw quality first
            raw_quality = extract_chord_quality(chord)
            debug_level = "info" if idx < 5 else "debug"  # Only log first few extensively

            if debug_level == "info":
                info(f"Extracted quality '{raw_quality}' from chord '{chord}'")

            # Simple mapping for common qualities
            if raw_quality in quality_to_idx:
                idx_to_quality[idx] = raw_quality
            # Handle variations that should map to standard qualities
            elif raw_quality in ["major", "M"]:
                idx_to_quality[idx] = "maj"
            elif raw_quality in ["minor", "m"]:
                idx_to_quality[idx] = "min"
            elif raw_quality in ["M7", "major7"]:
                idx_to_quality[idx] = "maj7"
            elif raw_quality in ["m7", "minor7"]:
                idx_to_quality[idx] = "min7"
            elif raw_quality in ["°", "o", "diminished"]:
                idx_to_quality[idx] = "dim"
            elif raw_quality in ["°7", "o7", "diminished7"]:
                idx_to_quality[idx] = "dim7"
            elif raw_quality in ["m7b5", "ø", "half-diminished"]:
                idx_to_quality[idx] = "hdim7"
            elif raw_quality in ["m6", "minor6"]:
                idx_to_quality[idx] = "min6"
            elif raw_quality in ["6", "maj6"]:
                idx_to_quality[idx] = "maj6"
            elif raw_quality in ["mmaj7", "mM7"]:
                idx_to_quality[idx] = "min-maj7"
            # Add more mappings for commonly encountered variations
            elif raw_quality.startswith("min"):
                idx_to_quality[idx] = "min"
            elif raw_quality.startswith("maj"):
                idx_to_quality[idx] = "maj"
            elif raw_quality.startswith("7"):
                idx_to_quality[idx] = "7"
            else:
                idx_to_quality[idx] = "other"
        except Exception as e:
            # Handle any errors and default to "other"
            idx_to_quality[idx] = "other"
            if idx < 10:  # Only log errors for first few chord indices to avoid spam
                info(f"Error processing chord {chord} at index {idx}: {e}")

    # Log quality mapping statistics
    quality_counts = {}
    for quality in idx_to_quality.values():
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

    info(f"Quality mapping statistics:")
    for quality, count in sorted(quality_counts.items(), key=lambda x: x[1], reverse=True):
        info(f"  {quality}: {count} chords")

    # Map predictions and targets to quality indices
    pred_qualities_idx = []
    target_qualities_idx = []

    for pred, target in zip(predictions, targets):
        # Convert numpy arrays to Python integers if needed
        if isinstance(pred, np.ndarray):
            # Handle arrays with multiple elements
            if pred.size > 1:
                # Take the first element or most common value
                pred = pred[0] if pred.size > 0 else 0
            else:
                # For single element arrays, use item()
                pred = pred.item()

        if isinstance(target, np.ndarray):
            # Handle arrays with multiple elements
            if target.size > 1:
                # Take the first element or most common value
                target = target[0] if target.size > 0 else 0
            else:
                # For single element arrays, use item()
                target = target.item()

        pred_quality = idx_to_quality.get(pred, "other")
        target_quality = idx_to_quality.get(target, "other")

        pred_qualities_idx.append(quality_to_idx[pred_quality])
        target_qualities_idx.append(quality_to_idx[target_quality])

    return np.array(pred_qualities_idx), np.array(target_qualities_idx), focus_qualities, quality_to_idx, idx_to_quality

def group_chords_by_quality(predictions, targets, idx_to_chord):
    """
    Group chord predictions and targets by quality.

    Args:
        predictions: List of predicted chord indices
        targets: List of target chord indices
        idx_to_chord: Dictionary mapping indices to chord names

    Returns:
        tuple: (pred_qualities, target_qualities, quality_names)
    """
    # Get the quality groups
    quality_groups = get_chord_quality_groups()

    # Map each chord to its quality
    idx_to_quality = {}
    for idx, chord in idx_to_chord.items():
        quality = map_chord_to_quality(chord)
        idx_to_quality[idx] = quality
        if quality not in quality_groups:
            quality_groups.append(quality)

    # Map predictions and targets to quality groups
    pred_qualities = []
    target_qualities = []

    for pred, target in zip(predictions, targets):
        # Get qualities (default to "Other" if not found)
        pred_quality = idx_to_quality.get(pred, "Other")
        target_quality = idx_to_quality.get(target, "Other")

        # Convert to indices in quality_groups
        pred_idx = quality_groups.index(pred_quality) if pred_quality in quality_groups else quality_groups.index("Other")
        target_idx = quality_groups.index(target_quality) if target_quality in quality_groups else quality_groups.index("Other")

        pred_qualities.append(pred_idx)
        target_qualities.append(target_idx)

    return np.array(pred_qualities), np.array(target_qualities), quality_groups

def calculate_quality_confusion_matrix(predictions, targets, idx_to_chord):
    """
    Calculate confusion matrix for chord qualities.

    Args:
        predictions: List of predicted chord indices
        targets: List of target chord indices
        idx_to_chord: Dictionary mapping indices to chord names

    Returns:
        tuple: (quality_cm, quality_counts, quality_accuracy, quality_groups)
    """
    # Group by quality
    pred_qualities, target_qualities, quality_groups = group_chords_by_quality(
        predictions, targets, idx_to_chord
    )

    # Calculate confusion matrix
    quality_cm = confusion_matrix(
        y_true=target_qualities,
        y_pred=pred_qualities,
        labels=list(range(len(quality_groups)))
    )

    # Count samples per quality group
    quality_counts = Counter(target_qualities)

    # Calculate accuracy per quality group
    quality_accuracy = {}
    for i, quality in enumerate(quality_groups):
        true_idx = np.where(target_qualities == i)[0]
        if len(true_idx) > 0:
            correct = np.sum(pred_qualities[true_idx] == i)
            accuracy = correct / len(true_idx)
            quality_accuracy[quality] = accuracy
        else:
            quality_accuracy[quality] = 0.0

    return quality_cm, quality_counts, quality_accuracy, quality_groups

def calculate_confusion_matrix(predictions, targets, idx_to_chord, checkpoint_dir, current_epoch=None):
    """
    Calculate, log, and save the confusion matrix.
    Prints chord quality groups for clearer visualization and also saves the
    full class confusion matrix every 10 epochs.
    Detailed class distribution and confusion matrix analysis are printed every 5 epochs.

    Args:
        predictions: List of predicted class indices
        targets: List of target class indices
        current_epoch: Current training epoch number (for periodic full matrix saving)
    """
    if not predictions or not targets:
        error("Cannot calculate confusion matrix: no predictions or targets")
        return

    # Determine if we should print detailed information
    # Print on: first epoch, last epoch, every 5th epoch, or when current_epoch is None
    is_first_epoch = current_epoch == 1
    is_last_epoch = False  # We don't know this, but we'll handle it separately
    is_print_epoch = current_epoch is None or is_first_epoch or is_last_epoch or current_epoch % 5 == 0

    try:
        # Count occurrences of each class in targets
        target_counter = Counter(targets)
        total_samples = len(targets)

        # Get the most common classes (up to 10) for standard printing
        most_common_classes = [cls for cls, _ in target_counter.most_common(10)]

        # Create a mapping of indices to chord names if available
        chord_names = {}
        if idx_to_chord:
            # First, build a reverse mapping from index to chord name
            for idx, chord in idx_to_chord.items():
                chord_names[idx] = chord

            # Also make sure all our most common classes are mapped
            for cls in most_common_classes:
                if cls not in chord_names:
                    if cls in idx_to_chord:
                        chord_names[cls] = idx_to_chord[cls]
                    else:
                        # If the class is not in idx_to_chord, use a consistent fallback
                        chord_names[cls] = f"Unknown-{cls}"
        else:
            # If no mapping is available, create generic labels
            for cls in most_common_classes:
                chord_names[cls] = f"Class-{cls}"

        # Only print detailed class distribution on print epochs
        if is_print_epoch:
            info("\nClass distribution in validation set (Top 10):")
            for cls in most_common_classes:
                try:
                    count = target_counter.get(cls, 0)
                    percentage = 100 * count / total_samples if total_samples > 0 else 0
                    chord_name = chord_names.get(cls, f"Class-{cls}")
                    info(f"  {chord_name}: {count} samples ({percentage:.2f}%)")
                except Exception as e:
                    error(f"Error processing class {cls}: {e}")

            # Calculate confusion matrix values for top classes (printed to log)
            confusion = {}
            for true_cls in most_common_classes:
                try:
                    # Get indices where true class equals this class
                    true_indices = [i for i, t in enumerate(targets) if t == true_cls]
                    if not true_indices:
                        continue

                    # Get predictions for these indices
                    cls_preds = [predictions[i] for i in true_indices]
                    pred_counter = Counter(cls_preds)

                    # Calculate accuracy for this class
                    correct = pred_counter.get(true_cls, 0)
                    total = len(true_indices)
                    accuracy = correct / total if total > 0 else 0

                    # Get the most commonly predicted class for this true class
                    most_predicted = pred_counter.most_common(1)[0][0] if pred_counter else true_cls

                    # Use the chord_names dictionary consistently
                    true_chord_name = chord_names.get(true_cls, f"Class-{true_cls}")
                    pred_chord_name = chord_names.get(most_predicted, f"Class-{most_predicted}")

                    confusion[true_chord_name] = {
                        'accuracy': accuracy,
                        'most_predicted': pred_chord_name,
                        'correct': correct,
                        'total': total
                    }
                except Exception as e:
                    error(f"Error processing confusion data for class {true_cls}: {e}")

            # Log confusion matrix information for top classes
            info("\nConfusion Matrix Analysis (Top 10 Classes):")
            info(f"{'True Class':<20} | {'Accuracy':<10} | {'Most Predicted':<20} | {'Correct/Total'}")
            info(f"{'-'*20} | {'-'*10} | {'-'*20} | {'-'*15}")

            for true_class, stats in confusion.items():
                info(f"{true_class:<20} | {stats['accuracy']:.4f}     | {stats['most_predicted']:<20} | {stats['correct']}/{stats['total']}")

            # Calculate overall metrics for these common classes
            common_correct = sum(confusion[cls]['correct'] for cls in confusion)
            common_total = sum(confusion[cls]['total'] for cls in confusion)
            common_acc = common_correct / common_total if common_total > 0 else 0
            info(f"\nAccuracy on most common classes: {common_acc:.4f} ({common_correct}/{common_total})")

        # NEW: Create and visualize chord quality group confusion matrix using chords.py
        if idx_to_chord:
            # Only print detailed quality information on print epochs
            if is_print_epoch:
                info("\n=== Creating chord quality group confusion matrix ===")

            try:
                # Use the visualization module to calculate quality statistics
                quality_cm, quality_counts, quality_accuracy, quality_groups = calculate_quality_confusion_matrix(
                    predictions, targets, idx_to_chord
                )

                # Log quality distribution only on print epochs (every 5 epochs)
                if is_print_epoch:
                    info("\nChord quality distribution:")
                    for i, quality in enumerate(quality_groups):
                        count = quality_counts.get(i, 0)
                        percentage = 100 * count / len(targets) if targets else 0
                        info(f"  {quality}: {count} samples ({percentage:.2f}%)")

                    # Log accuracies by quality
                    info("\nAccuracy by chord quality:")
                    for quality, acc in sorted(quality_accuracy.items(), key=lambda x: x[1], reverse=True):
                        info(f"  {quality}: {acc:.4f}")

                    # Add special reporting for 'N' (no chord) and 'X' (unknown chord)
                    n_chord_idx = None
                    x_chord_idx = None

                    # Find indices for 'N' and 'X' in quality_groups
                    for i, quality in enumerate(quality_groups):
                        if quality == "No Chord":
                            n_chord_idx = i
                        elif quality == "Unknown":
                            x_chord_idx = i

                    # Report counts for 'N' and 'X'
                    if n_chord_idx is not None:
                        n_count = quality_counts.get(n_chord_idx, 0)
                        # Use the quality name "No Chord" as the key, not the index
                        n_accuracy = quality_accuracy.get("No Chord", 0.0)
                        info(f"  'N' (No Chord): {n_count} samples, accuracy: {n_accuracy:.4f}")

                    if x_chord_idx is not None:
                        x_count = quality_counts.get(x_chord_idx, 0)
                        # Use the quality name "Unknown" as the key, not the index
                        x_accuracy = quality_accuracy.get("Unknown", 0.0)
                        info(f"  'X' (Unknown Chord): {x_count} samples, accuracy: {x_accuracy:.4f}")

                # Create and save chord quality confusion matrix
                title = f"Chord Quality Confusion Matrix - Epoch {current_epoch}"
                quality_cm_path = os.path.join(
                    checkpoint_dir,
                    f"confusion_matrix_quality_epoch_{current_epoch}.png"
                )

                # Plot using the visualization function
                try:
                    _, _, _, _, _ = plot_chord_quality_confusion_matrix(
                        predictions, targets, idx_to_chord,
                        title=title, save_path=quality_cm_path
                    )
                    if is_print_epoch:
                        info(f"Saved chord quality confusion matrix to {quality_cm_path}")
                except Exception as e:
                    error(f"Error plotting chord quality confusion matrix: {e}")
                    # Print normalized confusion matrix as text fallback on print epochs
                    if is_print_epoch:
                        info("\nChord Quality Confusion Matrix (normalized):")
                        normalized_cm = quality_cm.astype('float') / quality_cm.sum(axis=1)[:, np.newaxis]
                        normalized_cm = np.nan_to_num(normalized_cm)  # Replace NaN with zero
                        for i, row in enumerate(normalized_cm):
                            info(f"{quality_groups[i]:<10}: " + " ".join([f"{x:.2f}" for x in row]))

            except Exception as e:
                error(f"Error creating quality-based confusion matrix: {e}")
                error(traceback.format_exc())

        # Create and save the full confusion matrix every 10 epochs (less frequently)
        # Also save on the last epoch
        if current_epoch is None or current_epoch % 10 == 0:
            info(f"\nSaving full class confusion matrix (all 170 classes) for epoch {current_epoch}")

            try:
                # Create a mapping that includes ALL possible chord indices
                all_class_mapping = {}
                if idx_to_chord:
                    for idx, chord in idx_to_chord.items():
                        all_class_mapping[idx] = chord

                # Get a list of all unique classes
                all_classes = set(targets).union(set(predictions))

                # Ensure the classes are sorted for consistent visualization
                all_classes_list = sorted(list(all_classes))

                # Make sure all classes have labels
                for cls in all_classes:
                    if cls not in all_class_mapping:
                        all_class_mapping[cls] = f"Class-{cls}"

                # Generate the full confusion matrix using the visualization function
                full_title = f"Full Confusion Matrix - Epoch {current_epoch}"
                np_targets_full = np.array(targets)
                np_preds_full = np.array(predictions)

                # Set save path
                full_cm_path = os.path.join(
                    checkpoint_dir,
                    f"confusion_matrix_full_epoch_{current_epoch}.png"
                )

                # Generate full confusion matrix plot and save it
                fig_full = plot_confusion_matrix(
                    np_targets_full, np_preds_full,
                    class_names=all_class_mapping,
                    normalize=True,
                    title=full_title,
                    max_classes=None  # No limit on number of classes
                )

                os.makedirs(checkpoint_dir, exist_ok=True)
                fig_full.savefig(full_cm_path, dpi=300, bbox_inches='tight')
                info(f"Saved full confusion matrix to {full_cm_path}")
                plt.close(fig_full)

            except Exception as e:
                error(f"Error saving full confusion matrix visualization: {e}")
                error(traceback.format_exc())

    except Exception as e:
        error(f"Error calculating confusion matrix: {e}")
        error(traceback.format_exc())

def plot_chord_quality_confusion_matrix(predictions, targets, idx_to_chord,
                                       title=None, figsize=(10, 8), text_size=10,
                                       save_path=None, dpi=300):
    """
    Create and save a chord quality confusion matrix.

    Args:
        predictions: List of predicted chord indices
        targets: List of target chord indices
        idx_to_chord: Dictionary mapping indices to chord names
        title: Plot title
        figsize: Figure size (width, height) in inches
        text_size: Font size for text annotations
        save_path: Path to save the figure (if None, just returns the figure)
        dpi: DPI for saved figure

    Returns:
        tuple: (figure, quality_accuracy, quality_cm)
    """
    # Group by quality and calculate metrics
    quality_cm, quality_counts, quality_accuracy, quality_groups = calculate_quality_confusion_matrix(
        predictions, targets, idx_to_chord
    )

    # Create confusion matrix plot
    pred_qualities, target_qualities, _ = group_chords_by_quality(
        predictions, targets, idx_to_chord
    )

    fig = plot_confusion_matrix(
        target_qualities,
        pred_qualities,
        class_names=quality_groups,
        normalize=True,
        title=title,
        figsize=figsize,
        text_size=text_size
    )

    # Save the figure if requested
    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, quality_accuracy, quality_cm, quality_groups, quality_counts

def plot_chord_quality_distribution_accuracy(predictions, targets, idx_to_chord, save_path=None,
                                            figsize=(14, 8), title=None, dpi=300,
                                            focus_qualities=None):
    """
    Create a bar chart of chord quality distribution with overlaid accuracy line.

    Args:
        predictions: List of predicted chord indices
        targets: List of target chord indices
        idx_to_chord: Dictionary mapping indices to chord names
        save_path: Path to save the figure (if None, just returns the figure)
        figsize: Figure size (width, height) in inches
        title: Plot title
        dpi: DPI for saved figure
        focus_qualities: List of chord qualities to focus on

    Returns:
        matplotlib figure
    """
    if focus_qualities is None:
        focus_qualities = ["maj", "min", "dim", "aug", "min6", "maj6", "min7",
                          "min-maj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]

    # Group chords by quality
    pred_qualities, target_qualities, qualities_list, _, idx_to_quality = group_chords_by_focus_qualities(
        predictions, targets, idx_to_chord, focus_qualities
    )

    # Calculate counts for each quality
    quality_counts = Counter(target_qualities)

    # Calculate accuracy for each quality
    quality_accuracy = {}
    for i, quality in enumerate(qualities_list):
        # Find indices where target is this quality
        quality_indices = np.where(target_qualities == i)[0]
        if len(quality_indices) > 0:
            # Calculate correct predictions for this quality
            correct = np.sum(pred_qualities[quality_indices] == i)
            accuracy = correct / len(quality_indices)
        else:
            accuracy = 0
        quality_accuracy[quality] = accuracy

    # Prepare data for plotting
    filtered_qualities = []
    filtered_counts = []
    filtered_accuracies = []

    total_samples = len(targets)

    # Filter to only include focus qualities (excluding "other" and "N" for cleaner visualization)
    for quality in focus_qualities:
        if quality in qualities_list and quality not in ["other", "N"]:
            idx = qualities_list.index(quality)
            count = quality_counts.get(idx, 0)
            if count > 0:  # Only include qualities that appear in the data
                filtered_qualities.append(quality)
                filtered_counts.append(count)
                filtered_accuracies.append(quality_accuracy[quality])

    # Check if we have any data to plot
    if not filtered_counts:
        # Create an empty figure with a message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No chord quality data available for visualization",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()

        # Save the figure if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

        return fig

    # Calculate percentages
    percentages = [100 * count / total_samples for count in filtered_counts]

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot distribution bars
    bars = ax1.bar(filtered_qualities, percentages, alpha=0.7, color='steelblue', label='Distribution (%)')
    ax1.set_ylabel('Distribution (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, max(percentages) * 1.2 if percentages else 10)  # Add some headroom

    # Add percentage labels above bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='steelblue')

    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(filtered_qualities, filtered_accuracies, 'ro-', linewidth=2, markersize=8, label='Accuracy')
    ax2.set_ylabel('Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.0)

    # Add accuracy values as text
    for i, acc in enumerate(filtered_accuracies):
        ax2.annotate(f'{acc:.2f}',
                    xy=(i, acc),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='red')

    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Set title and adjust layout
    plt.title(title or "Chord Quality Distribution and Accuracy")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # Close the figure to free memory
        plt.close(fig)

    return fig

def plot_learning_curve(train_loss, val_loss=None, title='Learning Curve', figsize=(10, 6),
                       save_path=None, dpi=300):
    """
    Plot training and validation learning curves.

    Args:
        train_loss: List of training loss values
        val_loss: List of validation loss values (optional)
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, just returns the figure)
        dpi: DPI for saved figure

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot training loss
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, 'b-', label='Training Loss')

    # Plot validation loss if provided
    if val_loss:
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss')

    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save the figure if requested
    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # Close the figure to free memory
        plt.close(fig)

    return fig

def visualize_transitions(hmm_model, idx_to_chord, top_k=10, save_path='transitions.png',
                          figsize=(12, 6), cmap='viridis', title=None, dpi=300):
    """
    Visualize chord transition probabilities from an HMM model.

    Args:
        hmm_model: HMM model with transitions attribute (PyTorch tensor)
        idx_to_chord: Dictionary mapping indices to chord names
        top_k: Number of top transitions to display
        save_path: Path to save the visualization
        figsize: Figure size (width, height) in inches
        cmap: Color map for the bars
        title: Custom title (if None, a default title is used)
        dpi: DPI for saved figure

    Returns:
        matplotlib figure
    """
    # Get transition probabilities
    transitions = hmm_model.transitions.detach().cpu().numpy()

    # Convert from log space
    transitions = np.exp(transitions)

    # Find top k transitions
    top_pairs = []
    for i in range(transitions.shape[0]):
        for j in range(transitions.shape[1]):
            top_pairs.append((i, j, transitions[i, j]))

    top_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = top_pairs[:top_k]

    # Create labels
    labels = []
    values = []
    for i, j, v in top_pairs:
        # Handle cases where indices aren't in the chord mapping
        chord_i = idx_to_chord.get(i, f"Chord-{i}")
        chord_j = idx_to_chord.get(j, f"Chord-{j}")
        labels.append(f"{chord_i}→{chord_j}")
        values.append(v)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=labels, y=values, ax=ax, palette=cmap)
    plt.xticks(rotation=45, ha='right')
    plt.title(title or f"Top {top_k} Chord Transitions")
    plt.tight_layout()

    # Save the figure if requested
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # Close the figure to free memory
        plt.close(fig)

    return fig