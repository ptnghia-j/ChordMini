import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
import os
import re

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
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Replace NaN with zero
    
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

def map_chord_to_quality(chord_name):
    """
    Map a chord name to its quality group.
    
    Args:
        chord_name (str): The chord name (e.g., "C:maj", "A:min", "G:7", "N")
        
    Returns:
        str: The chord quality group name
    """
    # Try to import chord functions from chords.py
    try:
        from modules.utils.chords import get_chord_quality
        return get_chord_quality(chord_name)
    except (ImportError, AttributeError):
        # Fallback implementation if get_chord_quality isn't available
        
        # Handle special cases
        if chord_name in ["N", "X", "None", "Unknown"]:
            return "No Chord"
            
        # Standardize notation if it uses : separator
        chord_name = chord_name.replace(":", "")
            
        # Define quality mappings with regex patterns
        quality_mappings = [
            # Major quality group
            (r'^[A-G]$|maj$|^[A-G]maj$', 'Major'),
            # Minor quality group
            (r'min$|m$|^[A-G]m$', 'Minor'),
            # Dominant 7th quality group
            (r'7$|^[A-G]7$|dom7$', 'Dom7'),
            # Major 7th quality group
            (r'maj7$|^[A-G]maj7$', 'Maj7'),
            # Minor 7th quality group
            (r'min7$|m7$|^[A-G]m7$', 'Min7'),
            # Diminished quality group
            (r'dim$|°|^[A-G]dim$|^[A-G]o$', 'Dim'),
            # Diminished 7th quality group
            (r'dim7$|°7|^[A-G]dim7$|^[A-G]o7$', 'Dim7'),
            # Half-diminished quality group
            (r'hdim$|ø|m7b5$|^[A-G]ø$', 'Half-Dim'),
            # Augmented quality group
            (r'aug$|\+$|^[A-G]\+$|^[A-G]aug$', 'Aug'),
            # Suspended quality group
            (r'sus$|sus2$|sus4$', 'Sus'),
            # Extended chords 9th, 11th, 13th
            (r'9$|11$|13$', 'Extended')
        ]
            
        # Check each pattern
        for pattern, quality in quality_mappings:
            if re.search(pattern, chord_name):
                return quality
                
        # Default quality for any other chord
        return "Other"

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
    
    return fig
