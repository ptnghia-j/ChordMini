import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
from collections import Counter

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, title=None, max_classes=20, figsize=(12,10)):
    """
    Generate a confusion matrix visualization.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: Dictionary mapping class indices to class names, or None for default names
        normalize: Whether to normalize by row (true label frequency)
        title: Title for plot
        max_classes: Maximum number of classes to include (limit to most common)
        figsize: Figure size as (width, height)
        
    Returns:
        Figure object with the plot
    """

    
    if len(y_true) == 0 or len(y_pred) == 0:
        # Create an empty figure if there's no data
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data to plot", horizontalalignment='center', 
                verticalalignment='center', fontsize=14)
        return fig
    
    # Count true labels and get the most common ones
    counter = Counter(y_true)
    most_common = [cls for cls, _ in counter.most_common(max_classes)]
    
    # Mask to include only the most common classes
    mask_true = np.isin(y_true, most_common)
    mask_pred = np.isin(y_pred, most_common)
    
    # Get indices for both true and predicted being in the top classes
    indices = np.where(mask_true)[0]
    
    if len(indices) == 0:
        # If no samples match the filter, create empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No common classes found", horizontalalignment='center', 
                verticalalignment='center', fontsize=14)
        return fig
    
    # Filter the labels
    filtered_y_true = np.array(y_true)[indices]
    filtered_y_pred = np.array(y_pred)[indices]
    
    # Compute confusion matrix
    cm = confusion_matrix(filtered_y_true, filtered_y_pred, labels=most_common)
    
    # Normalize the confusion matrix if requested
    if normalize:
        # FIX: Handle division by zero when normalizing
        row_sums = cm.sum(axis=1)
        # Add small epsilon to avoid division by zero
        row_sums = np.where(row_sums == 0, 1e-10, row_sums)  # Replace zeros with small value
        cm = cm.astype('float') / row_sums[:, np.newaxis]
        
        # Additional fix: Replace NaN values with zeros for better visualization
        cm = np.nan_to_num(cm, nan=0.0)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate class labels for plotting
    if class_names is not None:
        # Convert class indices to names for the most common classes
        label_names = []
        for cls in most_common:
            # Handle cases where the class may not be in the mapping
            if class_names is None:
                label_names.append(f"Class-{cls}")
            elif isinstance(class_names, dict):
                label_names.append(class_names.get(cls, f"Class-{cls}"))
            else:
                try:
                    # For list/array-like class_names
                    label_names.append(class_names[cls])
                except (IndexError, TypeError):
                    label_names.append(f"Class-{cls}")
    else:
        # Default to class indices if no mapping provided
        label_names = [f"Class-{cls}" for cls in most_common]
    
    # Plot confusion matrix with improved handling of NaN values
    try:
        # Use robust min/max values to avoid seaborn warnings
        vmin = np.nanmin(cm[~np.isnan(cm)]) if np.any(~np.isnan(cm)) else 0
        vmax = np.nanmax(cm[~np.isnan(cm)]) if np.any(~np.isnan(cm)) else 1
        
        # Create heatmap with robust parameters
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names, ax=ax,
                    vmin=vmin, vmax=vmax)
    except ValueError as e:
        # Fallback to a simpler heatmap without vmin/vmax if there's an error
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.text(0.5, 0.5, f"Warning: {str(e)}", transform=ax.transAxes, 
                horizontalalignment='center', color='red', alpha=0.7)
        
    # Set title and labels
    if title:
        ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Tight layout to ensure all labels are visible
    fig.tight_layout()
    
    return fig

def plot_class_distribution(class_counts, class_names=None, title='Class Distribution', 
                           figsize=(12, 8), max_classes=20):
    """
    Plot a bar chart of class distribution.
    
    Args:
        class_counts: Dictionary or Counter with class_idx:count pairs
        class_names: Dictionary mapping class indices to names (optional)
        title: Plot title
        figsize: Figure size
        max_classes: Maximum number of classes to display
    
    Returns:
        matplotlib figure
    """
    # Sort by frequency
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to max_classes
    items_to_plot = sorted_items[:max_classes]
    
    # Extract class indices and counts
    classes, counts = zip(*items_to_plot)
    
    # Map class indices to names if provided
    if class_names:
        labels = [class_names.get(cls, f"Class-{cls}") for cls in classes]
    else:
        labels = [f"Class-{cls}" for cls in classes]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot bars
    bars = plt.bar(range(len(counts)), counts, align='center')
    
    # Add count numbers on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{counts[i]}', ha='center', va='bottom', rotation=0)
    
    # Set labels and title
    plt.xticks(range(len(counts)), labels, rotation=45, ha='right')
    plt.title(title, fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()

def plot_learning_curve(epochs, train_metrics, val_metrics=None, metric_name='Loss',
                        title='Learning Curve', figsize=(10, 6)):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        epochs: List of epoch numbers
        train_metrics: List of training metric values
        val_metrics: List of validation metric values (optional)
        metric_name: Name of the metric being plotted
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Plot training metrics
    plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    
    # Plot validation metrics if available
    if val_metrics:
        plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    
    # Add labels and legend
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.legend()
    
    # Add grid for readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return plt.gcf()
