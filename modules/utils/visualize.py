import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, 
                          title='Confusion Matrix', figsize=(12, 10), cmap=plt.cm.Blues,
                          max_classes=15):
    """
    Generate a confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        normalize: Whether to normalize values
        title: Plot title
        figsize: Figure size
        cmap: Color map
        max_classes: Maximum number of classes to display
    
    Returns:
        matplotlib figure
    """
    # Convert tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Get unique classes from both predictions and targets
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # If we have too many classes, focus on the most common ones
    if len(unique_classes) > max_classes:
        # Count class occurrences
        class_counts = np.bincount(y_true)
        # Get indices of top N most common classes
        common_classes = np.argsort(class_counts)[-max_classes:]
        # Create a mask for samples that belong to these classes
        mask = np.isin(y_true, common_classes)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        unique_classes = common_classes
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.2)
    
    # If class names are provided, use them
    if class_names and len(class_names) >= len(unique_classes):
        display_names = [class_names[i] for i in unique_classes]
    else:
        display_names = [str(i) for i in unique_classes]
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=display_names,
                yticklabels=display_names,
                linewidths=.5)
    
    plt.title(title, fontsize=16)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    
    # Attempt to improve readability by rotating tick labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Tight layout to ensure everything fits
    plt.tight_layout()
    
    return plt.gcf()

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
