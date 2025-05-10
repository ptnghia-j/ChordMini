"""
File utility functions for ChordMini project.

This module contains common file and path handling utilities used across
different training scripts in the ChordMini project.
"""

import os
import glob
import torch
from pathlib import Path
from modules.utils import logger

def count_files_in_subdirectories(directory, file_pattern):
    """
    Count files in a directory and all its subdirectories matching a pattern.
    
    Args:
        directory (str): The directory to search in
        file_pattern (str): The file pattern to match (e.g., "*.npy")
        
    Returns:
        int: The number of matching files found
    """
    if not directory or not os.path.exists(directory):
        return 0
    
    count = 0
    # Use Path.rglob for recursive search
    for file_path in Path(directory).rglob(file_pattern):
        if file_path.is_file():
            count += 1
    
    return count

def find_sample_files(directory, file_pattern, max_samples=5):
    """
    Find sample files in a directory and all its subdirectories matching a pattern.
    
    Args:
        directory (str): The directory to search in
        file_pattern (str): The file pattern to match (e.g., "*.npy")
        max_samples (int): Maximum number of sample files to return
        
    Returns:
        list: List of paths to sample files
    """
    if not directory or not os.path.exists(directory):
        return []
    
    samples = []
    # Use Path.rglob for recursive search
    for file_path in Path(directory).rglob(file_pattern):
        if file_path.is_file():
            samples.append(str(file_path))
            if len(samples) >= max_samples:
                break
    
    return samples

def resolve_path(path, storage_root=None, project_root=None):
    """
    Resolve a path that could be absolute, relative to storage_root, or relative to project_root.
    
    Args:
        path (str): The path to resolve
        storage_root (str): The storage root path
        project_root (str): The project root path
        
    Returns:
        str: The resolved absolute path
    """
    if not path:
        return None
    
    # If it's already absolute, return it directly
    if os.path.isabs(path):
        return path
    
    # Try as relative to storage_root first
    if storage_root:
        storage_path = os.path.join(storage_root, path)
        # Check existence for storage_root resolution
        if os.path.exists(storage_path):
            return storage_path
    
    # Then try as relative to project_root
    if project_root:
        project_path = os.path.join(project_root, path)
        # Check existence for project_root resolution
        if os.path.exists(project_path):
            return project_path
    
    # Fallback: prefer storage_root if provided, otherwise project_root
    if storage_root:
        return os.path.join(storage_root, path)
    
    return os.path.join(project_root, path) if project_root else path

def load_normalization_from_checkpoint(path, storage_root=None, project_root=None):
    """
    Load mean and std from a teacher checkpoint, or return (0.0, 1.0) if unavailable.
    
    Args:
        path (str): Path to the checkpoint file
        storage_root (str): Optional storage root path
        project_root (str): Optional project root path
        
    Returns:
        tuple: (mean, std) as float values
    """
    if not path:
        logger.warning("No teacher checkpoint specified for normalization. Using defaults (0.0, 1.0).")
        return 0.0, 1.0
    
    resolved_path = resolve_path(path, storage_root, project_root)
    if not os.path.exists(resolved_path):
        logger.warning(f"Teacher checkpoint for normalization not found at {resolved_path}. Using defaults (0.0, 1.0).")
        return 0.0, 1.0
    
    try:
        checkpoint = torch.load(resolved_path, map_location='cpu')
        mean = checkpoint.get('mean', 0.0)
        std = checkpoint.get('std', 1.0)
        mean = float(mean.item()) if hasattr(mean, 'item') else float(mean)
        std = float(std.item()) if hasattr(std, 'item') else float(std)
        
        if std == 0:
            logger.warning("Teacher checkpoint std is zero, using 1.0 instead.")
            std = 1.0
            
        logger.info(f"Loaded normalization from teacher checkpoint: mean={mean:.4f}, std={std:.4f}")
        return mean, std
    except Exception as e:
        logger.error(f"Error loading normalization from teacher checkpoint: {e}")
        logger.warning("Using default normalization parameters (mean=0.0, std=1.0).")
        return 0.0, 1.0

def find_data_directory(primary_path, alt_path, file_type, description):
    """
    Find directories containing data files with comprehensive fallback options.
    
    Args:
        primary_path (str): Primary path to check
        alt_path (str): Alternative path to check
        file_type (str): File type pattern to match
        description (str): Description of the data type for logging
        
    Returns:
        tuple: (path, count) - The found path and file count
    """
    paths_to_check = [
        primary_path,                          # Primary path from config/args
        alt_path,                              # Alternative path from config
        f"/mnt/storage/data/synth/{description}s",  # Common fallback location
        os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../../data/synth/{description}s")  # Project fallback
    ]

    # Filter out None paths
    paths_to_check = [p for p in paths_to_check if p]

    for path in paths_to_check:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Count files
        count = count_files_in_subdirectories(path, file_type)
        if count > 0:
            sample_files = find_sample_files(path, file_type, 3)
            logger.info(f"  {description.capitalize()}: {path} ({count} files)")
            if sample_files:
                logger.info(f"  Example files: {sample_files}")
            return path, count

    return paths_to_check[0] if paths_to_check else None, 0
