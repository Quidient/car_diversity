"""Utility functions for the pipeline."""

import numpy as np
import yaml
import json
from pathlib import Path
from typing import List, Union, Dict, Any
import logging
import shutil

logger = logging.getLogger(__name__)


def load_image_paths(image_dir: str) -> List[str]:
    """
    Load all image paths from a directory.

    Args:
        image_dir: Directory containing images

    Returns:
        List of image file paths
    """
    image_dir = Path(image_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.rglob(f'*{ext}'))
        image_paths.extend(image_dir.rglob(f'*{ext.upper()}'))

    return [str(p) for p in image_paths]


def save_results(
    data: Union[List, np.ndarray, Dict],
    output_path: Union[str, Path],
    mode: str = 'auto'
):
    """
    Save results to file.

    Args:
        data: Data to save
        output_path: Output file path
        mode: Save mode ('list', 'numpy', 'yaml', 'json', 'auto')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect mode from extension
    if mode == 'auto':
        ext = output_path.suffix.lower()
        if ext == '.npy':
            mode = 'numpy'
        elif ext in {'.yaml', '.yml'}:
            mode = 'yaml'
        elif ext == '.json':
            mode = 'json'
        elif ext == '.txt':
            mode = 'list'
        else:
            raise ValueError(f"Cannot auto-detect mode for extension: {ext}")

    # Save based on mode
    if mode == 'list':
        with open(output_path, 'w') as f:
            for item in data:
                f.write(f"{item}\n")

    elif mode == 'numpy':
        np.save(output_path, data)

    elif mode == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    elif mode == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    else:
        raise ValueError(f"Unknown save mode: {mode}")

    logger.info(f"Saved results to: {output_path}")


def load_results(
    input_path: Union[str, Path],
    mode: str = 'auto'
) -> Union[List, np.ndarray, Dict]:
    """
    Load results from file.

    Args:
        input_path: Input file path
        mode: Load mode ('list', 'numpy', 'yaml', 'json', 'auto')

    Returns:
        Loaded data
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Auto-detect mode
    if mode == 'auto':
        ext = input_path.suffix.lower()
        if ext == '.npy':
            mode = 'numpy'
        elif ext in {'.yaml', '.yml'}:
            mode = 'yaml'
        elif ext == '.json':
            mode = 'json'
        elif ext == '.txt':
            mode = 'list'

    # Load based on mode
    if mode == 'list':
        with open(input_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    elif mode == 'numpy':
        return np.load(input_path)

    elif mode == 'yaml':
        with open(input_path, 'r') as f:
            return yaml.safe_load(f)

    elif mode == 'json':
        with open(input_path, 'r') as f:
            return json.load(f)

    else:
        raise ValueError(f"Unknown load mode: {mode}")


def copy_selected_images(
    image_paths: List[str],
    output_dir: Union[str, Path],
    preserve_structure: bool = False
):
    """
    Copy selected images to output directory.

    Args:
        image_paths: List of image paths to copy
        output_dir: Output directory
        preserve_structure: Preserve original directory structure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying {len(image_paths)} images to {output_dir}...")

    for img_path in image_paths:
        src = Path(img_path)

        if preserve_structure:
            # Preserve directory structure
            dst = output_dir / src.name
        else:
            # Flat structure
            dst = output_dir / src.name

        # Handle name collisions
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = output_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(src, dst)

    logger.info(f"Copied {len(image_paths)} images")


def create_symlinks(
    image_paths: List[str],
    output_dir: Union[str, Path]
):
    """
    Create symlinks to selected images (faster than copying).

    Args:
        image_paths: List of image paths
        output_dir: Output directory for symlinks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating symlinks for {len(image_paths)} images...")

    for img_path in image_paths:
        src = Path(img_path).resolve()
        dst = output_dir / src.name

        # Handle name collisions
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = output_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        dst.symlink_to(src)

    logger.info(f"Created {len(image_paths)} symlinks")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(bytes_val: int) -> str:
    """
    Format bytes into human-readable string.

    Args:
        bytes_val: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def estimate_processing_time(
    n_images: int,
    batch_size: int = 64,
    time_per_batch: float = 0.5
) -> Dict[str, float]:
    """
    Estimate processing time for different pipeline stages.

    Args:
        n_images: Number of images
        batch_size: Batch size
        time_per_batch: Time per batch in seconds

    Returns:
        Dictionary with time estimates
    """
    n_batches = (n_images + batch_size - 1) // batch_size

    estimates = {
        'quality_filter': n_images * 0.002,  # ~2ms per image
        'feature_extraction': n_batches * time_per_batch,
        'pca_reduction': n_images * 0.0001,  # Fast
        'deduplication': n_images * 0.005,  # FAISS search
        'diversity_selection': n_images * 0.01,  # Depends on method
        'total': 0
    }

    estimates['total'] = sum(estimates.values())

    return estimates


def print_system_info():
    """Print system information for debugging."""
    import torch
    import platform

    logger.info("=" * 80)
    logger.info("System Information")
    logger.info("=" * 80)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA Available: Yes")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
    else:
        logger.info("CUDA Available: No")

    logger.info("=" * 80)
