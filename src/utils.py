"""Utility functions for the pipeline."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
)


def load_image_paths(image_dir: str, filter_list: str = None) -> list[str]:
    """Load all image paths from a directory recursively.

    Args:
        image_dir: Directory containing images.

    Returns:
        List of absolute image file paths.

    Raises:
        FileNotFoundError: If image_dir does not exist.
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = []
    if (filter_list is None):
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.extend(image_dir_path.rglob(f"*{ext}"))
            image_paths.extend(image_dir_path.rglob(f"*{ext.upper()}"))
    else:
        with open(filter_list, "r") as filter:
            image_paths = [line.rstrip() for line in filter]

    return [str(p) for p in image_paths]

def save_results(
    data: list[Any] | np.ndarray | dict[str, Any],
    output_path: str | Path,
    mode: str = "auto",
) -> None:
    """Save results to file in various formats.

    Args:
        data: Data to save (list, numpy array, or dictionary).
        output_path: Output file path.
        mode: Save mode. Options are 'list' (text file), 'numpy' (.npy),
            'yaml', 'json', or 'auto' (infer from extension).

    Raises:
        ValueError: If mode is unknown or cannot be auto-detected.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "auto":
        ext = output_path.suffix.lower()
        mode_map = {
            ".npy": "numpy",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".txt": "list",
        }
        mode = mode_map.get(ext)
        if mode is None:
            raise ValueError(f"Cannot auto-detect mode for extension: {ext}")

    if mode == "list":
        with open(output_path, "w") as f:
            for item in data:
                f.write(f"{item}\n")
    elif mode == "numpy":
        np.save(output_path, data)
    elif mode == "yaml":
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    elif mode == "json":
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unknown save mode: {mode}")

    logger.info(f"Saved results to: {output_path}")


def load_results(
    input_path: str | Path, mode: str = "auto"
) -> list[Any] | np.ndarray | dict[str, Any]:
    """Load results from file in various formats.

    Args:
        input_path: Input file path.
        mode: Load mode. Options are 'list', 'numpy', 'yaml', 'json',
            or 'auto' (infer from extension).

    Returns:
        Loaded data (list, numpy array, or dictionary).

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If mode is unknown or cannot be auto-detected.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if mode == "auto":
        ext = input_path.suffix.lower()
        mode_map = {
            ".npy": "numpy",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".txt": "list",
        }
        mode = mode_map.get(ext)
        if mode is None:
            raise ValueError(f"Cannot auto-detect mode for extension: {ext}")

    if mode == "list":
        with open(input_path) as f:
            return [line.strip() for line in f if line.strip()]
    elif mode == "numpy":
        return np.load(input_path)
    elif mode == "yaml":
        with open(input_path) as f:
            return yaml.safe_load(f)
    elif mode == "json":
        with open(input_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown load mode: {mode}")


def copy_selected_images(
    image_paths: list[str],
    output_dir: str | Path,
    preserve_structure: bool = False,
) -> None:
    """Copy selected images to output directory.

    Args:
        image_paths: List of image paths to copy.
        output_dir: Output directory.
        preserve_structure: If True, preserve original directory structure.
            Currently not implemented; all files are copied to a flat structure.

    Note:
        Files with duplicate names are automatically renamed with a numeric suffix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying {len(image_paths)} images to {output_dir}...")

    for img_path in image_paths:
        src = Path(img_path)
        dst = output_dir / src.name

        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = output_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(src, dst)

    logger.info(f"Copied {len(image_paths)} images")


def create_symlinks(image_paths: list[str], output_dir: str | Path) -> None:
    """Create symlinks to selected images (faster than copying).

    Args:
        image_paths: List of image paths.
        output_dir: Output directory for symlinks.

    Note:
        Symlinks with duplicate names are automatically renamed with a numeric suffix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating symlinks for {len(image_paths)} images...")

    for img_path in image_paths:
        src = Path(img_path).resolve()
        dst = output_dir / src.name

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
    """Format seconds into human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string (e.g., '42.5s', '3.2m', '1.50h').
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
    """Format bytes into human-readable string.

    Args:
        bytes_val: Size in bytes.

    Returns:
        Formatted size string (e.g., '1.50 MB', '2.34 GB').
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def estimate_processing_time(
    n_images: int, batch_size: int = 64, time_per_batch: float = 0.5
) -> dict[str, float]:
    """Estimate processing time for different pipeline stages.

    Args:
        n_images: Number of images to process.
        batch_size: Batch size for processing.
        time_per_batch: Time per batch in seconds.

    Returns:
        Dictionary mapping stage names to estimated time in seconds.
        Includes a 'total' key with sum of all estimates.
    """
    n_batches = (n_images + batch_size - 1) // batch_size

    estimates = {
        "quality_filter": n_images * 0.002,
        "feature_extraction": n_batches * time_per_batch,
        "pca_reduction": n_images * 0.0001,
        "deduplication": n_images * 0.005,
        "diversity_selection": n_images * 0.01,
        "total": 0,
    }

    estimates["total"] = sum(v for k, v in estimates.items() if k != "total")

    return estimates


def print_system_info() -> None:
    """Print system information for debugging."""
    import platform

    import torch

    logger.info("=" * 80)
    logger.info("System Information")
    logger.info("=" * 80)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info("CUDA Available: Yes")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
    else:
        logger.info("CUDA Available: No")

    logger.info("=" * 80)
