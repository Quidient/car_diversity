#!/usr/bin/env python3
"""Main entry point for car diversity selection pipeline."""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from src.evaluation import create_all_visualizations
from src.pipeline import DiverseCarImageSelector
from src.utils import (
    copy_selected_images,
    create_symlinks,
    format_time,
    print_system_info,
)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level (e.g., 'INFO', 'DEBUG', 'WARNING').
        log_file: Optional path to log file. If None, logs only to console.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()), format=log_format, handlers=handlers
    )


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def _get_default_config() -> dict[str, Any]:
    """Get default configuration dictionary.

    Returns:
        Dictionary with default configuration values.
    """
    return {
        "input": {"image_dir": "./images"},
        "output": {
            "output_dir": "./output",
            "copy_images": False,
            "create_symlinks": True,
        },
        "hardware": {
            "device": "cuda",
            "vram_limit_gb": 32,
            "batch_size": 64,
            "use_fp16": True,
        },
        "quality_filter": {"enabled": True},
        "dimensionality_reduction": {"enabled": True, "n_components": 512},
        "deduplication": {"enabled": True, "threshold": 0.95},
        "diversity_selection": {"n_select": 30000, "method": "hybrid"},
        "evaluation": {
            "enabled": True,
            "create_visualization": True,
            "visualization_method": "tsne",
        },
        "logging": {"level": "INFO", "log_file": None},
    }


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Select diverse car images from large dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python main.py --image-dir ./images

  # Run with custom config
  python main.py --config my_config.yaml

  # Quick run with random selection (for testing)
  python main.py --image-dir ./images --method random --n-select 1000

  # Run without quality filtering (if images are pre-filtered)
  python main.py --image-dir ./images --skip-quality-filter
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--image-dir", type=str, help="Directory containing images (overrides config)"
    )

    parser.add_argument(
        "--output-dir", type=str, help="Output directory (overrides config)"
    )

    parser.add_argument(
        "--n-select", type=int, help="Number of images to select (overrides config)"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "kmeans", "fps", "hybrid"],
        help="Selection method (overrides config)",
    )

    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")

    parser.add_argument(
        "--skip-quality-filter",
        action="store_true",
        help="Skip quality filtering stage",
    )

    parser.add_argument(
        "--skip-deduplication", action="store_true", help="Skip deduplication stage"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use (overrides config)",
    )

    parser.add_argument(
        "--visualization-method",
        type=str,
        default="tsne",
        choices=["tsne", "umap", "pca"],
        help="Device to use (overrides config)",
    )
    return parser


def _merge_config_with_args(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Merge configuration with command-line arguments.

    Command-line arguments take precedence over config file values.

    Args:
        config: Base configuration dictionary.
        args: Parsed command-line arguments.

    Returns:
        Merged configuration dictionary.
    """
    if args.image_dir:
        config["input"]["image_dir"] = args.image_dir
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.n_select:
        config["diversity_selection"]["n_select"] = args.n_select
    if args.method:
        config["diversity_selection"]["method"] = args.method
    if args.batch_size:
        config["hardware"]["batch_size"] = args.batch_size
    if args.device:
        config["hardware"]["device"] = args.device
    if args.skip_quality_filter:
        config["quality_filter"]["enabled"] = False
    if args.skip_deduplication:
        config["deduplication"]["enabled"] = False
    if args.visualization_method:
        config["evaluation"]["visualization_method"] = args.visualization_method

    return config


def _run_visualization(
    output_dir: Path, config: dict[str, Any], logger: logging.Logger
) -> None:
    """Create comprehensive visualizations of selection results.

    Args:
        output_dir: Directory containing pipeline outputs.
        config: Configuration dictionary.
        logger: Logger instance.
    """
    logger.info("Creating visualizations...")
    try:
        import numpy as np

        embeddings = np.load(output_dir / "embeddings_reduced.npy")
        selected_idx = np.load(output_dir / "selected_indices.npy")

        # Validate embeddings shape
        if embeddings.ndim != 2:
            logger.error(
                f"Invalid embeddings shape: {embeddings.shape}. Expected 2D array."
            )
            return

        # Validate selected indices
        if len(selected_idx) > len(embeddings):
            logger.error(
                f"More selected indices ({len(selected_idx)}) than embeddings ({len(embeddings)})"
            )
            return

        if len(selected_idx) > 0 and np.max(selected_idx) >= len(embeddings):
            logger.error(
                f"Selected index {np.max(selected_idx)} out of bounds for embeddings with {len(embeddings)} samples"
            )
            return

        logger.info(
            f"Loaded embeddings: {embeddings.shape}, selected: {len(selected_idx)}"
        )

        # Load metrics
        metrics_file = output_dir / "evaluation_metrics.yaml"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = yaml.safe_load(f)
        else:
            metrics = {}

        # Create all visualizations
        viz_dir = output_dir / "visualizations"
        create_all_visualizations(embeddings, selected_idx, metrics, viz_dir, config)

        logger.info(f"Visualizations saved to: {viz_dir}")
    except Exception as e:
        logger.warning(f"Failed to create visualizations: {e}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = _create_argument_parser()
    args = parser.parse_args()

    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        config = _get_default_config()

    config = _merge_config_with_args(config, args)

    # Setup logging
    setup_logging(
        level=config["logging"].get("level", "INFO"),
        log_file=config["logging"].get("log_file"),
    )

    logger = logging.getLogger(__name__)

    # Print system info
    print_system_info()

    # Validate input directory
    image_dir = Path(config["input"]["image_dir"])
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("Car Diversity Selection Pipeline")
    logger.info("=" * 80)
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Output directory: {config['output']['output_dir']}")
    logger.info(
        f"Target selection: {config['diversity_selection']['n_select']:,} images"
    )
    logger.info(f"Selection method: {config['diversity_selection']['method']}")
    logger.info("=" * 80)

    # Initialize pipeline
    selector = DiverseCarImageSelector(
        vram_limit_gb=config["hardware"].get("vram_limit_gb", 32),
        batch_size=config["hardware"].get("batch_size", 64),
        device=config["hardware"].get("device", "cuda"),
        use_fp16=config["hardware"].get("use_fp16", True),
    )

    # Run pipeline
    start_time = time.time()

    try:
        selected_images = selector.run_pipeline(
            image_dir=str(image_dir),
            n_select=config["diversity_selection"]["n_select"],
            output_dir=config["output"]["output_dir"],
            skip_quality_filter=not config["quality_filter"].get("enabled", True),
            skip_deduplication=not config["deduplication"].get("enabled", True),
            pca_components=config["dimensionality_reduction"].get("n_components", 512),
            dedup_threshold=config["deduplication"].get("threshold", 0.95),
            selection_method=config["diversity_selection"]["method"],
        )

        elapsed_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"Total time: {format_time(elapsed_time)}")
        logger.info(f"Selected images: {len(selected_images):,}")
        logger.info(f"Results saved to: {config['output']['output_dir']}")

        # Copy or symlink selected images
        output_dir = Path(config["output"]["output_dir"])

        if config["output"].get("copy_images", False):
            logger.info("Copying selected images...")
            copy_selected_images(selected_images, output_dir / "selected_images")

        if config["output"].get("create_symlinks", True):
            logger.info("Creating symlinks to selected images...")
            create_symlinks(selected_images, output_dir / "selected_images_symlinks")

        if config["evaluation"].get("create_visualization", True):
            _run_visualization(output_dir, config, logger)

        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
