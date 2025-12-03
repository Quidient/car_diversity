#!/usr/bin/env python3
"""Main entry point for car diversity selection pipeline."""

import argparse
import yaml
import logging
import time
from pathlib import Path

from src.pipeline import DiverseCarImageSelector
from src.utils import print_system_info, format_time, copy_selected_images, create_symlinks
from src.evaluation import visualize_selection


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Select diverse car images from large dataset',
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
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images (overrides config)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (overrides config)'
    )

    parser.add_argument(
        '--n-select',
        type=int,
        help='Number of images to select (overrides config)'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['random', 'kmeans', 'fps', 'hybrid'],
        help='Selection method (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--skip-quality-filter',
        action='store_true',
        help='Skip quality filtering stage'
    )

    parser.add_argument(
        '--skip-deduplication',
        action='store_true',
        help='Skip deduplication stage'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (overrides config)'
    )

    args = parser.parse_args()

    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        config = {
            'input': {'image_dir': './images'},
            'output': {'output_dir': './output', 'copy_images': False, 'create_symlinks': True},
            'hardware': {'device': 'cuda', 'vram_limit_gb': 32, 'batch_size': 64, 'use_fp16': True},
            'quality_filter': {'enabled': True},
            'dimensionality_reduction': {'enabled': True, 'n_components': 512},
            'deduplication': {'enabled': True, 'threshold': 0.95},
            'diversity_selection': {'n_select': 30000, 'method': 'hybrid'},
            'evaluation': {'enabled': True, 'create_visualization': True, 'visualization_method': 'tsne'},
            'logging': {'level': 'INFO', 'log_file': None}
        }

    # Override config with command-line arguments
    if args.image_dir:
        config['input']['image_dir'] = args.image_dir
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.n_select:
        config['diversity_selection']['n_select'] = args.n_select
    if args.method:
        config['diversity_selection']['method'] = args.method
    if args.batch_size:
        config['hardware']['batch_size'] = args.batch_size
    if args.device:
        config['hardware']['device'] = args.device
    if args.skip_quality_filter:
        config['quality_filter']['enabled'] = False
    if args.skip_deduplication:
        config['deduplication']['enabled'] = False

    # Setup logging
    setup_logging(
        level=config['logging'].get('level', 'INFO'),
        log_file=config['logging'].get('log_file')
    )

    logger = logging.getLogger(__name__)

    # Print system info
    print_system_info()

    # Validate input directory
    image_dir = Path(config['input']['image_dir'])
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("Car Diversity Selection Pipeline")
    logger.info("=" * 80)
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Output directory: {config['output']['output_dir']}")
    logger.info(f"Target selection: {config['diversity_selection']['n_select']:,} images")
    logger.info(f"Selection method: {config['diversity_selection']['method']}")
    logger.info("=" * 80)

    # Initialize pipeline
    selector = DiverseCarImageSelector(
        vram_limit_gb=config['hardware'].get('vram_limit_gb', 32),
        batch_size=config['hardware'].get('batch_size', 64),
        device=config['hardware'].get('device', 'cuda'),
        use_fp16=config['hardware'].get('use_fp16', True)
    )

    # Run pipeline
    start_time = time.time()

    try:
        selected_images = selector.run_pipeline(
            image_dir=str(image_dir),
            n_select=config['diversity_selection']['n_select'],
            output_dir=config['output']['output_dir'],
            skip_quality_filter=not config['quality_filter'].get('enabled', True),
            skip_deduplication=not config['deduplication'].get('enabled', True),
            pca_components=config['dimensionality_reduction'].get('n_components', 512),
            dedup_threshold=config['deduplication'].get('threshold', 0.95),
            selection_method=config['diversity_selection']['method']
        )

        elapsed_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"Total time: {format_time(elapsed_time)}")
        logger.info(f"Selected images: {len(selected_images):,}")
        logger.info(f"Results saved to: {config['output']['output_dir']}")

        # Copy or symlink selected images
        output_dir = Path(config['output']['output_dir'])

        if config['output'].get('copy_images', False):
            logger.info("Copying selected images...")
            copy_selected_images(selected_images, output_dir / "selected_images")

        if config['output'].get('create_symlinks', True):
            logger.info("Creating symlinks to selected images...")
            create_symlinks(selected_images, output_dir / "selected_images_symlinks")

        # Create visualization
        if config['evaluation'].get('create_visualization', True):
            logger.info("Creating visualization...")
            try:
                import numpy as np
                from src.utils import load_results

                # Load embeddings and indices
                embeddings = np.load(output_dir / "embeddings_reduced.npy")
                selected_idx = np.load(output_dir / "selected_indices.npy")

                visualize_selection(
                    embeddings,
                    selected_idx,
                    output_dir / "selection_visualization.png",
                    method=config['evaluation'].get('visualization_method', 'tsne')
                )
            except Exception as e:
                logger.warning(f"Failed to create visualization: {e}")

        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
