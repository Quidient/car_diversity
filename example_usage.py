#!/usr/bin/env python3
"""Example usage of the car diversity selection pipeline."""

import logging
from pathlib import Path
from src.pipeline import DiverseCarImageSelector
from src.feature_extractor import FeatureExtractor, CLIPFeatureExtractor, combine_features
from src.diversity_selector import DiversitySelector
from src.evaluation import evaluate_selection, compare_selection_methods, visualize_selection
from src.utils import save_results, load_results, create_symlinks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_pipeline():
    """Example 1: Basic pipeline usage."""
    logger.info("Example 1: Basic Pipeline")

    selector = DiverseCarImageSelector(
        vram_limit_gb=32,
        batch_size=128,  # RTX 5090 can handle larger batches
        device="cuda",
        use_fp16=True
    )

    selected_images = selector.run_pipeline(
        image_dir="./images",
        n_select=30000,
        output_dir="./output/example1",
        selection_method="hybrid"
    )

    logger.info(f"Selected {len(selected_images)} images")


def example_2_custom_feature_extraction():
    """Example 2: Custom feature extraction with multiple models."""
    logger.info("Example 2: Custom Feature Extraction")

    image_paths = ["./images/car1.jpg", "./images/car2.jpg"]  # Your image paths

    # Extract DINOv2 features
    dinov2_extractor = FeatureExtractor(
        model_name='dinov2-large',
        multilayer=True,
        layers=[4, 8, 12]
    )
    dinov2_features = dinov2_extractor.extract_features(image_paths)

    # Extract CLIP features
    clip_extractor = CLIPFeatureExtractor()
    clip_features = clip_extractor.extract_features(image_paths)

    # Combine features
    combined_features = combine_features(dinov2_features, clip_features, pca_components=1024)

    logger.info(f"Combined features shape: {combined_features.shape}")


def example_3_compare_methods():
    """Example 3: Compare different selection methods."""
    logger.info("Example 3: Compare Selection Methods")

    # Assume we have embeddings from feature extraction
    import numpy as np
    embeddings = np.load("./output/embeddings_reduced.npy")

    selector = DiversitySelector(device="cuda")

    # Test different methods
    selections = {
        'random': selector.select_diverse(embeddings, 5000, method='random'),
        'kmeans': selector.select_diverse(embeddings, 5000, method='kmeans'),
        'fps': selector.select_diverse(embeddings, 5000, method='fps'),
        'hybrid': selector.select_diverse(embeddings, 5000, method='hybrid'),
    }

    # Compare
    results = compare_selection_methods(embeddings, selections)

    # Save comparison
    save_results(results, "./output/method_comparison.yaml", mode='yaml')


def example_4_incremental_selection():
    """Example 4: Incremental selection (select more from existing pool)."""
    logger.info("Example 4: Incremental Selection")

    # Load previous results
    embeddings = np.load("./output/embeddings_reduced.npy")
    initial_selection = np.load("./output/selected_indices.npy")

    # Select additional diverse images
    selector = DiversitySelector(device="cuda")

    # Create a mask for unselected images
    import numpy as np
    all_indices = np.arange(len(embeddings))
    unselected_mask = ~np.isin(all_indices, initial_selection)
    unselected_embeddings = embeddings[unselected_mask]
    unselected_indices = all_indices[unselected_mask]

    # Select 5000 more diverse images from remaining pool
    additional_local = selector.select_diverse(unselected_embeddings, 5000, method='fps')
    additional_global = unselected_indices[additional_local]

    # Combine with initial selection
    final_selection = np.concatenate([initial_selection, additional_global])

    logger.info(f"Total selected: {len(final_selection)} images")


def example_5_quality_filter_only():
    """Example 5: Run only quality filtering (no selection)."""
    logger.info("Example 5: Quality Filter Only")

    from src.quality_filter import QualityFilter

    qf = QualityFilter(
        device="cuda",
        min_resolution=256,
        max_aspect_ratio=3.0,
        brisque_threshold=45.0,
        blur_threshold=100.0,
        car_confidence_threshold=0.6
    )

    valid_images = qf.filter_dataset("./images")

    logger.info(f"Filtered to {len(valid_images)} valid images")
    save_results(valid_images, "./output/quality_filtered.txt", mode='list')


def example_6_evaluation_and_visualization():
    """Example 6: Evaluate existing selection and create visualizations."""
    logger.info("Example 6: Evaluation and Visualization")

    import numpy as np

    # Load data
    embeddings = np.load("./output/embeddings_reduced.npy")
    selected_indices = np.load("./output/selected_indices.npy")

    # Evaluate
    metrics = evaluate_selection(embeddings, selected_indices)

    logger.info("Metrics:")
    logger.info(f"  Coverage radius: {metrics['coverage_radius_max']:.4f}")
    logger.info(f"  Intra-diversity: {metrics['intra_diversity_mean']:.4f}")

    # Create visualizations with different methods
    for method in ['pca', 'tsne']:
        visualize_selection(
            embeddings,
            selected_indices,
            f"./output/visualization_{method}.png",
            method=method
        )


def example_7_batch_processing():
    """Example 7: Process multiple datasets in batch."""
    logger.info("Example 7: Batch Processing")

    datasets = [
        ("./images/dataset1", "./output/dataset1", 10000),
        ("./images/dataset2", "./output/dataset2", 20000),
        ("./images/dataset3", "./output/dataset3", 15000),
    ]

    selector = DiverseCarImageSelector(
        vram_limit_gb=32,
        batch_size=128,
        device="cuda"
    )

    for image_dir, output_dir, n_select in datasets:
        logger.info(f"Processing {image_dir}...")

        selected_images = selector.run_pipeline(
            image_dir=image_dir,
            n_select=n_select,
            output_dir=output_dir,
            selection_method="hybrid"
        )

        logger.info(f"Completed {image_dir}: {len(selected_images)} images selected")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run example scripts")
    parser.add_argument(
        '--example',
        type=int,
        choices=range(1, 8),
        required=True,
        help='Example number to run (1-7)'
    )

    args = parser.parse_args()

    examples = {
        1: example_1_basic_pipeline,
        2: example_2_custom_feature_extraction,
        3: example_3_compare_methods,
        4: example_4_incremental_selection,
        5: example_5_quality_filter_only,
        6: example_6_evaluation_and_visualization,
        7: example_7_batch_processing,
    }

    examples[args.example]()
