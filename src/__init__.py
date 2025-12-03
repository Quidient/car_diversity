"""Car diversity selection package.

This package provides tools for selecting diverse subsets of images from large
datasets using deep feature extraction and submodular diversity optimization.

Modules:
    pipeline: Main pipeline orchestration.
    quality_filter: Multi-stage quality filtering.
    feature_extractor: Deep feature extraction using DINOv2 and CLIP.
    diversity_selector: Diversity selection algorithms.
    evaluation: Evaluation metrics and visualization.
    utils: Utility functions for file I/O and formatting.
"""

from .diversity_selector import DiversitySelector
from .evaluation import (
    compare_selection_methods,
    create_all_visualizations,
    evaluate_selection,
    visualize_coverage_distribution,
    visualize_diversity_matrix,
    visualize_selection,
    visualize_selection_metrics_dashboard,
)
from .feature_extractor import CLIPFeatureExtractor, FeatureExtractor
from .pipeline import DiverseCarImageSelector
from .quality_filter import QualityFilter
from .utils import load_image_paths, load_results, save_results

__version__ = "0.1.0"

__all__ = [
    "CLIPFeatureExtractor",
    "DiverseCarImageSelector",
    "DiversitySelector",
    "FeatureExtractor",
    "QualityFilter",
    "compare_selection_methods",
    "create_all_visualizations",
    "evaluate_selection",
    "load_image_paths",
    "load_results",
    "save_results",
    "visualize_coverage_distribution",
    "visualize_diversity_matrix",
    "visualize_selection",
    "visualize_selection_metrics_dashboard",
]
