"""Car diversity selection package."""

from .pipeline import DiverseCarImageSelector
from .quality_filter import QualityFilter
from .feature_extractor import FeatureExtractor, CLIPFeatureExtractor
from .diversity_selector import DiversitySelector
from .evaluation import evaluate_selection, compare_selection_methods
from .utils import load_image_paths, save_results, load_results

__version__ = "0.1.0"

__all__ = [
    "DiverseCarImageSelector",
    "QualityFilter",
    "FeatureExtractor",
    "CLIPFeatureExtractor",
    "DiversitySelector",
    "evaluate_selection",
    "compare_selection_methods",
    "load_image_paths",
    "save_results",
    "load_results",
]
