"""Main pipeline implementation for diverse car image selection."""

import logging
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm

from src.diversity_selector import DiversitySelector
from src.evaluation import evaluate_selection
from src.feature_extractor import FeatureExtractor
from src.quality_filter import QualityFilter
from src.utils import load_image_paths, save_results

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_DEDUP_THRESHOLD = 0.95
DEDUP_K_NEIGHBORS = 10


class DiverseCarImageSelector:
    """Complete pipeline for selecting diverse car images from a large dataset.

    This class orchestrates the entire pipeline from quality filtering through
    diversity selection, providing a high-level interface for image curation.

    Attributes:
        vram_limit: VRAM limit in GB.
        batch_size: Batch size for feature extraction.
        device: Device to use ('cuda' or 'cpu').
        use_fp16: Whether to use FP16 precision for faster processing.
        quality_filter: QualityFilter instance (lazy-initialized).
        feature_extractor: FeatureExtractor instance (lazy-initialized).
        diversity_selector: DiversitySelector instance (lazy-initialized).
    """

    def __init__(
        self,
        vram_limit_gb: int = 24,
        batch_size: int = 64,
        device: str = "cuda",
        use_fp16: bool = True,
    ) -> None:
        """Initialize the diverse car image selector.

        Args:
            vram_limit_gb: VRAM limit in GB (e.g., 32 for RTX 5090).
            batch_size: Batch size for feature extraction.
            device: Device to use ('cuda' or 'cpu').
            use_fp16: Whether to use FP16 for faster processing.
        """
        self.vram_limit = vram_limit_gb
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.use_fp16 = False

        if self.device == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        self.quality_filter: QualityFilter | None = None
        self.feature_extractor: FeatureExtractor | None = None
        self.diversity_selector: DiversitySelector | None = None

    def run_pipeline(
        self,
        image_dir: str,
        n_select: int = 30000,
        output_dir: str = "./output",
        skip_quality_filter: bool = False,
        skip_deduplication: bool = False,
        pca_components: int = 512,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        selection_method: str = "hybrid",
    ) -> list[str]:
        """Run complete pipeline: filter → embed → select.

        Args:
            image_dir: Directory containing images to process.
            n_select: Number of images to select.
            output_dir: Directory to save results.
            skip_quality_filter: If True, skip quality filtering (use if pre-filtered).
            skip_deduplication: If True, skip deduplication step.
            pca_components: Number of PCA components for dimensionality reduction.
            dedup_threshold: Cosine similarity threshold for deduplication (0-1).
            selection_method: Selection method ('hybrid', 'fps', 'kmeans', 'random').

        Returns:
            List of selected image paths.

        Raises:
            FileNotFoundError: If image_dir does not exist.
            ValueError: If invalid selection_method is specified.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Phase 1: Quality Filtering
        if skip_quality_filter:
            logger.info("Phase 1: Skipping quality filtering (loading all images)...")
            valid_images = load_image_paths(image_dir)
        else:
            logger.info("Phase 1: Quality filtering...")
            self.quality_filter = QualityFilter(
                device=self.device, batch_size=self.batch_size
            )
            valid_images = self.quality_filter.filter_dataset(image_dir)

            # Save filtered image list
            save_results(valid_images, output_path / "filtered_images.txt", mode="list")

        logger.info(f"Valid images after filtering: {len(valid_images):,}")

        if len(valid_images) < n_select:
            logger.warning(
                f"Only {len(valid_images)} valid images found, "
                f"but {n_select} requested. Adjusting to {len(valid_images)}."
            )
            n_select = len(valid_images)

        # Phase 2: Feature Extraction
        logger.info("Phase 2: Extracting DINOv2 features...")
        self.feature_extractor = FeatureExtractor(
            model_name="dinov2-large",
            device=self.device,
            batch_size=self.batch_size,
            use_fp16=self.use_fp16,
        )
        embeddings = self.feature_extractor.extract_features(valid_images)

        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Save raw embeddings
        np.save(output_path / "embeddings_raw.npy", embeddings)

        # Phase 3: Dimensionality Reduction
        logger.info("Phase 3: PCA dimensionality reduction...")
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_components)
        reduced = pca.fit_transform(embeddings)

        variance_retained = sum(pca.explained_variance_ratio_)
        logger.info(
            f"Reduced shape: {reduced.shape}, "
            f"variance retained: {variance_retained:.2%}"
        )

        # Save reduced embeddings and PCA model
        embeddings_file = output_path / "embeddings_reduced.npy"
        np.save(embeddings_file, reduced)

        # Verify the file was saved correctly
        try:
            loaded_check = np.load(embeddings_file)
            if loaded_check.shape != reduced.shape:
                logger.error(
                    f"Saved embeddings shape mismatch! "
                    f"Expected {reduced.shape}, got {loaded_check.shape}"
                )
                raise RuntimeError("Failed to save embeddings correctly")
            logger.debug(f"Verified embeddings saved correctly: {loaded_check.shape}")
        except Exception as e:
            logger.error(f"Failed to verify saved embeddings: {e}")
            raise

        import joblib

        joblib.dump(pca, output_path / "pca_model.pkl")

        # Phase 4: Deduplication
        if skip_deduplication:
            logger.info("Phase 4: Skipping deduplication...")
            deduplicated_embeddings = reduced
            deduplicated_images = valid_images
        else:
            logger.info("Phase 4: Removing near-duplicates...")
            keep_mask = self._deduplicate(reduced, threshold=dedup_threshold)
            deduplicated_embeddings = reduced[keep_mask]
            deduplicated_images = [
                img for img, keep in zip(valid_images, keep_mask) if keep
            ]

            logger.info(
                f"After deduplication: {len(deduplicated_images):,} "
                f"({len(valid_images) - len(deduplicated_images):,} removed)"
            )

            # Save deduplicated image list
            save_results(
                deduplicated_images,
                output_path / "deduplicated_images.txt",
                mode="list",
            )

        # Phase 5: Diverse Selection
        logger.info(
            f"Phase 5: Diversity-based selection ({selection_method} method)..."
        )
        self.diversity_selector = DiversitySelector(device=self.device)

        selected_idx = self.diversity_selector.select_diverse(
            deduplicated_embeddings, n_select=n_select, method=selection_method
        )

        selected_images = [deduplicated_images[i] for i in selected_idx]

        logger.info(f"Selected {len(selected_images):,} diverse images")

        # Save results
        save_results(selected_images, output_path / "selected_images.txt", mode="list")

        save_results(selected_idx, output_path / "selected_indices.npy", mode="numpy")

        # Phase 6: Evaluation
        logger.info("Phase 6: Evaluating selection quality...")
        metrics = evaluate_selection(
            deduplicated_embeddings, selected_idx,
        )

        save_results(metrics, output_path / "evaluation_metrics.yaml", mode="yaml")

        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Selected images: {len(selected_images):,}")
        logger.info("=" * 80)

        return selected_images

    def _deduplicate(
        self, embeddings: np.ndarray, threshold: float = DEFAULT_DEDUP_THRESHOLD
    ) -> np.ndarray:
        """Remove near-duplicate images based on cosine similarity.

        Uses FAISS to efficiently find near-duplicates. For each pair of images
        with similarity > threshold, keeps the first occurrence and removes
        subsequent duplicates.

        Args:
            embeddings: Feature embeddings of shape (N, D).
            threshold: Cosine similarity threshold (0-1). Higher values mean
                more strict deduplication (only very similar images removed).

        Returns:
            Boolean mask of shape (N,) indicating which images to keep.
        """
        embeddings_faiss = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings_faiss)

        index = faiss.IndexFlatIP(embeddings_faiss.shape[1])
        index.add(embeddings_faiss)

        similarities, indices = index.search(embeddings_faiss, k=DEDUP_K_NEIGHBORS)

        keep_mask = np.ones(len(embeddings_faiss), dtype=bool)

        logger.info("Identifying duplicates...")
        for i in tqdm(range(len(embeddings_faiss)), desc="Deduplication"):
            if not keep_mask[i]:
                continue

            for similarity, j in zip(similarities[i], indices[i]):
                if j > i and similarity > threshold:
                    keep_mask[j] = False

        return keep_mask
