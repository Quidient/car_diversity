"""Feature extraction using DINOv2 and optional CLIP."""

import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64
DEFAULT_PREFETCH_FACTOR = 2
MAX_WORKERS = 16


class ImageDataset(Dataset):
    """Dataset for parallel image loading and preprocessing.

    Performs image loading and preprocessing on worker threads for efficiency.

    Attributes:
        image_paths: List of image file paths.
        processor: Image processor for model input preparation.
    """

    def __init__(self, image_paths: list[str], processor: Any) -> None:
        """Initialize image dataset.

        Args:
            image_paths: List of image file paths.
            processor: Image processor (e.g., DINOv2 or CLIP processor).
        """
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        """Load and preprocess image.

        Args:
            idx: Image index.

        Returns:
            Preprocessed image tensor if successful, None otherwise.
        """
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return pixel_values
        except Exception as e:
            logger.debug(f"Error loading {path}: {e}")
            return None


def collate_fn(batch: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Filter out failed images and stack tensors.

    Args:
        batch: List of image tensors or None values.

    Returns:
        Stacked tensor batch, or None if all images failed.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.stack(batch)


class FeatureExtractor:
    """Extract deep features from images using DINOv2.

    Supports single-layer and multi-layer feature extraction with optional
    FP16 precision for faster inference.

    Attributes:
        device: Device to use ('cuda' or 'cpu').
        batch_size: Batch size for feature extraction.
        use_fp16: Whether to use FP16 precision.
        multilayer: Whether to extract features from multiple layers.
        layers: Layer indices for multi-layer extraction.
        model_name: Full model name (e.g., 'facebook/dinov2-large').
        model: Loaded model instance.
        processor: Image processor instance.
    """

    def __init__(
        self,
        model_name: str = "dinov2-large",
        device: str = "cuda",
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_fp16: bool = True,
        multilayer: bool = False,
        layers: list[int] | None = None,
    ) -> None:
        """Initialize feature extractor.

        Args:
            model_name: Model name shorthand or full HuggingFace identifier.
            device: Device to use ('cuda' or 'cpu').
            batch_size: Batch size for feature extraction.
            use_fp16: Whether to use FP16 precision (CUDA only).
            multilayer: If True, extract features from multiple layers.
            layers: Layer indices for multi-layer extraction (default: [4, 8, 12]).
        """
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and device == "cuda"
        self.multilayer = multilayer
        self.layers = layers if layers else [4, 8, 12]

        model_mapping = {
            "dinov2-small": "facebook/dinov2-small",
            "dinov2-base": "facebook/dinov2-base",
            "dinov2-large": "facebook/dinov2-large",
            "dinov2-giant": "facebook/dinov2-giant",
        }

        self.model_name = model_mapping.get(model_name, model_name)
        self._load_model()

    def _load_model(self) -> None:
        """Load DINOv2 model and processor.

        Raises:
            Exception: If model loading fails.
        """
        from transformers import AutoImageProcessor, AutoModel

        logger.info(f"Loading {self.model_name}...")

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name, use_fast=True
            )
            self.model = AutoModel.from_pretrained(self.model_name)

            self.model = self.model.to(self.device)
            self.model.eval()

            if self.use_fp16:
                self.model = self.model.half()
                logger.info("Using FP16 precision")

            logger.info(f"Model loaded successfully on {self.device}")

            if self.device == "cuda":
                param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
                logger.info(f"Model parameters: {param_count:.1f}M")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def extract_features(
        self, image_paths: list[str], show_progress: bool = True
    ) -> np.ndarray:
        """Extract features using optimized DataLoader.

        Args:
            image_paths: List of image file paths.
            show_progress: Whether to show progress bar.

        Returns:
            Feature array of shape (N, D) where N is number of images and
            D is feature dimension.

        Raises:
            ValueError: If no features were extracted from any images.
        """
        all_features = []

        dataset = ImageDataset(image_paths, self.processor)
        workers = self._get_optimal_workers()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        )

        iterator = tqdm(
            loader,
            desc="Extracting features",
            disable=not show_progress,
            total=len(loader),
        )

        with torch.no_grad():
            for batch_pixel_values in iterator:
                if batch_pixel_values is None:
                    continue

                pixel_values = batch_pixel_values.to(self.device, non_blocking=True)

                if self.use_fp16:
                    pixel_values = pixel_values.half()

                try:
                    features = self._forward_pass(pixel_values)
                    all_features.append(features)
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    continue

        if not all_features:
            raise ValueError("No features extracted from any images")

        all_features = np.vstack(all_features)
        return all_features

    def _forward_pass(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Run model inference on GPU.

        Args:
            pixel_values: Batch of preprocessed images.

        Returns:
            Feature array for the batch.
        """
        if self.use_fp16:
            with torch.autocast("cuda"):
                outputs = self.model(
                    pixel_values=pixel_values, output_hidden_states=self.multilayer
                )
        else:
            outputs = self.model(
                pixel_values=pixel_values, output_hidden_states=self.multilayer
            )

        if self.multilayer:
            features = torch.cat(
                [outputs.hidden_states[layer][:, 0, :] for layer in self.layers], dim=-1
            )
        else:
            features = outputs.last_hidden_state[:, 0, :]

        return features.cpu().float().numpy()

    def _get_optimal_workers(self) -> int:
        """Determine optimal number of DataLoader workers.

        Returns:
            Optimal number of workers (1-16).
        """
        try:
            available = len(os.sched_getaffinity(0))
        except AttributeError:
            available = os.cpu_count() or 1

        return max(1, min(available - 2, MAX_WORKERS))


class CLIPFeatureExtractor:
    """Extract CLIP features for semantic diversity.

    Attributes:
        device: Device to use ('cuda' or 'cpu').
        batch_size: Batch size for processing.
        model_name: CLIP model name.
        model: Loaded CLIP model.
        processor: CLIP processor.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 128,
        model_name: str = "openai/clip-vit-large-patch14",
    ) -> None:
        """Initialize CLIP feature extractor.

        Args:
            device: Device to use ('cuda' or 'cpu').
            batch_size: Batch size for processing.
            model_name: CLIP model name from HuggingFace.
        """
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        self._load_model()

    def _load_model(self) -> None:
        """Load CLIP model and processor.

        Raises:
            Exception: If model loading fails.
        """
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading {self.model_name}...")

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("CLIP model loaded successfully")

    def extract_features(
        self, image_paths: list[str], show_progress: bool = True
    ) -> np.ndarray:
        """Extract CLIP image features.

        Args:
            image_paths: List of image file paths.
            show_progress: Whether to show progress bar.

        Returns:
            Feature array of shape (N, D) where N is number of images.
        """
        all_features = []

        iterator = tqdm(
            range(0, len(image_paths), self.batch_size),
            desc="Extracting CLIP features",
            disable=not show_progress,
        )

        for i in iterator:
            batch_paths = image_paths[i : i + self.batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    logger.debug(f"Error loading {img_path}: {e}")
                    continue

            if not batch_images:
                continue

            try:
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)

                features = features.cpu().numpy()
                all_features.append(features)

            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
                continue

        all_features = np.vstack(all_features)
        logger.info(f"CLIP features shape: {all_features.shape}")

        return all_features


def combine_features(
    dinov2_features: np.ndarray, clip_features: np.ndarray, pca_components: int = 1024
) -> np.ndarray:
    """Combine DINOv2 and CLIP features with PCA reduction.

    Args:
        dinov2_features: DINOv2 embeddings of shape (N, D1).
        clip_features: CLIP embeddings of shape (N, D2).
        pca_components: Number of PCA components for final dimensionality.

    Returns:
        Combined and reduced features of shape (N, pca_components).
    """
    from sklearn.decomposition import PCA

    logger.info("Combining DINOv2 and CLIP features...")

    # Concatenate features
    combined = np.concatenate([dinov2_features, clip_features], axis=1)
    logger.info(f"Combined shape: {combined.shape}")

    # PCA reduction
    pca = PCA(n_components=pca_components)
    reduced = pca.fit_transform(combined)

    variance_retained = sum(pca.explained_variance_ratio_)
    logger.info(
        f"Reduced to {pca_components} dims, variance retained: {variance_retained:.2%}"
    )

    return reduced
