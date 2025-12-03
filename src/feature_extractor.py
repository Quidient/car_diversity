"""Feature extraction using DINOv2 and optional CLIP."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract deep features from images using DINOv2."""

    def __init__(
        self,
        model_name: str = 'dinov2-large',
        device: str = "cuda",
        batch_size: int = 64,
        use_fp16: bool = True,
        multilayer: bool = False,
        layers: Optional[List[int]] = None
    ):
        """
        Initialize feature extractor.

        Args:
            model_name: Model to use ('dinov2-small', 'dinov2-base', 'dinov2-large', 'dinov2-giant')
            device: Device to use ('cuda' or 'cpu')
            batch_size: Batch size for processing
            use_fp16: Use FP16 for faster processing
            multilayer: Extract features from multiple layers
            layers: Layer indices to extract from (default: [4, 8, 12])
        """
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and device == "cuda"
        self.multilayer = multilayer
        self.layers = layers if layers else [4, 8, 12]

        # Model name mapping
        model_mapping = {
            'dinov2-small': 'facebook/dinov2-small',
            'dinov2-base': 'facebook/dinov2-base',
            'dinov2-large': 'facebook/dinov2-large',
            'dinov2-giant': 'facebook/dinov2-giant'
        }

        self.model_name = model_mapping.get(model_name, model_name)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load DINOv2 model and processor."""
        from transformers import AutoImageProcessor, AutoModel

        logger.info(f"Loading {self.model_name}...")

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            self.model = self.model.to(self.device)
            self.model.eval()

            if self.use_fp16:
                self.model = self.model.half()
                logger.info("Using FP16 precision")

            logger.info(f"Model loaded successfully on {self.device}")

            # Print model info
            if self.device == "cuda":
                param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
                logger.info(f"Model parameters: {param_count:.1f}M")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def extract_features(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract features from a list of images.

        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar

        Returns:
            Feature embeddings as numpy array [N, D]
        """
        all_features = []

        iterator = tqdm(
            range(0, len(image_paths), self.batch_size),
            desc="Extracting features",
            disable=not show_progress
        )

        for i in iterator:
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []

            # Load batch images
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                except Exception as e:
                    logger.debug(f"Error loading {img_path}: {e}")
                    # Add a placeholder (will be filtered later)
                    continue

            if not batch_images:
                continue

            # Extract features
            try:
                features = self._extract_batch(batch_images)
                all_features.append(features)

            except Exception as e:
                logger.warning(f"Error processing batch starting at {i}: {e}")
                continue

        if not all_features:
            raise ValueError("No features extracted from any images")

        # Concatenate all features
        all_features = np.vstack(all_features)
        logger.info(f"Extracted features shape: {all_features.shape}")

        return all_features

    def _extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract features from a batch of images.

        Args:
            images: List of PIL images

        Returns:
            Feature embeddings [batch_size, feature_dim]
        """
        # Preprocess images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast('cuda'):
                    outputs = self.model(**inputs, output_hidden_states=self.multilayer)
            else:
                outputs = self.model(**inputs, output_hidden_states=self.multilayer)

        if self.multilayer:
            # Concatenate CLS tokens from multiple layers
            features = torch.cat([
                outputs.hidden_states[l][:, 0, :]
                for l in self.layers
            ], dim=-1)
        else:
            # Use final layer CLS token
            features = outputs.last_hidden_state[:, 0, :]

        # Convert to numpy
        features = features.cpu().float().numpy()

        return features


class CLIPFeatureExtractor:
    """Extract CLIP features for semantic diversity."""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 128,
        model_name: str = "openai/clip-vit-large-patch14"
    ):
        """
        Initialize CLIP feature extractor.

        Args:
            device: Device to use
            batch_size: Batch size for processing
            model_name: CLIP model name
        """
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        self._load_model()

    def _load_model(self):
        """Load CLIP model."""
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading {self.model_name}...")

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("CLIP model loaded successfully")

    def extract_features(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract CLIP image features."""
        all_features = []

        iterator = tqdm(
            range(0, len(image_paths), self.batch_size),
            desc="Extracting CLIP features",
            disable=not show_progress
        )

        for i in iterator:
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                except:
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
    dinov2_features: np.ndarray,
    clip_features: np.ndarray,
    pca_components: int = 1024
) -> np.ndarray:
    """
    Combine DINOv2 and CLIP features with PCA reduction.

    Args:
        dinov2_features: DINOv2 embeddings
        clip_features: CLIP embeddings
        pca_components: Number of PCA components

    Returns:
        Combined and reduced features
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
        f"Reduced to {pca_components} dims, "
        f"variance retained: {variance_retained:.2%}"
    )

    return reduced
