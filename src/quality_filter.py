"""Quality filtering pipeline for car images."""

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List
import logging

logger = logging.getLogger(__name__)


class QualityFilter:
    """Multi-stage quality filtering for car images."""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
        min_resolution: int = 256,
        max_aspect_ratio: float = 3.0,
        brisque_threshold: float = 45.0,
        blur_threshold: float = 100.0,
        car_confidence_threshold: float = 0.6
    ):
        """
        Initialize quality filter.

        Args:
            device: Device to use ('cuda' or 'cpu')
            batch_size: Batch size for processing
            min_resolution: Minimum image dimension
            max_aspect_ratio: Maximum width/height ratio
            brisque_threshold: BRISQUE quality threshold (lower is better)
            blur_threshold: Laplacian variance threshold (higher is better)
            car_confidence_threshold: CLIP confidence threshold for car detection
        """
        self.device = device
        self.batch_size = batch_size
        self.min_resolution = min_resolution
        self.max_aspect_ratio = max_aspect_ratio
        self.brisque_threshold = brisque_threshold
        self.blur_threshold = blur_threshold
        self.car_confidence_threshold = car_confidence_threshold

        # Initialize models lazily
        self._brisque_model = None
        self._clip_model = None
        self._clip_processor = None

    @property
    def brisque_model(self):
        """Lazy load BRISQUE model."""
        if self._brisque_model is None:
            try:
                import pyiqa
                self._brisque_model = pyiqa.create_metric('brisque', device=self.device)
                logger.info("BRISQUE model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BRISQUE model: {e}")
                self._brisque_model = False
        return self._brisque_model if self._brisque_model else None

    @property
    def clip_model(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model.eval()
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}")
                self._clip_model = False
        return self._clip_model if self._clip_model else None

    def filter_dataset(self, image_dir: str) -> List[str]:
        """
        Apply multi-stage quality filtering to all images in directory.

        Args:
            image_dir: Directory containing images

        Returns:
            List of valid image paths
        """
        image_paths = self._collect_image_paths(image_dir)
        logger.info(f"Found {len(image_paths):,} total images")

        # Stage 1: Fast screening (CPU)
        logger.info("Stage 1: Fast screening (resolution, aspect ratio)...")
        valid_paths = self._fast_filter(image_paths)
        logger.info(f"Passed fast filter: {len(valid_paths):,} ({len(valid_paths)/len(image_paths)*100:.1f}%)")

        # Stage 2: Quality assessment
        if self.brisque_model:
            logger.info("Stage 2: Quality assessment (BRISQUE, blur)...")
            valid_paths = self._quality_filter(valid_paths)
            logger.info(f"Passed quality filter: {len(valid_paths):,}")
        else:
            logger.warning("Skipping quality assessment (BRISQUE not available)")

        # # Stage 3: Car content validation
        # if self.clip_model:
        #     logger.info("Stage 3: Car content validation (CLIP)...")
        #     valid_paths = self._car_content_filter(valid_paths)
        #     logger.info(f"Passed car content filter: {len(valid_paths):,}")
        # else:
        #     logger.warning("Skipping car content validation (CLIP not available)")

        return valid_paths

    def _collect_image_paths(self, image_dir: str) -> List[str]:
        """Collect all image file paths from directory."""
        image_dir = Path(image_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.rglob(f'*{ext}'))
            image_paths.extend(image_dir.rglob(f'*{ext.upper()}'))

        return [str(p) for p in image_paths]

    def _fast_filter(self, image_paths: List[str]) -> List[str]:
        """Fast CPU-based filtering for resolution and aspect ratio."""
        valid_paths = []

        for img_path in tqdm(image_paths, desc="Fast filter"):
            try:
                img = Image.open(img_path)

                # Resolution filter
                if min(img.size) < self.min_resolution:
                    continue

                # Aspect ratio filter
                aspect = max(img.size) / min(img.size)
                if aspect > self.max_aspect_ratio:
                    continue

                valid_paths.append(img_path)

            except Exception as e:
                logger.debug(f"Error loading {img_path}: {e}")
                continue

        return valid_paths

    def _quality_filter(self, image_paths: List[str]) -> List[str]:
        """Quality assessment using BRISQUE and blur detection."""
        valid_paths = []

        for img_path in tqdm(image_paths, desc="Quality filter"):
            try:
                # Load image
                img_pil = Image.open(img_path).convert('RGB')
                img_np = np.array(img_pil)

                # Blur detection using Laplacian variance
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                if blur_score < self.blur_threshold:
                    continue

                # BRISQUE quality assessment
                if self.brisque_model:
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    img_tensor = img_tensor.to(self.device)

                    with torch.no_grad():
                        brisque_score = self.brisque_model(img_tensor).item()

                    if brisque_score > self.brisque_threshold:
                        continue

                valid_paths.append(img_path)

            except Exception as e:
                logger.debug(f"Error processing {img_path}: {e}")
                continue

        return valid_paths

    def _car_content_filter(self, image_paths: List[str]) -> List[str]:
        """Filter non-car images using CLIP zero-shot classification."""
        if not self.clip_model:
            return image_paths

        valid_paths = []

        car_prompts = [
            "a photograph of a car",
            "a photo of a vehicle automobile",
            "an image showing a car"
        ]
        non_car_prompts = [
            "a photo without any car",
            "an image with no vehicles"
        ]

        all_prompts = car_prompts + non_car_prompts

        # Process in batches
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Car content filter"):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []

            # Load batch images
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append((img_path, img))
                except Exception as e:
                    logger.debug(f"Error loading {img_path}: {e}")
                    continue

            if not batch_images:
                continue

            # Process batch
            try:
                paths, images = zip(*batch_images)

                inputs = self._clip_processor(
                    text=all_prompts,
                    images=list(images),
                    return_tensors="pt",
                    padding=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # Check car confidence for each image
                for idx, img_path in enumerate(paths):
                    car_score = probs[idx, :len(car_prompts)].mean().item()

                    if car_score > self.car_confidence_threshold:
                        valid_paths.append(img_path)

            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
                continue

        return valid_paths
