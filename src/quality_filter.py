"""Quality filtering pipeline for car images."""

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

MIN_RESOLUTION_DEFAULT = 256
MAX_ASPECT_RATIO_DEFAULT = 3.0
BRISQUE_THRESHOLD_DEFAULT = 45.0
BLUR_THRESHOLD_DEFAULT = 100.0
CAR_CONFIDENCE_THRESHOLD_DEFAULT = 0.5
QUALITY_FILTER_IMG_SIZE = (512, 512)


class QualityDataset(Dataset):
    """Dataset for quality filtering with CPU-based blur detection.

    Performs blur detection on native resolution images (CPU), then resizes
    for GPU-based quality assessment.

    Attributes:
        image_paths: List of image file paths.
        blur_threshold: Laplacian variance threshold for blur detection.
        img_size: Target size for GPU processing (width, height).
    """

    def __init__(
        self,
        image_paths: list[str],
        blur_threshold: float,
        img_size: tuple[int, int] = QUALITY_FILTER_IMG_SIZE,
    ) -> None:
        """Initialize quality dataset.

        Args:
            image_paths: List of image file paths.
            blur_threshold: Laplacian variance threshold for blur detection.
            img_size: Target size for GPU processing.
        """
        self.image_paths = image_paths
        self.blur_threshold = blur_threshold
        self.img_size = img_size

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str] | None:
        """Load and preprocess image.

        Args:
            idx: Image index.

        Returns:
            Tuple of (image_tensor, path) if successful, None otherwise.
        """
        path = self.image_paths[idx]
        try:
            img_pil = Image.open(path).convert("RGB")
            img_np_native = np.array(img_pil)

            gray = cv2.cvtColor(img_np_native, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score < self.blur_threshold:
                return None

            img_resized = img_pil.resize(self.img_size, Image.LANCZOS)
            img_np = np.array(img_resized)

            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

            return img_tensor, path

        except Exception:
            return None


def collate_fn(
    batch: list[tuple[torch.Tensor, str] | None],
) -> tuple[torch.Tensor, list[str]]:
    """Filter out None values and stack tensors.

    Args:
        batch: List of (tensor, path) tuples or None values.

    Returns:
        Tuple of (stacked_tensors, paths).
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), []

    tensors, paths = zip(*batch)
    return torch.stack(tensors), list(paths)


class QualityFilter:
    """Multi-stage quality filtering for car images.

    Applies resolution, aspect ratio, BRISQUE quality, and blur filters
    to remove low-quality images.

    Attributes:
        device: Device to use ('cuda' or 'cpu').
        batch_size: Batch size for processing.
        min_resolution: Minimum image dimension.
        max_aspect_ratio: Maximum width/height ratio.
        brisque_threshold: BRISQUE quality threshold (lower is better).
        blur_threshold: Laplacian variance threshold (higher is better).
        car_confidence_threshold: CLIP confidence threshold for car detection.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
        min_resolution: int = MIN_RESOLUTION_DEFAULT,
        max_aspect_ratio: float = MAX_ASPECT_RATIO_DEFAULT,
        brisque_threshold: float = BRISQUE_THRESHOLD_DEFAULT,
        blur_threshold: float = BLUR_THRESHOLD_DEFAULT,
        car_confidence_threshold: float = CAR_CONFIDENCE_THRESHOLD_DEFAULT,
    ) -> None:
        """Initialize quality filter.

        Args:
            device: Device to use ('cuda' or 'cpu').
            batch_size: Batch size for processing.
            min_resolution: Minimum image dimension in pixels.
            max_aspect_ratio: Maximum width/height ratio.
            brisque_threshold: BRISQUE quality threshold (lower is better).
            blur_threshold: Laplacian variance threshold (higher is better).
            car_confidence_threshold: CLIP confidence threshold for car detection.
        """
        self.device = device
        self.batch_size = batch_size
        self.min_resolution = min_resolution
        self.max_aspect_ratio = max_aspect_ratio
        self.brisque_threshold = brisque_threshold
        self.blur_threshold = blur_threshold
        self.car_confidence_threshold = car_confidence_threshold

        self._brisque_model: Any | None = None
        self._clip_model: Any | None = None
        self._clip_processor: Any | None = None

    @property
    def brisque_model(self) -> Any | None:
        """Lazy load BRISQUE model.

        Returns:
            BRISQUE model instance if loaded successfully, None otherwise.
        """
        if self._brisque_model is None:
            try:
                import pyiqa

                self._brisque_model = pyiqa.create_metric("brisque", device=self.device)
                logger.info("BRISQUE model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BRISQUE model: {e}")
                self._brisque_model = False
        return self._brisque_model if self._brisque_model else None

    @property
    def clip_model(self) -> Any | None:
        """Lazy load CLIP model.

        Returns:
            CLIP model instance if loaded successfully, None otherwise.
        """
        if self._clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor

                self._clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).to(self.device)
                self._clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self._clip_model.eval()
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}")
                self._clip_model = False
        return self._clip_model if self._clip_model else None

    def filter_dataset(self, image_dir: str) -> list[str]:
        """Apply multi-stage quality filtering to all images in directory.

        Args:
            image_dir: Directory containing images.

        Returns:
            List of valid image paths that passed all filters.
        """
        image_paths = self._collect_image_paths(image_dir)
        logger.info(f"Found {len(image_paths):,} total images")

        # Stage 1: Fast screening (CPU)
        logger.info("Stage 1: Fast screening (resolution, aspect ratio)...")
        valid_paths = self._fast_filter(image_paths)
        logger.info(
            f"Passed fast filter: {len(valid_paths):,} ({len(valid_paths) / len(image_paths) * 100:.1f}%)"
        )

        # Stage 2: Quality assessment
        if self.brisque_model:
            logger.info("Stage 2: Quality assessment (BRISQUE, blur)...")
            valid_paths = self._quality_filter(valid_paths)
            logger.info(f"Passed quality filter: {len(valid_paths):,}")
        else:
            logger.warning("Skipping quality assessment (BRISQUE not available)")

        # Stage 3: Car content validation
        if self.clip_model:
            logger.info("Stage 3: Car content validation (CLIP)...")
            valid_paths = self._car_content_filter(valid_paths)
            logger.info(f"Passed car content filter: {len(valid_paths):,}")
        else:
            logger.warning("Skipping car content validation (CLIP not available)")

        return valid_paths

    def _collect_image_paths(self, image_dir: str) -> list[str]:
        """Collect all image file paths from directory recursively.

        Args:
            image_dir: Directory to search.

        Returns:
            List of image file paths.
        """
        image_dir_path = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir_path.rglob(f"*{ext}"))
            image_paths.extend(image_dir_path.rglob(f"*{ext.upper()}"))

        return [str(p) for p in image_paths]

    def _fast_filter(self, image_paths: list[str]) -> list[str]:
        """Fast CPU-based filtering for resolution and aspect ratio.

        Args:
            image_paths: List of image paths to filter.

        Returns:
            List of paths that passed resolution and aspect ratio filters.
        """
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

    def _get_optimal_workers(self) -> int:
        """Determine optimal number of DataLoader workers.

        Respects container limits (Docker/Kubernetes) and applies heuristics
        to balance parallelism with overhead.

        Returns:
            Optimal number of workers (1-16).
        """
        try:
            # On Linux, this gets the specific cores assigned to this process
            # (Crucial if running in Docker or with taskset)
            available_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback for Windows/Mac
            available_cores = os.cpu_count() or 1

        # Heuristic 1: Reserve cores
        # Leave 2 cores free: 1 for the main process (GPU communication) + 1 for OS tasks
        workers = max(1, available_cores - 2)

        # Heuristic 2: Cap the limit
        # For image loading/preprocessing, diminishing returns usually hit around 16 workers.
        # More than that increases memory overhead and context switching cost.
        max_cap = 16

        min(workers, max_cap)

    def _quality_filter(self, image_paths: list[str]) -> list[str]:
        """Optimized quality assessment using DataLoader and batching.

        Args:
            image_paths: List of image paths to assess.

        Returns:
            List of paths that passed quality (BRISQUE and blur) filters.
        """
        valid_paths = []

        # 1. Setup Dataset and Loader
        dataset = QualityDataset(image_paths, self.blur_threshold)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._get_optimal_workers(),
            collate_fn=collate_fn,
            pin_memory=True,  # Faster CPU->GPU transfer
        )

        self.brisque_model.eval()

        # 2. Batch Processing Loop
        with torch.no_grad():
            for images, paths in tqdm(loader, desc="Quality filter (Batched)"):
                if images.numel() == 0:
                    continue

                # Move batch to GPU
                images = images.to(self.device, non_blocking=True)

                scores = self.brisque_model(images)

                if scores.ndim > 1:
                    scores = scores.squeeze()
                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)

                scores_np = scores.cpu().numpy()

                for score, path in zip(scores_np, paths):
                    if score <= self.brisque_threshold:
                        valid_paths.append(path)

        return valid_paths

    def _car_content_filter(self, image_paths: list[str]) -> list[str]:
        """Filter non-car images using CLIP zero-shot classification.

        Args:
            image_paths: List of image paths to filter.

        Returns:
            List of paths classified as containing cars.
        """
        if not self.clip_model:
            return image_paths

        valid_paths = []

        car_prompts = [
            "car",
            "vehicle",
            "automobile",
        ]
        non_car_prompts = ["an image without any car", "an image with no vehicles"]

        all_prompts = car_prompts #+ non_car_prompts

        # Process in batches
        for i in tqdm(
            range(0, len(image_paths), self.batch_size), desc="Car content filter"
        ):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_images = []

            # Load batch images
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
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
                    padding=True,
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # Check car confidence for each image
                for idx, img_path in enumerate(paths):
                    car_score = probs[idx, : len(car_prompts)].max().item()

                    if car_score > self.car_confidence_threshold:
                        valid_paths.append(img_path)

            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
                continue
        
        logger.info(f"% of Valid Paths - {(len(valid_paths)/len(paths))*100}%")

        return valid_paths
