"""Diversity selection algorithms for maximum coverage."""

import torch
import numpy as np
import faiss
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DiversitySelector:
    """Diversity-based selection using various algorithms."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize diversity selector.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device

    def select_diverse(
        self, embeddings: np.ndarray, n_select: int, method: str = "hybrid"
    ) -> np.ndarray:
        """
        Select diverse subset of images.

        Args:
            embeddings: Feature embeddings [N, D]
            n_select: Number of images to select
            method: Selection method ('hybrid', 'fps', 'kmeans', 'random')

        Returns:
            Indices of selected images
        """
        if method == "random":
            return self._random_selection(embeddings, n_select)
        elif method == "kmeans":
            return self._kmeans_selection(embeddings, n_select)
        elif method == "fps":
            return self._fps_selection(embeddings, n_select)
        elif method == "hybrid":
            return self._hybrid_selection(embeddings, n_select)
        elif method == "spectral":
            return self._spectral_selection(embeddings, n_select)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _random_selection(self, embeddings: np.ndarray, n_select: int) -> np.ndarray:
        """Random baseline selection."""
        logger.info("Using random selection (baseline)")
        return np.random.choice(len(embeddings), size=n_select, replace=False)

    def _kmeans_selection(self, embeddings: np.ndarray, n_select: int) -> np.ndarray:
        """
        K-means clustering + medoid selection.

        Args:
            embeddings: Feature embeddings
            n_select: Number of clusters/images to select

        Returns:
            Indices of cluster medoids
        """
        logger.info(f"Running k-means with {n_select} clusters...")

        try:
            # Try using GPU-accelerated cuML
            from cuml.cluster import KMeans

            embeddings_gpu = embeddings.astype(np.float32)
            kmeans = KMeans(n_clusters=n_select, max_iter=100, verbose=0)
            kmeans.fit(embeddings_gpu)

            cluster_centers = kmeans.cluster_centers_

            logger.info("Using cuML (GPU) for k-means")

        except ImportError:
            # Fall back to sklearn
            logger.info("cuML not available, using sklearn (CPU) for k-means")
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=n_select, batch_size=10000, max_iter=100, verbose=0
            )
            kmeans.fit(embeddings)
            cluster_centers = kmeans.cluster_centers_

        # Find medoids (closest real image to each cluster center)
        logger.info("Finding cluster medoids...")
        medoid_indices = []

        for center in tqdm(cluster_centers, desc="Finding medoids"):
            distances = np.linalg.norm(embeddings - center, axis=1)
            medoid_idx = np.argmin(distances)
            medoid_indices.append(medoid_idx)

        return np.array(medoid_indices)

    def _fps_selection(self, embeddings: np.ndarray, n_select: int) -> np.ndarray:
        """
        Farthest Point Sampling (FPS) with FAISS acceleration.

        Args:
            embeddings: Feature embeddings
            n_select: Number of points to select

        Returns:
            Indices of selected points
        """
        logger.info("Running Farthest Point Sampling (FPS)...")

        embeddings = embeddings.astype(np.float32)
        n, d = embeddings.shape

        # Initialize index (CPU-only for compatibility)
        index = faiss.IndexFlatL2(d)

        # Random first point
        selected = [np.random.randint(n)]
        index.add(embeddings[selected[0] : selected[0] + 1])

        # Track minimum distance to selected set
        min_distances = np.full(n, np.inf, dtype=np.float32)

        # Batch size for distance computation
        batch_size = 30000

        for iteration in tqdm(range(1, n_select), desc="FPS selection"):
            # Update distances in batches
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                batch = embeddings[batch_start:batch_end]

                # Distance to nearest selected point
                D, _ = index.search(batch, k=1)
                min_distances[batch_start:batch_end] = np.minimum(
                    min_distances[batch_start:batch_end], D.flatten()
                )

            # Select point with maximum min-distance
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)
            min_distances[next_idx] = 0  # Mark as selected

            # Add to index
            index.add(embeddings[next_idx : next_idx + 1])

            if iteration % 1000 == 0:
                max_dist = np.max(min_distances[min_distances < np.inf])
                logger.info(
                    f"Selected {iteration}/{n_select} images, "
                    f"max_min_dist: {max_dist:.4f}"
                )

        return np.array(selected)

    def _hybrid_selection(
        self, embeddings: np.ndarray, n_select: int, n_clusters: int | None = None
    ) -> np.ndarray:
        """
        Hybrid hierarchical selection: k-means clustering + FPS within clusters.

        This provides a good balance between computational efficiency and
        diversity guarantees.

        Args:
            embeddings: Feature embeddings
            n_select: Number of images to select
            n_clusters: Number of clusters (default: n_select // 10)

        Returns:
            Indices of selected images
        """
        if n_clusters is None:
            n_clusters = max(100, n_select // 10)

        logger.info(f"Hybrid selection: {n_clusters} clusters, {n_select} images")

        # Stage 1: Stratified clustering
        logger.info("Stage 1: K-means clustering...")

        try:
            from cuml.cluster import KMeans

            embeddings_gpu = embeddings.astype(np.float32)
            kmeans = KMeans(n_clusters=n_clusters, max_iter=100, verbose=0)
            labels = kmeans.fit_predict(embeddings_gpu)

            if hasattr(labels, "to_numpy"):
                labels = labels.to_numpy()
            elif hasattr(labels, "values"):
                labels = labels.values

            logger.info("Using cuML (GPU) for clustering")

        except ImportError:
            logger.info("Using sklearn (CPU) for clustering")
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, batch_size=10000, max_iter=100, verbose=0
            )
            labels = kmeans.fit_predict(embeddings)

        # Stage 2: Diverse selection within clusters
        logger.info("Stage 2: FPS within clusters...")

        selected_indices = []
        images_per_cluster = n_select // n_clusters

        for cluster_id in tqdm(range(n_clusters), desc="Cluster FPS"):
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_original_indices = np.where(cluster_mask)[0]

            if len(cluster_original_indices) == 0:
                continue

            # Adaptive allocation based on cluster size
            n_from_cluster = min(
                len(cluster_original_indices),
                max(1, int(images_per_cluster * 1.2)),  # Slight over-allocation
            )

            # FPS within cluster (simplified for speed)
            if len(cluster_original_indices) <= n_from_cluster:
                # Take all images from small clusters
                local_selected = np.arange(len(cluster_original_indices))
            else:
                # FPS within cluster
                local_selected = self._fast_fps_small(
                    cluster_embeddings, n_from_cluster
                )

            selected_indices.extend(cluster_original_indices[local_selected])

        selected_indices = np.array(selected_indices)

        # Stage 3: Global refinement if needed
        if len(selected_indices) < n_select:
            logger.info("Stage 3: Global refinement to reach target...")
            selected_indices = self._global_fps_refinement(
                embeddings, selected_indices, n_select
            )
        elif len(selected_indices) > n_select:
            # Trim excess with FPS on selected set
            logger.info(f"Trimming {len(selected_indices)} to {n_select}...")
            selected_embeddings = embeddings[selected_indices]
            keep_local = self._fast_fps_small(selected_embeddings, n_select)
            selected_indices = selected_indices[keep_local]

        logger.info(f"Final selection: {len(selected_indices)} images")

        return selected_indices
    
    def _spectral_selection(self, embeddings: np.ndarray, n_select: int) -> np.ndarray:
        """
        Spectral clustering + medoid selection.

        Args:
            embeddings: Feature embeddings
            n_select: Number of clusters/images to select

        Returns:
            Indices of cluster medoids
        """
        logger.info(f"Running spectral_clustering with {n_select} clusters...")

        # Fall back to sklearn
        logger.info("cuML not available, using sklearn (CPU) for spectral_clustering")
        from sklearn.cluster import SpectralClustering

        clusterer = SpectralClustering(
            n_clusters=n_select, verbose=0
        )
        clusterer.fit(embeddings)
        cluster_centers = clusterer.cluster_centers_

        # Find medoids (closest real image to each cluster center)
        logger.info("Finding cluster medoids...")
        medoid_indices = []

        for center in tqdm(cluster_centers, desc="Finding medoids"):
            distances = np.linalg.norm(embeddings - center, axis=1)
            medoid_idx = np.argmin(distances)
            medoid_indices.append(medoid_idx)

        return np.array(medoid_indices)

    def _fast_fps_small(self, embeddings: np.ndarray, n_select: int) -> np.ndarray:
        """
        Fast FPS for small sets using GPU (PyTorch).
        """
        n = len(embeddings)
        if n <= n_select:
            return np.arange(n)

        # 1. Move data to GPU once
        # Check if a GPU is available, otherwise fallback to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert to tensor and send to device
        data = torch.from_numpy(embeddings).to(device, dtype=torch.float32)

        # 2. Initialize
        # Distance to the set of selected points. Initialize to infinity.
        min_distances = torch.full(
            (n,), float("inf"), device=device, dtype=torch.float32
        )

        # Select the first point randomly
        selected_indices = []
        first_idx = np.random.randint(n)
        selected_indices.append(first_idx)

        # 3. Iterative Selection
        # We only need to compute distances from the *most recently* selected point
        # and update the running minimum.
        for _ in range(1, n_select):
            # Get the latest selected point vector
            last_selected_idx = selected_indices[-1]
            pivot = data[last_selected_idx].unsqueeze(0)  # Shape: (1, dim)

            # Compute L2 distance squared from pivot to ALL points
            # (x - y)^2
            dist_sq = torch.sum((data - pivot) ** 2, dim=1)

            # Update the minimum distance for every point
            # min_distances[i] = min(previous_min, dist_to_new_pivot)
            min_distances = torch.minimum(min_distances, dist_sq)

            # The next point is the one with the maximum distance to the set
            # We mask selected points by setting their dist to -1 (optional, but good for safety)
            # Note: Since distance to itself is 0, and we seek max, it naturally handles itself,
            # but pure 0s can be tricky if duplicates exist.

            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)

        return np.array(selected_indices)

    def _global_fps_refinement(
        self, embeddings: np.ndarray, initial_selection: np.ndarray, target_n: int
    ) -> np.ndarray:
        """
        Refine selection using global FPS to fill gaps.

        Args:
            embeddings: All embeddings
            initial_selection: Initially selected indices
            target_n: Target number of selections

        Returns:
            Refined selection
        """
        embeddings = embeddings.astype(np.float32)

        selected = set(initial_selection.tolist())
        remaining = set(range(len(embeddings))) - selected

        # Build FAISS index of selected points (CPU-only for compatibility)
        index = faiss.IndexFlatL2(embeddings.shape[1])

        index.add(embeddings[list(selected)])

        # Add points maximizing distance from current selection
        pbar = tqdm(total=target_n - len(selected), desc="Global refinement")

        while len(selected) < target_n and remaining:
            # Find farthest point from current selection
            remaining_list = list(remaining)
            remaining_emb = embeddings[remaining_list]

            D, _ = index.search(remaining_emb, k=1)
            farthest_local_idx = np.argmax(D)
            farthest_idx = remaining_list[farthest_local_idx]

            selected.add(farthest_idx)
            remaining.remove(farthest_idx)
            index.add(embeddings[farthest_idx : farthest_idx + 1])

            pbar.update(1)

        pbar.close()

        return np.array(list(selected))
