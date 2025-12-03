"""Evaluation metrics for diversity selection quality."""

import numpy as np
import faiss
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def evaluate_selection(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    image_paths: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate the quality of diverse selection.

    Args:
        embeddings: All embeddings
        selected_indices: Indices of selected images
        image_paths: Optional image paths for additional analysis

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Computing evaluation metrics...")

    metrics = {}

    # Basic statistics
    metrics['total_images'] = len(embeddings)
    metrics['selected_images'] = len(selected_indices)
    metrics['selection_ratio'] = len(selected_indices) / len(embeddings)

    # Coverage metrics
    coverage_metrics = compute_coverage_metrics(embeddings, selected_indices)
    metrics.update(coverage_metrics)

    # Diversity metrics
    diversity_metrics = compute_diversity_metrics(embeddings, selected_indices)
    metrics.update(diversity_metrics)

    # Distance distribution
    distance_dist = compute_distance_distribution(embeddings, selected_indices)
    metrics['distance_distribution'] = distance_dist

    # Log summary
    logger.info("Evaluation Summary:")
    logger.info(f"  Coverage radius (max): {metrics['coverage_radius_max']:.4f}")
    logger.info(f"  Coverage radius (mean): {metrics['coverage_radius_mean']:.4f}")
    logger.info(f"  Intra-selection diversity (mean): {metrics['intra_diversity_mean']:.4f}")
    logger.info(f"  Intra-selection diversity (std): {metrics['intra_diversity_std']:.4f}")

    return metrics


def compute_coverage_metrics(
    embeddings: np.ndarray,
    selected_indices: np.ndarray
) -> Dict[str, float]:
    """
    Compute coverage metrics (k-center quality).

    Measures how well the selected points cover the entire dataset.

    Args:
        embeddings: All embeddings
        selected_indices: Selected indices

    Returns:
        Dictionary with coverage metrics
    """
    embeddings = embeddings.astype(np.float32)
    selected_emb = embeddings[selected_indices]

    # Build FAISS index for selected points
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(selected_emb)

    # Find distance from each point to nearest selected point
    D, I = index.search(embeddings, k=1)
    distances = D.flatten()

    return {
        'coverage_radius_max': float(np.max(distances)),
        'coverage_radius_mean': float(np.mean(distances)),
        'coverage_radius_median': float(np.median(distances)),
        'coverage_radius_std': float(np.std(distances)),
        'coverage_radius_95th': float(np.percentile(distances, 95))
    }


def compute_diversity_metrics(
    embeddings: np.ndarray,
    selected_indices: np.ndarray
) -> Dict[str, float]:
    """
    Compute intra-selection diversity metrics.

    Measures how diverse the selected set is internally.

    Args:
        embeddings: All embeddings
        selected_indices: Selected indices

    Returns:
        Dictionary with diversity metrics
    """
    selected_emb = embeddings[selected_indices].astype(np.float32)
    n = len(selected_indices)

    # Sample pairs for efficiency (if too many points)
    max_pairs = 10000
    if n * (n - 1) // 2 > max_pairs:
        # Sample pairs
        np.random.seed(42)
        pairs = np.random.choice(n, size=(max_pairs, 2), replace=True)
    else:
        # All pairs
        idx = np.triu_indices(n, k=1)
        pairs = np.column_stack(idx)

    # Compute pairwise distances
    distances = np.linalg.norm(
        selected_emb[pairs[:, 0]] - selected_emb[pairs[:, 1]],
        axis=1
    )

    return {
        'intra_diversity_mean': float(np.mean(distances)),
        'intra_diversity_std': float(np.std(distances)),
        'intra_diversity_min': float(np.min(distances)),
        'intra_diversity_max': float(np.max(distances)),
        'intra_diversity_median': float(np.median(distances))
    }


def compute_distance_distribution(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    n_bins: int = 20
) -> Dict[str, List[float]]:
    """
    Compute distance distribution statistics.

    Args:
        embeddings: All embeddings
        selected_indices: Selected indices
        n_bins: Number of histogram bins

    Returns:
        Dictionary with distance distribution
    """
    embeddings = embeddings.astype(np.float32)
    selected_emb = embeddings[selected_indices]

    # Distance from each point to nearest selected point
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(selected_emb)
    D, _ = index.search(embeddings, k=1)
    distances = D.flatten()

    # Create histogram
    hist, bin_edges = np.histogram(distances, bins=n_bins)

    return {
        'histogram_counts': hist.tolist(),
        'histogram_edges': bin_edges.tolist()
    }


def compare_selection_methods(
    embeddings: np.ndarray,
    selections: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple selection methods.

    Args:
        embeddings: All embeddings
        selections: Dictionary mapping method name to selected indices

    Returns:
        Dictionary with metrics for each method
    """
    results = {}

    for method_name, selected_indices in selections.items():
        logger.info(f"Evaluating {method_name}...")

        metrics = evaluate_selection(embeddings, selected_indices)
        results[method_name] = metrics

    # Summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("Method Comparison Summary")
    logger.info("=" * 80)

    for method_name, metrics in results.items():
        logger.info(f"\n{method_name}:")
        logger.info(f"  Coverage radius (max): {metrics['coverage_radius_max']:.4f}")
        logger.info(f"  Coverage radius (mean): {metrics['coverage_radius_mean']:.4f}")
        logger.info(f"  Intra-diversity (mean): {metrics['intra_diversity_mean']:.4f}")

    return results


def visualize_selection(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    output_path: str,
    method: str = 'tsne'
):
    """
    Visualize selection using dimensionality reduction.

    Args:
        embeddings: All embeddings
        selected_indices: Selected indices
        output_path: Path to save visualization
        method: Reduction method ('tsne', 'umap', 'pca')
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping visualization")
        return

    logger.info(f"Creating visualization using {method}...")

    # Reduce to 2D
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            logger.warning("UMAP not available, falling back to PCA")
            method = 'pca'
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reduce dimensions
    if method == 'pca':
        reduced = reducer.fit_transform(embeddings)
    else:
        # Sample for efficiency with t-SNE/UMAP
        max_samples = 10000
        if len(embeddings) > max_samples:
            sample_idx = np.random.choice(len(embeddings), max_samples, replace=False)
            # Make sure selected indices are included
            sample_idx = np.union1d(sample_idx, selected_indices)
            embeddings_sample = embeddings[sample_idx]
            reduced_sample = reducer.fit_transform(embeddings_sample)

            # Map back to original indices
            reduced = np.zeros((len(embeddings), 2))
            reduced[sample_idx] = reduced_sample
            selected_mask = np.isin(sample_idx, selected_indices)
        else:
            reduced = reducer.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all points
    mask = np.ones(len(embeddings), dtype=bool)
    mask[selected_indices] = False

    ax.scatter(
        reduced[mask, 0],
        reduced[mask, 1],
        c='lightgray',
        s=1,
        alpha=0.5,
        label='Not selected'
    )

    # Plot selected points
    ax.scatter(
        reduced[selected_indices, 0],
        reduced[selected_indices, 1],
        c='red',
        s=10,
        alpha=0.8,
        label='Selected'
    )

    ax.set_title(f'Selection Visualization ({method.upper()})', fontsize=16)
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Visualization saved to: {output_path}")
