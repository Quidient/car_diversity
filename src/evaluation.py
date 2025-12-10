"""Evaluation metrics for diversity selection quality."""

import logging
from pathlib import Path
from typing import Any, Dict
from sklearn.preprocessing import normalize

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_selection(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate the quality of diverse selection.

    Args:
        embeddings: All embeddings
        selected_indices: Indices of selected images

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Computing evaluation metrics...")
    logger.info("Normalizing embeddings")
    # L2 normalize rows (axis=1) so every image vector has length 1.0
    embeddings = normalize(embeddings, norm="l2", axis=1)
    metrics = {}

    # Basic statistics
    metrics["total_images"] = len(embeddings)
    metrics["selected_images"] = len(selected_indices)
    metrics["selection_ratio"] = len(selected_indices) / len(embeddings)

    # Coverage metrics
    coverage_metrics = compute_coverage_metrics(embeddings, selected_indices)
    metrics.update(coverage_metrics)

    # Diversity metrics
    diversity_metrics = compute_diversity_metrics(embeddings, selected_indices)
    metrics.update(diversity_metrics)

    # Distance distribution
    distance_dist = compute_distance_distribution(embeddings, selected_indices)
    metrics["distance_distribution"] = distance_dist

    # Log summary
    logger.info("Evaluation Summary:")
    logger.info(f"  Coverage radius (max): {metrics['coverage_radius_max']:.4f}")
    logger.info(f"  Coverage radius (mean): {metrics['coverage_radius_mean']:.4f}")
    logger.info(
        f"  Intra-selection diversity (mean): {metrics['intra_diversity_mean']:.4f}"
    )
    logger.info(
        f"  Intra-selection diversity (std): {metrics['intra_diversity_std']:.4f}"
    )

    return metrics


def compute_coverage_metrics(
    embeddings: np.ndarray, selected_indices: np.ndarray
) -> dict[str, float]:
    """
    Compute coverage metrics (k-center quality).

    Measures how well the selected points cover the entire dataset.
    NOTE: FAISS IndexFlatL2 returns squared L2 distances. We apply sqrt
    to match the scale of the diversity metrics (Euclidean).

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
    distances_sq, _ = index.search(embeddings, k=1)

    # Convert squared L2 to Euclidean L2
    distances = np.sqrt(distances_sq.flatten())

    return {
        "coverage_radius_max": float(np.max(distances)),
        "coverage_radius_mean": float(np.mean(distances)),
        "coverage_radius_median": float(np.median(distances)),
        "coverage_radius_std": float(np.std(distances)),
        "coverage_p25": float(np.percentile(distances, 25)),
        "coverage_p50": float(np.percentile(distances, 50)),
        "coverage_p75": float(np.percentile(distances, 75)),
        "coverage_radius_95th": float(np.percentile(distances, 95)),
    }


def compute_diversity_metrics(
    embeddings: np.ndarray, selected_indices: np.ndarray
) -> dict[str, float]:
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

    # Compute pairwise distances (np.linalg.norm is standard Euclidean)
    distances = np.linalg.norm(
        selected_emb[pairs[:, 0]] - selected_emb[pairs[:, 1]], axis=1
    )

    return {
        "intra_diversity_mean": float(np.mean(distances)),
        "intra_diversity_std": float(np.std(distances)),
        "intra_diversity_min": float(np.min(distances)),
        "intra_diversity_max": float(np.max(distances)),
        "intra_diversity_median": float(np.median(distances)),
    }


def compute_distance_distribution(
    embeddings: np.ndarray, selected_indices: np.ndarray, n_bins: int = 20
) -> dict[str, list[float]]:
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
    D_sq, _ = index.search(embeddings, k=1)

    # Convert squared L2 to Euclidean L2
    distances = np.sqrt(D_sq.flatten())

    # Create histogram
    hist, bin_edges = np.histogram(distances, bins=n_bins)

    return {"histogram_counts": hist.tolist(), "histogram_edges": bin_edges.tolist()}


def compare_selection_methods(
    embeddings: np.ndarray, selections: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
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
    method: str = "tsne",
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
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    logger.info(f"Creating visualization using {method}...")

    # Reduce to 2D
    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        try:
            from umap import UMAP

            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            logger.warning("UMAP not available, falling back to PCA")
            method = "pca"
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reduce dimensions
    if method == "pca":
        reduced = reducer.fit_transform(embeddings)
    else:
        # Sample for efficiency with t-SNE/UMAP
        max_samples = 10000
        if len(embeddings) > max_samples:
            # Validate selected_indices before sampling
            valid_selected = selected_indices[selected_indices < len(embeddings)]
            if len(valid_selected) < len(selected_indices):
                logger.warning(
                    f"Filtered out {len(selected_indices) - len(valid_selected)} "
                    f"out-of-bounds selected indices"
                )

            sample_idx = np.random.choice(len(embeddings), max_samples, replace=False)
            # Make sure selected indices are included
            sample_idx = np.union1d(sample_idx, valid_selected)

            # Ensure sample_idx is within bounds
            sample_idx = sample_idx[sample_idx < len(embeddings)]

            embeddings_sample = embeddings[sample_idx]
            reduced_sample = reducer.fit_transform(embeddings_sample)

            # Validate shapes before assignment
            if len(sample_idx) != len(reduced_sample):
                raise ValueError(
                    f"Shape mismatch: sample_idx has {len(sample_idx)} elements "
                    f"but reduced_sample has {len(reduced_sample)} rows"
                )

            # Map back to original indices
            reduced = np.zeros((len(embeddings), 2))
            reduced[sample_idx] = reduced_sample
        else:
            reduced = reducer.fit_transform(embeddings)

    # Create plot
    _, ax = plt.subplots(figsize=(12, 10))

    # Validate selected_indices for plotting
    valid_plot_indices = selected_indices[selected_indices < len(embeddings)]
    if len(valid_plot_indices) < len(selected_indices):
        logger.warning(
            f"Skipping {len(selected_indices) - len(valid_plot_indices)} "
            f"out-of-bounds indices in plot"
        )

    # Plot all points
    mask = np.ones(len(embeddings), dtype=bool)
    mask[valid_plot_indices] = False

    ax.scatter(
        reduced[mask, 0],
        reduced[mask, 1],
        c="red",
        s=10,
        alpha=0.5,
        label="Not selected",
    )

    # Plot selected points
    ax.scatter(
        reduced[valid_plot_indices, 0],
        reduced[valid_plot_indices, 1],
        c="green",
        s=10,
        alpha=0.8,
        label="Selected",
    )

    ax.set_title(f"Selection Visualization ({method.upper()})", fontsize=16)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Visualization saved to: {output_path}")


def visualize_coverage_distribution(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    output_path: str,
) -> None:
    """Visualize distribution of coverage radii (distance to nearest selected point)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    logger.info("Creating coverage distribution visualization...")

    embeddings = embeddings.astype(np.float32)
    selected_emb = embeddings[selected_indices]

    # Build FAISS index for selected points
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(selected_emb)

    # Find distance from each point to nearest selected point
    distances_sq, _ = index.search(embeddings, k=1)

    # Convert squared L2 to Euclidean L2 for visualization consistency
    distances = np.sqrt(distances_sq.flatten())

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Coverage Distribution Analysis", fontsize=16, fontweight="bold")

    # 1. Histogram of distances
    axes[0, 0].hist(distances, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    axes[0, 0].axvline(
        np.mean(distances),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(distances):.3f}",
    )
    axes[0, 0].axvline(
        np.median(distances),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(distances):.3f}",
    )
    axes[0, 0].set_xlabel("Distance to Nearest Selected Point")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Coverage Radius Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Cumulative distribution
    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    axes[0, 1].plot(sorted_distances, cumulative, linewidth=2, color="steelblue")
    axes[0, 1].axhline(
        0.5, color="orange", linestyle="--", alpha=0.5, label="50th percentile"
    )
    axes[0, 1].axhline(
        0.95, color="red", linestyle="--", alpha=0.5, label="95th percentile"
    )
    axes[0, 1].set_xlabel("Distance to Nearest Selected Point")
    axes[0, 1].set_ylabel("Cumulative Probability")
    axes[0, 1].set_title("Cumulative Coverage Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Box plot
    axes[1, 0].boxplot(
        distances,
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "lightblue", "edgecolor": "black"},
        medianprops={"color": "red", "linewidth": 2},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )
    axes[1, 0].set_ylabel("Distance to Nearest Selected Point")
    axes[1, 0].set_title("Coverage Statistics")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 4. Summary statistics
    axes[1, 1].axis("off")
    stats_text = f"""
    Coverage Statistics
    {"=" * 30}

    Mean:        {np.mean(distances):.4f}
    Median:      {np.median(distances):.4f}
    Std Dev:     {np.std(distances):.4f}
    Min:         {np.min(distances):.4f}
    Max:         {np.max(distances):.4f}

    Percentiles:
    25th:        {np.percentile(distances, 25):.4f}
    50th:        {np.percentile(distances, 50):.4f}
    75th:        {np.percentile(distances, 75):.4f}
    95th:        {np.percentile(distances, 95):.4f}
    99th:        {np.percentile(distances, 99):.4f}
    """
    axes[1, 1].text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Coverage distribution saved to: {output_path}")


def visualize_diversity_matrix(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    output_path: str,
    max_samples: int = 500,
) -> None:
    """Visualize pairwise distance matrix within selected subset."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    logger.info("Creating diversity matrix visualization...")

    selected_emb = embeddings[selected_indices].astype(np.float32)

    # Sample if too many points
    if len(selected_indices) > max_samples:
        sample_idx = np.random.choice(len(selected_indices), max_samples, replace=False)
        selected_emb = selected_emb[sample_idx]
        title_suffix = f" (sampled {max_samples} points)"
    else:
        title_suffix = ""

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    distances = squareform(pdist(selected_emb, metric="euclidean"))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Intra-Selection Diversity Analysis{title_suffix}",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Heatmap of pairwise distances
    im = axes[0].imshow(distances, cmap="viridis", aspect="auto")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Sample Index")
    axes[0].set_title("Pairwise Distance Matrix")
    plt.colorbar(im, ax=axes[0], label="Euclidean Distance")

    # 2. Distribution of pairwise distances
    upper_tri_distances = distances[np.triu_indices_from(distances, k=1)]
    axes[1].hist(
        upper_tri_distances, bins=50, alpha=0.7, color="forestgreen", edgecolor="black"
    )
    axes[1].axvline(
        np.mean(upper_tri_distances),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(upper_tri_distances):.3f}",
    )
    axes[1].axvline(
        np.median(upper_tri_distances),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(upper_tri_distances):.3f}",
    )
    axes[1].set_xlabel("Pairwise Distance")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Pairwise Distances")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Diversity matrix saved to: {output_path}")


def visualize_selection_metrics_dashboard(
    metrics: Dict[str, Any],
    output_path: str,
) -> None:
    """Create a comprehensive dashboard of selection metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    logger.info("Creating metrics dashboard...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(
        "Diversity Selection Metrics Dashboard",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # 1. Coverage metrics (top row, left)
    ax1 = fig.add_subplot(gs[0, 0])
    coverage_metrics = {
        "Max": metrics.get("coverage_radius_max", 0),
        "95th": metrics.get("coverage_radius_95th", 0),
        "Mean": metrics.get("coverage_radius_mean", 0),
        "Median": metrics.get("coverage_radius_median", 0),
    }
    bars = ax1.bar(
        coverage_metrics.keys(),
        coverage_metrics.values(),
        color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"],
        edgecolor="black",
    )
    ax1.set_ylabel("Distance")
    ax1.set_title("Coverage Radius Metrics\n(Lower is Better)", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Diversity metrics (top row, middle)
    ax2 = fig.add_subplot(gs[0, 1])
    diversity_metrics = {
        "Mean": metrics.get("intra_diversity_mean", 0),
        "Median": metrics.get("intra_diversity_median", 0),
        "Max": metrics.get("intra_diversity_max", 0),
        "Min": metrics.get("intra_diversity_min", 0),
    }
    bars = ax2.bar(
        diversity_metrics.keys(),
        diversity_metrics.values(),
        color=["#3498db", "#2ecc71", "#9b59b6", "#f39c12"],
        edgecolor="black",
    )
    ax2.set_ylabel("Distance")
    ax2.set_title("Intra-Diversity Metrics\n(Higher is Better)", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Selection summary (top row, right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    summary_text = f"""
    Selection Summary
    {"=" * 35}

    Total Images:        {metrics.get("total_images", 0):,}
    Selected Images:     {metrics.get("selected_images", 0):,}
    Selection Ratio:     {metrics.get("selection_ratio", 0):.2%}

    Coverage Quality:
      Max Radius:        {metrics.get("coverage_radius_max", 0):.4f}
      Mean Radius:       {metrics.get("coverage_radius_mean", 0):.4f}
      Std Dev:           {metrics.get("coverage_radius_std", 0):.4f}

    Diversity Quality:
      Mean Distance:     {metrics.get("intra_diversity_mean", 0):.4f}
      Std Dev:           {metrics.get("intra_diversity_std", 0):.4f}
    """
    ax3.text(
        0.05,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    # 4. Distance distribution histogram (middle row, spans 2 columns)
    if "distance_distribution" in metrics:
        ax4 = fig.add_subplot(gs[1, :2])
        hist_data = metrics["distance_distribution"]
        bin_edges = hist_data["histogram_edges"]
        bin_counts = hist_data["histogram_counts"]
        bin_centers = [
            (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_counts))
        ]
        ax4.bar(
            bin_centers,
            bin_counts,
            width=np.diff(bin_edges),
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        ax4.set_xlabel("Distance to Nearest Selected Point")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Coverage Distance Distribution", fontweight="bold")
        ax4.grid(True, alpha=0.3)

    # 5. Quality assessment (middle row, right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    # Quality scoring
    max_expected_dist = 2.0
    coverage_score = max(
        0,
        min(
            100,
            100 * (1 - (metrics.get("coverage_radius_mean", 1) / max_expected_dist)),
        ),
    )
    max_diversity_dist = metrics.get("intra_diversity_max", 0)
    diversity_score = max(0, min(100, (metrics.get("intra_diversity_mean", 0)/max_diversity_dist)*100))
    overall_score = (coverage_score + diversity_score) / 2

    quality_text = f"""
    Quality Assessment
    {"=" * 35}

    Coverage Score:      {coverage_score:.1f}/100
      {"█" * int(coverage_score / 5)}{" " * (20 - int(coverage_score / 5))}

    Diversity Score:     {diversity_score:.1f}/100
      {"█" * int(diversity_score / 5)}{" " * (20 - int(diversity_score / 5))}

    Overall Score:       {overall_score:.1f}/100
      {"█" * int(overall_score / 5)}{" " * (20 - int(overall_score / 5))}

    Interpretation:
      Coverage:  {"Excellent" if coverage_score > 80 else "Good" if coverage_score > 60 else "Fair" if coverage_score > 40 else "Poor"}
      Diversity: {"Excellent" if diversity_score > 80 else "Good" if diversity_score > 60 else "Fair" if diversity_score > 40 else "Poor"}
    """
    ax5.text(
        0.05,
        0.5,
        quality_text,
        fontsize=9,
        family="monospace",
        verticalalignment="center",
    )

    # 6. Comparison gauge (bottom row, left)
    ax6 = fig.add_subplot(gs[2, 0])
    categories = ["Coverage\nRadius", "Diversity\nMean", "Selection\nRatio"]
    values = [
        metrics.get("coverage_radius_mean", 0),
        metrics.get("intra_diversity_mean", 0),
        metrics.get("selection_ratio", 0) * 10,  # Scale for visibility
    ]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    ax6.barh(categories, values, color=colors, edgecolor="black")
    ax6.set_xlabel("Normalized Value")
    ax6.set_title("Key Metrics Comparison", fontweight="bold")
    ax6.grid(True, alpha=0.3, axis="x")

    # 7. Percentile comparison (bottom row, middle)
    ax7 = fig.add_subplot(gs[2, 1])
    percentiles = [25, 50, 75, 95]

    # FIX 2: Use real pre-calculated percentiles
    coverage_values = [
        metrics.get("coverage_p25", 0),
        metrics.get("coverage_p50", metrics.get("coverage_radius_median", 0)),
        metrics.get("coverage_p75", 0),
        metrics.get("coverage_radius_95th", 0),
    ]

    ax7.plot(
        percentiles,
        coverage_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#3498db",
    )
    ax7.set_xlabel("Percentile")
    ax7.set_ylabel("Coverage Radius")
    ax7.set_title("Coverage by Percentile", fontweight="bold")
    ax7.grid(True, alpha=0.3)

    # 8. Recommendations (bottom row, right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")

    recommendations = []
    if metrics.get("coverage_radius_max", 0) > 2.0:
        recommendations.append("• Consider selecting more images")
        recommendations.append("  to improve coverage")
    if metrics.get("intra_diversity_mean", 0) < 3.0:
        recommendations.append("• Diversity seems low, try")
        recommendations.append("  using 'fps' method")
    if metrics.get("selection_ratio", 0) < 0.1:
        recommendations.append("• Very aggressive selection")
        recommendations.append("  ratio - ensure quality")
    if not recommendations:
        recommendations = [
            "• Selection quality looks good!",
            "• Metrics are well-balanced",
        ]

    rec_text = "Recommendations\n" + "=" * 35 + "\n\n" + "\n".join(recommendations)
    ax8.text(
        0.05, 0.5, rec_text, fontsize=10, family="monospace", verticalalignment="center"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Metrics dashboard saved to: {output_path}")


def create_all_visualizations(
    embeddings: np.ndarray,
    selected_indices: np.ndarray,
    metrics: Dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """Create all available visualizations."""
    logger.info("Creating comprehensive visualizations...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create all visualizations
    visualizations = [
        (
            "selection_scatter.png",
            visualize_selection,
            {"method": config["evaluation"]["visualization_method"]},
        ),
        ("coverage_distribution.png", visualize_coverage_distribution, {}),
        ("diversity_matrix.png", visualize_diversity_matrix, {}),
        (
            "metrics_dashboard.png",
            visualize_selection_metrics_dashboard,
            {"metrics": metrics},
        ),
    ]

    for filename, viz_func, extra_kwargs in visualizations:
        try:
            output_path = output_dir / filename
            if viz_func == visualize_selection_metrics_dashboard:
                viz_func(output_path=str(output_path), **extra_kwargs)
            elif viz_func == visualize_selection:
                viz_func(embeddings, selected_indices, str(output_path), **extra_kwargs)
            else:
                viz_func(embeddings, selected_indices, str(output_path), **extra_kwargs)
            logger.info(f"✓ Created {filename}")
        except Exception as e:
            logger.warning(f"Failed to create {filename}: {e}")

    logger.info(f"All visualizations saved to: {output_dir}")


def main() -> int:
    """Command-line interface for running evaluation independently.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate and visualize diversity selection results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation on existing pipeline outputs
  python -m src.evaluation output/

  # Specify custom output directory for visualizations
  python -m src.evaluation output/ --viz-dir custom_viz/

  # Use different visualization method
  python -m src.evaluation output/ --method pca

  # Skip visualizations, only compute metrics
  python -m src.evaluation output/ --no-viz
        """,
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory containing embeddings_reduced.npy and selected_indices.npy",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: OUTPUT_DIR/visualizations)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=["tsne", "umap", "pca"],
        help="Dimensionality reduction method for visualization (default: tsne)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualizations, only compute and save metrics",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute metrics, don't create visualizations",
    )

    args = parser.parse_args()

    # Validate output directory
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return 1

    # Load required files
    embeddings_file = output_dir / "embeddings_reduced.npy"
    indices_file = output_dir / "selected_indices.npy"

    if not embeddings_file.exists():
        logger.error(f"Embeddings file not found: {embeddings_file}")
        return 1

    if not indices_file.exists():
        logger.error(f"Selected indices file not found: {indices_file}")
        return 1

    try:
        logger.info(f"Loading embeddings from: {embeddings_file}")
        embeddings = np.load(embeddings_file)

        logger.info(f"Loading selected indices from: {indices_file}")
        selected_idx = np.load(indices_file)

        # Validate data
        if embeddings.ndim != 2:
            logger.error(
                f"Invalid embeddings shape: {embeddings.shape}. Expected 2D array."
            )
            return 1

        if len(selected_idx) > len(embeddings):
            logger.error(
                f"More selected indices ({len(selected_idx)}) than embeddings ({len(embeddings)})"
            )
            return 1

        if len(selected_idx) > 0 and np.max(selected_idx) >= len(embeddings):
            logger.error(
                f"Selected index {np.max(selected_idx)} out of bounds for {len(embeddings)} embeddings"
            )
            return 1

        logger.info(
            f"Loaded: {embeddings.shape[0]:,} embeddings, {len(selected_idx):,} selected"
        )

        # Compute evaluation metrics
        logger.info("Computing evaluation metrics...")
        metrics = evaluate_selection(embeddings, selected_idx)

        # Save metrics
        metrics_file = output_dir / "evaluation_metrics.yaml"
        logger.info(f"Saving metrics to: {metrics_file}")

        import yaml

        with open(metrics_file, "w") as f:
            yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)

        # Create visualizations unless skipped
        if not args.no_viz and not args.metrics_only:
            viz_dir = Path(args.viz_dir) if args.viz_dir else output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Creating visualizations in: {viz_dir}")

            # Create config dict for visualization method
            config = {
                "evaluation": {
                    "visualization_method": args.method,
                }
            }

            create_all_visualizations(
                embeddings, selected_idx, metrics, viz_dir, config
            )

            logger.info(f"✓ Visualizations saved to: {viz_dir}")
        else:
            logger.info("Skipping visualizations (--no-viz or --metrics-only specified)")

        logger.info("=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Metrics saved to: {metrics_file}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":

    raise SystemExit(main())
