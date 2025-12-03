# Car Diversity Selection Pipeline

A pipeline for selecting diverse images from large dataset using deep feature extraction and submodular diversity optimization.

## Overview

This pipeline implements a method for curating high-quality, diverse image datasets, combining:

- **Quality filtering**: Multi-stage filtering for resolution, blur, quality, and content
- **Deep feature extraction**: DINOv2 ViT-L embeddings for visual similarity
- **Diversity optimization**: Hybrid k-means + farthest point sampling for maximum coverage
- **GPU acceleration**: FAISS and cuML for efficient large-scale processing

---

## Why Diversity Matters More Than Scale

Recent research challenges the "more data is better" paradigm. Sorscher et al.'s NeurIPS 2022 paper demonstrates that strategic data pruning can shift model performance from power-law to exponential scaling curves. The **DataComp benchmark** (NeurIPS 2023) showed that training CLIP on carefully curated subsets achieved **79.2% ImageNet zero-shot accuracy**, outperforming models trained on larger datasets.

For automotive segmentation specifically, diversity ensures the network encounters varied:

- **Viewpoints**: front, rear, side, three-quarter angles
- **Part visibility**: full vehicles, close-ups of doors/hoods/trunks, partial occlusions
- **Conditions**: lighting variations, weather, image quality degradation
- **Vehicle types**: sedans, SUVs, trucks, damaged vehicles with dents/scratches

Without explicit diversity optimization, random sampling from scraped data typically overrepresents common scenarios (front-quarter views of clean sedans in daylight) while underrepresenting edge cases critical for real-world generalization.

---

## Project Structure

```
car_diversity/
├── pixi.toml                      # Pixi environment configuration
├── config.yaml                    # Pipeline configuration
├── main.py                        # Main entry point
├── run.sh                         # Convenience script
├── example_usage.py               # Usage examples
├── .gitignore                     # Git ignore rules
│
└── src/                           # Source code
    ├── __init__.py                # Package initialization
    ├── pipeline.py                # Main pipeline orchestration
    ├── quality_filter.py          # Quality filtering module
    ├── feature_extractor.py       # DINOv2/CLIP feature extraction
    ├── diversity_selector.py      # Diversity selection algorithms
    ├── evaluation.py              # Evaluation metrics
    └── utils.py                   # Utility functions
```

---

## Requirements

- NVIDIA GPU (RTX 4090/5090 recommended, requires CUDA-capable GPU)
- Linux (tested on Ubuntu 22.04+)
- [Pixi](https://pixi.sh/) for environment management

---

## Installation

### Quick Install (Recommended)

```bash
# 1. Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Install dependencies
pixi install

# 3. Test your setup
pixi run test
```

**Expected output:**
```
PyTorch: 2.9.1+cu128
CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 5090
```

### Environment Details

- **Python:** 3.10
- **PyTorch:** 2.9.1 with CUDA 12.8 support
- **GPU Support:** CUDA-enabled via pip
- **Key Packages:** DINOv2 (via transformers), FAISS-GPU, cuML/cuDF, OpenCV, scikit-learn, pandas

### Verifying Installation

```bash
pixi run python -c "
import torch
import torchvision
import transformers
import faiss
import cv2
import sklearn
print('All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### Manual Installation (Alternative)

If Pixi doesn't work, use regular pip:

```bash
python3.10 -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow tqdm scikit-learn opencv-python pyyaml pandas matplotlib seaborn
pip install faiss-gpu transformers accelerate pyiqa ftfy timm
pip install cuml-cu12 cudf-cu12

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Quick Start

### 1. Setup (One-time)

```bash
./run.sh setup
```

### 2. Test Your Setup

```bash
./run.sh test
```

### 3. Run Your First Selection

**Option A: Quick Test (5-10 minutes)**

```bash
./run.sh quick test_images
```

**Option B: Full Pipeline**

```bash
./run.sh run /path/to/your/million/images ./output 30000
```

### 4. Check Results

```bash
ls -lh output/

# Key files:
# - selected_images.txt           (list of selected paths)
# - selected_images_symlinks/     (symlinks to images)
# - evaluation_metrics.yaml       (quality metrics)
# - selection_visualization.png   (t-SNE plot)
```

---

## Pipeline Stages

### Stage 1: Quality Filtering

**Input:** raw images  
**Output:** clean images

Removes low-quality images via:
- Resolution filter
- Aspect ratio filter
- BRISQUE quality assessment
- Blur detection (Laplacian variance)
- CLIP-based car content validation ** Not working currently

### Stage 2: Feature Extraction

**Input:** N clean images  
**Output:** N × 1024 embeddings

**DINOv2 ViT-L/14** is the optimal choice for diversity-based selection. Unlike supervised models biased toward classification-relevant features, DINOv2's self-supervised training produces embeddings that capture both low-level visual properties and high-level semantic structure. Its **KoLeo regularizer** specifically encourages uniform, well-distributed feature representations—directly beneficial for diversity sampling.

**Multi-layer Feature Fusion (Optional):**

Different transformer layers capture different diversity dimensions:
- **Layers 1-4**: Texture, edges, local patterns (lighting/quality diversity)
- **Layers 5-8**: Parts and structural elements (compositional diversity)
- **Layers 9-12**: Semantic concepts (car type/scene diversity)

Enable with `multilayer: true` in config to concatenate features from multiple layers, then reduce via PCA.

### Stage 3: Dimensionality Reduction

**Input:** N × 1024 embeddings  
**Output:** N × 512 embeddings

PCA reduction for faster downstream processing.

### Stage 4: Deduplication

**Input:** possible duplicate images  
**Output:** unique images

FAISS-based near-duplicate removal:
- Cosine similarity threshold (default: 0.95)
- Preserves first occurrence

### Stage 5: Diversity Selection

**Input:** unique images  
**Output:** N diverse images

**Mathematical Foundations:**

The selection optimizes two complementary objectives:
- **Facility Location** (coverage): Measures how well the selected subset represents the full dataset. Greedy maximization achieves (1 - 1/e) ≈ 0.632 approximation guarantee.
- **k-Center** (minimax radius): Minimizes the maximum distance from any point to its nearest selected representative. Greedy farthest-point sampling achieves 2-approximation guarantee.

**Algorithm Comparison:**

| Algorithm                | Time Complexity | GPU Support | Guarantee      |
| ------------------------ | --------------- | ----------- | -------------- |
| Random sampling          | O(k)            | N/A         | None           |
| K-means + medoids        | O(n·k·d·iter)   | ✅ cuML      | None           |
| Farthest Point Sampling  | O(n·k)          | ✅ FAISS     | 2-approx       |
| Facility Location Greedy | O(n·k)          | ✅ SubModLib | (1-1/e)-approx |
| Exact DPP                | O(n³)           | ❌           | Optimal        |

**Selection Methods:**

| Method             | Time    | Quality      | Description                                                  |
| ------------------ | ------- | ------------ | ------------------------------------------------------------ |
| `hybrid` (default) | Medium  | Near-optimal | K-means clustering + FPS within clusters + global refinement |
| `fps`              | Long    | Best         | Pure farthest point sampling (2-approx guarantee)            |
| `kmeans`           | Short   | Good         | Cluster medoids                                              |
| `random`           | Instant | Baseline     | Random sampling                                              |

**Hybrid Method Details:**
1. **Stratified clustering**: K-means into N clusters
2. **Intra-cluster FPS**: Diverse selection within each cluster
3. **Global refinement**: Cross-cluster FPS to fill coverage gaps

### Stage 6: Evaluation

Computes diversity metrics and generates visualizations (t-SNE/PCA plots).

---

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
input:
  image_dir: "./images"

diversity_selection:
  n_select: 30000
  method: "hybrid"  # Options: random, kmeans, fps, hybrid

hardware:
  device: "cuda"
  batch_size: 64     # Increase for RTX 5090 (try 128)
  use_fp16: true
  vram_limit_gb: 32  # Set to your GPU VRAM

quality_filter:
  brisque_threshold: 45.0       # Lower = stricter
  blur_threshold: 100.0         # Higher = stricter
  car_confidence_threshold: 0.6 # Higher = stricter

feature_extraction:
  model_name: "dinov2-large"  # small/base/large/giant
  multilayer: false
  use_clip: false

dimensionality_reduction:
  n_components: 512

deduplication:
  threshold: 0.95  # Higher = more aggressive
```

---

## Usage Examples

### 1. Standard Run (30K images)

```bash
pixi run python main.py \
  --image-dir ./images \
  --output-dir ./output \
  --n-select 30000 \
  --method hybrid
```

### 2. Quick Test Run

```bash
pixi run python main.py \
  --image-dir ./test_images \
  --n-select 1000 \
  --method random \
  --skip-quality-filter
```

### 3. Maximum Quality (Pure FPS)

```bash
pixi run python main.py \
  --image-dir /data/car_images \
  --n-select 30000 \
  --method fps
```

### 4. Fast Processing (Pre-filtered images)

```bash
pixi run python main.py \
  --image-dir /data/pre_filtered_cars \
  --n-select 30000 \
  --skip-quality-filter \
  --skip-deduplication
```

### 5. Custom Configuration

```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml...
pixi run python main.py --config my_config.yaml
```

### 6. Using as a Python Library

```python
from src.pipeline import DiverseCarImageSelector

selector = DiverseCarImageSelector(
    vram_limit_gb=32,
    batch_size=128,
    device="cuda"
)

selected_images = selector.run_pipeline(
    image_dir="/path/to/images",
    n_select=30000,
    output_dir="./output",
    selection_method="hybrid"
)

print(f"Selected {len(selected_images)} images")
```

### 7. Comparing Selection Methods

```python
from src.diversity_selector import DiversitySelector
from src.evaluation import compare_selection_methods

selector = DiversitySelector(device="cuda")

selections = {
    'random': selector.select_diverse(embeddings, 30000, method='random'),
    'kmeans': selector.select_diverse(embeddings, 30000, method='kmeans'),
    'hybrid': selector.select_diverse(embeddings, 30000, method='hybrid'),
}

results = compare_selection_methods(embeddings, selections)
```

---

## Output Files

```
output/
├── filtered_images.txt              # Images passing quality filter
├── deduplicated_images.txt          # Images after deduplication
├── selected_images.txt              # Final selected images
├── selected_indices.npy             # Indices of selected images
├── embeddings_raw.npy               # Raw DINOv2 embeddings
├── embeddings_reduced.npy           # PCA-reduced embeddings
├── pca_model.pkl                    # Trained PCA model
├── evaluation_metrics.yaml          # Diversity metrics
├── selection_visualization.png      # t-SNE plot
├── selected_images_symlinks/        # Symlinks to selected images
└── selected_images/                 # Copied images (if enabled)
```

---

## Evaluation Metrics

### Coverage Metrics
- `coverage_radius_max`: Maximum distance to nearest selected point
- `coverage_radius_mean`: Average coverage quality
- **Lower is better** (tighter coverage)

### Diversity Metrics
- `intra_diversity_mean`: Average pairwise distance within selection
- `intra_diversity_std`: Diversity variance
- **Higher is better** (more diverse)

### Interpretation

Good selection should have:
- Low coverage radius (< 1.0 for normalized embeddings)
- High intra-diversity (> 5.0 for normalized embeddings)

### Expected Performance by Method

| Method                   | Coverage Radius | Guarantee      |
| ------------------------ | --------------- | -------------- |
| Random                   | Baseline        | None           |
| K-means                  | ~1.5x worse     | None           |
| Pure FPS                 | Best            | 2-approx       |
| Facility Location        | ~1.2x worse     | (1-1/e)-approx |
| **Hybrid (recommended)** | Near-best       | Combined       |

---

## FAISS Configuration Note

### Current Setup: CPU-Only FAISS

Due to CUDA/cuBLAS compatibility issues between FAISS-GPU 1.7.x and CUDA 12.8+/13.0, the pipeline currently uses **FAISS-CPU**.

**Affected Operations:**
- Deduplication (Stage 4)
- Diversity selection (Stage 5)

**Performance Impact:**
- CPU FAISS is slower than GPU FAISS for large datasets

---

## Troubleshooting

### CUDA Not Available

```bash
# Check GPU driver
nvidia-smi

# Reinstall environment
rm -rf .pixi pixi.lock
pixi install
pixi run test
```

### CUDA Out of Memory

```bash
# Reduce batch size
pixi run python main.py --batch-size 32
```

Or edit `config.yaml`:
```yaml
hardware:
  batch_size: 32
```

### Slow Processing

```bash
# Check GPU utilization
nvidia-smi

# Increase batch size if VRAM is underutilized
pixi run python main.py --batch-size 128
```

### Quality Filter Too Aggressive

```yaml
quality_filter:
  brisque_threshold: 60.0      # Increased (less strict)
  blur_threshold: 50.0         # Decreased (less strict)
```

### Import Errors

```bash
pixi run pip install --force-reinstall torch torchvision

# Or reinstall all
rm pixi.lock && pixi install
```
