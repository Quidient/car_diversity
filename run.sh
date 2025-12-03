#!/bin/bash
# Convenience script for running the car diversity selection pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if pixi is installed
if ! command -v pixi &> /dev/null; then
    print_error "Pixi is not installed. Please install it first:"
    echo "  curl -fsSL https://pixi.sh/install.sh | bash"
    exit 1
fi

# Commands
case "${1:-help}" in
    setup)
        print_info "Setting up environment..."
        pixi install
        print_info "Running system test..."
        pixi run test
        print_info "Setup complete! You can now run: ./run.sh help"
        ;;

    test)
        print_info "Testing system..."
        pixi run test
        ;;

    run)
        if [ -z "$2" ]; then
            print_error "Please specify image directory"
            echo "Usage: ./run.sh run <image_dir> [output_dir] [n_select]"
            exit 1
        fi

        IMAGE_DIR="$2"
        OUTPUT_DIR="${3:-./output}"
        N_SELECT="${4:-30000}"

        print_info "Running pipeline..."
        print_info "  Image directory: $IMAGE_DIR"
        print_info "  Output directory: $OUTPUT_DIR"
        print_info "  Target selection: $N_SELECT"

        pixi run python main.py \
            --image-dir "$IMAGE_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --n-select "$N_SELECT"
        ;;

    quick)
        if [ -z "$2" ]; then
            print_error "Please specify image directory"
            echo "Usage: ./run.sh quick <image_dir>"
            exit 1
        fi

        IMAGE_DIR="$2"

        print_info "Running quick test (random selection, 1000 images)..."
        pixi run python main.py \
            --image-dir "$IMAGE_DIR" \
            --output-dir ./output/quick_test \
            --n-select 1000 \
            --method random \
            --skip-quality-filter
        ;;

    compare)
        if [ -z "$2" ]; then
            print_error "Please specify image directory"
            echo "Usage: ./run.sh compare <image_dir> [n_select]"
            exit 1
        fi

        IMAGE_DIR="$2"
        N_SELECT="${3:-5000}"

        print_info "Comparing selection methods on $N_SELECT images..."

        for METHOD in random kmeans fps hybrid; do
            print_info "Testing method: $METHOD"
            pixi run python main.py \
                --image-dir "$IMAGE_DIR" \
                --output-dir "./output/compare_$METHOD" \
                --n-select "$N_SELECT" \
                --method "$METHOD" \
                --skip-quality-filter
        done

        print_info "Comparison complete! Check output/compare_*/ for results"
        ;;

    filter-only)
        if [ -z "$2" ]; then
            print_error "Please specify image directory"
            echo "Usage: ./run.sh filter-only <image_dir>"
            exit 1
        fi

        IMAGE_DIR="$2"

        print_info "Running quality filter only..."
        pixi run python example_usage.py --example 5
        ;;

    example)
        if [ -z "$2" ]; then
            print_error "Please specify example number (1-7)"
            echo "Usage: ./run.sh example <number>"
            echo ""
            echo "Available examples:"
            echo "  1 - Basic pipeline"
            echo "  2 - Custom feature extraction"
            echo "  3 - Compare selection methods"
            echo "  4 - Incremental selection"
            echo "  5 - Quality filter only"
            echo "  6 - Evaluation and visualization"
            echo "  7 - Batch processing"
            exit 1
        fi

        EXAMPLE_NUM="$2"
        print_info "Running example $EXAMPLE_NUM..."
        pixi run python example_usage.py --example "$EXAMPLE_NUM"
        ;;

    clean)
        print_warn "Cleaning output directory..."
        rm -rf output/
        print_info "Output directory cleaned"
        ;;

    gpu-info)
        print_info "GPU Information:"
        nvidia-smi
        ;;

    help|*)
        echo "Car Diversity Selection Pipeline"
        echo ""
        echo "Usage: ./run.sh <command> [arguments]"
        echo ""
        echo "Commands:"
        echo "  setup                           - Install dependencies and setup environment"
        echo "  test                            - Test system (GPU, PyTorch, etc.)"
        echo "  run <image_dir> [output] [n]    - Run full pipeline"
        echo "  quick <image_dir>               - Quick test with random sampling"
        echo "  compare <image_dir> [n]         - Compare all selection methods"
        echo "  filter-only <image_dir>         - Run quality filter only"
        echo "  example <1-7>                   - Run example script"
        echo "  clean                           - Clean output directory"
        echo "  gpu-info                        - Show GPU information"
        echo "  help                            - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run.sh setup"
        echo "  ./run.sh run ./my_images ./output 30000"
        echo "  ./run.sh quick ./test_images"
        echo "  ./run.sh compare ./my_images 5000"
        echo "  ./run.sh example 1"
        echo ""
        ;;
esac
