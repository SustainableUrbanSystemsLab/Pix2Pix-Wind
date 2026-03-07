#!/bin/bash
#SBATCH -J wind_preproc
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/preproc_%j.out
#SBATCH -e logs/preproc_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# CD to submission dir
cd $SLURM_SUBMIT_DIR

# 1. Install/Update Environment
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# 2. Define Paths
INPUT_DIR="${1:-input_csv}"
OUTPUT_DIR="${2:-datasets/wind}"
TEST_ANGLES="${3:-270 315}"

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Test angles: $TEST_ANGLES"

# 3. Run Preprocessing
echo "Starting data preprocessing..."
uv run python preprocess_csv.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --test_angles $TEST_ANGLES

echo "Job finished on $(date)"
