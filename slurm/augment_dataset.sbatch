#!/bin/bash
#SBATCH -J augment_wind
#SBATCH -A coa
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o augment_%j.out

# Load environment
cd $SLURM_SUBMIT_DIR

echo "Starting dataset augmentation Slurm job..."

# Install uv if not available
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# Paths
INPUT="${1:-datasets/wind}"
OUTPUT="${2:-datasets/wind_augmented}"

echo "Input: $INPUT"
echo "Output: $OUTPUT"

# Run preprocessing with augmentation (re-preprocess CSVs)
uv run python preprocess_csv.py \
    --input_dir input_csv \
    --output_dir "$OUTPUT" \
    --test_angles 270 315

echo "Job finished."
