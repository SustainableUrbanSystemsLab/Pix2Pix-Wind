#!/bin/bash
#SBATCH -J wind_test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err

# SBATCH Directives for ICE GPU
#SBATCH --gres=gpu:H200:1
#SBATCH --account=coa
#SBATCH --partition=ice-gpu

echo "Inference test started on $(date)"
echo "Running on node $(hostname)"

# Ensure logs directory exists
mkdir -p logs

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "Setting up environment..."
uv sync

echo "Starting pix2pixHD inference test..."
uv run python test.py \
    --name wind_pix2pixHD \
    --dataroot datasets/wind \
    --dataset_mode wind \
    --label_nc 0 \
    --input_nc 4 \
    --output_nc 4 \
    --no_instance \
    --resize_or_crop none \
    --which_epoch ${EPOCH:-latest} \
    --how_many ${NUM_SAMPLES:-50}

echo "Inference test finished on $(date)"
