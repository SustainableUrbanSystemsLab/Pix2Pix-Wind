#!/bin/bash
#SBATCH -J wind_vis
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o logs/vis_%j.out
#SBATCH -e logs/vis_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Go to the directory where sbatch was submitted from
cd $SLURM_SUBMIT_DIR

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

echo "Starting pix2pixHD Visualization..."
uv run python test.py \
    --name wind_pix2pixHD \
    --dataroot datasets/wind \
    --dataset_mode wind \
    --label_nc 0 \
    --input_nc 4 \
    --output_nc 4 \
    --no_instance \
    --resize_or_crop none \
    --results_dir ${OUTPUT_DIR:-results/vis_inference} \
    --which_epoch ${EPOCH:-latest} \
    --how_many ${NUM_SAMPLES:-10}

echo "Job finished on $(date)"
