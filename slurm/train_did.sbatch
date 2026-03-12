#!/bin/bash
#SBATCH -J wind_did
#SBATCH -A gts-pkastner3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH -t 11:50:00
#SBATCH -o logs/wind_did_%j.out
#SBATCH -e logs/wind_did_%j.err
#SBATCH --gres=gpu:H200:1

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Go to the directory where sbatch was submitted from
cd $SLURM_SUBMIT_DIR

# Ensure logs directory exists
mkdir -p logs

# Detect and remove Windows virtual environment if uploaded
if [ -d ".venv/Scripts" ]; then
    echo "Detected Windows .venv (Scripts folder exists). Removing it..."
    rm -rf .venv
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "Setting up environment..."
uv sync

# ── Step 1: Preprocess with DID features ──────────────────────────────
# We use a distinct folder so we don't mess with the standard model data
DATAROOT="datasets/wind_did"

if [ ! -d "$DATAROOT/train_A" ]; then
    echo "Generating DID-augmented dataset..."
    uv run python preprocess_did.py \
        --input_dir input_csv \
        --output_dir $DATAROOT \
        --test_angles 270 315
fi

# ── Step 2: Train pix2pixHD with 16-channel input ─────────────────────
echo "Starting DID-augmented training..."
uv run python train.py \
    --name wind_pix2pixHD_did \
    --dataroot $DATAROOT \
    --dataset_mode wind \
    --label_nc 0 \
    --input_nc 16 \
    --output_nc 1 \
    --no_instance \
    --no_vgg_loss \
    --resize_or_crop none \
    --batchSize 4 \
    --niter 500 \
    --niter_decay 0 \
    --lr 0.0002 \
    --save_epoch_freq 5 \
    --display_freq 10

echo "Job finished on $(date)"
