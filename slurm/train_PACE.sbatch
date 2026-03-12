#!/bin/bash
#SBATCH -J wind_p2p
#SBATCH -A gts-pkastner3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH -t 11:50:00
#SBATCH -o logs/wind_p2p_%j.out
#SBATCH -e logs/wind_p2p_%j.err
#SBATCH --gres=gpu:H200:1

echo "Job started on $(date)"
echo "Running on node $(hostname)"
echo "Number of GPUs: ${NUM_GPUS:-1}"

# Go to the directory where sbatch was submitted from
cd $SLURM_SUBMIT_DIR

# Ensure logs directory exists
mkdir -p logs

# CRITICAL: Detect and remove Windows virtual environment if uploaded
if [ -d ".venv/Scripts" ]; then
    echo "Detected Windows .venv (Scripts folder exists). Removing it to create Linux compatible environment..."
    rm -rf .venv
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies (creates venv based on pyproject.toml)
echo "Setting up environment..."
uv sync

# ── Step 1: Preprocess CSVs if dataset doesn't exist ──────────────────
if [ ! -d "datasets/wind/train_A" ]; then
    if [ ! -d "input_csv" ]; then
        echo "ERROR: Neither 'datasets/wind/train_A' nor 'input_csv' were found."
        echo "Please upload either your raw CSV files to 'input_csv/' or the preprocessed dataset to 'datasets/' before training."
        exit 1
    fi
    echo "Preprocessing CSV data..."
    uv run python preprocess_csv.py \
        --input_dir input_csv \
        --output_dir datasets/wind \
        --test_angles 270 315
    
    # Verify it actually produced data
    if [ ! -d "datasets/wind/train_A" ]; then
        echo "ERROR: Preprocessing failed to generate datasets/wind/train_A."
        exit 1
    fi
fi

# ── Step 2: Train pix2pixHD on wind data ──────────────────────────────
echo "Starting pix2pixHD training for Wind Comfort (Pix2Pix Format)..."
uv run python train.py \
    --name wind_pix2pixHD_p2p \
    --dataroot datasets/wind \
    --dataset_mode wind \
    --label_nc 0 \
    --input_nc 8 \
    --output_nc 1 \
    --no_instance \
    --no_vgg_loss \
    --resize_or_crop none \
    --batchSize ${BATCH_SIZE:-4} \
    --niter ${NITER:-500} \
    --niter_decay ${NITER_DECAY:-0} \
    --lr ${LR:-0.0002} \
    --save_epoch_freq 5 \
    --display_freq 1175

echo "Job finished on $(date)"
