#!/bin/bash
# Quick-launch script for local (non-Slurm) wind model training
# Usage: bash scripts/train_wind.sh
#
# Requires uv: https://docs.astral.sh/uv/

# Step 1: Preprocess CSVs if needed
if [ ! -d "datasets/wind/train_A" ]; then
    echo "Preprocessing CSV data first..."
    uv run python preprocess_csv.py \
        --input_dir input_csv \
        --output_dir datasets/wind \
        --test_angles 270 315
fi

# Step 2: Train
uv run python train.py \
    --name wind_pix2pixHD \
    --dataroot datasets/wind \
    --dataset_mode wind \
    --label_nc 0 \
    --input_nc 4 \
    --output_nc 4 \
    --no_instance \
    --no_vgg_loss \
    --resize_or_crop none \
    --batchSize 4 \
    --niter 100 \
    --niter_decay 100 \
    --save_epoch_freq 10
