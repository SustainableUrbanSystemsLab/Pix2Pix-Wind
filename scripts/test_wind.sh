#!/bin/bash
# Quick-launch script for local (non-Slurm) wind model testing
# Usage: bash scripts/test_wind.sh [epoch]
#
# Requires uv: https://docs.astral.sh/uv/

EPOCH="${1:-latest}"

uv run python test.py \
    --name wind_pix2pixHD \
    --dataroot datasets/wind \
    --dataset_mode wind \
    --label_nc 0 \
    --input_nc 4 \
    --output_nc 4 \
    --no_instance \
    --resize_or_crop none \
    --which_epoch "$EPOCH" \
    --how_many 50
