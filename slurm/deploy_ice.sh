#!/bin/bash

# ICE Deployment Wrapper for pix2pixHD Wind Comfort Training
# Usage: bash slurm/deploy_ice.sh --gpu [H100|A100|H200] [--ngpus 1|2] [--config config.toml]

GPU_TYPE="h200"
NUM_GPUS=1
CONFIG_FILE="config.toml"
SBATCH_SCRIPT="slurm/train.sbatch"

# Training hyperparameters (passed as environment variables)
BATCH_SIZE=4
LR="0.0002"
NITER=100
NITER_DECAY=100

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_TYPE="$2"; shift ;;
        --ngpus) NUM_GPUS="$2"; shift ;;
        --config) CONFIG_FILE="$2"; shift ;;
        --augmented) SBATCH_SCRIPT="slurm/train_augmented.sbatch" ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --niter) NITER="$2"; shift ;;
        --niter-decay) NITER_DECAY="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Convert to lowercase
GPU_TYPE_LOWER=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')

# Map to correct ICE gres names
case $GPU_TYPE_LOWER in
    h200) SLURM_GPU="H200:${NUM_GPUS}" ;;
    h100) SLURM_GPU="H100:${NUM_GPUS}" ;;
    a100) SLURM_GPU="A100:${NUM_GPUS}" ;;
    v100) SLURM_GPU="V100:${NUM_GPUS}" ;;
    *) SLURM_GPU="${GPU_TYPE}:${NUM_GPUS}" ;;
esac

# Read ICE config from config.toml
if [ -f "$CONFIG_FILE" ]; then
    ICE_ACCOUNT=$(sed -n '/^\[ice\]/,/^\[/p' "$CONFIG_FILE" | grep -E "^account\s*=" | sed 's/.*=\s*"\(.*\)".*/\1/' | tr -d ' ')
    ICE_PARTITION=$(sed -n '/^\[ice\]/,/^\[/p' "$CONFIG_FILE" | grep -E "^partition\s*=" | sed 's/.*=\s*"\(.*\)".*/\1/' | tr -d ' ')
else
    echo "Warning: $CONFIG_FILE not found."
    ICE_ACCOUNT="coa"
    ICE_PARTITION=""
fi

# Fallback if parsing failed
if [ -z "$ICE_ACCOUNT" ]; then
    ICE_ACCOUNT="coa"
fi

echo "=========================================="
echo " Preparing ICE Deployment (pix2pixHD Wind)"
echo " GPU Requested: $GPU_TYPE x $NUM_GPUS ($SLURM_GPU)"
echo " Config file: $CONFIG_FILE"
echo " Account: $ICE_ACCOUNT"
echo " Batch Size: $BATCH_SIZE"
echo " Learning Rate: $LR"
echo " Iterations: $NITER + $NITER_DECAY decay"
echo "=========================================="

mkdir -p logs

# Build sbatch command
SBATCH_CMD="sbatch --gres=gpu:$SLURM_GPU --account=$ICE_ACCOUNT --export=NUM_GPUS=$NUM_GPUS,BATCH_SIZE=$BATCH_SIZE,LR=$LR,NITER=$NITER,NITER_DECAY=$NITER_DECAY"

if [ -n "$ICE_PARTITION" ]; then
    SBATCH_CMD="$SBATCH_CMD --partition=$ICE_PARTITION"
fi

SBATCH_CMD="$SBATCH_CMD $SBATCH_SCRIPT"

# Submit
$SBATCH_CMD

echo "------------------------------------------"
echo "Job submitted on ICE. Check status with: squeue -u $USER"
