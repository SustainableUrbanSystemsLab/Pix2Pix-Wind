#!/bin/bash

# Usage: bash slurm/deploy_vis.sh [epoch]
# Example: bash slurm/deploy_vis.sh latest

EPOCH="${1:-latest}"
ACCOUNT="coa"
GPU="H200:1"

echo "Deploying Visualization Job..."
echo "Epoch: $EPOCH"

mkdir -p logs

sbatch --account=$ACCOUNT --gres=gpu:$GPU --export=EPOCH="$EPOCH" slurm/visualize.sbatch

echo "Submitted. Check output in logs/vis_*.out"
