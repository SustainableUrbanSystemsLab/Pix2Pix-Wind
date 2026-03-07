# 🌬️ Pix2PixHD for Wind Comfort ML

This repository contains a heavily stripped-down and specialized version of the original [NVIDIA pix2pixHD](https://github.com/NVIDIA/pix2pixHD) architecture, adapted specifically for **Wind Comfort Machine Learning Research**.

Unlike traditional image-to-image translation (which operates on 3-channel RGB images), this pipeline has been modified to map **8-channel physical geometry inputs** mapping to **1-channel wind deficit predictions**.

## 📊 Dataset Specification

The model expects inputs and outputs formatted as `.npy` arrays tightly normalized around `[-1, 1]` for the generator.

### Input Channels (8)
| Idx | Channel | Description |
| :--- | :--- | :--- |
| 0 | `SDF` | Signed Distance Field (distance to nearest wall) |
| 1 | `Bldg_height` | Height of the building at the specific pixel |
| 2 | `Z_relative` | Height slice of the CFD simulation |
| 3 | `U_over_Uref` | Background wind ratio (inlet profile) |
| 4 | `X_local` | X distance from building center |
| 5 | `Y_local` | Y distance from building center |
| 6 | `dir_sin` | Sine of Wind Direction |
| 7 | `dir_cos` | Cosine of Wind Direction |

### Output Target (1)
| Idx | Channel | Description |
| :--- | :--- | :--- |
| 0 | `mag_U` | Target Wake Deficit |

---

## 🚀 Quickstart & Scripts

We use `uv` for lightning-fast Python dependency management. Make sure `uv` is installed, and the environment will auto-sync.

### 1. Data Preprocessing
If you have new CFD raw `.csv` results in `input_csv/`, run the preprocessing script to generate the proper 8-channel input and 1-channel output arrays:
```bash
uv run preprocess_csv.py
```
*This will populate the `datasets/wind/` directory and compile a `stats.json` for normalization.*

### 2. Local Training
To test the model architecture locally (defaulting to CPU if no CUDA is available):
```bash
# Trains for 25 epochs, saving a checkpoint every 5 epochs
uv run python train.py --name pix2pix --dataset_mode wind --dataroot datasets/wind --input_nc 8 --output_nc 1 --niter 25 --niter_decay 0 --save_epoch_freq 5 --label_nc 0 --no_instance --display_freq 30 --gpu_ids -1 --nThreads 0
```

### 3. Training on HPC (PACE)
To submit a full-scale job on an HPC Slurm cluster (leveraging H200 GPUs):
```bash
sbatch slurm/train_PACE.sbatch
```

### 4. Visualizing Progress
To generate a side-by-side comparison GIF (Ground Truth vs. Model Prediction) of your training progress, run:
```bash
uv run python make_gif.py
```
*Outputs to `training_progress_comparison.gif`.*

---

## 🧹 Repository Cleanup
*Note: All legacy dataloaders for Cityscapes/Faces, unused TensorRT inference endpoints, and 1024p bash scripts have been purged to keep this repository clean and strictly focused on wind engineering datasets.*
