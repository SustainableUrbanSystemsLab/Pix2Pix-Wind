"""
preprocess_did.py
=================
Converts Wind Comfort CFD CSV files into paired 2D NumPy grids
WITH DID (Directional Integrated Distance) features included.

Outputs:
- train_A / test_A: shape (H, W, 16) -- [Original 8, DID 8]
- train_B / test_B: shape (H, W, 1)  -- [mag_U]

Usage:
    python preprocess_did.py --input_dir input_csv --output_dir datasets/wind_did --test_angles 270 315
"""
import argparse
import json
import os
import glob
import re
import time

import numpy as np
import pandas as pd
from compute_did import compute_did_features

# ── Column definitions ─────────────────────────────────────────────────────
INPUT_COLS = ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_local", "Y_local", "dir_sin", "dir_cos"]
OUTPUT_COLS = ["mag_U"]
DID_COLS = [f"DID_{i}" for i in range(8)]

def extract_angle(filename: str) -> int | None:
    m = re.search(r"ML_FormFlux_1_(\d+)\.csv$", os.path.basename(filename))
    return int(m.group(1)) if m else None

def csv_to_grid_did(csv_path: str):
    df = pd.read_csv(csv_path)

    if "X_local" not in df.columns:
        df["X_local"] = df["X"] - df["X"].mean()
    if "Y_local" not in df.columns:
        df["Y_local"] = df["Y"] - df["Y"].mean()

    xs = np.sort(df["X"].unique())
    ys = np.sort(df["Y"].unique())
    H, W = len(ys), len(xs)

    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: j for j, y in enumerate(ys)}

    input_grid = np.zeros((H, W, len(INPUT_COLS)), dtype=np.float32)
    output_grid = np.zeros((H, W, len(OUTPUT_COLS)), dtype=np.float32)

    # Fill basic grids
    for _, row in df.iterrows():
        xi = x_to_idx[row["X"]]
        yi = y_to_idx[row["Y"]]
        for ch, col in enumerate(INPUT_COLS):
            input_grid[yi, xi, ch] = row[col]
        for ch, col in enumerate(OUTPUT_COLS):
            output_grid[yi, xi, ch] = row[col]

    # --- Compute DID features ---
    # Derive mask: SDF <= 0 is obstacle
    # Assuming first column of INPUT_COLS is SDF
    sdf = input_grid[:, :, 0]
    mask = sdf <= 0
    
    start_time = time.time()
    did_grid = compute_did_features(mask) # (H, W, 8)
    # Normalize DID to [0, 1] for better learning?
    # Original paper often uses log or just standard normalization.
    # We will let the stats collector handle it if we want, 
    # but DID can be very large. We'll clip at some max distance.
    max_d = np.sqrt(H**2 + W**2)
    did_grid = np.clip(did_grid / (max_d / 4.0), 0, 1) # Heuristic scaling
    
    elapsed = time.time() - start_time
    
    # Concatenate [Standard, DID]
    full_input = np.concatenate([input_grid, did_grid], axis=-1) # (H, W, 16)

    return full_input, output_grid, elapsed

def compute_stats(arrays: list[np.ndarray]) -> dict:
    stacked = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
    return {
        "min": stacked.min(axis=0).tolist(),
        "max": stacked.max(axis=0).tolist(),
    }

def main():
    parser = argparse.ArgumentParser(description="Preprocess Wind CSV with DID features")
    parser.add_argument("--input_dir", type=str, default="input_csv")
    parser.add_argument("--output_dir", type=str, default="datasets/wind_did")
    parser.add_argument("--test_angles", type=int, nargs="+", default=[270, 315])
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "ML_FormFlux_1_*.csv")
    all_csvs = sorted(glob.glob(pattern))
    gt_csvs = [f for f in all_csvs if "_pred" not in os.path.basename(f)]

    if not gt_csvs:
        print(f"ERROR: No CSV files found matching {pattern}")
        return

    print(f"Found {len(gt_csvs)} ground-truth CSV files. Computing DID...")

    for split in ["train", "test"]:
        for suffix in ["_A", "_B"]:
            os.makedirs(os.path.join(args.output_dir, split + suffix), exist_ok=True)

    all_inputs = []
    all_outputs = []
    file_info = []

    for csv_path in gt_csvs:
        angle = extract_angle(csv_path)
        if angle is None: continue

        split = "test" if angle in args.test_angles else "train"
        print(f"  Angle {angle} -> {split} set...")

        input_grid, output_grid, dt = csv_to_grid_did(csv_path)
        print(f"    DID compute time: {dt:.2f}s")
        
        all_inputs.append(input_grid)
        all_outputs.append(output_grid)

        a_path = os.path.join(args.output_dir, f"{split}_A", f"{angle}.npy")
        b_path = os.path.join(args.output_dir, f"{split}_B", f"{angle}.npy")
        np.save(a_path, input_grid)
        np.save(b_path, output_grid)

        file_info.append({
            "angle": angle,
            "split": split,
            "shape": list(input_grid.shape[:2]),
            "input_channels": input_grid.shape[-1],
            "did_compute_time": dt
        })

    input_stats = compute_stats(all_inputs)
    output_stats = compute_stats(all_outputs)

    stats = {
        "input_columns": INPUT_COLS + DID_COLS,
        "input_stats": input_stats,
        "output_stats": output_stats,
        "files": file_info,
    }

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Saved DID-augmented dataset to {args.output_dir}")

if __name__ == "__main__":
    main()
