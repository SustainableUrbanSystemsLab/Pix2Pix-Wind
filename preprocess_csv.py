"""
preprocess_csv.py
=================
Converts Wind Comfort CFD CSV files into paired 2D NumPy grids
suitable for pix2pixHD training.

Usage:
    python preprocess_csv.py --input_dir input_csv --output_dir datasets/wind --test_angles 270 315
"""
import argparse
import json
import os
import glob
import re

import numpy as np
import pandas as pd


# ── Column definitions ─────────────────────────────────────────────────────
INPUT_COLS = ["SDF", "Bldg_height", "Z_relative", "U_over_Uref", "X_local", "Y_local", "dir_sin", "dir_cos"]
OUTPUT_COLS = ["mag_U"]


def extract_angle(filename: str) -> int | None:
    """Extract the wind angle from a filename like ML_FormFlux_1_45.csv."""
    m = re.search(r"ML_FormFlux_1_(\d+)\.csv$", os.path.basename(filename))
    return int(m.group(1)) if m else None


def csv_to_grid(csv_path: str):
    """
    Read a CSV file and pivot into 2D grids.

    Returns
    -------
    input_grid  : np.ndarray, shape (H, W, len(INPUT_COLS))
    output_grid : np.ndarray, shape (H, W, len(OUTPUT_COLS))
    """
    df = pd.read_csv(csv_path)

    # Compute X_local and Y_local if they are needed in INPUT_COLS
    if "X_local" not in df.columns:
        df["X_local"] = df["X"] - df["X"].mean()
    if "Y_local" not in df.columns:
        df["Y_local"] = df["Y"] - df["Y"].mean()

    # Determine the regular grid from unique X and Y values
    xs = np.sort(df["X"].unique())
    ys = np.sort(df["Y"].unique())
    H, W = len(ys), len(xs)

    # Build coordinate → index mappings
    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: j for j, y in enumerate(ys)}

    # Pre-allocate grids (fill with 0 for any missing points)
    input_grid = np.zeros((H, W, len(INPUT_COLS)), dtype=np.float32)
    output_grid = np.zeros((H, W, len(OUTPUT_COLS)), dtype=np.float32)

    # Fill grids row by row
    for _, row in df.iterrows():
        xi = x_to_idx[row["X"]]
        yi = y_to_idx[row["Y"]]
        for ch, col in enumerate(INPUT_COLS):
            input_grid[yi, xi, ch] = row[col]
        for ch, col in enumerate(OUTPUT_COLS):
            output_grid[yi, xi, ch] = row[col]

    return input_grid, output_grid


def compute_stats(arrays: list[np.ndarray]) -> dict:
    """
    Compute per-channel min and max across a list of arrays.
    Each array is expected to have shape (H, W, C).
    Returns dict with 'min' and 'max' lists of length C.
    """
    stacked = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
    return {
        "min": stacked.min(axis=0).tolist(),
        "max": stacked.max(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Wind CSV files to NumPy grids for pix2pixHD")
    parser.add_argument("--input_dir", type=str, default="input_csv",
                        help="Directory containing ML_FormFlux CSV files")
    parser.add_argument("--output_dir", type=str, default="datasets/wind",
                        help="Output directory for processed data")
    parser.add_argument("--test_angles", type=int, nargs="+", default=[270, 315],
                        help="Wind angles to reserve for testing")
    args = parser.parse_args()

    # Find all ground-truth CSVs (exclude _pred files)
    pattern = os.path.join(args.input_dir, "ML_FormFlux_1_*.csv")
    all_csvs = sorted(glob.glob(pattern))
    gt_csvs = [f for f in all_csvs if "_pred" not in os.path.basename(f)]

    if not gt_csvs:
        print(f"ERROR: No CSV files found matching {pattern}")
        import sys
        sys.exit(1)

    print(f"Found {len(gt_csvs)} ground-truth CSV files")

    # Create output directories
    for split in ["train", "test"]:
        for suffix in ["_A", "_B"]:
            os.makedirs(os.path.join(args.output_dir, split + suffix), exist_ok=True)

    # Process each CSV
    all_inputs = []
    all_outputs = []
    file_info = []

    for csv_path in gt_csvs:
        angle = extract_angle(csv_path)
        if angle is None:
            print(f"  Skipping {csv_path} (could not extract angle)")
            continue

        split = "test" if angle in args.test_angles else "train"
        print(f"  Processing angle {angle} -> {split} set ...")

        input_grid, output_grid = csv_to_grid(csv_path)
        all_inputs.append(input_grid)
        all_outputs.append(output_grid)

        # Save
        a_path = os.path.join(args.output_dir, f"{split}_A", f"{angle}.npy")
        b_path = os.path.join(args.output_dir, f"{split}_B", f"{angle}.npy")
        np.save(a_path, input_grid)
        np.save(b_path, output_grid)

        file_info.append({
            "angle": angle,
            "split": split,
            "shape": list(input_grid.shape[:2]),
            "input_channels": len(INPUT_COLS),
            "output_channels": len(OUTPUT_COLS),
        })

        print(f"    Grid shape: {input_grid.shape[0]}×{input_grid.shape[1]}, "
              f"Input channels: {len(INPUT_COLS)}, Output channels: {len(OUTPUT_COLS)}")

    # Compute and save normalization statistics
    input_stats = compute_stats(all_inputs)
    output_stats = compute_stats(all_outputs)

    stats = {
        "input_columns": INPUT_COLS,
        "output_columns": OUTPUT_COLS,
        "input_stats": input_stats,
        "output_stats": output_stats,
        "files": file_info,
    }

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n* Saved normalization statistics to {stats_path}")
    print(f"* Train samples: {sum(1 for fi in file_info if fi['split'] == 'train')}")
    print(f"* Test  samples: {sum(1 for fi in file_info if fi['split'] == 'test')}")
    print("Done!")


if __name__ == "__main__":
    main()
