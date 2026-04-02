"""
evaluate.py
===========
Evaluate pix2pixHD wind predictions against ground truth in PHYSICAL UNITS.

Unlike test.py which saves visualizations as uint8 images (losing all
physical meaning), this script:
  1. Runs inference on the test set
  2. Denormalizes predictions from [-1, 1] back to original wind speed (m/s)
  3. Loads original ground truth .npy files and denormalizes them too
  4. Computes metrics (MAE, RMSE, R², MAPE) in real physical units
  5. Saves per-sample heatmap comparisons and a summary report

Usage:
    python evaluate.py \
        --name wind_pix2pixHD \
        --dataroot datasets/wind \
        --dataset_mode wind \
        --input_nc 8 --output_nc 1 \
        --label_nc 0 --no_instance \
        --resize_or_crop none \
        --which_epoch latest
"""

import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model


def denormalize(data, vmin, vmax):
    """Reverse the [-1, 1] normalization back to [vmin, vmax]."""
    denom = vmax - vmin
    denom[denom == 0] = 1.0
    return (data + 1.0) / 2.0 * denom + vmin


def compute_metrics(gt, pred, mask=None):
    """
    Compute evaluation metrics on physical-scale values.

    Parameters
    ----------
    gt   : np.ndarray — ground truth in physical units
    pred : np.ndarray — prediction in physical units
    mask : np.ndarray (bool), optional — if provided, only evaluate where True
           (e.g., to exclude zero-padded regions)

    Returns
    -------
    dict with MAE, RMSE, R², MAPE, max_error
    """
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]

    gt = gt.flatten()
    pred = pred.flatten()

    diff = gt - pred
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    max_err = np.max(np.abs(diff))

    # R² (coefficient of determination)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # MAPE (mean absolute percentage error), avoiding division by zero
    nonzero = np.abs(gt) > 1e-6
    if nonzero.sum() > 0:
        mape = np.mean(np.abs(diff[nonzero] / gt[nonzero])) * 100.0
    else:
        mape = float('nan')

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'MaxError': max_err,
        'N_pixels': len(gt),
    }


def make_comparison_figure(gt_phys, pred_phys, diff, sample_name, output_stats, save_path):
    """Create a side-by-side GT vs Pred vs Error heatmap figure."""
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    vmin_val = 0.0
    vmax_val = output_stats['max'][0]

    # Ground Truth
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(gt_phys, cmap='viridis', vmin=vmin_val, vmax=vmax_val)
    ax0.set_title(f'Ground Truth\n(mag_U, m/s)', fontsize=12)
    ax0.axis('off')

    # Prediction
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(pred_phys, cmap='viridis', vmin=vmin_val, vmax=vmax_val)
    ax1.set_title(f'Prediction\n(mag_U, m/s)', fontsize=12)
    ax1.axis('off')

    # Absolute Error
    ax2 = fig.add_subplot(gs[2])
    err_max = np.percentile(np.abs(diff), 99)  # clip for visibility
    im2 = ax2.imshow(np.abs(diff), cmap='hot', vmin=0, vmax=max(err_max, 0.01))
    ax2.set_title(f'|Error| (m/s)\nMAE={np.mean(np.abs(diff)):.4f}', fontsize=12)
    ax2.axis('off')

    # Colorbars
    cax = fig.add_subplot(gs[3])
    fig.colorbar(im0, cax=cax, label='Wind speed (m/s)')

    fig.suptitle(f'Sample: {sample_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # ── Parse options ──────────────────────────────────────────────────────
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    # ── Load normalization stats ───────────────────────────────────────────
    stats_path = opt.wind_stats if opt.wind_stats else os.path.join(opt.dataroot, 'stats.json')
    if not os.path.exists(stats_path):
        print(f"ERROR: stats.json not found at {stats_path}")
        print("Cannot denormalize without normalization statistics.")
        sys.exit(1)

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    output_min = np.array(stats['output_stats']['min'], dtype=np.float32)
    output_max = np.array(stats['output_stats']['max'], dtype=np.float32)
    input_min = np.array(stats['input_stats']['min'], dtype=np.float32)
    input_max = np.array(stats['input_stats']['max'], dtype=np.float32)

    print(f"\n{'='*60}")
    print(f"EVALUATION IN PHYSICAL UNITS")
    print(f"{'='*60}")
    print(f"Output column(s): {stats['output_columns']}")
    print(f"Output range: [{output_min[0]:.4f}, {output_max[0]:.4f}] m/s")
    print(f"Stats file: {stats_path}")
    print(f"{'='*60}\n")

    # ── Load data & model ──────────────────────────────────────────────────
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    # ── Output directory ───────────────────────────────────────────────────
    eval_dir = os.path.join(opt.results_dir, opt.name,
                            f'{opt.phase}_{opt.which_epoch}_evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    npy_dir = os.path.join(eval_dir, 'predictions_npy')
    os.makedirs(npy_dir, exist_ok=True)
    fig_dir = os.path.join(eval_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ── Run inference & evaluate ───────────────────────────────────────────
    all_results = []

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        # Get sample name from path
        img_path = data['path'][0] if isinstance(data['path'], list) else data['path']
        sample_name = os.path.splitext(os.path.basename(img_path))[0]
        # Handle NPZ paths like "path.npz::test_A_0"
        if '::' in sample_name:
            sample_name = sample_name.split('::')[-1]

        print(f"[{i+1}] Evaluating sample: {sample_name}")

        # ── Forward pass ───────────────────────────────────────────────────
        generated = model.inference(data['label'], data['inst'], data['image'])

        # ── Extract tensors → numpy ────────────────────────────────────────
        # Prediction: (1, C, H, W) → (H, W, C)
        pred_norm = generated.data[0].cpu().float().numpy()  # (C, H, W), in [-1, 1]
        pred_norm = np.transpose(pred_norm, (1, 2, 0))       # (H, W, C)

        # Ground truth (from dataset): (1, C, H, W) → (H, W, C)
        gt_tensor = data['image']
        if isinstance(gt_tensor, torch.Tensor) and gt_tensor.dim() >= 3:
            gt_norm = gt_tensor[0].cpu().float().numpy()      # (C, H, W), in [-1, 1]
            gt_norm = np.transpose(gt_norm, (1, 2, 0))        # (H, W, C)
        else:
            print(f"  WARNING: No ground truth for sample {sample_name}, skipping.")
            continue

        # ── Denormalize to physical units ──────────────────────────────────
        pred_phys = denormalize(pred_norm, output_min, output_max)
        gt_phys   = denormalize(gt_norm, output_min, output_max)

        # ── Determine valid mask (exclude padding) ─────────────────────────
        # The original grid is 504x504. Padded regions have gt ≈ min value
        # after denormalization. We can detect padding from the input label.
        orig_H, orig_W = None, None
        for fi in stats.get('files', []):
            if fi['split'] == opt.phase:
                orig_H, orig_W = fi['shape']
                break

        if orig_H and orig_W:
            # Crop away padding to evaluate only on real data
            pred_phys_eval = pred_phys[:orig_H, :orig_W, :]
            gt_phys_eval   = gt_phys[:orig_H, :orig_W, :]
        else:
            pred_phys_eval = pred_phys
            gt_phys_eval   = gt_phys

        # ── Compute metrics ────────────────────────────────────────────────
        # For single-channel output, squeeze to 2D for analysis
        gt_2d   = gt_phys_eval[:, :, 0]
        pred_2d = pred_phys_eval[:, :, 0]
        diff_2d = gt_2d - pred_2d

        metrics = compute_metrics(gt_2d, pred_2d)

        # Also compute metrics excluding obstacle pixels (where gt ≈ 0)
        fluid_mask = gt_2d > 1e-4
        metrics_fluid = compute_metrics(gt_2d, pred_2d, mask=fluid_mask)

        result = {
            'sample': sample_name,
            'all_pixels': metrics,
            'fluid_only': metrics_fluid,
        }
        all_results.append(result)

        # Print per-sample results
        print(f"  All pixels:  MAE={metrics['MAE']:.4f} m/s  RMSE={metrics['RMSE']:.4f} m/s  "
              f"R²={metrics['R2']:.4f}  MAPE={metrics['MAPE']:.1f}%")
        print(f"  Fluid only:  MAE={metrics_fluid['MAE']:.4f} m/s  RMSE={metrics_fluid['RMSE']:.4f} m/s  "
              f"R²={metrics_fluid['R2']:.4f}  MAPE={metrics_fluid['MAPE']:.1f}%")

        # ── Save raw predictions ───────────────────────────────────────────
        np.save(os.path.join(npy_dir, f'{sample_name}_pred.npy'), pred_phys_eval)
        np.save(os.path.join(npy_dir, f'{sample_name}_gt.npy'), gt_phys_eval)

        # ── Save comparison figure ─────────────────────────────────────────
        make_comparison_figure(
            gt_2d, pred_2d, diff_2d, sample_name,
            stats['output_stats'],
            os.path.join(fig_dir, f'{sample_name}_comparison.png')
        )

    # ── Aggregate & print summary ──────────────────────────────────────────
    if not all_results:
        print("\nNo samples were evaluated. Check your dataroot and phase settings.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY — Physical Units (m/s)")
    print(f"{'='*80}")
    print(f"Model: {opt.name}  |  Epoch: {opt.which_epoch}  |  Samples: {len(all_results)}")
    print(f"Output range in training data: [{output_min[0]:.4f}, {output_max[0]:.4f}] m/s")
    print(f"{'='*80}")

    # Per-sample table
    header = f"{'Sample':<20} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8} | {'MAPE%':>8} | {'MaxErr':>8}"
    print(f"\n--- All Pixels ---")
    print(header)
    print('-' * len(header))
    for r in all_results:
        m = r['all_pixels']
        print(f"{r['sample']:<20} | {m['MAE']:>8.4f} | {m['RMSE']:>8.4f} | "
              f"{m['R2']:>8.4f} | {m['MAPE']:>7.1f}% | {m['MaxError']:>8.4f}")

    print(f"\n--- Fluid Region Only (excluding obstacles) ---")
    print(header)
    print('-' * len(header))
    for r in all_results:
        m = r['fluid_only']
        print(f"{r['sample']:<20} | {m['MAE']:>8.4f} | {m['RMSE']:>8.4f} | "
              f"{m['R2']:>8.4f} | {m['MAPE']:>7.1f}% | {m['MaxError']:>8.4f}")

    # Aggregate means
    def avg_metric(key, region='all_pixels'):
        vals = [r[region][key] for r in all_results if not np.isnan(r[region][key])]
        return np.mean(vals) if vals else float('nan')

    print(f"\n--- Averages ---")
    print(f"{'Region':<25} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8} | {'MAPE%':>8}")
    print('-' * 70)
    print(f"{'All pixels':<25} | {avg_metric('MAE'):>8.4f} | {avg_metric('RMSE'):>8.4f} | "
          f"{avg_metric('R2'):>8.4f} | {avg_metric('MAPE'):>7.1f}%")
    print(f"{'Fluid only':<25} | {avg_metric('MAE','fluid_only'):>8.4f} | {avg_metric('RMSE','fluid_only'):>8.4f} | "
          f"{avg_metric('R2','fluid_only'):>8.4f} | {avg_metric('MAPE','fluid_only'):>7.1f}%")

    # ── Save JSON report ───────────────────────────────────────────────────
    report = {
        'model': opt.name,
        'epoch': opt.which_epoch,
        'dataroot': opt.dataroot,
        'output_range_ms': {'min': float(output_min[0]), 'max': float(output_max[0])},
        'samples': [],
        'averages': {
            'all_pixels': {k: float(avg_metric(k, 'all_pixels'))
                           for k in ['MAE', 'RMSE', 'R2', 'MAPE']},
            'fluid_only': {k: float(avg_metric(k, 'fluid_only'))
                           for k in ['MAE', 'RMSE', 'R2', 'MAPE']},
        }
    }
    for r in all_results:
        report['samples'].append({
            'name': r['sample'],
            'all_pixels': {k: float(v) for k, v in r['all_pixels'].items()},
            'fluid_only': {k: float(v) for k, v in r['fluid_only'].items()},
        })

    report_path = os.path.join(eval_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Results saved to: {eval_dir}")
    print(f"  - Figures: {fig_dir}/")
    print(f"  - Raw predictions (npy): {npy_dir}/")
    print(f"  - JSON report: {report_path}")


if __name__ == '__main__':
    main()
