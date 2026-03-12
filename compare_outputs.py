import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

def load_npy(path):
    return np.load(path)

def compute_metrics(gt, pred):
    mae = np.mean(np.abs(gt - pred))
    rmse = np.sqrt(np.mean((gt - pred)**2))
    return mae, rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground truth .npy files (test_B)')
    parser.add_argument('--std_results', type=str, required=True, help='Path to standard model predictions')
    parser.add_argument('--did_results', type=str, required=True, help='Path to DID model predictions')
    parser.add_argument('--output_img', type=str, default='did_comparison.png')
    args = parser.parse_args()

    # Find common files
    gt_files = sorted([f for f in os.listdir(args.gt_dir) if f.endswith('.npy')])
    
    results = []
    
    fig, axes = plt.subplots(len(gt_files), 3, figsize=(15, 5 * len(gt_files)))
    if len(gt_files) == 1: axes = [axes]

    for i, fname in enumerate(gt_files):
        angle = os.path.splitext(fname)[0]
        gt = load_npy(os.path.join(args.gt_dir, fname))
        
        # We assume the results scripts save as [angle]_synthesized_image.npy or similar
        # If they only save as images, we'd need to load and denormalize.
        # For precise metrics, the user should use a test script that saves .npy.
        
        # To make this useful, I'll advise the user to save .npy in a modified test script.
        # But for now, let's assume they exist.
        std_path = os.path.join(args.std_results, fname)
        did_path = os.path.join(args.did_results, fname)
        
        if not os.path.exists(std_path) or not os.path.exists(did_path):
            print(f"Skipping angle {angle}: missing pred files ({std_path} or {did_path})")
            continue
            
        std_pred = load_npy(std_path)
        did_pred = load_npy(did_path)
        
        mae_std, rmse_std = compute_metrics(gt, std_pred)
        mae_did, rmse_did = compute_metrics(gt, did_pred)
        
        results.append({
            "angle": angle,
            "std": {"mae": mae_std, "rmse": rmse_std},
            "did": {"mae": mae_did, "rmse": rmse_did}
        })
        
        # Visualization
        axes[i][0].imshow(gt[..., 0], cmap='viridis')
        axes[i][0].set_title(f"GT (Angle {angle})")
        
        axes[i][1].imshow(std_pred[..., 0], cmap='viridis')
        axes[i][1].set_title(f"Standard\nMAE: {mae_std:.4f}")
        
        axes[i][2].imshow(did_pred[..., 0], cmap='viridis')
        axes[i][2].set_title(f"DID\nMAE: {mae_did:.4f}")

    plt.tight_layout()
    plt.savefig(args.output_img)
    print(f"Comparison image saved to {args.output_img}")
    
    # Print summary table
    print("\n" + "="*50)
    print(f"{'Angle':<10} | {'Std MAE':<10} | {'DID MAE':<10} | {'Improvement %':<10}")
    print("-"*50)
    for res in results:
        imp = (res['std']['mae'] - res['did']['mae']) / res['std']['mae'] * 100
        print(f"{res['angle']:<10} | {res['std']['mae']:<10.4f} | {res['did']['mae']:<10.4f} | {imp:<10.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
