import numpy as np
import scipy.ndimage as ndimage
import argparse
import time

def compute_did_features(mask, n_sectors=8, max_dist=None):
    """
    Computes 8-channel DID features from a binary mask.
    mask: (H, W) where 1 is obstacle, 0 is free space.
    Returns: (H, W, 8) tensor.
    """
    H, W = mask.shape
    if max_dist is None:
        max_dist = np.sqrt(H**2 + W**2)
    
    # 1. Extract boundary pixels
    # Boundary is where pixel is 1 but has at least one 0 neighbor
    eroded = ndimage.binary_erosion(mask)
    boundary = mask ^ eroded
    boundary_indices = np.argwhere(boundary) # (N_b, 2)
    
    if len(boundary_indices) == 0:
        return np.ones((H, W, n_sectors)) * max_dist

    # 2. Get all pixel coordinates
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixels = np.stack([yy, xx], axis=-1).reshape(-1, 2) # (H*W, 2)
    
    # 3. DID computation (Vectorized-ish)
    # To keep memory footprint low, we process in chunks or use a more efficient approach
    # than a full distance matrix.
    
    did = np.full((H * W, n_sectors), max_dist, dtype=np.float32)
    
    sector_width = 2 * np.pi / n_sectors
    # Use overlapping sectors (centers offset by half width)
    # we'll use n_sectors centers: 0, 45, 90...
    sector_centers = np.arange(n_sectors) * (2 * np.pi / n_sectors)
    
    # Optimization: Only compute for free-space pixels? 
    # Usually DID is defined everywhere, but inside obstacles it might be 0 or small.
    # We follow the request: "treat each pixel center as a query location".
    
    # We'll use a block-based approach to avoid O(HW * Nb) memory explosion
    block_size = 1024
    for i in range(0, len(pixels), block_size):
        p_block = pixels[i:i+block_size] # (B, 2)
        
        # d_pos: (B, Nb, 2)
        d_pos = boundary_indices[None, :, :] - p_block[:, None, :]
        
        # dists: (B, Nb)
        dists = np.linalg.norm(d_pos, axis=2)
        
        # angles: (B, Nb) in [-pi, pi]
        angles = np.arctan2(d_pos[:, :, 0], d_pos[:, :, 1])
        
        for s in range(n_sectors):
            center = sector_centers[s]
            # Use wrapped angle difference for overlapping sectors
            # Sector width is usually 2*pi/n_sectors, but "overlapping" implies wider.
            # We'll use width = 2 * (2*pi/n_sectors) for 50% overlap.
            width = 2 * (2 * np.pi / n_sectors)
            
            angle_diff = (angles - center + np.pi) % (2 * np.pi) - np.pi
            mask_sector = np.abs(angle_diff) <= (width / 2)
            
            # For each pixel in block, find min dist in this sector
            # dists is (B, Nb), mask_sector is (B, Nb)
            sector_dists = np.where(mask_sector, dists, max_dist)
            did[i:i+block_size, s] = np.min(sector_dists, axis=1)

    return did.reshape(H, W, n_sectors)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run a small test')
    args = parser.parse_args()
    
    if args.test:
        print("Running DID computation test...")
        # Create a 64x64 mask with a circle in the middle
        H, W = 64, 64
        mask = np.zeros((H, W), dtype=bool)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        mask[(yy-32)**2 + (xx-32)**2 <= 10**2] = True
        
        start = time.time()
        did = compute_did_features(mask)
        end = time.time()
        
        print(f"Test size: {H}x{W}, Compute time: {end-start:.4f}s")
        print(f"DID shape: {did.shape}")
        print(f"DID mean: {did.mean():.4f}")
        print(f"DID max: {did.max():.4f}")
        
if __name__ == "__main__":
    main()
