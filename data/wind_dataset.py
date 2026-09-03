"""
WindDataset
===========
PyTorch dataset for loading preprocessed Wind Comfort NumPy grids
produced by preprocess_csv.py.
"""
import json
import os

import numpy as np
import torch
from data.base_dataset import BaseDataset


class WindDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # Support loading directly from a single .npz file or from a directory of .npy files
        self.is_npz = self.root.endswith('.npz') and os.path.isfile(self.root)
        
        if self.is_npz:
            print(f"Loading from NPZ archive: {self.root}")
            self.npz_data = np.load(self.root)
            # Find all keys belonging to this phase (e.g., 'train_A_0', 'train_A_1')
            self.A_keys = sorted([k for k in self.npz_data.files if k.startswith(f"{opt.phase}_A_")])
            self.B_keys = sorted([k for k in self.npz_data.files if k.startswith(f"{opt.phase}_B_")])
            self.dataset_size = len(self.A_keys)
        else:
            # Determine directory names
            dir_A = os.path.join(opt.dataroot, opt.phase + "_A")
            dir_B = os.path.join(opt.dataroot, opt.phase + "_B")
    
            # Collect .npy file paths
            self.A_paths = sorted([
                os.path.join(dir_A, f) for f in os.listdir(dir_A)
                if f.endswith(".npy")
            ]) if os.path.isdir(dir_A) else []
    
            self.B_paths = []
            if opt.isTrain or os.path.isdir(dir_B):
                self.B_paths = sorted([
                    os.path.join(dir_B, f) for f in os.listdir(dir_B)
                    if f.endswith(".npy")
                ]) if os.path.isdir(dir_B) else []
    
            self.dataset_size = len(self.A_paths)

        if self.dataset_size == 0:
            if self.is_npz:
                 raise ValueError(f"No arrays found for phase '{opt.phase}' in {self.root}. Check the .npz keys.")
            else:
                 raise ValueError(f"No .npy files found in {dir_A}. The dataset directory might be empty. "
                                  "Please ensure your data was properly uploaded or preprocessed.")

        # Load normalization statistics
        stats_path = opt.wind_stats if opt.wind_stats else (
            os.path.join(os.path.dirname(self.root), "stats.json") if self.is_npz 
            else os.path.join(opt.dataroot, "stats.json")
        )
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            self.input_min = np.array(stats["input_stats"]["min"], dtype=np.float32)
            self.input_max = np.array(stats["input_stats"]["max"], dtype=np.float32)
            self.output_min = np.array(stats["output_stats"]["min"], dtype=np.float32)
            self.output_max = np.array(stats["output_stats"]["max"], dtype=np.float32)
        else:
            print(f"WARNING: stats.json not found at {stats_path}. Using identity normalization.")
            nc_in = opt.input_nc
            nc_out = opt.output_nc
            self.input_min = np.zeros(nc_in, dtype=np.float32)
            self.input_max = np.ones(nc_in, dtype=np.float32)
            self.output_min = np.zeros(nc_out, dtype=np.float32)
            self.output_max = np.ones(nc_out, dtype=np.float32)

    def _normalize(self, data: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
        """Normalize data from [vmin, vmax] to [-1, 1]."""
        denom = vmax - vmin
        denom[denom == 0] = 1.0  # avoid division by zero for constant channels
        normalized = 2.0 * (data - vmin) / denom - 1.0
        return normalized.astype(np.float32)

    def _pad_to_power_of_2(self, arr: np.ndarray, base: int = 16) -> np.ndarray:
        """Pad array so H and W are divisible by base (needed for downsampling)."""
        H, W = arr.shape[:2]
        pad_h = (base - H % base) % base
        pad_w = (base - W % base) % base
        if pad_h > 0 or pad_w > 0:
            if arr.ndim == 3:
                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            else:
                arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        return arr

    def __getitem__(self, index):
        if self.is_npz:
            A_key = self.A_keys[index]
            A = self.npz_data[A_key] # shape (H, W, C_in)
            A_path = f"{self.root}::{A_key}"
            
            B_tensor = 0
            if len(self.B_keys) > 0:
                B_key = self.B_keys[index]
                B = self.npz_data[B_key]
            else:
                B = None
        else:
            # Load input (A) array
            A_path = self.A_paths[index]
            A = np.load(A_path)  # shape (H, W, C_in)
    
            # Load output (B) array
            B_tensor = 0
            if len(self.B_paths) > 0:
                B_path = self.B_paths[index]
                B = np.load(B_path)  # shape (H, W, C_out)
            else:
                B = None

        # Data augmentation: random horizontal flip
        if self.opt.isTrain and not self.opt.no_flip:
            if np.random.random() > 0.5:
                A = np.flip(A, axis=1).copy()
                if B is not None:
                    B = np.flip(B, axis=1).copy()
                
                # Fix: Negate X-components to reflect physical horizontal flip
                A[:, :, 4] *= -1.0  # X_local
                A[:, :, 6] *= -1.0  # dir_sin
                
                # Fix: Swap left/right 45-degree angle DID sectors if 16-channel DID data is used (this stops hallucinated wind from the left/right)
                if A.shape[-1] == 16:
                    temp_A = A.copy()
                    A[:, :, 8] = temp_A[:, :, 12]  # East(0) <-> West(4)
                    A[:, :, 12] = temp_A[:, :, 8]
                    A[:, :, 9] = temp_A[:, :, 11]  # NE(1) <-> NW(3)
                    A[:, :, 11] = temp_A[:, :, 9]
                    A[:, :, 15] = temp_A[:, :, 13] # SE(7) <-> SW(5)
                    A[:, :, 13] = temp_A[:, :, 15]

            # Data augmentation: random vertical flip
            if np.random.random() > 0.5:
                A = np.flip(A, axis=0).copy()
                if B is not None:
                    B = np.flip(B, axis=0).copy()
                
                # Fix: Negate Y-components to reflect physical vertical flip
                A[:, :, 5] *= -1.0  # Y_local
                A[:, :, 7] *= -1.0  # dir_cos
                
                # Fix: Swap top/bottom DID sectors (this stops hallucinated wind from the top/bottom)
                if A.shape[-1] == 16:
                    temp_A = A.copy()
                    A[:, :, 10] = temp_A[:, :, 14] # North(2) <-> South(6)
                    A[:, :, 14] = temp_A[:, :, 10]
                    A[:, :, 9]  = temp_A[:, :, 15] # NE(1) <-> SE(7)
                    A[:, :, 15] = temp_A[:, :, 9]
                    A[:, :, 11] = temp_A[:, :, 13] # NW(3) <-> SW(5)
                    A[:, :, 13] = temp_A[:, :, 11]

        # Pad to make dimensions divisible by 2^n_downsample_global. Pad BEFORE
        # normalizing: a raw 0 is exactly what out-of-domain cells hold, so the pad
        # band normalizes to the same value; padding after normalization put the
        # mid-range value (SDF -205 m, Bldg_height 76 m, mag_U 1.87) in the band.
        base = 2 ** self.opt.n_downsample_global
        if self.opt.netG == 'local':
            base *= (2 ** self.opt.n_local_enhancers)
        A = self._pad_to_power_of_2(A, int(base))
        if B is not None:
            B = self._pad_to_power_of_2(B, int(base))

        # Normalize to [-1, 1]
        A = self._normalize(A, self.input_min, self.input_max)
        if B is not None:
            B = self._normalize(B, self.output_min, self.output_max)

        # Convert to tensors: (H, W, C) → (C, H, W)
        A_tensor = torch.from_numpy(A.transpose(2, 0, 1)).float()
        if B is not None:
            B_tensor = torch.from_numpy(B.transpose(2, 0, 1)).float()

        # inst and feat are not used for wind data
        inst_tensor = 0
        feat_tensor = 0

        input_dict = {
            'label': A_tensor,
            'inst': inst_tensor,
            'image': B_tensor,
            'feat': feat_tensor,
            'path': A_path,
        }

        return input_dict

    def __len__(self):
        sz = len(self.A_keys) if self.is_npz else len(self.A_paths)
        return sz // max(self.opt.batchSize, 1) * max(self.opt.batchSize, 1)

    def name(self):
        return 'WindDataset'
