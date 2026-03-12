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

        # Determine directory names
        dir_A = os.path.join(opt.dataroot, opt.phase + "_A")
        dir_B = os.path.join(opt.dataroot, opt.phase + "_B")

        # Collect .npy file paths
        self.A_paths = sorted([
            os.path.join(dir_A, f) for f in os.listdir(dir_A)
            if f.endswith(".npy")
        ])

        self.B_paths = []
        if opt.isTrain or os.path.isdir(dir_B):
            self.B_paths = sorted([
                os.path.join(dir_B, f) for f in os.listdir(dir_B)
                if f.endswith(".npy")
            ])

        self.dataset_size = len(self.A_paths)
        if self.dataset_size == 0:
            raise ValueError(f"No .npy files found in {dir_A}. The dataset directory might be empty. "
                             "Please ensure your data was properly uploaded or preprocessed.")

        # Load normalization statistics
        stats_path = opt.wind_stats if opt.wind_stats else os.path.join(opt.dataroot, "stats.json")
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

        # Normalize to [-1, 1]
        A = self._normalize(A, self.input_min, self.input_max)
        if B is not None:
            B = self._normalize(B, self.output_min, self.output_max)

        # Pad to make dimensions divisible by 2^n_downsample_global
        base = 2 ** self.opt.n_downsample_global
        if self.opt.netG == 'local':
            base *= (2 ** self.opt.n_local_enhancers)
        A = self._pad_to_power_of_2(A, int(base))
        if B is not None:
            B = self._pad_to_power_of_2(B, int(base))

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
        return len(self.A_paths) // max(self.opt.batchSize, 1) * max(self.opt.batchSize, 1)

    def name(self):
        return 'WindDataset'
