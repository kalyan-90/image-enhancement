"""
PyTorch Dataset Loader for LOL-v1 (Low-Light Dataset).
Guarantees clean, leak-free partitioning between:
- Training set: 437 image pairs
- Validation set: 48 image pairs
- Test set: 15 image pairs (eval15)
"""

import os
import json
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LOLDataset(Dataset):
    """
    Dataset class for LOL-v1 dataset.
    """
    def __init__(self, root_dir, split="train", patch_size=48, split_config_path=None):
        """
        Args:
            root_dir: Path to dataset/LOL directory.
            split: 'train', 'val', or 'test'.
            patch_size: Square patch size for training (None for full resolution).
            split_config_path: Path to configs/train_val_split.json.
        """
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size if split == "train" else None
        
        if split in ["train", "val"]:
            self.low_dir = os.path.join(root_dir, "train", "low")
            self.high_dir = os.path.join(root_dir, "train", "high")
            
            # Load deterministic partition
            if split_config_path is None:
                split_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "configs",
                    "train_val_split.json"
                )
            
            if os.path.exists(split_config_path):
                with open(split_config_path, "r") as f:
                    config = json.load(f)
                if split == "train":
                    self.image_names = sorted(config["train_images"])
                else:
                    self.image_names = sorted(config["val_images"])
            else:
                # Fallback if split JSON is missing: first 437 train, last 48 val
                all_images = sorted([
                    f for f in os.listdir(self.low_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                if split == "train":
                    self.image_names = all_images[:437]
                else:
                    self.image_names = all_images[437:]
        elif split == "test":
            self.low_dir = os.path.join(root_dir, "test", "low")
            self.high_dir = os.path.join(root_dir, "test", "high")
            self.image_names = sorted([
                f for f in os.listdir(self.low_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
        else:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        if len(self.image_names) == 0:
            raise FileNotFoundError(f"No images found for split '{split}' in {self.low_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]
        low_path = os.path.join(self.low_dir, name)
        high_path = os.path.join(self.high_dir, name)

        low_image = Image.open(low_path).convert("RGB")
        high_image = Image.open(high_path).convert("RGB")

        if low_image.size != high_image.size:
            raise ValueError(f"Image dimension mismatch for {name}: {low_image.size} vs {high_image.size}")

        low_arr = np.asarray(low_image, dtype=np.float32) / 255.0
        high_arr = np.asarray(high_image, dtype=np.float32) / 255.0

        # Training data augmentation: random patch crop and flips
        if self.split == "train" and self.patch_size is not None:
            h, w, _ = low_arr.shape
            p = self.patch_size
            if h >= p and w >= p:
                top = random.randint(0, h - p)
                left = random.randint(0, w - p)
                low_arr = low_arr[top:top+p, left:left+p]
                high_arr = high_arr[top:top+p, left:left+p]

            if random.random() > 0.5:
                low_arr = np.fliplr(low_arr).copy()
                high_arr = np.fliplr(high_arr).copy()

            if random.random() > 0.5:
                low_arr = np.flipud(low_arr).copy()
                high_arr = np.flipud(high_arr).copy()

        # Convert to tensor: (C, H, W)
        low_tensor = torch.from_numpy(low_arr).permute(2, 0, 1)
        high_tensor = torch.from_numpy(high_arr).permute(2, 0, 1)

        return low_tensor, high_tensor, name
