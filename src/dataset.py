"""
src/dataset.py
==============
DesertDataset  — paired RGB + segmentation mask (train / val)
TestDataset    — RGB only (inference)

CRITICAL FIX:
  Raw mask pixel values: {100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}
  Model needs consecutive indices: 0-9
  ID_TO_IDX handles this remapping via a fast lookup table.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# ── Raw ID → class index mapping ─────────────────────────────────────────────
ID_TO_IDX = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}

# Fast lookup array (index by raw pixel value)
_LUT = np.zeros(10001, dtype=np.int64)
for _raw, _idx in ID_TO_IDX.items():
    _LUT[_raw] = _idx

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky",
]
NUM_CLASSES = len(CLASS_NAMES)

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def remap_mask(mask_array: np.ndarray) -> np.ndarray:
    """Convert raw-ID mask → 0-based index mask."""
    return _LUT[mask_array].astype(np.int64)


class DesertDataset(Dataset):
    """Train / val dataset — returns (image_tensor, mask_tensor)."""

    def __init__(self, rgb_dir: str, seg_dir: str, transforms=None):
        self.rgb_dir    = rgb_dir
        self.seg_dir    = seg_dir
        self.transforms = transforms

        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
        if not os.path.isdir(seg_dir):
            raise FileNotFoundError(f"SEG dir not found: {seg_dir}")

        self.filenames = sorted([
            f for f in os.listdir(rgb_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        if len(self.filenames) == 0:
            raise RuntimeError(f"No images found in {rgb_dir}")

        missing = [f for f in self.filenames
                   if not os.path.exists(os.path.join(seg_dir, f))]
        if missing:
            print(f"  ⚠️  {len(missing)} mask(s) missing in {seg_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname    = self.filenames[idx]
        image    = Image.open(os.path.join(self.rgb_dir, fname)).convert("RGB")
        mask_raw = np.array(Image.open(os.path.join(self.seg_dir, fname)))
        mask     = remap_mask(mask_raw)   # ← critical fix

        if self.transforms:
            image, mask = self.transforms(image, mask)

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(
                np.array(image).transpose(2, 0, 1)).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask


class TestDataset(Dataset):
    """Inference dataset — returns (image_tensor, filename, original_size)."""

    def __init__(self, rgb_dir: str, transforms=None):
        self.rgb_dir    = rgb_dir
        self.transforms = transforms

        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"Test dir not found: {rgb_dir}")

        self.filenames = sorted([
            f for f in os.listdir(rgb_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        if len(self.filenames) == 0:
            raise RuntimeError(f"No images found in {rgb_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname     = self.filenames[idx]
        image     = Image.open(os.path.join(self.rgb_dir, fname)).convert("RGB")
        orig_size = image.size  # (W, H)

        if self.transforms:
            image = self.transforms(image)

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(
                np.array(image).transpose(2, 0, 1)).float() / 255.0

        return image, fname, orig_size
