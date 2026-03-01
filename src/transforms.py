"""
src/transforms.py  —  Duality AI Desert Segmentation
Compatible with albumentations >= 1.4 AND >= 2.0
"""

import numpy as np
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUM = True
    _VER = tuple(int(x) for x in A.__version__.split(".")[:2])
    print(f"  albumentations {A.__version__} detected")
except ImportError:
    _ALBUM = False
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    import torch, random


# ── Version-safe constructors ─────────────────────────────────────────────────

def _rrc(size):
    """RandomResizedCrop: height/width → size in albumentations >= 1.4"""
    kwargs = dict(scale=(0.3, 1.0), ratio=(0.75, 1.33), p=1.0)
    try:
        return A.RandomResizedCrop(size=(size, size), **kwargs)
    except TypeError:
        return A.RandomResizedCrop(height=size, width=size, **kwargs)

def _resize(size):
    try:
        return A.Resize(height=size, width=size)
    except TypeError:
        return A.Resize(height=size, width=size)

def _shadow():
    try:
        return A.RandomShadow(shadow_roi=(0,0,1,1),
            num_shadows=(1,2), shadow_dimension=5, p=0.25)
    except TypeError:
        try:
            return A.RandomShadow(shadow_roi=(0,0,1,1),
                num_shadows_lower=1, num_shadows_upper=2,
                shadow_dimension=5, p=0.25)
        except TypeError:
            return A.RandomShadow(p=0.25)

def _fog():
    try:
        return A.RandomFog(fog_coef_range=(0.05, 0.15),
            alpha_coef=0.1, p=0.1)
    except TypeError:
        try:
            return A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15,
                alpha_coef=0.1, p=0.1)
        except TypeError:
            return A.RandomFog(p=0.1)

def _noise():
    try:
        return A.GaussNoise(std_range=(0.01, 0.05), p=1.0)
    except TypeError:
        try:
            return A.GaussNoise(var_limit=(10.0, 60.0), p=1.0)
        except TypeError:
            return A.GaussNoise(p=1.0)

def _dropout():
    try:
        return A.CoarseDropout(num_holes_range=(1, 8),
            hole_height_range=(16, 64), hole_width_range=(16, 64),
            fill_value=0, p=0.3)
    except TypeError:
        try:
            return A.CoarseDropout(max_holes=8, max_height=64, max_width=64,
                min_holes=1, min_height=16, min_width=16,
                fill_value=0, p=0.3)
        except TypeError:
            return A.CoarseDropout(p=0.3)

def _ssr():
    try:
        return A.Affine(translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            scale=(0.85, 1.15), rotate=(-20, 20), mode=0, p=0.5)
    except Exception:
        try:
            return A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15,
                rotate_limit=20, border_mode=0, p=0.5)
        except Exception:
            return A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15,
                rotate_limit=20, p=0.5)


# ── Pipelines ─────────────────────────────────────────────────────────────────

def _train_pipeline(size):
    return A.Compose([
        _rrc(size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        _ssr(),
        A.OneOf([
            A.RandomBrightnessContrast(0.35, 0.35, p=1.0),
            A.HueSaturationValue(20, 35, 25, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.75),
        _shadow(),
        _fog(),
        A.OneOf([
            _noise(),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3,7), p=1.0),
        ], p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        _dropout(),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

def _val_pipeline(size):
    return A.Compose([
        _resize(size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

def _test_pipeline(size):
    return A.Compose([
        _resize(size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])


# ── Wrappers ──────────────────────────────────────────────────────────────────

class _PairedWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, image, mask=None):
        import torch
        img_np = np.array(image)
        if mask is not None:
            out = self.pipeline(image=img_np, mask=mask)
            return out["image"], torch.from_numpy(
                np.array(out["mask"])).long()
        return self.pipeline(image=img_np)["image"]


class _TestWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, image):
        return self.pipeline(image=np.array(image))["image"]


# ── Torchvision fallback ──────────────────────────────────────────────────────

class _TVTrain:
    def __init__(self, size):
        self.size  = size
        self.img_t = T.Compose([
            T.ColorJitter(0.3, 0.3, 0.3, 0.1),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __call__(self, image, mask):
        import torch
        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask  = Image.fromarray(mask.astype(np.int32)).resize(
            (self.size, self.size), Image.NEAREST)
        if random.random() > 0.5:
            image = TF.hflip(image); mask = TF.hflip(mask)
        return self.img_t(image), torch.from_numpy(np.array(mask)).long()


class _TVVal:
    def __init__(self, size):
        self.size  = size
        self.img_t = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __call__(self, image, mask=None):
        import torch
        image = image.resize((self.size, self.size), Image.BILINEAR)
        if mask is not None:
            mask = Image.fromarray(mask.astype(np.int32)).resize(
                (self.size, self.size), Image.NEAREST)
            return self.img_t(image), torch.from_numpy(np.array(mask)).long()
        return self.img_t(image)


# ── Public API ────────────────────────────────────────────────────────────────

def get_train_transforms(size: int = 512):
    if _ALBUM:
        return _PairedWrapper(_train_pipeline(size))
    print("albumentations not found — using torchvision fallback")
    return _TVTrain(size)

def get_val_transforms(size: int = 512):
    if _ALBUM:
        return _PairedWrapper(_val_pipeline(size))
    return _TVVal(size)

def get_test_transforms(size: int = 512):
    if _ALBUM:
        return _TestWrapper(_test_pipeline(size))
    return _TVVal(size)