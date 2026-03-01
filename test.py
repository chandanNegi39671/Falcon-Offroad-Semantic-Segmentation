"""
Duality AI — Offroad Segmentation
Inference / Test Script
=======================
Run:  python test.py
      python test.py --seg-dir val/Segmentation   (to compute IoU)

Outputs:
  runs/predictions/*_pred.png        — colour segmentation mask
  runs/predictions/overlays/*        — semi-transparent blend on original
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm import tqdm

from src.dataset    import TestDataset, CLASS_NAMES, NUM_CLASSES, remap_mask
from src.transforms import get_test_transforms
from src.metrics    import IoUMetric

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test-dir",   default="test/Color_Images")
parser.add_argument("--seg-dir",    default=None,
                    help="Ground-truth masks folder — enables IoU scoring")
parser.add_argument("--model",      default="runs/best_model.pth")
parser.add_argument("--encoder",    default="mit_b2")
parser.add_argument("--output-dir", default="runs/predictions")
parser.add_argument("--img-size",   type=int, default=512)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Colour map (one colour per class) ─────────────────────────────────────────
COLOR_MAP = np.array([
    [ 34, 139,  34],   # 0 Trees          Forest Green
    [  0, 200, 100],   # 1 Lush Bushes    Lime Green
    [210, 180, 140],   # 2 Dry Grass      Tan
    [139,  90,  43],   # 3 Dry Bushes     Brown
    [169, 169, 169],   # 4 Ground Clutter Grey
    [255, 215,   0],   # 5 Flowers        Gold
    [101,  67,  33],   # 6 Logs           Dark Brown
    [128, 128, 128],   # 7 Rocks          Stone Grey
    [194, 178, 128],   # 8 Landscape      Sand
    [135, 206, 235],   # 9 Sky            Sky Blue
], dtype=np.uint8)

overlay_dir = os.path.join(args.output_dir, "overlays")
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
if not os.path.exists(args.model):
    raise FileNotFoundError(
        f"Model not found: {args.model}\n"
        "Run 'python train.py' first.")

model = smp.DeepLabV3Plus(
    encoder_name    = args.encoder,
    encoder_weights = None,
    in_channels     = 3,
    classes         = NUM_CLASSES,
)
model.load_state_dict(torch.load(args.model, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print(f"  ✅ Loaded : {args.model}")
print(f"  Encoder  : {args.encoder}")
print(f"  Device   : {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
test_dataset = TestDataset(args.test_dir, get_test_transforms(args.img_size))
print(f"  Images   : {len(test_dataset)}\n")

use_gt = args.seg_dir is not None and os.path.isdir(args.seg_dir)
metric = IoUMetric(NUM_CLASSES) if use_gt else None
if use_gt:
    print(f"  GT masks found → IoU will be computed\n")

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    for img_tensor, fname, orig_size in tqdm(test_dataset, desc="Predicting"):
        orig_w, orig_h = orig_size

        out = model(img_tensor.unsqueeze(0).to(DEVICE))

        # Resize back to ORIGINAL image resolution
        out = nn.functional.interpolate(
            out, size=(orig_h, orig_w),
            mode="bilinear", align_corners=False)

        pred = out.argmax(dim=1).squeeze().cpu().numpy()   # (H, W)

        # Colour mask
        colour = COLOR_MAP[pred]
        out_name = os.path.splitext(fname)[0] + "_pred.png"
        Image.fromarray(colour).save(os.path.join(args.output_dir, out_name))

        # Overlay blend
        orig_arr  = np.array(
            Image.open(os.path.join(args.test_dir, fname)).convert("RGB"),
            dtype=np.float32)
        overlay   = (0.55 * orig_arr + 0.45 * colour.astype(np.float32))
        overlay   = overlay.clip(0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(
            os.path.join(overlay_dir,
                         os.path.splitext(fname)[0] + "_overlay.png"))

        # IoU update
        if use_gt:
            gt_path = os.path.join(args.seg_dir, fname)
            if os.path.exists(gt_path):
                gt  = remap_mask(np.array(Image.open(gt_path)))
                gtt = torch.from_numpy(gt).long().unsqueeze(0)
                prt = torch.from_numpy(pred).long().unsqueeze(0)
                metric.update_preds(prt, gtt)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  ✅ Colour masks → {args.output_dir}/")
print(f"  🎨 Overlays     → {overlay_dir}/")

if use_gt and metric:
    print("\n  IoU Results on val set:")
    metric.pretty_print(CLASS_NAMES)

print(f"\n  Colour legend:")
for i, name in enumerate(CLASS_NAMES):
    r, g, b = COLOR_MAP[i]
    print(f"    {i}  {name:<18} RGB({r:3d},{g:3d},{b:3d})")

print(f"\n  Next step → python visualize.py")
