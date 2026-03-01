"""
Duality AI — Dataset Validation Script
=======================================
Run:  python check_data.py

Checks:
  • File counts: train / val / test
  • Mask pixel IDs match exactly {100,200,300,500,550,600,700,800,7100,10000}
  • RGB / mask size alignment
  • Class pixel distribution across full training set
  • Saves data_check.png and class_distribution.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

TRAIN_RGB = "train/Color_Images"
TRAIN_SEG = "train/Segmentation"
VAL_RGB   = "val/Color_Images"
VAL_SEG   = "val/Segmentation"
TEST_DIR  = "test/Color_Images"

EXPECTED_IDS = {100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}
ID_ORDER     = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]

CLASS_NAMES = {
    100:"Trees", 200:"Lush Bushes", 300:"Dry Grass",
    500:"Dry Bushes", 550:"Ground Clutter", 600:"Flowers",
    700:"Logs", 800:"Rocks", 7100:"Landscape", 10000:"Sky",
}
COLOR_MAP = {
    100:(34,139,34), 200:(0,200,100), 300:(210,180,140),
    500:(139,90,43), 550:(169,169,169), 600:(255,215,0),
    700:(101,67,33), 800:(128,128,128), 7100:(194,178,128), 10000:(135,206,235),
}

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(path):
    if not os.path.isdir(path):
        return []
    return sorted([f for f in os.listdir(path)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS])


print("=" * 62)
print("  Duality AI — Dataset Validation")
print("=" * 62)
for label, path in [("Train RGB", TRAIN_RGB), ("Train SEG", TRAIN_SEG),
                     ("Val   RGB", VAL_RGB),   ("Val   SEG", VAL_SEG),
                     ("Test  IMG", TEST_DIR)]:
    print(f"  {label:<12}: {len(list_images(path)):>5} images")
print("=" * 62)

# ── Mask validation ───────────────────────────────────────────────────────────
rgb_files    = list_images(TRAIN_RGB)
all_ids      = set()
size_errors  = []
class_counts = defaultdict(int)
total_pixels = 0

print(f"\n  Validating all {len(rgb_files)} training masks...")

for fname in tqdm(rgb_files, desc="  Checking", ncols=70):
    rgb_path = os.path.join(TRAIN_RGB, fname)
    seg_path = os.path.join(TRAIN_SEG, fname)

    if not os.path.exists(seg_path):
        print(f"    ⚠️  Missing mask: {fname}")
        continue

    rgb_img = Image.open(rgb_path).convert("RGB")
    seg_img = Image.open(seg_path)
    mask    = np.array(seg_img)

    if rgb_img.size != seg_img.size:
        size_errors.append(f"  {fname}: RGB={rgb_img.size} SEG={seg_img.size}")

    ids = set(np.unique(mask).tolist())
    all_ids |= ids

    for raw_id in ids:
        if raw_id in EXPECTED_IDS:
            class_counts[raw_id] += int((mask == raw_id).sum())
    total_pixels += mask.size

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n  Unique IDs found : {sorted(all_ids)}")
unknown = all_ids - EXPECTED_IDS
if unknown:
    print(f"  ⚠️  Unknown IDs  : {unknown}  ← FIX THIS before training")
else:
    print(f"  ✅ All IDs are correct!")

if size_errors:
    print(f"\n  ⚠️  Size mismatches ({len(size_errors)}):")
    for e in size_errors[:5]:
        print(f"    {e}")
else:
    print(f"  ✅ All RGB/SEG sizes match!")

print(f"\n  Class distribution across training set:")
print(f"  {'Class':<20} {'Pixels':>14}  {'%':>6}  Bar")
for raw_id in ID_ORDER:
    cnt  = class_counts.get(raw_id, 0)
    pct  = 100.0 * cnt / total_pixels if total_pixels else 0
    bar  = "█" * int(pct / 2)
    name = CLASS_NAMES[raw_id]
    print(f"  {name:<20} {cnt:>14,}  {pct:>5.1f}%  {bar}")

# ── Visual ────────────────────────────────────────────────────────────────────
if rgb_files:
    sample_rgb  = np.array(Image.open(
        os.path.join(TRAIN_RGB, rgb_files[0])).convert("RGB"))
    sample_mask = np.array(Image.open(
        os.path.join(TRAIN_SEG, rgb_files[0])))

    colour_vis = np.zeros((*sample_mask.shape, 3), dtype=np.uint8)
    for raw_id, colour in COLOR_MAP.items():
        colour_vis[sample_mask == raw_id] = colour

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(sample_rgb)
    axes[0].set_title("RGB Image", fontsize=12); axes[0].axis("off")
    axes[1].imshow(sample_mask, cmap="tab20")
    axes[1].set_title("Raw Mask (pixel IDs)", fontsize=12); axes[1].axis("off")
    axes[2].imshow(colour_vis)
    axes[2].set_title("Colour-coded Mask", fontsize=12); axes[2].axis("off")

    patches = [mpatches.Patch(
        color=np.array(COLOR_MAP[i]) / 255.0,
        label=CLASS_NAMES[i]) for i in ID_ORDER if i in class_counts]
    axes[2].legend(handles=patches, loc="lower right",
                   fontsize=7, framealpha=0.85, ncol=2)

    plt.tight_layout()
    plt.savefig("data_check.png", dpi=150)
    plt.close()
    print(f"\n  ✅ Visual saved → data_check.png")

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(11, 4))
    names_list   = [CLASS_NAMES[i] for i in ID_ORDER]
    values_list  = [100.0 * class_counts.get(i, 0) / total_pixels for i in ID_ORDER]
    colours_list = [np.array(COLOR_MAP[i]) / 255.0 for i in ID_ORDER]
    ax.bar(names_list, values_list, color=colours_list, edgecolor="white")
    ax.set_title("Class Pixel Distribution — Training Set", fontsize=13)
    ax.set_ylabel("% of total pixels")
    ax.set_ylim(0, max(values_list) * 1.2 if values_list else 1)
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(values_list):
        if v > 0.2:
            ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.close()
    print(f"  📊 Class distribution → class_distribution.png")

print(f"\n  Next step → python train.py")
