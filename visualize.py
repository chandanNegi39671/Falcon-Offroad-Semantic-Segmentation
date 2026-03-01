"""
Duality AI — Segmentation Visualizer
=====================================
Run:  python visualize.py          (first 5 images)
      python visualize.py --n 20   (first 20 images)
      python visualize.py --n 999  (all images)

Outputs for each image:
  runs/visualizations/*_viz.png         — RGB | Mask | Overlay + class bar
  runs/visualizations/_summary_mosaic.png
  runs/visualizations/_class_distribution.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--rgb-dir",  default="test/Color_Images")
parser.add_argument("--pred-dir", default="runs/predictions")
parser.add_argument("--out-dir",  default="runs/visualizations")
parser.add_argument("--n",        type=int, default=5)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky",
]
COLOR_MAP = np.array([
    [ 34, 139,  34], [  0, 200, 100], [210, 180, 140],
    [139,  90,  43], [169, 169, 169], [255, 215,   0],
    [101,  67,  33], [128, 128, 128], [194, 178, 128],
    [135, 206, 235],
], dtype=np.uint8)
COLORS_NORM = COLOR_MAP / 255.0


def class_dist(pred_colour):
    total = pred_colour.shape[0] * pred_colour.shape[1]
    dist  = np.zeros(len(CLASS_NAMES))
    for i, c in enumerate(COLOR_MAP):
        dist[i] = np.all(pred_colour == c, axis=-1).sum() / total
    return dist


rgb_files = sorted([
    f for f in os.listdir(args.rgb_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:args.n]

if not rgb_files:
    print(f"No images found in {args.rgb_dir}")
    exit(1)

all_dists    = []
saved_paths  = []

for fname in rgb_files:
    pred_name = os.path.splitext(fname)[0] + "_pred.png"
    rgb_path  = os.path.join(args.rgb_dir,  fname)
    pred_path = os.path.join(args.pred_dir, pred_name)

    if not os.path.exists(pred_path):
        print(f"  ⚠️  Prediction missing: {pred_name} — run test.py first")
        continue

    rgb     = np.array(Image.open(rgb_path).convert("RGB"))
    pred    = np.array(Image.open(pred_path).convert("RGB"))
    overlay = (0.55 * rgb.astype(np.float32) +
               0.45 * pred.astype(np.float32)).clip(0, 255).astype(np.uint8)
    dist    = class_dist(pred)
    all_dists.append(dist)

    # ── Main figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 7), facecolor="#111827")
    gs  = fig.add_gridspec(2, 3, height_ratios=[5, 1.4],
                            hspace=0.08, wspace=0.05)

    for col, (img, title) in enumerate(
            [(rgb, "Original RGB"),
             (pred, "Predicted Mask"),
             (overlay, "Overlay")]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=13, color="white", pad=7)
        ax.axis("off")

    # Legend row
    ax_leg = fig.add_subplot(gs[1, :2])
    ax_leg.set_facecolor("#111827"); ax_leg.axis("off")
    patches = [mpatches.Patch(facecolor=COLORS_NORM[i],
                              edgecolor="#111827", label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    ax_leg.legend(handles=patches, loc="center", ncol=5,
                  fontsize=9, framealpha=0.0, labelcolor="white")

    # Class bar
    ax_bar = fig.add_subplot(gs[1, 2])
    ax_bar.set_facecolor("#111827")
    ax_bar.barh(CLASS_NAMES, dist * 100,
                color=COLORS_NORM, edgecolor="#111827", height=0.8)
    ax_bar.set_xlim(0, 100)
    ax_bar.tick_params(axis="x", colors="#6b7280", labelsize=7)
    ax_bar.tick_params(axis="y", colors="white",   labelsize=7)
    ax_bar.set_xlabel("% pixels", color="#6b7280", fontsize=8)
    for spine in ax_bar.spines.values():
        spine.set_color("#1f2937")

    fig.suptitle(fname, fontsize=11, color="#9ca3af", y=0.98)

    save_path = os.path.join(
        args.out_dir, os.path.splitext(fname)[0] + "_viz.png")
    plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor="#111827")
    plt.close()
    print(f"  ✅ {save_path}")
    saved_paths.append(save_path)

# ── Summary mosaic ────────────────────────────────────────────────────────────
if len(saved_paths) >= 2:
    cols    = min(3, len(saved_paths))
    rows    = (len(saved_paths) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    axs     = np.array(ax).flatten() if rows * cols > 1 else [ax]
    for i, p in enumerate(saved_paths):
        axs[i].imshow(np.array(Image.open(p)))
        axs[i].set_title(os.path.basename(p)[:30], fontsize=7)
        axs[i].axis("off")
    for j in range(len(saved_paths), len(axs)):
        axs[j].axis("off")
    plt.suptitle("Prediction Summary", fontsize=13)
    plt.tight_layout()
    mosaic = os.path.join(args.out_dir, "_summary_mosaic.png")
    plt.savefig(mosaic, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\n  📋 Summary mosaic → {mosaic}")

# ── Aggregate class distribution ─────────────────────────────────────────────
if all_dists:
    mean_dist = np.mean(all_dists, axis=0)
    fig, ax   = plt.subplots(figsize=(11, 4))
    ax.set_facecolor("#f8f9fa")
    bars = ax.bar(CLASS_NAMES, mean_dist * 100,
                  color=COLORS_NORM, edgecolor="white", width=0.8)
    ax.set_title("Average Class Distribution — Test Images", fontsize=13)
    ax.set_ylabel("% pixels (mean)")
    ax.set_ylim(0, max(mean_dist) * 115 if max(mean_dist) > 0 else 1)
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, mean_dist):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val*100:.1f}%", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    dist_chart = os.path.join(args.out_dir, "_class_distribution.png")
    plt.savefig(dist_chart, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  📊 Class distribution → {dist_chart}")

print(f"\n  ✅ All visualizations saved → {args.out_dir}/")
