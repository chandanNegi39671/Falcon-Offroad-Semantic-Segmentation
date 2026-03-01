"""
Duality AI — Offroad Segmentation
Main Training Script — mit_b2 encoder
======================================
Run:  python train.py

Changes vs previous version:
  • Encoder: resnet50 → mit_b2  (+8-12% mIoU expected)
  • FutureWarnings fixed (torch.amp instead of torch.cuda.amp)
  • Differential LR: encoder 5x lower than decoder for transformers
  • LR default lowered to 6e-5 (better for ViT-style encoders)
  • Tensor copy warning fixed in dataset
  • Early stopping patience = 15 (transformers need more epochs)
  • Rocks class weight boosted 1.8 → 3.5 (worst-performing class)
  • ShiftScaleRotate → Affine + albumentations 2.x API fixes (no warnings)
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.dataset    import DesertDataset, CLASS_NAMES, NUM_CLASSES
from src.transforms import get_train_transforms, get_val_transforms
from src.losses     import CombinedLoss, DATASET_CLASS_WEIGHTS
from src.metrics    import IoUMetric

# ── Weight override: boost Rocks (index 7) — worst-performing class ───────────
# Default 1.8 → 3.5 based on observed IoU ~0.19-0.26 in validation
DATASET_CLASS_WEIGHTS = DATASET_CLASS_WEIGHTS.clone()
DATASET_CLASS_WEIGHTS[7] = 3.5   # Rocks

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train desert segmentation — mit_b2")
parser.add_argument("--train-rgb",  default="train/Color_Images")
parser.add_argument("--train-seg",  default="train/Segmentation")
parser.add_argument("--val-rgb",    default="val/Color_Images")
parser.add_argument("--val-seg",    default="val/Segmentation")
parser.add_argument("--epochs",     type=int,   default=60)
parser.add_argument("--batch-size", type=int,   default=4)
parser.add_argument("--img-size",   type=int,   default=512)
parser.add_argument("--lr",         type=float, default=6e-5)
parser.add_argument("--encoder",    default="mit_b2",
                    help="mit_b2 | mit_b4 | efficientnet-b4 | resnet50")
parser.add_argument("--save-dir",   default="runs")
parser.add_argument("--resume", default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--patience",   type=int,   default=15)
parser.add_argument("--workers",    type=int,   default=0)
args = parser.parse_args()

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")
os.makedirs(args.save_dir, exist_ok=True)

print("=" * 62)
print("  Duality AI — Desert Segmentation  [mit_b2 encoder]")
print("=" * 62)
print(f"  Device      : {DEVICE}  |  AMP = {USE_AMP}")
print(f"  Encoder     : {args.encoder}")
print(f"  Epochs      : {args.epochs}  |  Patience : {args.patience}")
print(f"  Batch size  : {args.batch_size}  |  Img size : {args.img_size}x{args.img_size}")
print(f"  LR          : {args.lr}")
print(f"\n  Class weights (rare classes boosted):")
names_short = ["Trees","LushBushes","DryGrass","DryBushes","GndClutter",
               "Flowers","Logs","Rocks","Landscape","Sky"]
for i, (n, w) in enumerate(zip(names_short, DATASET_CLASS_WEIGHTS)):
    bar = "▓" * min(int(w * 6), 30)
    print(f"    {i}  {n:<18} {w:.3f}  {bar}")
print("=" * 62)

# ── DATA ──────────────────────────────────────────────────────────────────────
train_dataset = DesertDataset(
    args.train_rgb, args.train_seg, get_train_transforms(args.img_size))
val_dataset   = DesertDataset(
    args.val_rgb,   args.val_seg,   get_val_transforms(args.img_size))

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True,  num_workers=args.workers,
    pin_memory=(DEVICE == "cuda"), drop_last=True)

val_loader = DataLoader(
    val_dataset, batch_size=2,
    shuffle=False, num_workers=args.workers)

print(f"\n  Train : {len(train_dataset):,} images  |  Val : {len(val_dataset):,} images")

# ── MODEL ─────────────────────────────────────────────────────────────────────
model = smp.DeepLabV3Plus(
    encoder_name    = args.encoder,
    encoder_weights = "imagenet",
    in_channels     = 3,
    classes         = NUM_CLASSES,
)
model = model.to(DEVICE)
if args.resume and os.path.exists(args.resume):
    model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
    print(f"  ✅ Resumed from: {args.resume}")

n_params   = sum(p.numel() for p in model.parameters()) / 1e6
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"  Model : DeepLabV3+ ({args.encoder})")
print(f"          {n_params:.1f}M total params  |  {n_trainable:.1f}M trainable\n")

# ── LOSS ──────────────────────────────────────────────────────────────────────
criterion = CombinedLoss(
    num_classes  = NUM_CLASSES,
    ce_weight    = DATASET_CLASS_WEIGHTS.to(DEVICE),
    gamma        = 2.0,
    focal_weight = 1.0,
    dice_weight  = 0.5,
    use_focal    = True,
)

# ── OPTIMISER — differential LR ───────────────────────────────────────────────
# Transformer encoders (mit_b*) need lower LR on encoder to preserve pretrained
# features. Decoder and head train at full LR.
optimizer = AdamW([
    {"params": model.encoder.parameters(),          "lr": args.lr * 0.2},
    {"params": model.decoder.parameters(),          "lr": args.lr},
    {"params": model.segmentation_head.parameters(),"lr": args.lr},
], weight_decay=1e-4)

# Cosine warm restarts — escapes local minima. T_0=20 for transformer encoders.
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6)

# ── AMP — FutureWarning fixed ─────────────────────────────────────────────────
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
metric = IoUMetric(NUM_CLASSES)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
best_miou     = 0.0
no_improve    = 0
train_losses  = []
val_mious     = []
val_per_class = []
start_time    = time.time()

print("  Starting training...\n")

for epoch in range(1, args.epochs + 1):

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    t0         = time.time()

    loop = tqdm(train_loader,
                desc=f"Ep {epoch:03d}/{args.epochs} [Train]",
                leave=False, ncols=88)

    for images, masks in loop:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # FutureWarning fix: use torch.amp instead of torch.cuda.amp
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = nn.functional.interpolate(
                    outputs, size=masks.shape[-2:],
                    mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    metric.reset()

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE,  non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                outputs = model(images)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = nn.functional.interpolate(
                        outputs, size=masks.shape[-2:],
                        mode="bilinear", align_corners=False)
            metric.update(outputs, masks)

    scheduler.step(epoch)

    # ── Print results ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\nEpoch {epoch:03d}/{args.epochs}  |  "
          f"Loss: {avg_loss:.4f}  |  Time: {elapsed:.0f}s")

    per_class_iou, miou = metric.pretty_print(CLASS_NAMES, best_miou)
    val_mious.append(miou)
    val_per_class.append(per_class_iou)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, "last_model.pth"))

    if miou > best_miou:
        best_miou  = miou
        no_improve = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, "best_model.pth"))
        print(f"  ✅ New best!  mIoU = {miou*100:.2f}%  → saved best_model.pth\n")
    else:
        no_improve += 1
        print(f"  ⏳ No improvement for {no_improve}/{args.patience} epochs\n")
        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

# ── SUMMARY ───────────────────────────────────────────────────────────────────
total_min = (time.time() - start_time) / 60
print(f"\n{'='*62}")
print(f"  Training complete  |  {total_min:.1f} min  |  Best mIoU: {best_miou*100:.2f}%")
print(f"{'='*62}")

# ── SAVE GRAPHS ───────────────────────────────────────────────────────────────
epochs_ran = len(train_losses)
fig, axes  = plt.subplots(1, 3, figsize=(19, 5))
fig.suptitle(
    f"Duality AI — DeepLabV3+ ({args.encoder}) | Best mIoU: {best_miou*100:.2f}%",
    fontsize=13, fontweight="bold")

# Loss
axes[0].plot(range(1, epochs_ran + 1), train_losses,
             color="#ef4444", linewidth=2)
axes[0].set_title("Training Loss", fontsize=12)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# mIoU
axes[1].plot(range(1, epochs_ran + 1), [v * 100 for v in val_mious],
             color="#3b82f6", linewidth=2)
axes[1].axhline(y=93, color="#22c55e", linestyle="--",
                alpha=0.7, label="Target 93%")
axes[1].set_title("Validation mIoU", fontsize=12)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU (%)")
axes[1].set_ylim(0, 100)
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Per-class IoU bar (final epoch)
colors  = ["#228B22","#00C864","#D2B48C","#8B5A2B","#A9A9A9",
           "#FFD700","#654321","#808080","#C2B280","#87CEEB"]
last_pc = [v if not np.isnan(v) else 0.0 for v in val_per_class[-1]]
bars = axes[2].barh(
    CLASS_NAMES, [v * 100 for v in last_pc],
    color=colors, edgecolor="white", height=0.72)
axes[2].set_xlim(0, 100)
axes[2].axvline(x=93, color="#22c55e", linestyle="--",
                alpha=0.7, label="Target 93%")
axes[2].set_title("Per-Class IoU (final epoch)", fontsize=12)
axes[2].set_xlabel("IoU (%)")
for bar, val in zip(bars, last_pc):
    if val > 1:
        axes[2].text(bar.get_width() + 0.8,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val*100:.1f}%", va="center", fontsize=8)
axes[2].grid(True, axis="x", alpha=0.3)

plt.tight_layout()
graph_path = os.path.join(args.save_dir, "training_graphs.png")
plt.savefig(graph_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  📊 Graphs saved → {graph_path}")
print(f"  🏆 Best model  → {os.path.join(args.save_dir, 'best_model.pth')}")
print(f"\n  Next step → python test.py")
