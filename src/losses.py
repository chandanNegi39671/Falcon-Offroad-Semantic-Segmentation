"""
src/losses.py
=============
CombinedLoss = FocalLoss(gamma=2) + DiceLoss

Weights computed from YOUR real dataset (2857 images, check_data.py output):
  Logs=0.1% of pixels → weight 4.440  (boosted 22x vs Sky)
  Sky=37.6% of pixels → weight 0.203  (down-weighted)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Pre-computed from your 2857-image dataset
DATASET_CLASS_WEIGHTS = torch.tensor([
    0.660,   # 0  Trees           3.5%
    0.510,   # 1  Lush Bushes     5.9%
    0.285,   # 2  Dry Grass      18.9%
    1.185,   # 3  Dry Bushes      1.1%
    0.592,   # 4  Ground Clutter  4.4%
    0.740,   # 5  Flowers         2.8%
    7.0,   # 6  Logs            0.1%  <- 22x heavier than Sky
    1.8,   # 7  Rocks           1.2%
    0.251,   # 8  Landscape      24.4%
    0.203,   # 9  Sky            37.6%
], dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=2 focuses gradient on hard/misclassified pixels.
    """

    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        weight = self.weight
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)

        ce_loss = F.cross_entropy(
            logits, targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        p_t        = torch.exp(-ce_loss)
        focal_loss = (1.0 - p_t) ** self.gamma * ce_loss

        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index)
            return focal_loss[mask].mean() if mask.any() else focal_loss.sum() * 0
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.
    ignore_empty=True skips classes absent from the batch —
    prevents NaN when Logs or Rocks don't appear in a mini-batch.
    """

    def __init__(self, num_classes=10, smooth=1.0, ignore_empty=True):
        super().__init__()
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_empty = ignore_empty

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        probs      = F.softmax(logits, dim=1)

        targets_oh = (
            F.one_hot(targets.clamp(0, C - 1), C)
            .permute(0, 3, 1, 2)
            .float()
        )

        dice_scores = []
        for c in range(C):
            p = probs[:, c]
            t = targets_oh[:, c]

            if self.ignore_empty and t.sum() == 0:
                continue

            intersection = (p * t).sum(dim=(1, 2))
            union        = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice_c       = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_c.mean())

        if not dice_scores:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return 1.0 - torch.stack(dice_scores).mean()


class CombinedLoss(nn.Module):
    """
    CombinedLoss = focal_weight * FocalLoss + dice_weight * DiceLoss

    Usage in train.py:
        from src.losses import CombinedLoss, DATASET_CLASS_WEIGHTS
        criterion = CombinedLoss(
            num_classes  = NUM_CLASSES,
            ce_weight    = DATASET_CLASS_WEIGHTS.to(DEVICE),
            use_focal    = True,
        )
    """

    def __init__(
        self,
        num_classes  = 10,
        ce_weight    = None,
        gamma        = 2.0,
        focal_weight = 1.0,
        dice_weight  = 0.5,
        ignore_index = -100,
        use_focal    = True,
    ):
        super().__init__()
        self.focal_w = focal_weight
        self.dice_w  = dice_weight
        self.dice    = DiceLoss(num_classes=num_classes, ignore_empty=True)

        if use_focal:
            self.ce = FocalLoss(
                gamma=gamma,
                weight=ce_weight,
                ignore_index=ignore_index,
            )
        else:
            self.ce = nn.CrossEntropyLoss(
                weight=ce_weight,
                ignore_index=ignore_index,
            )

    def forward(self, logits, targets):
        return (self.focal_w * self.ce(logits, targets) +
                self.dice_w  * self.dice(logits, targets))


if __name__ == "__main__":
    print("=" * 52)
    print("  src/losses.py  —  sanity check")
    print("=" * 52)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    logits  = torch.randn(2, 10, 64, 64, device=device, requires_grad=True)
    targets = torch.randint(0, 10, (2, 64, 64), device=device)

    criterion = CombinedLoss(
        num_classes = 10,
        ce_weight   = DATASET_CLASS_WEIGHTS.to(device),
        use_focal   = True,
    )
    loss = criterion(logits, targets)
    loss.backward()

    print(f"  CombinedLoss : {loss.item():.4f}")
    print(f"  Backward     : OK")
    print()

    names = ["Trees", "LushBushes", "DryGrass", "DryBushes", "GndClutter",
             "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
    for i, (n, w) in enumerate(zip(names, DATASET_CLASS_WEIGHTS)):
        bar = "█" * min(int(w * 10), 44)
        print(f"  {i}  {n:<18} {w:.3f}  {bar}")

    print()
    print("  losses.py is ready")
    print("=" * 52)
