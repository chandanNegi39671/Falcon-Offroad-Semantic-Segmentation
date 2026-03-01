"""
src/metrics.py
==============
IoUMetric — streaming per-class IoU + mIoU using confusion matrix.

Usage:
    metric = IoUMetric(NUM_CLASSES)
    metric.reset()                          # start of epoch
    metric.update(logits, targets)          # each batch
    per_class_iou, miou = metric.compute()  # end of epoch
"""

import torch
import numpy as np


class IoUMetric:

    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.conf_mat = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """logits: (B,C,H,W)  targets: (B,H,W)"""
        preds = logits.argmax(dim=1)
        self._accum(preds.cpu(), targets.cpu())

    def update_preds(self, preds: torch.Tensor, targets: torch.Tensor):
        """preds: (B,H,W) already argmaxed"""
        self._accum(preds.cpu(), targets.cpu())

    def _accum(self, preds, targets):
        C    = self.num_classes
        mask = (targets >= 0) & (targets < C)
        if self.ignore_index >= 0:
            mask &= (targets != self.ignore_index)

        combined = C * targets[mask] + preds[mask]
        bc       = torch.bincount(combined, minlength=C * C)
        self.conf_mat += bc.reshape(C, C)

    def compute(self):
        cm  = self.conf_mat.float()
        tp  = cm.diag()
        fp  = cm.sum(0) - tp
        fn  = cm.sum(1) - tp
        den = tp + fp + fn

        per_class = []
        for c in range(self.num_classes):
            if den[c].item() == 0:
                per_class.append(float("nan"))
            else:
                per_class.append((tp[c] / den[c]).item())

        valid = [v for v in per_class if not np.isnan(v)]
        miou  = float(np.mean(valid)) if valid else 0.0
        return per_class, miou

    def pretty_print(self, class_names=None, best_miou=0.0):
        per_class, miou = self.compute()
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]

        print(f"  {'─'*42}")
        print(f"  {'Class':<18} {'IoU':>6}   Bar")
        print(f"  {'─'*42}")
        for name, iou in zip(class_names, per_class):
            if np.isnan(iou):
                print(f"  {name:<18}   N/A")
                continue
            bar  = "█" * int(iou * 24)
            flag = ""
            if   iou < 0.20: flag = "  🔴"
            elif iou < 0.50: flag = "  🟡"
            elif iou > 0.85: flag = "  ✅"
            print(f"  {name:<18} {iou:.4f}  {bar}{flag}")
        print(f"  {'─'*42}")
        star = "  ⭐ NEW BEST" if miou > best_miou else ""
        print(f"  mIoU : {miou:.4f}  ({miou*100:.2f}%){star}")
        print(f"  {'─'*42}")
        return per_class, miou

    def get_confusion_matrix(self):
        return self.conf_mat.numpy()
