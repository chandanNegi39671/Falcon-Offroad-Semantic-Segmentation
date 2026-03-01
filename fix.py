"""
fix.py
======
Utility script — run this ONCE if you have any of these issues:

  1. Wrong data paths  (data/train instead of train, etc.)
  2. torch.cuda.amp FutureWarnings still appearing
  3. Tensor copy UserWarning in dataset.py

Run:  python fix.py
"""

import os
import re

# ── Files to patch ────────────────────────────────────────────────────────────
TARGET_FILES = [
    "train.py",
    "test.py",
    "check_data.py",
    "visualize.py",
    "src/dataset.py",
    "src/losses.py",
    "src/transforms.py",
    "src/metrics.py",
]

# ── Fix 1: Wrong data paths ───────────────────────────────────────────────────
PATH_FIXES = [
    ("data/train", "train"),
    ("data/val",   "val"),
    ("data/test",  "test"),
]

# ── Fix 2: torch.cuda.amp FutureWarnings ─────────────────────────────────────
AMP_FIXES = [
    # GradScaler
    (
        r'torch\.cuda\.amp\.GradScaler\(',
        'torch.amp.GradScaler("cuda", '
    ),
    # autocast
    (
        r'torch\.cuda\.amp\.autocast\(',
        'torch.amp.autocast("cuda", '
    ),
    # import style — amp.GradScaler(enabled=...)
    (
        r'amp\.GradScaler\(enabled=',
        'torch.amp.GradScaler("cuda", enabled='
    ),
    # import style — amp.autocast(enabled=...)
    (
        r'amp\.autocast\(enabled=',
        'torch.amp.autocast("cuda", enabled='
    ),
]

# ── Fix 3: Tensor copy UserWarning in dataset ─────────────────────────────────
# torch.tensor(mask) → torch.from_numpy(mask).clone()
TENSOR_FIXES = [
    (
        r'torch\.tensor\(mask,\s*dtype=torch\.long\)',
        'torch.from_numpy(mask.copy()).long()'
    ),
    (
        r'torch\.tensor\(mask\)',
        'torch.from_numpy(mask.copy()).long()'
    ),
]

# ── Fix 4: Encoder name consistency ───────────────────────────────────────────
# Make sure encoder is mit_b2 everywhere (not resnet50 leftover)
ENCODER_FIXES = [
    # Only fix if it's the default string value — don't touch argparse choices
    (
        'default="resnet50"',
        'default="mit_b2"'
    ),
]


def apply_fixes(filepath, fixes, use_regex=False):
    if not os.path.exists(filepath):
        return False, "file not found"

    with open(filepath, "r", encoding="utf-8") as f:
        original = f.read()

    content = original
    changes = []

    for old, new in fixes:
        if use_regex:
            new_content = re.sub(old, new, content)
        else:
            new_content = content.replace(old, new)

        if new_content != content:
            changes.append(old if not use_regex else old[:40])
            content = new_content

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True, changes

    return False, []


# ── Run all fixes ─────────────────────────────────────────────────────────────
print("=" * 56)
print("  fix.py — patching all project files")
print("=" * 56)

total_changed = 0

for fpath in TARGET_FILES:
    file_changed = False
    file_log     = []

    # Path fixes
    changed, what = apply_fixes(fpath, PATH_FIXES, use_regex=False)
    if changed:
        file_changed = True
        file_log.append(f"paths: {what}")

    # AMP fixes
    changed, what = apply_fixes(fpath, AMP_FIXES, use_regex=True)
    if changed:
        file_changed = True
        file_log.append(f"AMP warnings: fixed")

    # Tensor fixes (mostly dataset.py)
    changed, what = apply_fixes(fpath, TENSOR_FIXES, use_regex=True)
    if changed:
        file_changed = True
        file_log.append(f"tensor copy: fixed")

    # Encoder fix (train.py and test.py defaults)
    changed, what = apply_fixes(fpath, ENCODER_FIXES, use_regex=False)
    if changed:
        file_changed = True
        file_log.append(f"encoder default: resnet50 → mit_b2")

    if file_changed:
        total_changed += 1
        print(f"  ✅ {fpath}")
        for log in file_log:
            print(f"       └─ {log}")
    else:
        print(f"  ─  {fpath}  (nothing to fix)")

print()
if total_changed > 0:
    print(f"  {total_changed} file(s) patched successfully.")
else:
    print("  All files already clean — nothing needed fixing.")

print()
print("  Fixes applied:")
print("   [1] data/train → train  |  data/val → val  |  data/test → test")
print("   [2] torch.cuda.amp FutureWarnings → torch.amp")
print("   [3] torch.tensor(mask) UserWarning → torch.from_numpy")
print("   [4] Default encoder: resnet50 → mit_b2")
print()
print("  Next step → python train.py")
print("=" * 56)
