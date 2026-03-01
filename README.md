# Duality AI — Offroad Semantic Scene Segmentation

DeepLabV3+ with **mit_b2** (MixTransformer) encoder for desert environment segmentation.  
Dataset: 2,857 train / 317 val / 1,002 test images.  
Target: **≥ 93% mIoU**

---

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── dataset.py       ← ID→index remapping (critical fix)
│   ├── transforms.py    ← albumentations augmentation pipeline
│   ├── losses.py        ← FocalLoss + DiceLoss + real class weights
│   └── metrics.py       ← streaming IoU metric
├── train.py             ← main training script
├── test.py              ← inference on test images
├── visualize.py         ← side-by-side result visualization
├── check_data.py        ← dataset validation
└── runs/
    ├── best_model.pth
    ├── last_model.pth
    └── training_graphs.png
```

---

## Setup

```bash
conda create -n EDU python=3.10 -y
conda activate EDU

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependencies
pip install segmentation-models-pytorch albumentations matplotlib tqdm pillow numpy
```

### Install mit_b2 encoder weights
```bash
pip install timm
```

---

## Execution Order

```bash
# 1. Validate dataset
python check_data.py

# 2. Train (mit_b2 encoder, 60 epochs)
python train.py

# 3. Run inference on test images
python test.py

# 4. Visualize results
python visualize.py --n 20
```

---

## Class Reference

| Index | Raw ID | Class          | Pixels (train) | % | Weight |
|-------|--------|----------------|----------------|---|--------|
| 0 | 100 | Trees | 52M | 3.5% | 0.660 |
| 1 | 200 | Lush Bushes | 87M | 5.9% | 0.510 |
| 2 | 300 | Dry Grass | 279M | 18.9% | 0.285 |
| 3 | 500 | Dry Bushes | 16M | 1.1% | 1.185 |
| 4 | 550 | Ground Clutter | 65M | 4.4% | 0.592 |
| 5 | 600 | Flowers | 41M | 2.8% | 0.740 |
| 6 | 700 | Logs | 1.1M | 0.1% | **4.440** |
| 7 | 800 | Rocks | 17M | 1.2% | 1.134 |
| 8 | 7100 | Landscape | 362M | 24.4% | 0.251 |
| 9 | 10000 | Sky | 557M | 37.6% | 0.203 |

> **Critical:** Mask pixel values (100, 200 … 10000) are remapped to 0-9 indices
> inside `src/dataset.py` via a fast lookup table before any training step.

---

## Why mit_b2 Over ResNet50

| Encoder | Expected mIoU | Params | Notes |
|---|---|---|---|
| resnet50 | ~68-72% | 26.7M | Good baseline |
| **mit_b2** | **~85-93%** | **25M** | Hierarchical Vision Transformer, better at multi-scale desert features |
| mit_b4 | ~90-95% | 62M | More VRAM needed |

---

## Submission Checklist

- [ ] `runs/best_model.pth`
- [ ] `train.py`, `test.py`, `visualize.py`, `check_data.py`
- [ ] `src/` folder (all 4 modules)
- [ ] `README.md`
- [ ] `runs/training_graphs.png`
- [ ] Performance report PDF (max 8 pages)
- [ ] ZIP → private GitHub repo
- [ ] Add collaborators: `Maazsyedm`, `rebekah-bogdanoff`, `egold010`
