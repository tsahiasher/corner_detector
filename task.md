## 1) Network 1 — Coarse corner detector

**Purpose:** quickly find approximate 4 corners on the full image

* Architecture: tiny CNN (MobileNet-style, depthwise separable convs)
* Input: ~384×384 image
* Output: 4 corner coordinates

**Size:**

* ~0.5M – 1M parameters (very small)
* Runs fast because it downsamples aggressively

---

## 2) Network 2 — Corner refiner (shared)

**Purpose:** get **sub-pixel accurate corner** from small patches

* Architecture: very small U-Net–like CNN
* Input: 64×64 patch (one corner at a time)
* Output: heatmap + offset

**Size:**

* ~0.2M – 0.5M parameters
* Same network reused 4 times (not 4 separate models)

---

## Total model size

* Combined: **~0.7M – 1.5M parameters**
* This is **orders of magnitude smaller** than ResNet50 (~25M)


## Scripts
python coarse\train.py --epochs 50 --batch_size 480

python coarse\test.py --weights coarse\runs\20260318_120441\checkpoints\best.pt

python coarse\export_torchscript.py --weights coarse\runs\20260318_120441\checkpoints\best.pt --output coarse\runs\20260318_120441\checkpoints\coarse_quad_net.pt

python coarse\run_torchscript_image.py --model coarse\runs\20260318_120441\checkpoints\coarse_quad_net.pt --image your_id_card.jpg

