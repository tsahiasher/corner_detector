## 1) Stage 1 — BoundingBox Corner Detector

**Purpose:** Rapidly localize 4 corners on the full image.

* **Architecture**: YOLO-Pose style (ResNet-18 + FPN + Dense Head)
* **Input**: 384×384 image
* **Output**: Bounding box + 4 corners + Confidence scores
* **Size**: ~11.5M parameters

---

## 2) Stage 1.5 — Orientation Classifier (OrientNet)

**Purpose**: Classify card rotation (0/90/180/270) to ensure consistent physical corner identity.

* **Architecture**: MobileNet-style classifier
* **Input**: 128×128 rectified card crop
* **Output**: Rotation class (4-way)
* **Size**: ~0.05M parameters

---

## 3) Stage 2 — Corner Refiner (Iterative Refiner)

**Purpose**: Achieve sub-pixel accuracy using differentiable zoom.

* **Architecture**: Two-stage iterative network (Global + Local Zoom + Fine Head)
* **Input**: 96×96 corner patches
* **Output**: High-precision sub-pixel offsets via `SoftArgmax2D`
* **Size**: ~0.5M parameters
* Same network reused 4 times (one per corner)

---

## Training Commands

### Stage 1: BoundingBox
```bash
python boundingbox/train.py --epochs 100 --batch_size 32 --mine_hard
python boundingbox/test.py --weights boundingbox/runs/latest/checkpoints/best.pt
python boundingbox/export_torchscript.py --weights boundingbox/runs/latest/checkpoints/best.pt
```

### Stage 1.5: Orientation
```bash
python orient/train.py --epochs 30 --batch_size 64
python orient/test.py --weights orient/runs/latest/checkpoints/best.pt
```

### Stage 2: Refiner
```bash
python refiner/train.py --epochs 50 --batch_size 128
python refiner/test.py --weights refiner/runs/latest/checkpoints/best.pt
```

### Full Pipeline Inference
```bash
python run_torchscript_image.py \
    --boundingbox_model boundingbox/runs/latest/checkpoints/boundingbox_quad_net.pt \
    --refiner_model refiner/runs/latest/checkpoints/patch_refiner.pt \
    --input your_image.jpg
```
